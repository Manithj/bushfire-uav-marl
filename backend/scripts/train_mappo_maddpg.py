"""Train MAPPO and MADDPG to fly to fires and extinguish them.

The previous short run learned detection/approach but not the hover
that finishes suppression. This trainer:

1. Collects greedy expert trajectories with known active fires.
2. Oversamples near-fire HOVER so the clone stays on the front.
3. Behavioral-clones until holdout extinguishes (measured, not invented).
4. Fine-tunes MAPPO and MADDPG with a BC regularizer so they do not
   forget suppression. Best checkpoint is the one with the highest
   holdout extinguish count.

Inference uses the same actor as training. A last-metre hover lock
(if a known fire is already inside suppression range) matches greedy
and is applied at eval so the saved metric is what the UI will do.
"""

from __future__ import annotations

import argparse
import copy
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.marl.inference import ActorNet
from app.marl.observation import OBS_DIM
from app.schemas.scenario import (
    CommunicationConfig,
    EnvironmentConfig,
    FireSourceConfig,
    LocationConfig,
    PolicyConfig,
    ScenarioConfig,
    SensorConfig,
    UAVFleetConfig,
)
from app.simulation.environment import WildfireEnv

N_ACTIONS = 9
HIDDEN = 128
HOVER = 8
HOVER_NORM = 0.012
N_UAVS = 4
DURATION_S = 220.0


class CentralCritic(nn.Module):
    def __init__(self, n_agents: int, obs_dim: int = OBS_DIM, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim * n_agents, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class CentralQ(nn.Module):
    def __init__(self, n_agents: int, obs_dim: int = OBS_DIM, n_actions: int = N_ACTIONS, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_agents * (obs_dim + n_actions), hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs_cat: torch.Tensor, act_cat: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs_cat, act_cat], dim=-1)).squeeze(-1)


def make_scenario(seed: int, allow_global_state: bool = True, n_fires: int = 4) -> ScenarioConfig:
    rng = np.random.default_rng(seed)
    fires = []
    for i in range(n_fires):
        fires.append(
            FireSourceConfig(
                fire_id=f"fire_{i+1:02d}",
                latitude=float(rng.uniform(-37.538, -37.502)),
                longitude=float(rng.uniform(145.275, 145.345)),
                initial_intensity=0.8,
                ignition_time=0.0 if i < 2 else float(rng.uniform(0.0, 12.0)),
                radius_m=70.0,
            )
        )
    return ScenarioConfig(
        name="suppress-train",
        location=LocationConfig(region_id="kinglake"),
        fires=fires,
        environment=EnvironmentConfig(
            duration_s=DURATION_S,
            timestep_s=1.0,
            fire_cell_size_m=80.0,
            coverage_cell_size_m=120.0,
            wind_speed_kmh=14.0,
            wind_from="NW",
            base_ros_mps=0.22,
        ),
        uavs=UAVFleetConfig(count=N_UAVS, cruise_speed_mps=16.0, altitude_m=120.0),
        sensor=SensorConfig(
            sensor_range_m=800.0,
            suppression_range_m=120.0,
            false_positive_probability=0.0,
        ),
        communication=CommunicationConfig(range_m=4500.0),
        policy=PolicyConfig(name="greedy", allow_global_state=allow_global_state),
        seed=seed,
    )


def _stats(env: WildfireEnv) -> dict:
    assert env.fire is not None
    return {
        "detected": sum(1 for f in env.fire.sources if f.detected),
        "extinguished": sum(1 for f in env.fire.sources if f.suppressed),
        "total": len(env.fire.sources),
        "coverage": env.coverage.coverage_fraction() if env.coverage else 0.0,
        "reward": env.metrics.episode_reward,
        "cells": sum(u.cells_extinguished for u in env.uavs),
    }


def _near_fire(vec: np.ndarray) -> bool:
    for i in range(3):
        dx = float(vec[11 + 3 * i])
        dy = float(vec[11 + 3 * i + 1])
        inten = float(vec[11 + 3 * i + 2])
        if inten > 1e-6 and float(np.hypot(dx, dy)) <= HOVER_NORM:
            return True
    return False


def _has_fire(vec: np.ndarray) -> bool:
    return float(np.abs(vec[11:20]).sum()) > 1e-6


def _act_actor(actor: ActorNet, vec: np.ndarray, hover_lock: bool) -> int:
    if hover_lock and _near_fire(vec):
        return HOVER
    x = torch.from_numpy(np.asarray(vec, dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        return int(torch.argmax(actor(x), dim=-1).item())


def collect_expert(n_episodes: int, seed0: int) -> tuple[np.ndarray, np.ndarray]:
    obs_list: list[np.ndarray] = []
    act_list: list[int] = []
    for ep in range(n_episodes):
        sc = make_scenario(seed0 + ep, allow_global_state=True)
        env = WildfireEnv(sc)
        obs, _ = env.reset(seed=sc.seed)
        while not env._terminated:
            states = {u.uav_id: {"agent_index": i, "n_agents": len(env.uavs)} for i, u in enumerate(env.uavs)}
            assert env.policy is not None
            actions = env.policy.act_many(obs, states)
            for aid, vec in obs.items():
                act = int(actions[aid])
                if act == HOVER and not _has_fire(vec):
                    continue
                copies = 5 if (act == HOVER and _near_fire(vec)) else 1
                if _has_fire(vec) and act != HOVER:
                    copies = 2
                for _ in range(copies):
                    obs_list.append(vec.astype(np.float32))
                    act_list.append(act)
            obs, *_ = env.step(actions)
        st = _stats(env)
        print(f"  expert {ep+1}/{n_episodes} det={st['detected']}/{st['total']} ext={st['extinguished']} cells={st['cells']}")
    return np.stack(obs_list), np.asarray(act_list, dtype=np.int64)


def behavioral_clone(actor: ActorNet, xs: np.ndarray, ys: np.ndarray, epochs: int, batch: int) -> float:
    opt = torch.optim.Adam(actor.parameters(), lr=8e-4)
    last = 0.0
    n = xs.shape[0]
    w = torch.ones(N_ACTIONS)
    w[HOVER] = 2.2
    for epoch in range(epochs):
        perm = np.random.permutation(n)
        losses = []
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            x = torch.from_numpy(xs[idx])
            y = torch.from_numpy(ys[idx])
            loss = F.cross_entropy(actor(x), y, weight=w)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
        last = float(np.mean(losses))
        print(f"  bc epoch {epoch+1}/{epochs} loss={last:.4f}")
    return last


def evaluate(actor: ActorNet, seed: int, hover_lock: bool = True) -> dict:
    sc = make_scenario(seed, allow_global_state=True)
    env = WildfireEnv(sc)
    env.reset(seed=seed)
    while not env._terminated:
        obs = env._observe_all()
        actions = {aid: _act_actor(actor, vec, hover_lock) for aid, vec in obs.items()}
        env.step(actions)
    return _stats(env)


def holdout(actor: ActorNet, seeds: list[int]) -> list[dict]:
    return [evaluate(actor, s) for s in seeds]


def _mean_ext(evals: list[dict]) -> float:
    return float(np.mean([e["extinguished"] for e in evals]))


def _print_evals(tag: str, seeds: list[int], evals: list[dict]) -> None:
    for s, ev in zip(seeds, evals):
        print(
            f"  {tag} seed {s}: det {ev['detected']}/{ev['total']} "
            f"ext={ev['extinguished']} cells={ev['cells']} R={ev['reward']:.1f}"
        )


def bc_until_extinguish(actor: ActorNet, xs: np.ndarray, ys: np.ndarray, seeds: list[int]) -> tuple[float, list[dict]]:
    loss = behavioral_clone(actor, xs, ys, epochs=12, batch=256)
    evals = holdout(actor, seeds)
    _print_evals("bc", seeds, evals)
    extra = 0
    while _mean_ext(evals) < 2.5 and extra < 4:
        extra += 1
        print(f"  bc retry {extra}: holdout ext={_mean_ext(evals):.2f} < 2.5")
        loss = behavioral_clone(actor, xs, ys, epochs=6, batch=256)
        evals = holdout(actor, seeds)
        _print_evals("bc", seeds, evals)
    return loss, evals


def _bc_batch_loss(actor: ActorNet, xs: np.ndarray, ys: np.ndarray, n: int = 256) -> torch.Tensor:
    idx = np.random.choice(len(ys), size=min(n, len(ys)), replace=False)
    x = torch.from_numpy(xs[idx])
    y = torch.from_numpy(ys[idx])
    w = torch.ones(N_ACTIONS)
    w[HOVER] = 2.0
    return F.cross_entropy(actor(x), y, weight=w)


def train_mappo(
    actor: ActorNet,
    xs: np.ndarray,
    ys: np.ndarray,
    episodes: int,
    seed0: int,
    holdout_seeds: list[int],
) -> tuple[ActorNet, list[dict]]:
    actor = copy.deepcopy(actor)
    critic = CentralCritic(n_agents=N_UAVS)
    opt_a = torch.optim.Adam(actor.parameters(), lr=8e-5)
    opt_c = torch.optim.Adam(critic.parameters(), lr=2e-4)
    best = {k: v.detach().clone() for k, v in actor.state_dict().items()}
    best_ext = _mean_ext(holdout(actor, holdout_seeds))
    print(f"  mappo start holdout ext={best_ext:.2f}")

    for ep in range(episodes):
        sc = make_scenario(seed0 + ep, allow_global_state=True)
        env = WildfireEnv(sc)
        obs, _ = env.reset(seed=sc.seed)
        logps: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        rewards: list[float] = []
        ents: list[torch.Tensor] = []
        teacher = 0.35 * max(0.0, 1.0 - ep / max(episodes * 0.6, 1))
        while not env._terminated:
            keys = list(obs.keys())
            batch = np.stack([obs[a] for a in keys], axis=0)
            x = torch.from_numpy(batch)
            logits = actor(x)
            dist = torch.distributions.Categorical(logits=logits)
            if np.random.random() < teacher:
                states = {u.uav_id: {"agent_index": i, "n_agents": len(env.uavs)} for i, u in enumerate(env.uavs)}
                assert env.policy is not None
                expert = env.policy.act_many(obs, states)
                acts = torch.tensor([int(expert[a]) for a in keys], dtype=torch.int64)
            else:
                acts = dist.sample()
            logps.append(dist.log_prob(acts).mean())
            ents.append(dist.entropy().mean())
            values.append(critic(x.reshape(1, -1)).squeeze())
            action = {aid: int(acts[i].item()) for i, aid in enumerate(keys)}
            obs, rew, *_ = env.step(action)
            rewards.append(float(np.mean([rew[k] for k in keys if k != "__all__"])))
        R = 0.0
        returns: list[float] = []
        for r in reversed(rewards):
            R = r + 0.99 * R
            returns.append(R)
        returns.reverse()
        tgt = torch.tensor(returns, dtype=torch.float32)
        vals = torch.stack(values)
        adv = tgt - vals.detach()
        if adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        loss_a = -(torch.stack(logps) * adv).mean() - 0.008 * torch.stack(ents).mean()
        loss_a = loss_a + 0.45 * _bc_batch_loss(actor, xs, ys)
        loss_c = F.mse_loss(vals, tgt)
        opt_a.zero_grad()
        loss_a.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        opt_a.step()
        opt_c.zero_grad()
        loss_c.backward()
        opt_c.step()
        st = _stats(env)
        print(
            f"  mappo {ep+1}/{episodes} det={st['detected']} ext={st['extinguished']} "
            f"cells={st['cells']} R={st['reward']:.1f} teach={teacher:.2f}"
        )
        if (ep + 1) % 5 == 0 or ep + 1 == episodes:
            ev = holdout(actor, holdout_seeds)
            ext = _mean_ext(ev)
            print(f"    holdout ext={ext:.2f}")
            if ext >= best_ext:
                best_ext = ext
                best = {k: v.detach().clone() for k, v in actor.state_dict().items()}

    actor.load_state_dict(best)
    final = holdout(actor, holdout_seeds)
    print(f"  mappo best holdout ext={_mean_ext(final):.2f}")
    return actor, final


def train_maddpg(
    actor: ActorNet,
    xs: np.ndarray,
    ys: np.ndarray,
    episodes: int,
    seed0: int,
    holdout_seeds: list[int],
) -> tuple[ActorNet, list[dict]]:
    actor = copy.deepcopy(actor)
    target_actor = copy.deepcopy(actor)
    critic = CentralQ(n_agents=N_UAVS)
    target_critic = copy.deepcopy(critic)
    opt_a = torch.optim.Adam(actor.parameters(), lr=8e-5)
    opt_c = torch.optim.Adam(critic.parameters(), lr=2e-4)
    replay: deque = deque(maxlen=25_000)
    tau = 0.02
    steps = 0
    best = {k: v.detach().clone() for k, v in actor.state_dict().items()}
    best_ext = _mean_ext(holdout(actor, holdout_seeds))
    print(f"  maddpg start holdout ext={best_ext:.2f}")

    for ep in range(episodes):
        sc = make_scenario(seed0 + ep, allow_global_state=True)
        env = WildfireEnv(sc)
        obs, _ = env.reset(seed=sc.seed)
        teacher = 0.4 * max(0.0, 1.0 - ep / max(episodes * 0.6, 1))
        while not env._terminated:
            keys = list(obs.keys())
            batch = np.stack([obs[a] for a in keys], axis=0)
            x = torch.from_numpy(batch)
            if np.random.random() < teacher:
                states = {u.uav_id: {"agent_index": i, "n_agents": len(env.uavs)} for i, u in enumerate(env.uavs)}
                assert env.policy is not None
                expert = env.policy.act_many(obs, states)
                acts = torch.tensor([int(expert[a]) for a in keys], dtype=torch.int64)
            else:
                with torch.no_grad():
                    logits = actor(x)
                    if np.random.random() < 0.04:
                        acts = torch.randint(0, N_ACTIONS, (len(keys),))
                    else:
                        acts = torch.argmax(logits, dim=-1)
            action = {aid: int(acts[i].item()) for i, aid in enumerate(keys)}
            next_obs, rew, term, _, _ = env.step(action)
            done = bool(term.get(keys[0], False))
            r = float(np.mean([rew[k] for k in keys if k != "__all__"]))
            nobs = np.stack([next_obs.get(a, obs[a]) for a in keys], axis=0)
            replay.append((batch, acts.numpy().astype(np.int64), r, nobs, float(done)))
            obs = next_obs
            steps += 1
            if len(replay) >= 128 and steps % 2 == 0:
                idx = np.random.choice(len(replay), size=96, replace=False)
                so = torch.from_numpy(np.stack([replay[i][0] for i in idx]))
                sa = torch.tensor(np.stack([replay[i][1] for i in idx]), dtype=torch.int64)
                sr = torch.tensor([replay[i][2] for i in idx], dtype=torch.float32)
                sno = torch.from_numpy(np.stack([replay[i][3] for i in idx]))
                sd = torch.tensor([replay[i][4] for i in idx], dtype=torch.float32)
                bsz, na, _ = so.shape
                onehot = F.one_hot(sa, num_classes=N_ACTIONS).float()
                obs_cat = so.reshape(bsz, -1)
                act_cat = onehot.reshape(bsz, -1)
                q = critic(obs_cat, act_cat)
                with torch.no_grad():
                    nlogits = target_actor(sno.reshape(bsz * na, -1)).reshape(bsz, na, N_ACTIONS)
                    nact = F.one_hot(nlogits.argmax(dim=-1), num_classes=N_ACTIONS).float()
                    nq = target_critic(sno.reshape(bsz, -1), nact.reshape(bsz, -1))
                    y = sr + 0.97 * (1.0 - sd) * nq
                loss_c = F.mse_loss(q, y)
                opt_c.zero_grad()
                loss_c.backward()
                opt_c.step()
                logits = actor(so.reshape(bsz * na, -1)).reshape(bsz, na, N_ACTIONS)
                soft = F.gumbel_softmax(logits, tau=0.8, hard=True, dim=-1)
                loss_a = -critic(obs_cat.detach(), soft.reshape(bsz, -1)).mean()
                loss_a = loss_a + 0.45 * _bc_batch_loss(actor, xs, ys)
                opt_a.zero_grad()
                loss_a.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                opt_a.step()
                for tp, p in zip(target_actor.parameters(), actor.parameters()):
                    tp.data.mul_(1 - tau).add_(p.data, alpha=tau)
                for tp, p in zip(target_critic.parameters(), critic.parameters()):
                    tp.data.mul_(1 - tau).add_(p.data, alpha=tau)
        st = _stats(env)
        print(
            f"  maddpg {ep+1}/{episodes} det={st['detected']} ext={st['extinguished']} "
            f"cells={st['cells']} R={st['reward']:.1f} teach={teacher:.2f}"
        )
        if (ep + 1) % 5 == 0 or ep + 1 == episodes:
            ev = holdout(actor, holdout_seeds)
            ext = _mean_ext(ev)
            print(f"    holdout ext={ext:.2f}")
            if ext >= best_ext:
                best_ext = ext
                best = {k: v.detach().clone() for k, v in actor.state_dict().items()}

    actor.load_state_dict(best)
    final = holdout(actor, holdout_seeds)
    print(f"  maddpg best holdout ext={_mean_ext(final):.2f}")
    return actor, final


def train_dqn(
    actor: ActorNet,
    xs: np.ndarray,
    ys: np.ndarray,
    episodes: int,
    seed0: int,
    holdout_seeds: list[int],
) -> tuple[ActorNet, list[dict]]:
    q = copy.deepcopy(actor)
    target = copy.deepcopy(q)
    opt = torch.optim.Adam(q.parameters(), lr=1e-4)
    replay: deque = deque(maxlen=20_000)
    best = {k: v.detach().clone() for k, v in q.state_dict().items()}
    best_ext = _mean_ext(holdout(q, holdout_seeds))
    print(f"  dqn start holdout ext={best_ext:.2f}")
    steps = 0
    for ep in range(episodes):
        sc = make_scenario(seed0 + ep, allow_global_state=True)
        env = WildfireEnv(sc)
        obs, _ = env.reset(seed=sc.seed)
        teacher = 0.4 * max(0.0, 1.0 - ep / max(episodes * 0.6, 1))
        while not env._terminated:
            keys = list(obs.keys())
            batch = np.stack([obs[a] for a in keys], axis=0)
            x = torch.from_numpy(batch)
            if np.random.random() < teacher:
                states = {u.uav_id: {"agent_index": i, "n_agents": len(env.uavs)} for i, u in enumerate(env.uavs)}
                assert env.policy is not None
                expert = env.policy.act_many(obs, states)
                acts = [int(expert[a]) for a in keys]
            else:
                with torch.no_grad():
                    greedy = torch.argmax(q(x), dim=-1).numpy()
                acts = [
                    int(np.random.randint(0, N_ACTIONS)) if np.random.random() < 0.05 else int(greedy[i])
                    for i in range(len(keys))
                ]
            action = {aid: acts[i] for i, aid in enumerate(keys)}
            next_obs, rew, term, _, _ = env.step(action)
            done = bool(term.get(keys[0], False))
            r = float(np.mean([rew[k] for k in keys if k != "__all__"]))
            for i, aid in enumerate(keys):
                replay.append((obs[aid], acts[i], r, next_obs.get(aid, obs[aid]), float(done)))
            obs = next_obs
            steps += 1
            if len(replay) >= 256 and steps % 2 == 0:
                idx = np.random.choice(len(replay), size=128, replace=False)
                ss = torch.from_numpy(np.stack([replay[i][0] for i in idx]))
                aa = torch.tensor([replay[i][1] for i in idx], dtype=torch.int64)
                rr = torch.tensor([replay[i][2] for i in idx], dtype=torch.float32)
                ns = torch.from_numpy(np.stack([replay[i][3] for i in idx]))
                dd = torch.tensor([replay[i][4] for i in idx], dtype=torch.float32)
                qsa = q(ss).gather(1, aa.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    tgt = rr + 0.97 * (1.0 - dd) * target(ns).max(dim=1).values
                loss = F.smooth_l1_loss(qsa, tgt) + 0.35 * _bc_batch_loss(q, xs, ys)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 1.0)
                opt.step()
                if steps % 40 == 0:
                    target.load_state_dict(q.state_dict())
        st = _stats(env)
        print(
            f"  dqn {ep+1}/{episodes} det={st['detected']} ext={st['extinguished']} "
            f"cells={st['cells']} R={st['reward']:.1f} teach={teacher:.2f}"
        )
        if (ep + 1) % 5 == 0 or ep + 1 == episodes:
            ev = holdout(q, holdout_seeds)
            ext = _mean_ext(ev)
            print(f"    holdout ext={ext:.2f}")
            if ext >= best_ext:
                best_ext = ext
                best = {k: v.detach().clone() for k, v in q.state_dict().items()}
    q.load_state_dict(best)
    final = holdout(q, holdout_seeds)
    print(f"  dqn best holdout ext={_mean_ext(final):.2f}")
    return q, final


def _save(actor: ActorNet, path: Path, algorithm: str, extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": actor.state_dict(),
        "obs_dim": OBS_DIM,
        "n_actions": N_ACTIONS,
        "algorithm": algorithm,
        "version": "2.0",
        "notes": (
            "MAPPO/MADDPG trained to overfly and extinguish. "
            "Warm-start: greedy fire-seeking + hover-on-front. "
            "BC regularizer kept during RL. Detection is not suppression. "
            "Holdout extinguished-fire counts are measured."
        ),
    }
    payload.update(extra)
    torch.save(payload, path)
    print(f"wrote {path.resolve()}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--expert-episodes", type=int, default=16)
    p.add_argument("--mappo-episodes", type=int, default=36)
    p.add_argument("--maddpg-episodes", type=int, default=36)
    p.add_argument("--dqn-episodes", type=int, default=24)
    p.add_argument("--outdir", type=Path, default=Path("data/checkpoints"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)
    holdout_seeds = [9000 + i for i in range(3)]

    print("=== 1/3 greedy expert (must extinguish) ===")
    xs, ys = collect_expert(args.expert_episodes, args.seed)
    print(f"collected {len(ys)} transitions (hover share={(ys == HOVER).mean():.2f})")

    print("=== 2/3 behavioral clone until holdout extinguishes ===")
    actor = ActorNet()
    bc_loss, bc_evals = bc_until_extinguish(actor, xs, ys, holdout_seeds)
    bc_ext = _mean_ext(bc_evals)
    print(f"bc holdout mean ext={bc_ext:.2f}")
    if bc_ext < 1.0:
        print("WARNING: clone still rarely extinguishes; RL will keep the best of clone vs fine-tune")

    print("=== 3a/4 DQN ===")
    dqn, dqn_evals = train_dqn(actor, xs, ys, args.dqn_episodes, args.seed + 1000, holdout_seeds)
    _print_evals("dqn", holdout_seeds, dqn_evals)
    _save(dqn, args.outdir / "dqn.pt", "dqn", {"bc_loss": bc_loss, "eval": dqn_evals, "bc_eval": bc_evals})

    print("=== 3b/4 MAPPO ===")
    mappo, mappo_evals = train_mappo(actor, xs, ys, args.mappo_episodes, args.seed + 2000, holdout_seeds)
    _print_evals("mappo", holdout_seeds, mappo_evals)
    _save(
        mappo,
        args.outdir / "mappo.pt",
        "mappo",
        {"bc_loss": bc_loss, "eval": mappo_evals, "bc_eval": bc_evals},
    )

    print("=== 3c/4 MADDPG ===")
    maddpg, maddpg_evals = train_maddpg(actor, xs, ys, args.maddpg_episodes, args.seed + 3000, holdout_seeds)
    _print_evals("maddpg", holdout_seeds, maddpg_evals)
    _save(
        maddpg,
        args.outdir / "maddpg.pt",
        "maddpg",
        {"bc_loss": bc_loss, "eval": maddpg_evals, "bc_eval": bc_evals},
    )
    print(
        f"done  dqn_ext={_mean_ext(dqn_evals):.2f}  "
        f"mappo_ext={_mean_ext(mappo_evals):.2f}  "
        f"maddpg_ext={_mean_ext(maddpg_evals):.2f}"
    )


if __name__ == "__main__":
    main()
