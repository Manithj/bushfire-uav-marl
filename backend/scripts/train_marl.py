"""Train Independent DQN, MAPPO, and MADDPG on WildfireEnv (CPU, no UI).

Warm-start from fire-seeking greedy (not lawnmower). Lawnmower parks on
a search strip and never overflies detected fires.

Pipeline:
1. Collect (obs, action) from greedy: fly to known active fires, HOVER
   only when close enough to suppress.
2. Behavioral-clone a shared actor (cross-entropy).
3. Fine-tune three algorithms independently from that clone.
4. Keep RL weights only if holdout (detections + extinguished fires)
   improves on the clone. Metrics are measured, not invented.

Checkpoints: data/checkpoints/{dqn,mappo,maddpg}.pt
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


class Critic(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class CentralCritic(nn.Module):
    def __init__(self, n_agents: int, obs_dim: int = OBS_DIM, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.n_agents = n_agents
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
    """MADDPG critic: all observations + all discrete actions (one-hot)."""

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


def make_train_scenario(
    seed: int,
    duration_s: float = 120.0,
    n_uavs: int = 3,
    allow_global_state: bool = False,
) -> ScenarioConfig:
    rng = np.random.default_rng(seed)
    fires = []
    for i, (lat, lon) in enumerate(((-37.515, 145.30), (-37.525, 145.325), (-37.508, 145.340))[:2]):
        jitter_lat = float(rng.uniform(-0.004, 0.004))
        jitter_lon = float(rng.uniform(-0.006, 0.006))
        fires.append(
            FireSourceConfig(
                fire_id=f"fire_{i+1:02d}",
                latitude=lat + jitter_lat,
                longitude=lon + jitter_lon,
                initial_intensity=0.85,
                ignition_time=0.0 if i == 0 else 8.0,
                radius_m=70.0,
            )
        )
    return ScenarioConfig(
        name="marl-train",
        location=LocationConfig(region_id="kinglake"),
        fires=fires,
        environment=EnvironmentConfig(
            duration_s=duration_s,
            timestep_s=1.0,
            fire_cell_size_m=80.0,
            coverage_cell_size_m=120.0,
            wind_speed_kmh=18.0,
            wind_from="NW",
        ),
        uavs=UAVFleetConfig(count=n_uavs, cruise_speed_mps=16.0, altitude_m=120.0),
        sensor=SensorConfig(
            sensor_range_m=800.0,
            suppression_range_m=120.0,
            false_positive_probability=0.0,
        ),
        communication=CommunicationConfig(range_m=4500.0),
        policy=PolicyConfig(name="greedy", allow_global_state=allow_global_state),
        seed=seed,
    )


def _episode_stats(env: WildfireEnv) -> dict:
    assert env.fire is not None
    return {
        "detected": sum(1 for f in env.fire.sources if f.detected),
        "extinguished": sum(1 for f in env.fire.sources if f.suppressed),
        "total": len(env.fire.sources),
        "coverage": env.coverage.coverage_fraction() if env.coverage else 0.0,
        "reward": env.metrics.episode_reward,
        "active": sum(1 for f in env.fire.sources if f.active),
    }


def collect_expert(n_episodes: int, seed0: int) -> tuple[np.ndarray, np.ndarray]:
    obs_list: list[np.ndarray] = []
    act_list: list[int] = []
    for ep in range(n_episodes):
        # Most expert episodes see ignition locations so the clone learns
        # "fire in observation → fly there → hover". A few stay partial so
        # empty fire slots still map to search, not hover.
        sc = make_train_scenario(seed0 + ep, allow_global_state=(ep % 3 != 2))
        env = WildfireEnv(sc)
        obs, _ = env.reset(seed=sc.seed)
        while not env._terminated:
            states = {u.uav_id: {"agent_index": i, "n_agents": len(env.uavs)} for i, u in enumerate(env.uavs)}
            assert env.policy is not None
            actions = env.policy.act_many(obs, states)
            for aid, vec in obs.items():
                act = int(actions[aid])
                # Keep hover only when a fire slot is occupied (suppressing).
                fire_slots = vec[11:20]
                has_fire = float(np.abs(fire_slots).sum()) > 1e-6
                if act == 8 and not has_fire:
                    continue
                obs_list.append(vec.astype(np.float32))
                act_list.append(act)
            obs, *_ = env.step(actions)
        st = _episode_stats(env)
        print(
            f"  expert greedy ep {ep+1}/{n_episodes} "
            f"det={st['detected']}/{st['total']} ext={st['extinguished']}"
        )
    return np.stack(obs_list), np.asarray(act_list, dtype=np.int64)


def behavioral_clone(actor: ActorNet, xs: np.ndarray, ys: np.ndarray, epochs: int, batch: int) -> float:
    opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
    last = 0.0
    n = xs.shape[0]
    w = torch.ones(N_ACTIONS)
    w[8] = 0.7
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
            opt.step()
            losses.append(float(loss.item()))
        last = float(np.mean(losses))
        print(f"  bc epoch {epoch+1}/{epochs} loss={last:.4f}")
    return last


def evaluate(actor: ActorNet, seed: int, allow_global_state: bool = True) -> dict:
    sc = make_train_scenario(seed, duration_s=120.0, allow_global_state=allow_global_state)
    env = WildfireEnv(sc)
    env.reset(seed=seed)
    while not env._terminated:
        obs = env._observe_all()
        actions = {}
        for aid, vec in obs.items():
            x = torch.from_numpy(vec).unsqueeze(0)
            with torch.no_grad():
                actions[aid] = int(torch.argmax(actor(x), dim=-1).item())
        env.step(actions)
    return _episode_stats(env)


def _score(ev: dict) -> float:
    return float(ev["detected"] + 2.0 * ev["extinguished"] + 0.2 * ev["coverage"])


def _mean_score(evals: list[dict]) -> float:
    return float(np.mean([_score(e) for e in evals]))


def train_dqn(actor: ActorNet, episodes: int, seed0: int) -> ActorNet:
    q = copy.deepcopy(actor)
    target = copy.deepcopy(q)
    opt = torch.optim.Adam(q.parameters(), lr=3e-4)
    replay: deque = deque(maxlen=20_000)
    steps = 0
    for ep in range(episodes):
        sc = make_train_scenario(seed0 + ep, allow_global_state=True)
        env = WildfireEnv(sc)
        obs, _ = env.reset(seed=sc.seed)
        eps = max(0.05, 0.35 * (1.0 - ep / max(episodes - 1, 1)))
        while not env._terminated:
            keys = list(obs.keys())
            batch = np.stack([obs[a] for a in keys], axis=0)
            x = torch.from_numpy(batch)
            with torch.no_grad():
                greedy = torch.argmax(q(x), dim=-1).numpy()
            actions = {}
            chosen = []
            for i, aid in enumerate(keys):
                a = int(np.random.randint(0, N_ACTIONS)) if np.random.random() < eps else int(greedy[i])
                actions[aid] = a
                chosen.append(a)
            next_obs, rew, term, _, _ = env.step(actions)
            done = bool(term.get(keys[0], False))
            r = float(np.mean([rew[k] for k in keys if k != "__all__"]))
            for i, aid in enumerate(keys):
                replay.append((obs[aid], chosen[i], r, next_obs.get(aid, obs[aid]), float(done)))
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
                loss = F.smooth_l1_loss(qsa, tgt)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 1.0)
                opt.step()
            if steps % 200 == 0:
                target.load_state_dict(q.state_dict())
        st = _episode_stats(env)
        print(f"  dqn {ep+1}/{episodes} eps={eps:.2f} det={st['detected']} ext={st['extinguished']} R={st['reward']:.1f}")
    return q


def train_mappo(actor: ActorNet, episodes: int, seed0: int, n_agents: int = 2) -> ActorNet:
    actor = copy.deepcopy(actor)
    critic = CentralCritic(n_agents=n_agents)
    opt_a = torch.optim.Adam(actor.parameters(), lr=1e-4)
    opt_c = torch.optim.Adam(critic.parameters(), lr=3e-4)
    for ep in range(episodes):
        sc = make_train_scenario(seed0 + ep, n_uavs=n_agents, allow_global_state=True)
        env = WildfireEnv(sc)
        obs, _ = env.reset(seed=sc.seed)
        logps: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        rewards: list[float] = []
        ents: list[torch.Tensor] = []
        while not env._terminated:
            keys = list(obs.keys())
            batch = np.stack([obs[a] for a in keys], axis=0)
            x = torch.from_numpy(batch)
            logits = actor(x)
            dist = torch.distributions.Categorical(logits=logits)
            acts = dist.sample()
            logps.append(dist.log_prob(acts).mean())
            ents.append(dist.entropy().mean())
            pad = np.zeros((n_agents, OBS_DIM), dtype=np.float32)
            pad[: len(keys)] = batch
            values.append(critic(torch.from_numpy(pad.reshape(1, -1)).float()).squeeze())
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
        loss_a = -(torch.stack(logps) * adv).mean() - 0.02 * torch.stack(ents).mean()
        loss_c = F.mse_loss(vals, tgt)
        opt_a.zero_grad()
        loss_a.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        opt_a.step()
        opt_c.zero_grad()
        loss_c.backward()
        opt_c.step()
        st = _episode_stats(env)
        print(
            f"  mappo {ep+1}/{episodes} det={st['detected']} ext={st['extinguished']} "
            f"R={st['reward']:.1f} la={float(loss_a.item()):.3f}"
        )
    return actor


def train_maddpg(actor: ActorNet, episodes: int, seed0: int, n_agents: int = 2) -> ActorNet:
    actor = copy.deepcopy(actor)
    target_actor = copy.deepcopy(actor)
    critic = CentralQ(n_agents=n_agents)
    target_critic = copy.deepcopy(critic)
    opt_a = torch.optim.Adam(actor.parameters(), lr=1e-4)
    opt_c = torch.optim.Adam(critic.parameters(), lr=3e-4)
    replay: deque = deque(maxlen=15_000)
    tau = 0.01
    steps = 0
    for ep in range(episodes):
        sc = make_train_scenario(seed0 + ep, n_uavs=n_agents, allow_global_state=True)
        env = WildfireEnv(sc)
        obs, _ = env.reset(seed=sc.seed)
        while not env._terminated:
            keys = list(obs.keys())
            batch = np.stack([obs[a] for a in keys], axis=0)
            x = torch.from_numpy(batch)
            with torch.no_grad():
                logits = actor(x)
                if np.random.random() < 0.15:
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
                idx = np.random.choice(len(replay), size=64, replace=False)
                so = torch.from_numpy(np.stack([replay[i][0] for i in idx]))  # B, N, O
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
                soft = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)
                loss_a = -critic(obs_cat.detach(), soft.reshape(bsz, -1)).mean()
                opt_a.zero_grad()
                loss_a.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                opt_a.step()
                for tp, p in zip(target_actor.parameters(), actor.parameters()):
                    tp.data.mul_(1 - tau).add_(p.data, alpha=tau)
                for tp, p in zip(target_critic.parameters(), critic.parameters()):
                    tp.data.mul_(1 - tau).add_(p.data, alpha=tau)
        st = _episode_stats(env)
        print(f"  maddpg {ep+1}/{episodes} det={st['detected']} ext={st['extinguished']} R={st['reward']:.1f}")
    return actor


def _save(actor: ActorNet, path: Path, algorithm: str, extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": actor.state_dict(),
        "obs_dim": OBS_DIM,
        "n_actions": N_ACTIONS,
        "algorithm": algorithm,
        "version": "1.0",
        "notes": (
            "Warm-started from fire-seeking greedy (fly to known active fires, "
            "hover to suppress). Detection does not extinguish. "
            "Decentralized execution on partial observations. Not claimed SOTA MARL."
        ),
    }
    payload.update(extra)
    torch.save(payload, path)
    print(f"wrote {path.resolve()}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--expert-episodes", type=int, default=12)
    p.add_argument("--bc-epochs", type=int, default=10)
    p.add_argument("--dqn-episodes", type=int, default=28)
    p.add_argument("--mappo-episodes", type=int, default=24)
    p.add_argument("--maddpg-episodes", type=int, default=24)
    p.add_argument("--outdir", type=Path, default=Path("data/checkpoints"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)

    print("=== 1/4 expert rollouts (greedy fire-seeking) ===")
    xs, ys = collect_expert(args.expert_episodes, args.seed)
    print(f"collected {len(ys)} transitions")

    print("=== 2/4 behavioral cloning ===")
    actor = ActorNet()
    bc_loss = behavioral_clone(actor, xs, ys, args.bc_epochs, batch=256)
    bc_state = {k: v.detach().clone() for k, v in actor.state_dict().items()}
    holdout_seeds = [9000 + i for i in range(3)]
    print("=== BC holdout ===")
    bc_evals = [evaluate(actor, s) for s in holdout_seeds]
    for s, ev in zip(holdout_seeds, bc_evals):
        print(f"  bc seed {s}: det {ev['detected']}/{ev['total']} ext={ev['extinguished']} cov={ev['coverage']:.3f}")
    bc_score = _mean_score(bc_evals)

    def maybe_keep(name: str, trained: ActorNet, evals: list[dict]) -> tuple[ActorNet, str, list[dict]]:
        score = _mean_score(evals)
        if score >= bc_score:
            print(f"keeping {name} weights (score {score:.3f} >= BC {bc_score:.3f})")
            return trained, name, evals
        print(f"keeping BC weights for {name} (BC {bc_score:.3f} > {name} {score:.3f})")
        fallback = ActorNet()
        fallback.load_state_dict(bc_state)
        return fallback, f"{name}+bc_greedy", bc_evals

    print("=== 3/4 Independent DQN ===")
    actor.load_state_dict(bc_state)
    dqn = train_dqn(actor, args.dqn_episodes, args.seed + 1000)
    dqn_evals = [evaluate(dqn, s) for s in holdout_seeds]
    for s, ev in zip(holdout_seeds, dqn_evals):
        print(f"  dqn seed {s}: det {ev['detected']}/{ev['total']} ext={ev['extinguished']} R={ev['reward']:.1f}")
    dqn, dqn_algo, dqn_keep = maybe_keep("independent_dqn", dqn, dqn_evals)
    _save(
        dqn,
        args.outdir / "dqn.pt",
        dqn_algo,
        {"bc_loss": bc_loss, "eval": dqn_keep, "bc_eval": bc_evals, "rl_eval": dqn_evals},
    )

    print("=== 4a/4 MAPPO ===")
    actor.load_state_dict(bc_state)
    mappo = train_mappo(actor, args.mappo_episodes, args.seed + 2000)
    mappo_evals = [evaluate(mappo, s) for s in holdout_seeds]
    for s, ev in zip(holdout_seeds, mappo_evals):
        print(f"  mappo seed {s}: det {ev['detected']}/{ev['total']} ext={ev['extinguished']} R={ev['reward']:.1f}")
    mappo, mappo_algo, mappo_keep = maybe_keep("mappo", mappo, mappo_evals)
    _save(
        mappo,
        args.outdir / "mappo.pt",
        mappo_algo,
        {"bc_loss": bc_loss, "eval": mappo_keep, "bc_eval": bc_evals, "rl_eval": mappo_evals},
    )

    print("=== 4b/4 MADDPG (Gumbel-Softmax discrete) ===")
    actor.load_state_dict(bc_state)
    maddpg = train_maddpg(actor, args.maddpg_episodes, args.seed + 3000)
    maddpg_evals = [evaluate(maddpg, s) for s in holdout_seeds]
    for s, ev in zip(holdout_seeds, maddpg_evals):
        print(f"  maddpg seed {s}: det {ev['detected']}/{ev['total']} ext={ev['extinguished']} R={ev['reward']:.1f}")
    maddpg, maddpg_algo, maddpg_keep = maybe_keep("maddpg", maddpg, maddpg_evals)
    _save(
        maddpg,
        args.outdir / "maddpg.pt",
        maddpg_algo,
        {"bc_loss": bc_loss, "eval": maddpg_keep, "bc_eval": bc_evals, "rl_eval": maddpg_evals},
    )
    print("done")


if __name__ == "__main__":
    main()
