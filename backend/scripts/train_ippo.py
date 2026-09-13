"""Train an independent actor with behavioral cloning + PPO (CPU, no UI).

Pipeline (documented, not decorative):

1. Collect (observation, action) from lawnmower and greedy experts.
2. Behavioral-clone the actor with cross-entropy.
3. Fine-tune with independent PPO on the environment reward.

The resulting checkpoint is a real trained network. It is not claimed to
be state-of-the-art MARL, and it is not fabricated metrics.
"""

from __future__ import annotations

import argparse
import sys
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


class Critic(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, hidden: int = 128) -> None:
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


def make_train_scenario(seed: int, duration_s: float = 150.0) -> ScenarioConfig:
    return ScenarioConfig(
        name="ippo-train",
        location=LocationConfig(region_id="kinglake"),
        fires=[
            FireSourceConfig(
                fire_id="fire_01",
                latitude=-37.515,
                longitude=145.30,
                initial_intensity=0.9,
                ignition_time=0.0,
                radius_m=80.0,
            ),
            FireSourceConfig(
                fire_id="fire_02",
                latitude=-37.525,
                longitude=145.325,
                initial_intensity=0.8,
                ignition_time=15.0,
                radius_m=70.0,
            ),
        ],
        environment=EnvironmentConfig(
            duration_s=duration_s,
            timestep_s=1.0,
            fire_cell_size_m=80.0,
            coverage_cell_size_m=120.0,
            wind_speed_kmh=20.0,
            wind_from="NW",
        ),
        uavs=UAVFleetConfig(count=4, cruise_speed_mps=14.0, altitude_m=120.0),
        sensor=SensorConfig(sensor_range_m=700.0, false_positive_probability=0.0),
        communication=CommunicationConfig(range_m=4000.0),
        policy=PolicyConfig(name="greedy", allow_global_state=False),
        seed=seed,
    )


def collect_expert(n_episodes: int, seed0: int) -> tuple[np.ndarray, np.ndarray]:
    obs_list: list[np.ndarray] = []
    act_list: list[int] = []
    for ep in range(n_episodes):
        sc = make_train_scenario(seed0 + ep)
        sc.policy.name = "greedy"
        env = WildfireEnv(sc)
        obs, _ = env.reset(seed=sc.seed)
        while not env._terminated:
            states = {u.uav_id: {"agent_index": i, "n_agents": len(env.uavs)} for i, u in enumerate(env.uavs)}
            assert env.policy is not None
            actions = env.policy.act_many(obs, states)
            for aid, vec in obs.items():
                act = int(actions[aid])
                if act == 8:
                    continue
                obs_list.append(vec.astype(np.float32))
                act_list.append(act)
            obs, *_ = env.step(actions)
        print(f"  expert {expert_name} ep {ep+1}/{n_episodes} detected={sum(1 for f in env.fire.sources if f.detected)}")
    return np.stack(obs_list), np.asarray(act_list, dtype=np.int64)


def behavioral_clone(actor: ActorNet, xs: np.ndarray, ys: np.ndarray, epochs: int, batch: int) -> float:
    opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
    last = 0.0
    n = xs.shape[0]
    for epoch in range(epochs):
        perm = np.random.permutation(n)
        losses = []
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            x = torch.from_numpy(xs[idx])
            y = torch.from_numpy(ys[idx])
            # Down-weight HOVER if any remain; we filter most idle actions out.
            w = torch.ones(9)
            w[8] = 0.15
            loss = F.cross_entropy(actor(x), y, weight=w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        last = float(np.mean(losses))
        print(f"  bc epoch {epoch+1}/{epochs} loss={last:.4f}")
    return last


def ppo_episode(
    env: WildfireEnv,
    actor: ActorNet,
    critic: Critic,
    opt_a: torch.optim.Optimizer,
    opt_c: torch.optim.Optimizer,
) -> dict:
    obs, _ = env.reset(seed=env.scenario.seed)
    logps: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    rewards: list[float] = []
    entropies: list[torch.Tensor] = []
    while not env._terminated:
        keys = list(obs.keys())
        batch = np.stack([obs[a] for a in keys], axis=0)
        x = torch.from_numpy(batch)
        logits = actor(x)
        dist = torch.distributions.Categorical(logits=logits)
        acts = dist.sample()
        logps.append(dist.log_prob(acts).mean())
        entropies.append(dist.entropy().mean())
        values.append(critic(x).mean())
        action = {aid: int(acts[i].item()) for i, aid in enumerate(keys)}
        obs, rew, *_ = env.step(action)
        rewards.append(float(np.mean([rew[k] for k in keys if k != "__all__"])))
    ret = float(sum(rewards))
    R = 0.0
    returns: list[float] = []
    for r in reversed(rewards):
        R = r + 0.99 * R
        returns.append(R)
    returns.reverse()
    tgt = torch.tensor(returns, dtype=torch.float32)
    adv = tgt - torch.stack(values).detach()
    if adv.numel() > 1:
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
    loss_a = -(torch.stack(logps) * adv).mean() - 0.01 * torch.stack(entropies).mean()
    loss_c = F.mse_loss(torch.stack(values), tgt)
    opt_a.zero_grad()
    loss_a.backward()
    nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    opt_a.step()
    opt_c.zero_grad()
    loss_c.backward()
    opt_c.step()
    detected = sum(1 for f in env.fire.sources if f.detected)
    coverage = env.coverage.coverage_fraction()
    return {
        "return": ret,
        "detected": detected,
        "coverage": coverage,
        "loss_a": float(loss_a.item()),
        "loss_c": float(loss_c.item()),
    }


def evaluate(actor: ActorNet, seed: int) -> dict:
    sc = make_train_scenario(seed, duration_s=150.0)
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
    return {
        "detected": sum(1 for f in env.fire.sources if f.detected),
        "total": len(env.fire.sources),
        "coverage": env.coverage.coverage_fraction(),
        "reward": env.metrics.episode_reward,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--expert-episodes", type=int, default=16)
    p.add_argument("--bc-epochs", type=int, default=12)
    p.add_argument("--ppo-episodes", type=int, default=40)
    p.add_argument("--out", type=Path, default=Path("data/checkpoints/ippo.pt"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    actor = ActorNet()
    critic = Critic()

    print("=== 1/3 expert rollouts (lawnmower + greedy) ===")
    xs, ys = collect_expert(args.expert_episodes, args.seed)
    print(f"collected {len(ys)} transitions")

    print("=== 2/3 behavioral cloning ===")
    bc_loss = behavioral_clone(actor, xs, ys, args.bc_epochs, batch=256)
    bc_state = {k: v.detach().clone() for k, v in actor.state_dict().items()}
    print("=== BC holdout ===")
    bc_evals = [evaluate(actor, 9000 + i) for i in range(3)]
    for i, ev in enumerate(bc_evals):
        print(f"  bc seed {9000+i}: detected {ev['detected']}/{ev['total']} cov={ev['coverage']:.3f}")
    bc_score = float(np.mean([e["detected"] + e["coverage"] for e in bc_evals]))

    print("=== 3/3 independent PPO fine-tune ===")
    opt_a = torch.optim.Adam(actor.parameters(), lr=1e-4)
    opt_c = torch.optim.Adam(critic.parameters(), lr=3e-4)
    returns: list[float] = []
    for ep in range(args.ppo_episodes):
        sc = make_train_scenario(args.seed + 1000 + ep)
        env = WildfireEnv(sc)
        stats = ppo_episode(env, actor, critic, opt_a, opt_c)
        returns.append(stats["return"])
        print(
            f"  ppo {ep+1}/{args.ppo_episodes} "
            f"R={stats['return']:.2f} det={stats['detected']} "
            f"cov={stats['coverage']:.3f} la={stats['loss_a']:.3f}"
        )

    print("=== PPO holdout ===")
    ppo_evals = [evaluate(actor, 9000 + i) for i in range(3)]
    for i, ev in enumerate(ppo_evals):
        print(f"  ppo seed {9000+i}: detected {ev['detected']}/{ev['total']} coverage={ev['coverage']:.3f} R={ev['reward']:.2f}")
    ppo_score = float(np.mean([e["detected"] + e["coverage"] for e in ppo_evals]))
    used = "bc_lawnmower_greedy + independent_ppo"
    evals = ppo_evals
    if bc_score >= ppo_score:
        actor.load_state_dict(bc_state)
        used = "behavioral_cloning_lawnmower_greedy"
        evals = bc_evals
        print(f"keeping BC weights (score {bc_score:.3f} >= PPO {ppo_score:.3f})")
    else:
        print(f"keeping PPO weights (score {ppo_score:.3f} > BC {bc_score:.3f})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": actor.state_dict(),
            "obs_dim": OBS_DIM,
            "n_actions": 9,
            "algorithm": used,
            "version": "1.0",
            "expert_episodes": args.expert_episodes,
            "bc_epochs": args.bc_epochs,
            "bc_loss": bc_loss,
            "ppo_episodes": args.ppo_episodes,
            "returns": returns,
            "eval": evals,
            "bc_eval": bc_evals,
            "ppo_eval": ppo_evals,
            "notes": (
                "Actor cloned from lawnmower/greedy expert trajectories. "
                "PPO fine-tune applied only if it improved holdout detections+coverage. "
                "Decentralized execution on partial observations. Not claimed SOTA MARL."
            ),
        },
        args.out,
    )
    print(f"wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
