"""Headless evaluation of every named policy on random multi-fire Kinglake scenes.

Metrics are measured from WildfireEnv state. Extinguished = suppressed sources.
Detection is not counted as a success.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.baselines.registry import POLICY_NAMES
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

# Kinglake AOI inset so ignitions stay inside the envelope.
_LAT = (-37.538, -37.502)
_LON = (145.275, 145.345)


def random_fires(rng: np.random.Generator, n: int) -> list[FireSourceConfig]:
    fires = []
    for i in range(n):
        fires.append(
            FireSourceConfig(
                fire_id=f"fire_{i+1:02d}",
                latitude=float(rng.uniform(*_LAT)),
                longitude=float(rng.uniform(*_LON)),
                initial_intensity=float(rng.uniform(0.7, 0.95)),
                ignition_time=0.0 if i < 2 else float(rng.uniform(0.0, 15.0)),
                radius_m=70.0,
            )
        )
    return fires


def make_scenario(policy: str, seed: int, n_fires: int, duration_s: float) -> ScenarioConfig:
    rng = np.random.default_rng(seed)
    return ScenarioConfig(
        name=f"eval-{policy}-{seed}",
        location=LocationConfig(region_id="kinglake"),
        fires=random_fires(rng, n_fires),
        environment=EnvironmentConfig(
            duration_s=duration_s,
            timestep_s=1.0,
            fire_cell_size_m=80.0,
            coverage_cell_size_m=120.0,
            wind_speed_kmh=18.0,
            wind_from="NW",
            base_ros_mps=0.22,
            stochastic_fire=True,
        ),
        uavs=UAVFleetConfig(count=6, cruise_speed_mps=16.0, altitude_m=120.0),
        sensor=SensorConfig(
            sensor_range_m=800.0,
            suppression_range_m=120.0,
            false_positive_probability=0.0,
        ),
        communication=CommunicationConfig(range_m=4500.0),
        policy=PolicyConfig(name=policy, allow_global_state=True),
        seed=seed,
    )


def run_one(policy: str, seed: int, n_fires: int, duration_s: float) -> dict:
    sc = make_scenario(policy, seed, n_fires, duration_s)
    env = WildfireEnv(sc)
    env.reset(seed=seed)
    while not env._terminated:
        env.step_policy()
    assert env.fire is not None and env.policy is not None
    return {
        "policy": policy,
        "seed": seed,
        "detected": sum(1 for f in env.fire.sources if f.detected),
        "extinguished": sum(1 for f in env.fire.sources if f.suppressed),
        "active": sum(1 for f in env.fire.sources if f.active),
        "total": len(env.fire.sources),
        "cells": int(sum(u.cells_extinguished for u in env.uavs)),
        "coverage": float(env.coverage.coverage_fraction()) if env.coverage else 0.0,
        "reward": float(env.metrics.episode_reward),
        "trained": bool(env.policy.info.trained),
        "algo": env.policy.info.algorithm,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--policies", nargs="*", default=list(POLICY_NAMES))
    p.add_argument("--seeds", type=int, nargs="*", default=[101, 202])
    p.add_argument("--n-fires", type=int, default=4)
    p.add_argument("--duration", type=float, default=240.0)
    p.add_argument("--out", type=Path, default=Path("data/eval_policies.json"))
    args = p.parse_args()

    rows = []
    for name in args.policies:
        for seed in args.seeds:
            row = run_one(name, seed, args.n_fires, args.duration)
            rows.append(row)
            print(
                f"{name:12s} seed={seed}  det={row['detected']}/{row['total']}  "
                f"ext={row['extinguished']}/{row['total']}  cells={row['cells']}  "
                f"cov={row['coverage']:.2f}  trained={row['trained']}",
                flush=True,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {args.out.resolve()}")
    print("\n=== means ===")
    for name in args.policies:
        sub = [r for r in rows if r["policy"] == name]
        ext = float(np.mean([r["extinguished"] for r in sub]))
        det = float(np.mean([r["detected"] for r in sub]))
        print(f"{name:12s}  mean det={det:.2f}  mean ext={ext:.2f} / {args.n_fires}")


if __name__ == "__main__":
    main()
