"""Headless verification of a deterministic Kinglake scenario (seed=42)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def build() -> ScenarioConfig:
    return ScenarioConfig(
        name="deterministic-verify",
        location=LocationConfig(region_id="kinglake"),
        fires=[
            FireSourceConfig(
                fire_id="fire_01",
                latitude=-37.50,
                longitude=145.30,
                initial_intensity=0.9,
                ignition_time=0.0,
                radius_m=80.0,
            ),
            FireSourceConfig(
                fire_id="fire_02",
                latitude=-37.52,
                longitude=145.34,
                initial_intensity=0.75,
                ignition_time=10.0,
                radius_m=60.0,
            ),
        ],
        environment=EnvironmentConfig(
            wind_speed_kmh=25.0,
            wind_from="NW",
            duration_s=90.0,
            timestep_s=1.0,
            fire_cell_size_m=80.0,
            coverage_cell_size_m=120.0,
        ),
        uavs=UAVFleetConfig(count=4, altitude_m=120.0),
        sensor=SensorConfig(sensor_range_m=600.0, false_positive_probability=0.0),
        communication=CommunicationConfig(range_m=2500.0),
        policy=PolicyConfig(name="lawnmower"),
        seed=42,
    )


def fingerprint(seed: int) -> dict:
    env = WildfireEnv(build())
    env.reset(seed=seed)
    while not env._terminated:
        env.step_policy()
    state = env.public_state()
    return {
        "time_s": state["time_s"],
        "coverage": state["metrics"]["coverage_fraction"],
        "detected": state["metrics"]["detected_fires"],
        "reward": state["metrics"]["episode_reward"],
        "distance": state["metrics"]["total_distance_m"],
        "burned_m2": state["metrics"]["burned_area_m2"],
        "collisions": state["metrics"]["collisions"],
        "uav_xy": [(round(u.body.x, 3), round(u.body.y, 3)) for u in env.uavs],
    }


def main() -> None:
    a = fingerprint(42)
    b = fingerprint(42)
    print(json.dumps(a, indent=2))
    assert a == b, "seed=42 was not reproducible"
    print("OK: seed=42 reproduced identically (lawnmower, Kinglake, 90 s).")


if __name__ == "__main__":
    main()
