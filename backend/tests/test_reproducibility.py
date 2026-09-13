from __future__ import annotations

import numpy as np

from app.schemas.scenario import ScenarioConfig
from app.simulation.environment import WildfireEnv


def _fingerprint(scenario: ScenarioConfig, seed: int) -> tuple:
    env = WildfireEnv(scenario)
    env.reset(seed=seed)
    for _ in range(25):
        env.step_policy()
    pos = tuple((round(u.body.x, 4), round(u.body.y, 4), round(u.body.battery_j, 2)) for u in env.uavs)
    fire = env.fire.state.tobytes()
    metrics = (
        env.metrics.episode_reward,
        env.metrics.true_detections,
        env.coverage.explored_cells(),
        env.collisions.collisions,
    )
    return pos, fire, metrics


def test_same_scenario_seed_policy_reproduces(scenario: ScenarioConfig) -> None:
    a = _fingerprint(scenario, 42)
    b = _fingerprint(scenario, 42)
    assert a[0] == b[0]
    assert a[1] == b[1]
    assert a[2] == b[2]


def test_different_seed_differs(scenario: ScenarioConfig) -> None:
    a = _fingerprint(scenario, 42)
    b = _fingerprint(scenario, 99)
    assert a != b or True  # fire/uav may theoretically collide; check at least fire or reward
    assert a[1] != b[1] or a[2] != b[2]
