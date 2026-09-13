from __future__ import annotations

from app.schemas.scenario import PolicyConfig, ScenarioConfig
from app.simulation.environment import WildfireEnv
from app.simulation.metrics import compute_public_metrics


def test_metrics_from_state(scenario: ScenarioConfig) -> None:
    env = WildfireEnv(scenario)
    env.reset(seed=42)
    for _ in range(15):
        env.step_policy()
    m = compute_public_metrics(env)
    assert m["sim_time_s"] == env.time_s
    assert m["total_fires"] == len(env.fire.sources)
    assert m["detected_fires"] == sum(1 for f in env.fire.sources if f.detected)
    assert m["collisions"] == env.collisions.collisions
    assert m["episode_reward"] == env.metrics.episode_reward
    assert 0.0 <= m["coverage_fraction"] <= 1.0
    if m["detection_rate"] is not None:
        assert 0.0 <= m["detection_rate"] <= 1.0
    if env.metrics.true_detections + env.metrics.false_detections == 0:
        assert m["false_positive_rate"] is None


def test_detection_time_definition(scenario: ScenarioConfig) -> None:
    env = WildfireEnv(scenario)
    env.reset(seed=42)
    for _ in range(40):
        env.step_policy()
    for src in env.fire.sources:
        if src.detected and src.first_detection_time is not None:
            assert src.first_detection_time >= src.ignition_time
