from __future__ import annotations

from app.marl.observation import OBS_DIM
from app.schemas.scenario import ScenarioConfig
from app.simulation.environment import WildfireEnv


def test_reset_and_step(scenario: ScenarioConfig) -> None:
    env = WildfireEnv(scenario)
    obs, info = env.reset(seed=42)
    assert set(obs) == {u.uav_id for u in env.uavs}
    for v in obs.values():
        assert v.shape == (OBS_DIM,)
    actions = {aid: 0 for aid in obs}
    nobs, rewards, term, trunc, info = env.step(actions)
    assert env.time_s == scenario.environment.timestep_s
    assert all(a in rewards for a in nobs)
    assert all(t is False or t is True for t in term.values())


def test_partial_observability_default(scenario: ScenarioConfig) -> None:
    scenario.policy.allow_global_state = False
    env = WildfireEnv(scenario)
    env.reset(seed=42)
    # Before any detection, known-fire slots should be zero
    obs = env._observe_all()
    for vec in obs.values():
        fire_slots = vec[11:20]
        assert float(abs(fire_slots).sum()) == 0.0


def test_headless_run_completes(scenario: ScenarioConfig) -> None:
    scenario.environment.duration_s = 20.0
    env = WildfireEnv(scenario)
    env.reset(seed=42)
    steps = 0
    while not env._terminated and steps < 40:
        env.step_policy()
        steps += 1
    assert env._terminated
    assert env.time_s >= 20.0 - 1e-6
