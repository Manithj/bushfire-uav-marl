from __future__ import annotations

import numpy as np

from app.baselines.greedy_policy import GreedyPolicy
from app.schemas.scenario import UAVPlacement
from app.simulation.fire_model import EXTINGUISHED, BURNING, FireModel, FireModelConfig
from app.simulation.environment import WildfireEnv
from app.schemas.scenario import ScenarioConfig


def test_suppress_in_range_extinguishes_cells() -> None:
    rng = np.random.default_rng(0)
    m = FireModel.create(0.0, 0.0, 2000.0, 2000.0, FireModelConfig(cell_size_m=50.0, stochastic=False), rng)
    m.ignite("f", 1000.0, 1000.0, -37.5, 145.3, 0.9, 0.0, 60.0)
    assert np.any(m.state == BURNING)
    n, newly = m.suppress(1000.0, 1000.0, 200.0, 2.0, "uav_00", 2.0)
    assert n > 0
    assert np.any(m.state == EXTINGUISHED)
    assert m.sources[0].suppressed
    assert newly == ["f"]


def test_out_of_suppression_range_does_not_extinguish() -> None:
    rng = np.random.default_rng(0)
    m = FireModel.create(0.0, 0.0, 2000.0, 2000.0, FireModelConfig(cell_size_m=50.0, stochastic=False), rng)
    m.ignite("f", 1000.0, 1000.0, -37.5, 145.3, 0.9, 0.0, 60.0)
    n, newly = m.suppress(1000.0 + 600.0, 1000.0, 200.0, 2.0, "uav_00", 2.0)
    assert n == 0
    assert newly == []
    assert not np.any(m.state == EXTINGUISHED)
    assert m.sources[0].active
    assert not m.sources[0].suppressed


def test_overfly_extinguishes_in_env(scenario: ScenarioConfig) -> None:
    scenario.uavs.count = 1
    scenario.uavs.placements = [
        UAVPlacement(latitude=-37.50, longitude=145.30, heading_deg=0.0),
    ]
    scenario.sensor.sensor_range_m = 800.0
    scenario.sensor.suppression_range_m = 120.0
    scenario.policy.name = "greedy"
    scenario.environment.duration_s = 10.0
    env = WildfireEnv(scenario)
    env.reset(seed=42)
    aid = env.uavs[0].uav_id
    for _ in range(8):
        env.step({aid: 8})
    assert np.any(env.fire.state == EXTINGUISHED)
    assert env.uavs[0].cells_extinguished > 0
    assert env.fire.sources[0].suppressed


def test_detection_without_overfly_does_not_extinguish(scenario: ScenarioConfig) -> None:
    """UAV ~550 m east of the fire: inside 800 m detect, outside 150 m suppress."""
    scenario.uavs.count = 1
    lat, lon = -37.50, 145.30
    dlon = 550.0 / (111320.0 * np.cos(np.deg2rad(lat)))
    scenario.uavs.placements = [
        UAVPlacement(latitude=lat, longitude=lon + dlon, heading_deg=0.0),
    ]
    scenario.sensor.sensor_range_m = 800.0
    scenario.sensor.suppression_range_m = 150.0
    scenario.sensor.detection_probability = 1.0
    scenario.sensor.false_negative_probability = 0.0
    scenario.sensor.false_positive_probability = 0.0
    scenario.policy.name = "greedy"
    scenario.environment.duration_s = 6.0
    env = WildfireEnv(scenario)
    env.reset(seed=42)
    aid = env.uavs[0].uav_id
    for _ in range(5):
        env.step({aid: 8})
    src = env.fire.sources[0]
    assert src.detected
    assert not src.suppressed
    assert not np.any(env.fire.state == EXTINGUISHED)
    assert env.uavs[0].cells_extinguished == 0


def test_greedy_hovers_when_close_to_known_fire() -> None:
    obs = np.zeros(57, dtype=np.float32)
    obs[11:14] = [0.008, 0.008, 0.8]
    assert GreedyPolicy().act(obs, "uav_00") == 8


def test_greedy_flies_toward_known_fire() -> None:
    obs = np.zeros(57, dtype=np.float32)
    obs[11:14] = [0.2, 0.0, 0.8]  # east
    assert GreedyPolicy().act(obs, "uav_00") == 2  # E


def test_overlap_extinguishes_whole_fire_and_stops_spread() -> None:
    rng = np.random.default_rng(0)
    m = FireModel.create(0.0, 0.0, 2000.0, 2000.0, FireModelConfig(cell_size_m=50.0, stochastic=False), rng)
    m.ignite("f", 1000.0, 1000.0, -37.5, 145.3, 0.9, 0.0, 60.0)
    for t in range(20):
        m.step(1.0, 8.0, 315.0, float(t + 1))
    assert int(np.sum(m.state == BURNING)) > 1
    n, newly = m.suppress(1000.0, 1000.0, 120.0, 1.0, "uav_00", 21.0)
    assert n > 0
    assert newly == ["f"]
    assert m.sources[0].suppressed
    assert not m.sources[0].active
    assert not np.any(m.state == BURNING)
    after = int(np.sum((m.state == BURNING) | (m.state == EXTINGUISHED)))
    m.step(1.0, 8.0, 315.0, 22.0)
    later = int(np.sum((m.state == BURNING) | (m.state == EXTINGUISHED)))
    assert later == after
    assert not np.any(m.state == BURNING)
