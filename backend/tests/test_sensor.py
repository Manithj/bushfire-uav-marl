from __future__ import annotations

import numpy as np

from app.simulation.fire_model import FireModel, FireModelConfig
from app.simulation.sensor_model import sense_fires
from app.simulation.uav import UAVAgent, UAVConfig
from app.simulation.uav_dynamics import UAVBodyState, UAVDynamicsConfig


def _setup(p_det: float, p_fn: float, p_fp: float, range_m: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    fire = FireModel.create(0.0, 0.0, 2000.0, 2000.0, FireModelConfig(cell_size_m=50.0, stochastic=False), rng)
    fire.ignite("fire_01", 1000.0, 1000.0, -37.5, 145.3, 1.0, 0.0, 80.0)
    dyn = UAVDynamicsConfig()
    cfg = UAVConfig(
        uav_id="uav_00",
        dynamics=dyn,
        sensor_range_m=range_m,
        detection_probability=p_det,
        false_positive_probability=p_fp,
        false_negative_probability=p_fn,
        observation_noise_m=0.0,
    )
    body = UAVBodyState(x=1000.0, y=1000.0, z=120.0, heading=0.0, speed=0.0, battery_j=dyn.battery_capacity_j)
    uav = UAVAgent(config=cfg, body=body, lat=-37.5, lon=145.3)
    return uav, fire, rng


def test_in_range_can_detect() -> None:
    uav, fire, rng = _setup(1.0, 0.0, 0.0, 500.0, seed=1)
    ev = sense_fires(uav, fire, rng, 1.0, lambda x, y: (-37.5, 145.3))
    assert any(e.is_true and e.fire_id == "fire_01" for e in ev)


def test_out_of_range_no_true_detection() -> None:
    uav, fire, rng = _setup(1.0, 0.0, 0.0, 50.0, seed=1)
    uav.body.x = 0.0
    uav.body.y = 0.0
    ev = sense_fires(uav, fire, rng, 1.0, lambda x, y: (-37.5, 145.3))
    assert not any(e.is_true for e in ev)


def test_false_negative_can_suppress() -> None:
    uav, fire, rng = _setup(1.0, 1.0, 0.0, 500.0, seed=3)
    ev = sense_fires(uav, fire, rng, 1.0, lambda x, y: (-37.5, 145.3))
    assert not any(e.is_true for e in ev)


def test_false_positive_when_no_fire_in_range() -> None:
    uav, fire, rng = _setup(1.0, 0.0, 1.0, 50.0, seed=4)
    uav.body.x = 0.0
    uav.body.y = 0.0
    ev = sense_fires(uav, fire, rng, 1.0, lambda x, y: (-37.5, 145.3))
    assert any(not e.is_true for e in ev)
