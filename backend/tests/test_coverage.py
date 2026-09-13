from __future__ import annotations

from app.simulation.coverage import CoverageModel
from app.simulation.uav import UAVAgent, UAVConfig
from app.simulation.uav_dynamics import UAVBodyState, UAVDynamicsConfig


def _uav(x: float, y: float, rng: float) -> UAVAgent:
    dyn = UAVDynamicsConfig()
    cfg = UAVConfig(uav_id="uav_00", dynamics=dyn, sensor_range_m=rng)
    body = UAVBodyState(x=x, y=y, z=100.0, heading=0.0, speed=0.0, battery_j=dyn.battery_capacity_j)
    return UAVAgent(config=cfg, body=body, lat=0.0, lon=0.0)


def test_explored_area_increases() -> None:
    cov = CoverageModel.create(0.0, 0.0, 2000.0, 2000.0, 100.0, n_agents=1)
    assert cov.coverage_fraction() == 0.0
    cov.update([_uav(1000.0, 1000.0, 250.0)], 1.0)
    assert cov.explored_cells() > 0
    assert 0.0 < cov.coverage_fraction() <= 1.0


def test_inaccessible_not_counted() -> None:
    cov = CoverageModel.create(0.0, 0.0, 1000.0, 1000.0, 100.0, n_agents=1)
    cov.inaccessible[:, :] = True
    cov.update([_uav(500.0, 500.0, 800.0)], 1.0)
    assert cov.eligible_cells() == 0
    assert cov.coverage_fraction() == 0.0
