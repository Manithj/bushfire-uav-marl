from __future__ import annotations

import numpy as np

from app.simulation.uav_dynamics import UAVBodyState, UAVDynamicsConfig, apply_action, battery_fraction


def test_moves_north_and_respects_speed_cap() -> None:
    cfg = UAVDynamicsConfig(max_speed_mps=10.0, cruise_speed_mps=10.0)
    s = UAVBodyState(x=0.0, y=0.0, z=100.0, heading=0.0, speed=0.0, battery_j=cfg.battery_capacity_j)
    s = apply_action(s, 0.0, 10.0, 100.0, cfg, 1.0, -1e5, 1e5, -1e5, 1e5)
    assert s.y > 0.0
    assert abs(s.x) < 1e-6
    assert s.speed <= cfg.max_speed_mps + 1e-9


def test_battery_decreases() -> None:
    cfg = UAVDynamicsConfig()
    s = UAVBodyState(x=0.0, y=0.0, z=100.0, heading=0.0, speed=8.0, battery_j=cfg.battery_capacity_j)
    s2 = apply_action(s, 0.0, 8.0, 100.0, cfg, 2.0, -1e5, 1e5, -1e5, 1e5)
    assert s2.battery_j < s.battery_j
    assert 0.0 <= battery_fraction(s2, cfg) <= 1.0


def test_stays_inside_bounds() -> None:
    cfg = UAVDynamicsConfig(max_speed_mps=40.0)
    s = UAVBodyState(x=5.0, y=5.0, z=100.0, heading=0.0, speed=40.0, battery_j=cfg.battery_capacity_j)
    for _ in range(20):
        s = apply_action(s, 0.0, 40.0, 100.0, cfg, 1.0, 0.0, 10.0, 0.0, 10.0)
    assert 0.0 <= s.x <= 10.0
    assert 0.0 <= s.y <= 10.0


def test_turn_rate_limit() -> None:
    cfg = UAVDynamicsConfig(max_turn_rate_rps=0.4)
    s = UAVBodyState(x=0.0, y=0.0, z=100.0, heading=0.0, speed=5.0, battery_j=cfg.battery_capacity_j)
    s2 = apply_action(s, np.pi, 5.0, 100.0, cfg, 1.0, -1e5, 1e5, -1e5, 1e5)
    assert abs(s2.heading - 0.0) <= cfg.max_turn_rate_rps + 1e-6
