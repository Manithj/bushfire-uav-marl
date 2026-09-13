from __future__ import annotations

from app.simulation.collision import CollisionMonitor
from app.simulation.uav import UAVAgent, UAVConfig, UAVStatus
from app.simulation.uav_dynamics import UAVBodyState, UAVDynamicsConfig


def _uav(uid: str, x: float, y: float, z: float = 100.0) -> UAVAgent:
    dyn = UAVDynamicsConfig()
    cfg = UAVConfig(uav_id=uid, dynamics=dyn, min_separation_m=40.0)
    body = UAVBodyState(x=x, y=y, z=z, heading=0.0, speed=5.0, battery_j=dyn.battery_capacity_j)
    return UAVAgent(config=cfg, body=body, lat=0.0, lon=0.0)


def test_near_collision_from_geometry() -> None:
    mon = CollisionMonitor(collision_radius_m=8.0)
    ev = mon.check([_uav("a", 0.0, 0.0), _uav("b", 20.0, 0.0)], 1.0)
    assert any(not e.is_collision for e in ev)
    assert mon.near_events >= 1


def test_collision_from_geometry() -> None:
    mon = CollisionMonitor(collision_radius_m=8.0)
    ev = mon.check([_uav("a", 0.0, 0.0), _uav("b", 3.0, 0.0)], 1.0)
    assert any(e.is_collision for e in ev)
    assert mon.collisions >= 1


def test_altitude_separation_avoids_event() -> None:
    mon = CollisionMonitor(collision_radius_m=8.0, vertical_threshold_m=15.0)
    ev = mon.check([_uav("a", 0.0, 0.0, 100.0), _uav("b", 3.0, 0.0, 140.0)], 1.0)
    assert ev == []
