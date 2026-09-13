from __future__ import annotations

import numpy as np

from app.simulation.communication import CommunicationModel
from app.simulation.sensor_model import DetectionEvent
from app.simulation.uav import UAVAgent, UAVConfig
from app.simulation.uav_dynamics import UAVBodyState, UAVDynamicsConfig


def _uav(uid: str, x: float, y: float, comm: float) -> UAVAgent:
    dyn = UAVDynamicsConfig()
    cfg = UAVConfig(uav_id=uid, dynamics=dyn, communication_range_m=comm)
    body = UAVBodyState(x=x, y=y, z=100.0, heading=0.0, speed=0.0, battery_j=dyn.battery_capacity_j)
    return UAVAgent(config=cfg, body=body, lat=0.0, lon=0.0)


def test_link_exists_within_range() -> None:
    a = _uav("a", 0.0, 0.0, 1000.0)
    b = _uav("b", 200.0, 0.0, 1000.0)
    c = _uav("c", 5000.0, 0.0, 1000.0)
    model = CommunicationModel()
    links = model.update_links([a, b, c])
    pairs = {(l.a, l.b) for l in links}
    assert ("a", "b") in pairs or ("b", "a") in pairs
    assert all("c" not in (l.a, l.b) for l in links)


def test_message_propagates_to_neighbour() -> None:
    a = _uav("a", 0.0, 0.0, 1000.0)
    b = _uav("b", 100.0, 0.0, 1000.0)
    model = CommunicationModel(failure_probability=0.0)
    model.update_links([a, b])
    ev = DetectionEvent(1.0, "a", True, "fire_01", 0, 0, 0, 0, 1.0, 10.0, 0.9)
    msgs = model.broadcast_detection(ev, [a, b], np.random.default_rng(0))
    assert any(m.delivered and m.receiver == "b" for m in msgs)
    assert "fire_01" in b.known_fire_ids
    assert model.received >= 1
