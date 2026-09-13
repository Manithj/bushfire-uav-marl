"""Explicit inter-agent communication.

A link exists between UAV i and j iff planimetric distance ≤ min(range_i, range_j)
and both have remaining battery. Messages are fire-detection broadcasts.

One-hop only by default. Multi-hop flooding is optional and off unless enabled.
Communication failures: a message on an existing link is dropped with
probability p_fail (seeded). If no link exists, the send is a failure.

Do not claim communication occurs unless this module processes a message.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.simulation.sensor_model import DetectionEvent
from app.simulation.uav import UAVAgent


@dataclass(frozen=True)
class CommLink:
    a: str
    b: str
    distance_m: float


@dataclass
class CommMessage:
    time_s: float
    sender: str
    receiver: str
    payload_type: str
    fire_id: str | None
    delivered: bool
    reason: str


@dataclass
class CommunicationModel:
    failure_probability: float = 0.0
    multi_hop: bool = False
    links: list[CommLink] = field(default_factory=list)
    messages: list[CommMessage] = field(default_factory=list)
    sent: int = 0
    received: int = 0
    failures: int = 0

    def reset_step(self) -> None:
        self.links = []

    def update_links(self, uavs: list[UAVAgent]) -> list[CommLink]:
        links: list[CommLink] = []
        n = len(uavs)
        for i in range(n):
            a = uavs[i]
            if a.body.battery_j <= 0:
                continue
            for j in range(i + 1, n):
                b = uavs[j]
                if b.body.battery_j <= 0:
                    continue
                d = float(np.hypot(a.body.x - b.body.x, a.body.y - b.body.y))
                rng = min(a.config.communication_range_m, b.config.communication_range_m)
                if d <= rng:
                    links.append(CommLink(a=a.uav_id, b=b.uav_id, distance_m=d))
        self.links = links
        return links

    def neighbours(self, uav_id: str) -> list[str]:
        out: list[str] = []
        for link in self.links:
            if link.a == uav_id:
                out.append(link.b)
            elif link.b == uav_id:
                out.append(link.a)
        return out

    def connected_fraction(self, n_uavs: int) -> float | None:
        if n_uavs <= 1:
            return None
        possible = n_uavs * (n_uavs - 1) / 2.0
        if possible <= 0:
            return None
        return float(len(self.links) / possible)

    def broadcast_detection(
        self,
        event: DetectionEvent,
        uavs: list[UAVAgent],
        rng: np.random.Generator,
    ) -> list[CommMessage]:
        by_id = {u.uav_id: u for u in uavs}
        sender = by_id.get(event.uav_id)
        if sender is None:
            return []
        produced: list[CommMessage] = []
        targets = self.neighbours(event.uav_id)
        if self.multi_hop:
            targets = [u.uav_id for u in uavs if u.uav_id != event.uav_id]
        if not targets:
            self.sent += 1
            self.failures += 1
            msg = CommMessage(
                time_s=event.time_s,
                sender=event.uav_id,
                receiver="",
                payload_type="detection",
                fire_id=event.fire_id,
                delivered=False,
                reason="no_neighbour_in_range",
            )
            self.messages.append(msg)
            produced.append(msg)
            return produced

        for recv_id in targets:
            self.sent += 1
            sender.messages_sent += 1
            in_graph = recv_id in self.neighbours(event.uav_id)
            drop = (not in_graph) or (rng.random() < self.failure_probability)
            if drop:
                self.failures += 1
                msg = CommMessage(
                    time_s=event.time_s,
                    sender=event.uav_id,
                    receiver=recv_id,
                    payload_type="detection",
                    fire_id=event.fire_id,
                    delivered=False,
                    reason="link_failure" if in_graph else "not_in_range",
                )
            else:
                self.received += 1
                recv = by_id[recv_id]
                recv.messages_received += 1
                if event.is_true and event.fire_id:
                    recv.known_fire_ids.add(event.fire_id)
                msg = CommMessage(
                    time_s=event.time_s,
                    sender=event.uav_id,
                    receiver=recv_id,
                    payload_type="detection",
                    fire_id=event.fire_id,
                    delivered=True,
                    reason="ok",
                )
            self.messages.append(msg)
            produced.append(msg)
        return produced
