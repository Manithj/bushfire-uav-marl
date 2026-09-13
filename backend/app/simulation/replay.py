"""Replay log: enough state to reproduce and replay a run.

Stores scenario JSON, seed, policy identity, per-step UAV/fire/metrics
summaries, detections, comms, and the event log. Fire grids are stored
sparsely every snapshot_stride steps to bound memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReplayFrame:
    time_s: float
    step: int
    uavs: list[dict[str, Any]]
    fires: list[dict[str, Any]]
    links: list[dict[str, Any]]
    metrics: dict[str, Any]
    coverage_fraction: float
    burning_cells: int
    burned_cells: int


@dataclass
class ReplayLog:
    scenario: dict[str, Any]
    seed: int
    policy_name: str
    policy_version: str
    checkpoint_path: str | None
    timestep_s: float
    frames: list[ReplayFrame] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    detections: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    snapshot_stride: int = 1

    def record(self, env, stride: int = 1) -> None:
        if env.step_count % stride != 0 and not env._terminated:
            return
        state = env.public_state()
        self.frames.append(
            ReplayFrame(
                time_s=state["time_s"],
                step=state["step"],
                uavs=[
                    {
                        "uav_id": u["uav_id"],
                        "latitude": u["latitude"],
                        "longitude": u["longitude"],
                        "altitude_m": u["altitude_m"],
                        "heading_deg": u["heading_deg"],
                        "speed_mps": u["speed_mps"],
                        "battery_fraction": u["battery_fraction"],
                        "status": u["status"],
                    }
                    for u in state["uavs"]
                ],
                fires=[
                    {
                        "fire_id": f["fire_id"],
                        "latitude": f["latitude"],
                        "longitude": f["longitude"],
                        "active": f["active"],
                        "detected": f["detected"],
                        "perimeter": f["perimeter"],
                    }
                    for f in state["fires"]
                ],
                links=state["communication_links"],
                metrics=state["metrics"],
                coverage_fraction=state["coverage_fraction"],
                burning_cells=state["fire_grid"]["burning_cells"],
                burned_cells=state["fire_grid"]["burned_cells"],
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "seed": self.seed,
            "policy_name": self.policy_name,
            "policy_version": self.policy_version,
            "checkpoint_path": self.checkpoint_path,
            "timestep_s": self.timestep_s,
            "frames": [f.__dict__ for f in self.frames],
            "events": self.events,
            "detections": self.detections,
            "messages": self.messages,
        }


def replay_from_env(env) -> ReplayLog:
    pol = env.policy
    log = ReplayLog(
        scenario=env.scenario.model_dump(),
        seed=env.scenario.seed,
        policy_name=pol.info.name if pol else "unknown",
        policy_version=pol.info.version if pol else "",
        checkpoint_path=pol.info.checkpoint_path if pol else None,
        timestep_s=env.dt,
    )
    log.events = env.events.as_dicts()
    log.detections = [
        {
            "time_s": d.time_s,
            "uav_id": d.uav_id,
            "is_true": d.is_true,
            "fire_id": d.fire_id,
            "latitude": d.lat,
            "longitude": d.lon,
            "probability_used": d.probability_used,
        }
        for d in env.detections
    ]
    log.messages = [
        {
            "time_s": m.time_s,
            "sender": m.sender,
            "receiver": m.receiver,
            "delivered": m.delivered,
            "fire_id": m.fire_id,
            "reason": m.reason,
        }
        for m in env.comms.messages
    ]
    return log
