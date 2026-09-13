"""Simulation event log. Every entry is produced by the engine, not the UI."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SimEvent:
    time_s: float
    kind: str
    message: str
    agent_id: str | None = None
    fire_id: str | None = None
    data: dict | None = None


@dataclass
class EventLog:
    events: list[SimEvent] = field(default_factory=list)

    def add(
        self,
        time_s: float,
        kind: str,
        message: str,
        agent_id: str | None = None,
        fire_id: str | None = None,
        data: dict | None = None,
    ) -> SimEvent:
        ev = SimEvent(
            time_s=time_s,
            kind=kind,
            message=message,
            agent_id=agent_id,
            fire_id=fire_id,
            data=data,
        )
        self.events.append(ev)
        return ev

    def since(self, index: int) -> list[SimEvent]:
        return self.events[index:]

    def as_dicts(self, limit: int | None = None) -> list[dict]:
        items = self.events if limit is None else self.events[-limit:]
        return [
            {
                "time_s": e.time_s,
                "kind": e.kind,
                "message": e.message,
                "agent_id": e.agent_id,
                "fire_id": e.fire_id,
                "data": e.data,
            }
            for e in items
        ]
