"""In-process simulation manager for the API (not used during training)."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from app.schemas.scenario import ScenarioConfig
from app.simulation.environment import WildfireEnv
from app.simulation.replay import ReplayLog, replay_from_env

SimStatus = Literal["created", "running", "paused", "stopped", "finished"]


@dataclass
class SimulationHandle:
    simulation_id: str
    created_at: str
    scenario: ScenarioConfig
    env: WildfireEnv
    status: SimStatus = "created"
    speed: float = 1.0
    replay: ReplayLog | None = None
    trails: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    subscribers: set[asyncio.Queue] = field(default_factory=set)
    event_cursor: int = 0
    _task: asyncio.Task | None = None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class SimulationManager:
    def __init__(self) -> None:
        self._sims: dict[str, SimulationHandle] = {}
        self._scenarios: dict[str, ScenarioConfig] = {}

    def save_scenario(self, scenario: ScenarioConfig) -> str:
        sid = str(uuid.uuid4())
        self._scenarios[sid] = scenario
        return sid

    def list_scenarios(self) -> list[dict[str, Any]]:
        return [{"scenario_id": k, "name": v.name, "seed": v.seed} for k, v in self._scenarios.items()]

    def get_scenario(self, scenario_id: str) -> ScenarioConfig | None:
        return self._scenarios.get(scenario_id)

    def create(self, scenario: ScenarioConfig) -> SimulationHandle:
        env = WildfireEnv(scenario)
        env.reset(seed=scenario.seed)
        hid = str(uuid.uuid4())
        handle = SimulationHandle(
            simulation_id=hid,
            created_at=datetime.now(timezone.utc).isoformat(),
            scenario=scenario,
            env=env,
        )
        self._record_trails(handle)
        handle.replay = replay_from_env(env)
        handle.replay.record(env)
        self._sims[hid] = handle
        return handle

    def get(self, simulation_id: str) -> SimulationHandle | None:
        return self._sims.get(simulation_id)

    def list_sims(self) -> list[dict[str, Any]]:
        return [
            {
                "simulation_id": h.simulation_id,
                "status": h.status,
                "time_s": h.env.time_s,
                "policy": h.scenario.policy.name,
                "seed": h.scenario.seed,
            }
            for h in self._sims.values()
        ]

    async def start(self, simulation_id: str) -> SimulationHandle:
        h = self._require(simulation_id)
        if h.status in ("finished",):
            return h
        if h.status == "stopped":
            h.env.reset(seed=h.scenario.seed)
            h.trails.clear()
            self._record_trails(h)
        h.status = "running"
        if h._task is None or h._task.done():
            h._task = asyncio.create_task(self._run_loop(h), name=f"sim-{h.simulation_id}")
        await self._broadcast(h, {"type": "status", "status": h.status, "state": h.env.public_state()})
        return h

    async def pause(self, simulation_id: str) -> SimulationHandle:
        h = self._require(simulation_id)
        if h.status == "running":
            h.status = "paused"
            await self._broadcast(h, {"type": "status", "status": h.status, "state": h.env.public_state()})
        return h

    async def resume(self, simulation_id: str) -> SimulationHandle:
        return await self.start(simulation_id)

    async def stop(self, simulation_id: str) -> SimulationHandle:
        h = self._require(simulation_id)
        h.status = "stopped"
        if h._task and not h._task.done():
            h._task.cancel()
        await self._broadcast(h, {"type": "status", "status": h.status, "state": h.env.public_state()})
        return h

    async def reset(self, simulation_id: str) -> SimulationHandle:
        h = self._require(simulation_id)
        if h._task and not h._task.done():
            h._task.cancel()
            h._task = None
        h.env.reset(seed=h.scenario.seed)
        h.status = "created"
        h.trails.clear()
        self._record_trails(h)
        h.replay = replay_from_env(h.env)
        h.replay.record(h.env)
        h.event_cursor = 0
        await self._broadcast(h, {"type": "status", "status": h.status, "state": self.public(h)})
        return h

    def set_speed(self, simulation_id: str, speed: float) -> SimulationHandle:
        h = self._require(simulation_id)
        if speed not in (0.5, 1.0, 2.0, 5.0):
            raise ValueError("speed must be 0.5, 1, 2, or 5")
        h.speed = speed
        return h

    async def step_once(self, simulation_id: str, n: int = 1) -> SimulationHandle:
        h = self._require(simulation_id)
        if h.status == "finished":
            return h
        if h.status == "running":
            h.status = "paused"
        for _ in range(max(1, n)):
            if h.env._terminated:
                break
            h.env.step_policy()
            self._after_step(h)
        if h.env._terminated:
            h.status = "finished"
        await self._broadcast(h, {"type": "tick", "state": self.public(h)})
        return h

    def public(self, h: SimulationHandle) -> dict[str, Any]:
        state = h.env.public_state()
        state["status"] = h.status
        state["speed"] = h.speed
        state["simulation_id"] = h.simulation_id
        state["trails"] = {
            uid: [{"latitude": a, "longitude": b} for a, b in pts[-400:]]
            for uid, pts in h.trails.items()
        }
        state["events"] = [
            {
                "time_s": e.time_s,
                "kind": e.kind,
                "message": e.message,
                "agent_id": e.agent_id,
                "fire_id": e.fire_id,
            }
            for e in h.env.events.events
        ]
        return state

    def _require(self, simulation_id: str) -> SimulationHandle:
        h = self._sims.get(simulation_id)
        if h is None:
            raise KeyError(simulation_id)
        return h

    def _record_trails(self, h: SimulationHandle) -> None:
        for u in h.env.uavs:
            h.trails.setdefault(u.uav_id, []).append((u.lat, u.lon))

    def _after_step(self, h: SimulationHandle) -> None:
        self._record_trails(h)
        if h.replay is None:
            h.replay = replay_from_env(h.env)
        h.replay.record(h.env, stride=1)
        if h.env._terminated:
            h.replay.events = h.env.events.as_dicts()
            h.status = "finished"

    async def _run_loop(self, h: SimulationHandle) -> None:
        try:
            while h.status == "running" and not h.env._terminated:
                async with h._lock:
                    h.env.step_policy()
                    self._after_step(h)
                    payload = {"type": "tick", "state": self.public(h)}
                await self._broadcast(h, payload)
                await asyncio.sleep(max(0.02, h.env.dt / max(h.speed, 0.1)))
            if h.env._terminated:
                h.status = "finished"
                await self._broadcast(h, {"type": "status", "status": h.status, "state": self.public(h)})
        except asyncio.CancelledError:
            return

    async def subscribe(self, simulation_id: str) -> asyncio.Queue:
        h = self._require(simulation_id)
        q: asyncio.Queue = asyncio.Queue(maxsize=8)
        h.subscribers.add(q)
        await q.put({"type": "snapshot", "state": self.public(h)})
        return q

    def unsubscribe(self, simulation_id: str, q: asyncio.Queue) -> None:
        h = self._sims.get(simulation_id)
        if h:
            h.subscribers.discard(q)

    async def _broadcast(self, h: SimulationHandle, payload: dict[str, Any]) -> None:
        dead = []
        for q in h.subscribers:
            if q.full():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            h.subscribers.discard(q)


MANAGER = SimulationManager()
