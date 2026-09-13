"""Greedy nearest-known-fire / exploratory search.

If the observation contains an active known fire, steer toward it.
When the UAV is close enough to overfly (normalized distance below
suppression hover threshold), HOVER so intensity can be driven down.

Detection alone does not extinguish. Hovering far from the fire does
not extinguish. The UAV must close to suppression range.

Otherwise fly toward the least-covered local patch cell.
Uses communicated known fires when they appear in the observation;
does not invent global knowledge.
"""

from __future__ import annotations

import numpy as np

from app.geography.coordinate_transform import heading_from_vector
from app.marl.action import ACTION_NAMES
from app.marl.observation import N_KNOWN_FIRES, PATCH
from app.marl.policy import Policy, PolicyInfo

_NAME_TO_IDX = {n: i for i, n in enumerate(ACTION_NAMES)}
_HEADING_ACTIONS = list(range(8))

# Observation dx/dy are / AOI extent. Must sit on the fire point
# (default overlap 120 m). 0.012 * 8 km ≈ 96 m.
_HOVER_NORM = 0.012
_WAYPOINTS = (
    (0.22, 0.22),
    (0.78, 0.22),
    (0.78, 0.78),
    (0.22, 0.78),
    (0.50, 0.50),
    (0.50, 0.22),
    (0.50, 0.78),
    (0.22, 0.50),
)


def _heading_to_action(heading: float) -> int:
    idx = int((heading + np.pi / 8.0) // (np.pi / 4.0)) % 8
    return _HEADING_ACTIONS[idx]


def known_fire_slots(observation: np.ndarray) -> list[tuple[float, float, float]]:
    fires: list[tuple[float, float, float]] = []
    for i in range(N_KNOWN_FIRES):
        dx = float(observation[11 + i * 3])
        dy = float(observation[11 + i * 3 + 1])
        inten = float(observation[11 + i * 3 + 2])
        if inten <= 1e-6 and abs(dx) < 1e-8 and abs(dy) < 1e-8:
            continue
        fires.append((float(np.hypot(dx, dy)), dx, dy))
    return fires


def prosecute_known_fire(observation: np.ndarray, agent_index: int = 0) -> int | None:
    """Steer to an assigned known fire; HOVER when overlapping. None if no fire."""
    fires = known_fire_slots(observation)
    if not fires:
        return None
    fires.sort(key=lambda t: t[0])
    dist, dx, dy = fires[int(agent_index) % len(fires)]
    if dist <= _HOVER_NORM:
        return _NAME_TO_IDX["HOVER"]
    return _heading_to_action(heading_from_vector(dx, dy))


class GreedyPolicy(Policy):
    def __init__(self) -> None:
        self._wp: dict[str, int] = {}
        self.info = PolicyInfo(
            name="greedy",
            kind="baseline",
            algorithm="nearest_known_fire",
            checkpoint_path=None,
            version="1.1",
            cooperative=True,
            uses_communication=True,
            trained=False,
            notes=(
                "Steers toward an assigned active fire in the observation "
                "(own detections + communicated reports) and HOVERs when "
                "close enough to suppress. Detection does not put the fire "
                "out. Otherwise searches the AOI via staggered waypoints."
            ),
        )

    def reset(self, seed: int | None = None) -> None:
        del seed
        self._wp.clear()

    def act(self, observation: np.ndarray, agent_id: str, state=None) -> int:
        idx = int(state.get("agent_index", 0)) if state else 0
        fire_act = prosecute_known_fire(observation, idx)
        if fire_act is not None:
            return fire_act

        start = 11 + N_KNOWN_FIRES * 3 + 3 * 4
        patch = observation[start : start + PATCH * PATCH].reshape(PATCH, PATCH)
        half = PATCH // 2
        candidates: list[tuple[float, int, int]] = []
        for r in range(PATCH):
            for c in range(PATCH):
                if patch[r, c] == 0.0:
                    dc = c - half
                    dr = r - half
                    dy = -dr
                    d2 = dc * dc + dy * dy
                    if d2 > 0:
                        candidates.append((d2, dc, dy))
        if candidates:
            candidates.sort()
            _, dc, dy = candidates[0]
            return _heading_to_action(heading_from_vector(float(dc), float(dy)))

        x_norm = float(observation[0])
        y_norm = float(observation[1])
        base = int(state.get("agent_index", 0)) if state else 0
        i = self._wp.get(agent_id, base % len(_WAYPOINTS))
        tx, ty = _WAYPOINTS[i]
        if float(np.hypot(tx - x_norm, ty - y_norm)) < 0.07:
            i = (i + 1) % len(_WAYPOINTS)
            self._wp[agent_id] = i
            tx, ty = _WAYPOINTS[i]
        return _heading_to_action(heading_from_vector(tx - x_norm, ty - y_norm))
