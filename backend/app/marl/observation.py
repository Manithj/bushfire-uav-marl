"""Partial-observability observation builder.

Default execution observation is local / communication-constrained.
Global fire locations are included only when allow_global_state=True
(training privilege or an explicitly configured experiment).

Vector layout (length = OBS_DIM):

    [0]  x_norm          own easting / extent_x in [0, 1]
    [1]  y_norm          own northing / extent_y in [0, 1]
    [2]  z_norm          altitude / max_altitude
    [3]  sin(heading)
    [4]  cos(heading)
    [5]  speed_norm
    [6]  battery_frac
    [7]  wind_speed_norm ( / 40 m/s )
    [8]  sin(downwind)
    [9]  cos(downwind)
    [10] t_norm
    [11..19]  3 known fires: dx_norm, dy_norm, intensity   (0 if absent)
    [20..31]  3 nearby UAVs (comm range): dx, dy, sinψ, cosψ
    [32..56]  5×5 local coverage patch around the agent

Normalization is documented here and must stay consistent with training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from gymnasium import spaces

from app.geography.coordinate_transform import heading_components
from app.simulation.uav import UAVAgent

N_KNOWN_FIRES = 3
N_NEAR_UAVS = 3
PATCH = 5
OBS_DIM = 11 + N_KNOWN_FIRES * 3 + N_NEAR_UAVS * 4 + PATCH * PATCH  # 57


@dataclass(frozen=True)
class ObservationSpec:
    dim: int
    layout: list[str]
    partial: bool
    allow_global_state: bool
    communication_assumption: str
    normalization: str


def observation_space() -> spaces.Box:
    return spaces.Box(low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)


def observation_spec(allow_global_state: bool) -> ObservationSpec:
    layout = [
        "x_norm", "y_norm", "z_norm", "sin_heading", "cos_heading",
        "speed_norm", "battery_frac", "wind_speed_norm", "sin_downwind",
        "cos_downwind", "t_norm",
    ]
    for i in range(N_KNOWN_FIRES):
        layout += [f"fire{i}_dx", f"fire{i}_dy", f"fire{i}_intensity"]
    for i in range(N_NEAR_UAVS):
        layout += [f"uav{i}_dx", f"uav{i}_dy", f"uav{i}_sin", f"uav{i}_cos"]
    for r in range(PATCH):
        for c in range(PATCH):
            layout.append(f"cov_{r}_{c}")
    return ObservationSpec(
        dim=OBS_DIM,
        layout=layout,
        partial=not allow_global_state,
        allow_global_state=allow_global_state,
        communication_assumption=(
            "Nearby UAV features include only agents within communication range. "
            "Known-fire features include fires this agent has detected or received "
            "via a delivered communication message"
            + (" plus all true fire sources when allow_global_state is True." if allow_global_state else ".")
        ),
        normalization=(
            "Positions divided by AOI extent, clipped to [-1, 1]. "
            "Trigonometric heading encodings. Battery in [0, 1]. "
            "Wind speed divided by 40 m/s. Time divided by episode duration."
        ),
    )


def build_observation(
    agent: UAVAgent,
    all_uavs: list[UAVAgent],
    known_fires: list[tuple[float, float, float]],
    coverage_patch: np.ndarray,
    wind_speed_mps: float,
    downwind_rad: float,
    time_norm: float,
    extent_x: float,
    extent_y: float,
    max_altitude: float,
    max_speed: float,
    origin_x: float,
    origin_y: float,
) -> np.ndarray:
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    x_norm = (agent.body.x - origin_x) / max(extent_x, 1.0)
    y_norm = (agent.body.y - origin_y) / max(extent_y, 1.0)
    east, north = heading_components(agent.body.heading)
    # heading_components returns (sin, cos) of nav heading = (east, north)
    obs[0] = np.clip(x_norm, 0.0, 1.0)
    obs[1] = np.clip(y_norm, 0.0, 1.0)
    obs[2] = np.clip(agent.body.z / max(max_altitude, 1.0), 0.0, 1.0)
    obs[3] = east  # sin(heading)
    obs[4] = north  # cos(heading)
    obs[5] = np.clip(agent.body.speed / max(max_speed, 1e-6), 0.0, 1.0)
    obs[6] = np.clip(agent.battery_fraction(), 0.0, 1.0)
    obs[7] = np.clip(wind_speed_mps / 40.0, 0.0, 2.0)
    obs[8] = float(np.sin(downwind_rad))
    obs[9] = float(np.cos(downwind_rad))
    obs[10] = float(np.clip(time_norm, 0.0, 1.0))

    fires_sorted = sorted(
        known_fires,
        key=lambda f: (f[0] - agent.body.x) ** 2 + (f[1] - agent.body.y) ** 2,
    )
    base = 11
    for i in range(N_KNOWN_FIRES):
        if i >= len(fires_sorted):
            break
        fx, fy, inten = fires_sorted[i]
        obs[base + i * 3 + 0] = np.clip((fx - agent.body.x) / max(extent_x, 1.0), -1.0, 1.0)
        obs[base + i * 3 + 1] = np.clip((fy - agent.body.y) / max(extent_y, 1.0), -1.0, 1.0)
        obs[base + i * 3 + 2] = np.clip(inten, 0.0, 1.0)

    others = []
    for other in all_uavs:
        if other.uav_id == agent.uav_id:
            continue
        d = float(np.hypot(other.body.x - agent.body.x, other.body.y - agent.body.y))
        if d <= agent.config.communication_range_m:
            others.append((d, other))
    others.sort(key=lambda t: t[0])
    base = 11 + N_KNOWN_FIRES * 3
    for i in range(N_NEAR_UAVS):
        if i >= len(others):
            break
        other = others[i][1]
        oe, on = heading_components(other.body.heading)
        obs[base + i * 4 + 0] = np.clip((other.body.x - agent.body.x) / max(extent_x, 1.0), -1.0, 1.0)
        obs[base + i * 4 + 1] = np.clip((other.body.y - agent.body.y) / max(extent_y, 1.0), -1.0, 1.0)
        obs[base + i * 4 + 2] = oe
        obs[base + i * 4 + 3] = on

    patch = np.asarray(coverage_patch, dtype=np.float32).reshape(-1)
    start = 11 + N_KNOWN_FIRES * 3 + N_NEAR_UAVS * 4
    obs[start : start + PATCH * PATCH] = patch[: PATCH * PATCH]
    return obs


def local_coverage_patch(coverage, agent: UAVAgent) -> np.ndarray:
    """5×5 explored flags around the agent, row-major, unknown=0."""
    col = int((agent.body.x - coverage.origin_x) / coverage.cell_size_m)
    row = int((coverage.origin_y + coverage.extent_y - agent.body.y) / coverage.cell_size_m)
    half = PATCH // 2
    patch = np.zeros((PATCH, PATCH), dtype=np.float32)
    for dr in range(-half, half + 1):
        for dc in range(-half, half + 1):
            r, c = row + dr, col + dc
            if 0 <= r < coverage.height and 0 <= c < coverage.width:
                if coverage.inaccessible[r, c]:
                    patch[dr + half, dc + half] = -1.0
                elif coverage.explored[r, c]:
                    patch[dr + half, dc + half] = 1.0
    return patch
