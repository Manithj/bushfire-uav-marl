"""Kinematic UAV model in the local ENU frame.

Assumptions (documented, not aerodynamic CFD):
- Point-mass kinematic model with bounded speed, climb rate, and turn rate.
- No wind drift on the airframe (wind affects fire only). This is a
  simplification, not a claim that wind does not affect UAVs.
- Battery energy is depleted by a hover + speed-dependent power model.

Heading: 0 = North, clockwise (navigation).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.geography.coordinate_transform import heading_components, wrap_angle


@dataclass
class UAVDynamicsConfig:
    max_speed_mps: float = 18.0
    cruise_speed_mps: float = 12.0
    min_speed_mps: float = 0.0
    max_climb_mps: float = 3.0
    max_turn_rate_rps: float = 0.7
    min_altitude_m: float = 40.0
    max_altitude_m: float = 250.0
    # Power model: P = P_hover + k_speed * v^2  (Watts)
    hover_power_w: float = 180.0
    speed_power_coeff: float = 0.35
    battery_capacity_j: float = 180_000.0  # 50 Wh
    return_to_base_fraction: float = 0.30


@dataclass
class UAVBodyState:
    x: float
    y: float
    z: float
    heading: float
    speed: float
    battery_j: float
    distance_m: float = 0.0
    airborne_s: float = 0.0
    idle_s: float = 0.0


def apply_action(
    state: UAVBodyState,
    target_heading: float,
    target_speed: float,
    target_altitude: float,
    cfg: UAVDynamicsConfig,
    dt: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> UAVBodyState:
    """Integrate one kinematic step. All constraints are enforced here."""
    if dt <= 0:
        return state

    heading_err = wrap_angle(target_heading - state.heading)
    max_turn = cfg.max_turn_rate_rps * dt
    dpsi = float(np.clip(heading_err, -max_turn, max_turn))
    heading = (state.heading + dpsi) % (2.0 * np.pi)

    speed = float(np.clip(target_speed, cfg.min_speed_mps, cfg.max_speed_mps))
    # First-order speed response
    speed = state.speed + float(np.clip(speed - state.speed, -6.0 * dt, 6.0 * dt))
    speed = float(np.clip(speed, 0.0, cfg.max_speed_mps))

    alt_err = target_altitude - state.z
    dz = float(np.clip(alt_err, -cfg.max_climb_mps * dt, cfg.max_climb_mps * dt))
    z = float(np.clip(state.z + dz, cfg.min_altitude_m, cfg.max_altitude_m))

    east, north = heading_components(heading)
    x = state.x + east * speed * dt
    y = state.y + north * speed * dt
    x = float(np.clip(x, x_min, x_max))
    y = float(np.clip(y, y_min, y_max))

    dist = float(np.hypot(x - state.x, y - state.y))
    power_w = cfg.hover_power_w + cfg.speed_power_coeff * speed * speed
    energy = power_w * dt
    battery = max(0.0, state.battery_j - energy)

    airborne = state.airborne_s + dt
    idle = state.idle_s + (dt if speed < 0.4 else 0.0)

    return UAVBodyState(
        x=x,
        y=y,
        z=z,
        heading=heading,
        speed=speed,
        battery_j=battery,
        distance_m=state.distance_m + dist,
        airborne_s=airborne,
        idle_s=idle,
    )


def battery_fraction(state: UAVBodyState, cfg: UAVDynamicsConfig) -> float:
    if cfg.battery_capacity_j <= 0:
        return 0.0
    return float(np.clip(state.battery_j / cfg.battery_capacity_j, 0.0, 1.0))
