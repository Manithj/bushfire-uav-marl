"""Explicit action space.

Discrete-9 (default):
    0 N, 1 NE, 2 E, 3 SE, 4 S, 5 SW, 6 W, 7 NW, 8 HOVER

Each action is mapped to a target heading and speed, then validated
against UAV dynamic constraints in uav_dynamics.apply_action.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from gymnasium import spaces

from app.simulation.uav_dynamics import UAVDynamicsConfig

ACTION_NAMES = ("N", "NE", "E", "SE", "S", "SW", "W", "NW", "HOVER")
N_DISCRETE_ACTIONS = 9

# Navigation heading in radians: 0 = North, clockwise.
_HEADINGS = tuple(i * np.pi / 4.0 for i in range(8))


@dataclass(frozen=True)
class Command:
    heading: float
    speed: float
    altitude: float
    action_index: int
    name: str


def discrete_space() -> spaces.Discrete:
    return spaces.Discrete(N_DISCRETE_ACTIONS)


def decode_discrete(
    action: int,
    altitude: float,
    cfg: UAVDynamicsConfig,
    current_heading: float,
) -> Command:
    if action < 0 or action >= N_DISCRETE_ACTIONS:
        raise ValueError(f"Invalid discrete action {action}")
    if action == 8:
        return Command(
            heading=current_heading,
            speed=0.0,
            altitude=altitude,
            action_index=8,
            name="HOVER",
        )
    return Command(
        heading=float(_HEADINGS[action]),
        speed=cfg.cruise_speed_mps,
        altitude=altitude,
        action_index=action,
        name=ACTION_NAMES[action],
    )


def validate_action(action: int) -> int:
    a = int(action)
    if a < 0 or a >= N_DISCRETE_ACTIONS:
        raise ValueError(f"Action {action} outside discrete-9 space")
    return a
