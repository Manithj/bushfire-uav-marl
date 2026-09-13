"""UAV agent state and configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from app.simulation.uav_dynamics import UAVBodyState, UAVDynamicsConfig, battery_fraction


class UAVStatus(str, Enum):
    IDLE = "idle"
    SEARCHING = "searching"
    INVESTIGATING = "investigating"
    SUPPRESSING = "suppressing"
    RETURNING = "returning"
    LANDED = "landed"
    COLLIDED = "collided"
    BATTERY_DEPLETED = "battery_depleted"


@dataclass
class UAVConfig:
    uav_id: str
    dynamics: UAVDynamicsConfig
    sensor_range_m: float = 500.0
    field_of_view_deg: float = 360.0
    detection_probability: float = 0.85
    false_positive_probability: float = 0.002
    false_negative_probability: float = 0.10
    observation_noise_m: float = 15.0
    communication_range_m: float = 2000.0
    min_separation_m: float = 40.0
    start_lat: float | None = None
    start_lon: float | None = None
    start_altitude_m: float = 120.0
    start_heading_deg: float = 0.0
    suppression_range_m: float = 120.0


@dataclass
class UAVAgent:
    config: UAVConfig
    body: UAVBodyState
    lat: float
    lon: float
    status: UAVStatus = UAVStatus.SEARCHING
    last_action: int = 8
    last_reward: float = 0.0
    detections_true: int = 0
    detections_false: int = 0
    covered_cells: int = 0
    actions_taken: int = 0
    known_fire_ids: set[str] = field(default_factory=set)
    messages_sent: int = 0
    messages_received: int = 0
    near_collisions: int = 0
    collisions: int = 0
    cells_extinguished: int = 0
    fires_suppressed: int = 0

    @property
    def uav_id(self) -> str:
        return self.config.uav_id

    def battery_fraction(self) -> float:
        return battery_fraction(self.body, self.config.dynamics)

    def should_return_to_base(self) -> bool:
        return self.battery_fraction() <= self.config.dynamics.return_to_base_fraction
