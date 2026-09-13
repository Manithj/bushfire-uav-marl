"""Serializable scenario configuration (JSON)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LocationConfig(BaseModel):
    region_id: str = "kinglake"
    south: float | None = None
    west: float | None = None
    north: float | None = None
    east: float | None = None


class FireSourceConfig(BaseModel):
    fire_id: str | None = None
    latitude: float
    longitude: float
    initial_intensity: float = Field(0.8, ge=0.05, le=1.0)
    ignition_time: float = Field(0.0, ge=0.0)
    radius_m: float = Field(80.0, ge=10.0)


class EnvironmentConfig(BaseModel):
    wind_speed_kmh: float = Field(20.0, ge=0.0, le=120.0)
    wind_from: str = Field("NW", description="Meteorological FROM direction, 8-wind or degrees.")
    wind_from_deg: float | None = Field(
        None, description="If set, overrides wind_from compass label. 0=N, clockwise, FROM."
    )
    duration_s: float = Field(1800.0, ge=10.0, le=20_000.0)
    timestep_s: float = Field(1.0, ge=0.2, le=5.0)
    fire_cell_size_m: float = Field(50.0, ge=20.0, le=200.0)
    coverage_cell_size_m: float = Field(80.0, ge=30.0, le=300.0)
    stochastic_fire: bool = True
    base_ros_mps: float = Field(0.35, ge=0.01, le=3.0)


class UAVFleetConfig(BaseModel):
    count: int = Field(4, ge=1, le=24)
    altitude_m: float = Field(120.0, ge=40.0, le=250.0)
    max_speed_mps: float = Field(18.0, ge=2.0, le=40.0)
    cruise_speed_mps: float = Field(12.0, ge=1.0, le=40.0)
    battery_capacity_j: float = Field(180_000.0, ge=10_000.0)
    return_to_base_fraction: float = Field(0.30, ge=0.05, le=0.8)
    min_separation_m: float = Field(40.0, ge=5.0, le=200.0)
    placements: list[UAVPlacement] = Field(default_factory=list)


class UAVPlacement(BaseModel):
    uav_id: str | None = None
    latitude: float
    longitude: float
    heading_deg: float = 0.0
    altitude_m: float | None = None


class SensorConfig(BaseModel):
    sensor_range_m: float = Field(800.0, ge=50.0, le=3000.0)
    field_of_view_deg: float = Field(360.0, ge=30.0, le=360.0)
    detection_probability: float = Field(0.85, ge=0.0, le=1.0)
    false_positive_probability: float = Field(0.002, ge=0.0, le=0.5)
    false_negative_probability: float = Field(0.10, ge=0.0, le=0.9)
    observation_noise_m: float = Field(15.0, ge=0.0, le=200.0)
    suppression_range_m: float = Field(
        120.0,
        ge=20.0,
        le=2000.0,
        description="UAV must overlap the fire point (planimetric distance ≤ this). Detection range is separate and does not extinguish.",
    )


class CommunicationConfig(BaseModel):
    range_m: float = Field(4500.0, ge=50.0, le=20_000.0)
    failure_probability: float = Field(0.0, ge=0.0, le=1.0)
    multi_hop: bool = False


class RewardConfig(BaseModel):
    alpha_detection: float = 8.0
    beta_coverage: float = 0.4
    gamma_energy: float = 0.04
    delta_collision: float = 8.0
    epsilon_overlap: float = 0.3
    zeta_extinguish_cell: float = 0.35
    eta_suppress_fire: float = 16.0


class PolicyConfig(BaseModel):
    name: str = "mappo"
    checkpoint_path: str | None = None
    allow_global_state: bool = False
    algorithm: str | None = None


class ScenarioConfig(BaseModel):
    name: str = "Victoria study scenario"
    location: LocationConfig = Field(default_factory=LocationConfig)
    fires: list[FireSourceConfig] = Field(default_factory=list)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    uavs: UAVFleetConfig = Field(default_factory=UAVFleetConfig)
    sensor: SensorConfig = Field(default_factory=SensorConfig)
    communication: CommunicationConfig = Field(default_factory=CommunicationConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    seed: int = 42
    action_space: Literal["discrete9"] = "discrete9"

    model_config = {"extra": "forbid"}


COMPASS_FROM = {
    "N": 0.0,
    "NE": 45.0,
    "E": 90.0,
    "SE": 135.0,
    "S": 180.0,
    "SW": 225.0,
    "W": 270.0,
    "NW": 315.0,
}


def resolve_wind_from_deg(env: EnvironmentConfig) -> float:
    if env.wind_from_deg is not None:
        return float(env.wind_from_deg) % 360.0
    key = env.wind_from.strip().upper()
    if key not in COMPASS_FROM:
        raise ValueError(f"Unknown wind_from '{env.wind_from}'. Use N,NE,E,SE,S,SW,W,NW or wind_from_deg.")
    return COMPASS_FROM[key]
