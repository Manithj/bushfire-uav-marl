"""Shared fixtures. Scientific tests use a small, fast Kinglake AOI."""

from __future__ import annotations

import pytest

from app.schemas.scenario import (
    CommunicationConfig,
    EnvironmentConfig,
    FireSourceConfig,
    LocationConfig,
    PolicyConfig,
    ScenarioConfig,
    SensorConfig,
    UAVFleetConfig,
)


@pytest.fixture
def scenario() -> ScenarioConfig:
    return ScenarioConfig(
        name="unit-test",
        location=LocationConfig(region_id="kinglake"),
        fires=[
            FireSourceConfig(
                fire_id="fire_01",
                latitude=-37.50,
                longitude=145.30,
                initial_intensity=0.9,
                ignition_time=0.0,
                radius_m=80.0,
            )
        ],
        environment=EnvironmentConfig(
            wind_speed_kmh=25.0,
            wind_from="NW",
            duration_s=60.0,
            timestep_s=1.0,
            fire_cell_size_m=80.0,
            coverage_cell_size_m=120.0,
            stochastic_fire=True,
        ),
        uavs=UAVFleetConfig(count=3, altitude_m=120.0, cruise_speed_mps=12.0),
        sensor=SensorConfig(
            sensor_range_m=600.0,
            detection_probability=0.9,
            false_positive_probability=0.0,
            false_negative_probability=0.05,
        ),
        communication=CommunicationConfig(range_m=3000.0, failure_probability=0.0),
        policy=PolicyConfig(name="greedy"),
        seed=42,
    )
