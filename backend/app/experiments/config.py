"""Experiment configuration."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.scenario import ScenarioConfig


class ExperimentConfig(BaseModel):
    name: str = "comparison"
    scenario: ScenarioConfig
    policies: list[str] = Field(default_factory=lambda: ["dqn", "mappo", "maddpg"])
    seeds: list[int] = Field(default_factory=lambda: list(range(1, 6)))
    checkpoint_path: str | None = None
    max_steps: int | None = None
