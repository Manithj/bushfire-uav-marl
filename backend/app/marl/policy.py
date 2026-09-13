"""Algorithm-agnostic policy interface.

    action = policy.act(observation, agent_id, state=None)

Does not hard-code MAPPO / MADDPG / QMIX. Those can wrap this interface.
A missing checkpoint must not be silently replaced by fabricated metrics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PolicyInfo:
    name: str
    kind: str  # "baseline" | "learned" | "heuristic"
    algorithm: str | None
    checkpoint_path: str | None
    version: str
    cooperative: bool
    uses_communication: bool
    notes: str
    trained: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class Policy(ABC):
    info: PolicyInfo

    def reset(self, seed: int | None = None) -> None:
        return None

    @abstractmethod
    def act(
        self,
        observation: np.ndarray,
        agent_id: str,
        state: dict[str, Any] | None = None,
    ) -> int:
        """Return a discrete-9 action index."""

    def act_many(
        self,
        observations: dict[str, np.ndarray],
        states: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, int]:
        states = states or {}
        return {
            aid: self.act(obs, aid, states.get(aid))
            for aid, obs in observations.items()
        }


class UntrainedMARLPolicy(Policy):
    """Placeholder when a checkpoint is requested but not present.

    Acts as a seeded random policy and is labeled untrained.
    Never presented as a trained MARL result.
    """

    def __init__(self, seed: int = 0, requested_path: str | None = None) -> None:
        self._rng = np.random.default_rng(seed)
        self.info = PolicyInfo(
            name="marl_untrained",
            kind="learned",
            algorithm=None,
            checkpoint_path=requested_path,
            version="0.0.0-untrained",
            cooperative=False,
            uses_communication=False,
            trained=False,
            notes=(
                "No trained checkpoint is loaded. Actions are random. "
                "Do not interpret this as a MARL result."
            ),
        )

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def act(self, observation: np.ndarray, agent_id: str, state=None) -> int:
        del observation, agent_id, state
        return int(self._rng.integers(0, 9))
