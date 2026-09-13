"""Policy factory.

Public UI names are the trained actors only. Greedy stays constructable
for training scripts (expert rollouts). It is not listed in POLICY_NAMES.
"""

from __future__ import annotations

from app.baselines.greedy_policy import GreedyPolicy
from app.marl.inference import load_policy
from app.marl.policy import Policy

POLICY_NAMES = (
    "dqn",
    "mappo",
    "maddpg",
)

_LEARNED = {"dqn", "mappo", "maddpg", "ippo"}


def make_policy(name: str, seed: int = 0, checkpoint_path: str | None = None) -> Policy:
    key = name.strip().lower()
    if key == "greedy":
        return GreedyPolicy()
    if key in _LEARNED:
        return load_policy(checkpoint_path, seed=seed, algorithm=key)
    raise KeyError(f"Unknown policy '{name}'. Known: {', '.join(POLICY_NAMES)}")
