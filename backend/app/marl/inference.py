"""Load a trained policy checkpoint if present.

Expected checkpoint keys (flexible):
    state_dict | actor | model
    obs_dim, n_actions, algorithm (optional)

If the file is missing or incompatible, returns UntrainedMARLPolicy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from app.marl.observation import OBS_DIM
from app.marl.policy import Policy, PolicyInfo, UntrainedMARLPolicy

CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "data" / "checkpoints"
ALGO_FILES = {
    "dqn": "dqn.pt",
    "mappo": "mappo.pt",
    "maddpg": "maddpg.pt",
    "ippo": "ippo.pt",
}
PREFERRED = ("mappo.pt", "dqn.pt", "maddpg.pt", "ippo.pt")


class ActorNet(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, hidden: int = 128, n_actions: int = 9) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TorchActorPolicy(Policy):
    def __init__(
        self,
        actor: ActorNet,
        path: str,
        algorithm: str | None,
        version: str = "1.0",
    ) -> None:
        self.actor = actor
        self.actor.eval()
        raw = (algorithm or "marl").lower()
        if "maddpg" in raw:
            label = "maddpg"
        elif "mappo" in raw:
            label = "mappo"
        elif "dqn" in raw:
            label = "dqn"
        elif "ippo" in raw:
            label = "ippo"
        else:
            label = "marl"
        self.info = PolicyInfo(
            name=label,
            kind="learned",
            algorithm=algorithm,
            checkpoint_path=path,
            version=version,
            cooperative=True,
            uses_communication=True,
            trained=True,
            notes=(
                "Feed-forward actor loaded from checkpoint. "
                f"Algorithm: {algorithm or 'unspecified'}. "
                "Decentralized execution: each agent uses its own observation only. "
                "A green detection does not extinguish. If a known fire is already "
                "inside suppression range the actor hovers so the overflight can finish."
            ),
        )

    def act(self, observation: np.ndarray, agent_id: str, state=None) -> int:
        del agent_id, state
        for i in range(3):
            dx = float(observation[11 + i * 3])
            dy = float(observation[11 + i * 3 + 1])
            inten = float(observation[11 + i * 3 + 2])
            if inten > 1e-6 and float(np.hypot(dx, dy)) <= 0.012:
                return 8
        x = torch.from_numpy(np.asarray(observation, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            logits = self.actor(x)
            return int(torch.argmax(logits, dim=-1).item())


def resolve_checkpoint(path: str | None, algorithm: str | None = None) -> Path | None:
    if path:
        p = Path(path)
        if p.exists():
            return p
    if algorithm:
        fname = ALGO_FILES.get(algorithm.lower())
        if fname:
            p = CHECKPOINT_DIR / fname
            if p.exists():
                return p
    for name in PREFERRED:
        p = CHECKPOINT_DIR / name
        if p.exists():
            return p
    cwd = Path.cwd() / "data" / "checkpoints"
    for name in PREFERRED:
        p = cwd / name
        if p.exists():
            return p
    return None


def load_policy(path: str | None, seed: int = 0, algorithm: str | None = None) -> Policy:
    resolved = resolve_checkpoint(path, algorithm=algorithm)
    if resolved is None:
        return UntrainedMARLPolicy(seed=seed, requested_path=path)
    try:
        blob = torch.load(resolved, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(resolved, map_location="cpu")
    except Exception:
        return UntrainedMARLPolicy(seed=seed, requested_path=path)

    if not isinstance(blob, dict):
        return UntrainedMARLPolicy(seed=seed, requested_path=path)

    state = blob.get("state_dict") or blob.get("actor") or blob.get("model")
    obs_dim = int(blob.get("obs_dim", OBS_DIM))
    n_actions = int(blob.get("n_actions", 9))
    algo = blob.get("algorithm") or algorithm
    version = str(blob.get("version", "1.0"))
    if state is None:
        return UntrainedMARLPolicy(seed=seed, requested_path=path)
    actor = ActorNet(obs_dim=obs_dim, n_actions=n_actions)
    try:
        actor.load_state_dict(state)
    except Exception:
        return UntrainedMARLPolicy(seed=seed, requested_path=path)
    return TorchActorPolicy(actor, path=str(resolved), algorithm=algo, version=version)
