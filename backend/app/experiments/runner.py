"""Headless multi-seed experiment runner. No visualization."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from app.experiments.config import ExperimentConfig
from app.experiments.results import format_mean_sd, trial_table
from app.schemas.scenario import ScenarioConfig
from app.simulation.environment import WildfireEnv


def run_episode(scenario: ScenarioConfig, max_steps: int | None = None) -> dict[str, Any]:
    env = WildfireEnv(scenario)
    env.reset(seed=scenario.seed)
    steps = 0
    limit = max_steps if max_steps is not None else int(scenario.environment.duration_s / scenario.environment.timestep_s) + 5
    while not env._terminated and steps < limit:
        env.step_policy()
        steps += 1
    metrics = env.public_state()["metrics"]
    metrics["policy"] = env.policy.info.name if env.policy else None
    metrics["seed"] = scenario.seed
    metrics["trained_policy"] = env.policy.info.trained if env.policy else False
    return metrics


def run_experiment(cfg: ExperimentConfig, progress=None) -> dict[str, Any]:
    experiment_id = str(uuid.uuid4())
    started = datetime.now(timezone.utc).isoformat()
    policy_trials: dict[str, list[dict[str, Any]]] = {p: [] for p in cfg.policies}
    total = len(cfg.policies) * len(cfg.seeds)
    done = 0
    for policy in cfg.policies:
        for seed in cfg.seeds:
            sc = cfg.scenario.model_copy(deep=True)
            sc.seed = int(seed)
            sc.policy.name = policy
            if cfg.checkpoint_path:
                sc.policy.checkpoint_path = cfg.checkpoint_path
            trial = run_episode(sc, max_steps=cfg.max_steps)
            trial["policy"] = policy
            policy_trials[policy].append(trial)
            done += 1
            if progress is not None:
                progress(done, total, policy, seed)

    table = trial_table(policy_trials)
    return {
        "experiment_id": experiment_id,
        "name": cfg.name,
        "timestamp": started,
        "scenario": cfg.scenario.model_dump(),
        "seeds": cfg.seeds,
        "policies": cfg.policies,
        "checkpoint_path": cfg.checkpoint_path,
        "n_trials_per_policy": len(cfg.seeds),
        "trials": policy_trials,
        "summary": table,
        "report_text": render_report(cfg, table),
        "significance_tested": False,
        "notes": (
            "Intervals are descriptive (mean ± sample SD, and t-based 95% CI). "
            "No hypothesis test was performed; do not claim statistical significance."
        ),
    }


def render_report(cfg: ExperimentConfig, table: dict) -> str:
    lines = [
        "Experiment Summary",
        "==================",
        "",
        f"Scenario: {cfg.scenario.name}",
        f"Region:   {cfg.scenario.location.region_id}",
        f"Agents:   {cfg.scenario.uavs.count}",
        f"Trials:   {len(cfg.seeds)} seeds {cfg.seeds}",
        f"Policies: {', '.join(cfg.policies)}",
        "",
        "Values are mean ± sample standard deviation across seeds.",
        "No statistical significance test was performed.",
        "",
    ]
    keys = [
        ("detection_rate", "Detection rate"),
        ("mean_detection_delay_s", "Detection time (s)"),
        ("coverage_fraction", "Coverage"),
        ("energy_used_fraction", "Energy used"),
        ("total_distance_m", "Distance (m)"),
        ("collisions", "Collisions"),
        ("messages_sent", "Messages sent"),
        ("episode_reward", "Episode reward"),
    ]
    for policy in cfg.policies:
        lines.append(f"{policy}:")
        for key, label in keys:
            lines.append(f"  {label:<22} {format_mean_sd(table[policy][key])}")
        lines.append("")
    return "\n".join(lines)
