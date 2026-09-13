"""Statistical summaries. No significance claims without a test."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


METRIC_KEYS = (
    "detection_rate",
    "mean_detection_delay_s",
    "time_to_first_detection_s",
    "coverage_fraction",
    "total_distance_m",
    "energy_used_fraction",
    "collisions",
    "messages_sent",
    "communication_link_fraction",
    "episode_reward",
    "false_positive_rate",
    "miss_rate",
    "search_overlap_fraction",
)


def summarize(values: list[float | None]) -> dict[str, float | None]:
    xs = [float(v) for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not xs:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "median": None,
            "min": None,
            "max": None,
            "ci95_low": None,
            "ci95_high": None,
        }
    arr = np.asarray(xs, dtype=np.float64)
    n = arr.size
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    if n > 1:
        sem = stats.sem(arr)
        ci = stats.t.interval(0.95, n - 1, loc=mean, scale=sem)
        ci_low, ci_high = float(ci[0]), float(ci[1])
    else:
        ci_low = ci_high = None
    return {
        "n": int(n),
        "mean": mean,
        "std": std,
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }


def format_mean_sd(summary: dict[str, float | None], digits: int = 3) -> str:
    if summary["mean"] is None:
        return "N/A"
    if summary["n"] <= 1:
        return f"{summary['mean']:.{digits}g}"
    return f"{summary['mean']:.{digits}g} ± {summary['std']:.{digits}g}"


def trial_table(policy_trials: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, dict[str, float | None]]]:
    table: dict[str, dict[str, dict[str, float | None]]] = {}
    for policy, trials in policy_trials.items():
        table[policy] = {}
        for key in METRIC_KEYS:
            table[policy][key] = summarize([t.get(key) for t in trials])
    return table
