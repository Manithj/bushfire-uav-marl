"""Research metrics derived only from simulation state.

Every metric has: definition, unit, source. If a value cannot be
computed, the public payload uses null (rendered as N/A).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MetricSpec:
    id: str
    name: str
    definition: str
    unit: str
    source: str
    group: str


METRIC_SPECS: list[MetricSpec] = [
    MetricSpec(
        "sim_time_s",
        "Simulation time",
        "Elapsed simulated time from episode start.",
        "s",
        "environment.time_s",
        "simulation",
    ),
    MetricSpec(
        "active_fires",
        "Active fires",
        "Count of ignition sources that still have at least one BURNING cell.",
        "count",
        "fire_model.sources.active",
        "fire",
    ),
    MetricSpec(
        "total_fires",
        "Total fire sources",
        "Number of configured ignition sources.",
        "count",
        "fire_model.sources",
        "fire",
    ),
    MetricSpec(
        "detected_fires",
        "Detected fires",
        "Ignition sources with at least one true positive detection.",
        "count",
        "fire_model.sources.detected",
        "detection",
    ),
    MetricSpec(
        "detection_rate",
        "Detection rate",
        "detected_fires / total_fires. Undefined if total_fires = 0.",
        "fraction",
        "detected_fires / total_fires",
        "detection",
    ),
    MetricSpec(
        "miss_rate",
        "Miss rate",
        "undetected_fires / total_fires. Undefined if total_fires = 0.",
        "fraction",
        "1 - detection_rate",
        "detection",
    ),
    MetricSpec(
        "false_positive_rate",
        "False positive rate",
        "false_positive_detections / total_detection_reports. Undefined if no reports.",
        "fraction",
        "sensor detection events",
        "detection",
    ),
    MetricSpec(
        "time_to_first_detection_s",
        "Time to first detection",
        "Simulation time of the first true fire detection, measured from t = 0.",
        "s",
        "first true DetectionEvent.time_s",
        "detection",
    ),
    MetricSpec(
        "mean_detection_delay_s",
        "Average detection time",
        "Mean over detected fires of (first_detection_time − ignition_time).",
        "s",
        "fire_model.sources first_detection_time",
        "detection",
    ),
    MetricSpec(
        "coverage_fraction",
        "Coverage",
        "explored_eligible_cells / eligible_search_cells.",
        "fraction",
        "coverage_model.coverage_fraction()",
        "coverage",
    ),
    MetricSpec(
        "explored_area_m2",
        "Explored area",
        "Eligible cells observed by at least one UAV sensor, times cell area.",
        "m^2",
        "coverage_model",
        "coverage",
    ),
    MetricSpec(
        "eligible_area_m2",
        "Eligible search area",
        "AOI cells that are not marked inaccessible.",
        "m^2",
        "coverage_model",
        "coverage",
    ),
    MetricSpec(
        "burned_area_m2",
        "Burned area",
        "Cells in BURNING or BURNED state times fire-grid cell area.",
        "m^2",
        "fire_model.burned_area_m2()",
        "fire",
    ),
    MetricSpec(
        "total_distance_m",
        "Total distance",
        "Sum of path length integrated from UAV kinematics.",
        "m",
        "uav.body.distance_m",
        "uav",
    ),
    MetricSpec(
        "mean_speed_mps",
        "Average speed",
        "total_distance / total_airborne_time. Undefined if airborne time is 0.",
        "m/s",
        "uav.body",
        "uav",
    ),
    MetricSpec(
        "mean_battery_fraction",
        "Average battery remaining",
        "Mean over UAVs of battery_J / battery_capacity_J.",
        "fraction",
        "uav.battery_fraction()",
        "uav",
    ),
    MetricSpec(
        "energy_used_fraction",
        "Energy used",
        "1 − mean_battery_fraction (relative to full charge at t = 0).",
        "fraction",
        "1 - mean_battery_fraction",
        "uav",
    ),
    MetricSpec(
        "collisions",
        "Collisions",
        "Count of pairwise events with horizontal distance < collision_radius and |Δz| < threshold.",
        "count",
        "collision_monitor.collisions",
        "safety",
    ),
    MetricSpec(
        "near_collisions",
        "Near-collisions",
        "Count of pairwise events closer than configured min_separation but not colliding.",
        "count",
        "collision_monitor.near_events",
        "safety",
    ),
    MetricSpec(
        "min_inter_agent_distance_m",
        "Minimum inter-agent distance",
        "Minimum planimetric UAV–UAV distance observed so far.",
        "m",
        "collision_monitor.min_separation_observed_m",
        "safety",
    ),
    MetricSpec(
        "communication_link_fraction",
        "Communication connectivity",
        "Active links / possible undirected pairs. Undefined if fewer than 2 UAVs.",
        "fraction",
        "communication_model.connected_fraction",
        "communication",
    ),
    MetricSpec(
        "messages_sent",
        "Messages sent",
        "Detection broadcast attempts (including failed deliveries).",
        "count",
        "communication_model.sent",
        "communication",
    ),
    MetricSpec(
        "messages_received",
        "Messages received",
        "Successfully delivered detection broadcasts.",
        "count",
        "communication_model.received",
        "communication",
    ),
    MetricSpec(
        "communication_failures",
        "Communication failures",
        "Broadcasts dropped due to no neighbour, out of range, or link failure draw.",
        "count",
        "communication_model.failures",
        "communication",
    ),
    MetricSpec(
        "search_overlap_fraction",
        "Redundant exploration",
        "Fraction of eligible cells observed by two or more UAVs.",
        "fraction",
        "coverage_model.overlap_fraction()",
        "multi_agent",
    ),
    MetricSpec(
        "mean_pairwise_jaccard",
        "Mean pairwise coverage Jaccard",
        "Mean Jaccard index of explored-cell sets over UAV pairs.",
        "fraction",
        "coverage_model.pairwise_overlap_fraction()",
        "multi_agent",
    ),
    MetricSpec(
        "episode_reward",
        "Episode reward",
        "Sum of per-step team rewards. Optimisation signal, not detection accuracy.",
        "1",
        "sum of reward.compute_reward",
        "rl",
    ),
    MetricSpec(
        "extinguished_fires",
        "Extinguished fires",
        "Ignition sources whose last BURNING cells were put out by a UAV inside suppression range. Detection alone does not count.",
        "count",
        "fire_model.sources.suppressed",
        "suppression",
    ),
    MetricSpec(
        "extinguished_area_m2",
        "Extinguished area",
        "Cells in EXTINGUISHED state times fire-grid cell area. BURNED (natural burnout) is excluded.",
        "m^2",
        "fire_model.extinguished_area_m2()",
        "suppression",
    ),
    MetricSpec(
        "suppression_rate",
        "Suppression rate",
        "extinguished_fires / total_fires. Undefined if total_fires = 0. Not the same as detection rate.",
        "fraction",
        "extinguished_fires / total_fires",
        "suppression",
    ),
    MetricSpec(
        "true_detections",
        "True detections",
        "Count of sensor reports that match a simulated burning fire source.",
        "count",
        "detection events is_true=True",
        "detection",
    ),
    MetricSpec(
        "false_detections",
        "False detections",
        "Count of sensor reports with no matching simulated fire.",
        "count",
        "detection events is_true=False",
        "detection",
    ),
]


def _div(num: float, den: float) -> float | None:
    if den == 0:
        return None
    return num / den


@dataclass
class MetricsAccumulator:
    episode_reward: float = 0.0
    step_rewards: list[float] = field(default_factory=list)
    true_detections: int = 0
    false_detections: int = 0
    time_to_first_detection_s: float | None = None
    reward_components: dict[str, float] = field(default_factory=lambda: {
        "detection": 0.0,
        "coverage": 0.0,
        "energy": 0.0,
        "collision": 0.0,
        "overlap": 0.0,
        "extinguish": 0.0,
        "suppress": 0.0,
    })

    def add_reward(self, r: float) -> None:
        self.episode_reward += r
        self.step_rewards.append(r)

    def record_detection(self, is_true: bool, time_s: float) -> None:
        if is_true:
            self.true_detections += 1
            if self.time_to_first_detection_s is None:
                self.time_to_first_detection_s = time_s
        else:
            self.false_detections += 1


def compute_public_metrics(env) -> dict:
    """Build the dashboard payload from live environment state."""
    fires = env.fire.sources
    n_fires = len(fires)
    n_detected = sum(1 for f in fires if f.detected)
    delays = [
        f.first_detection_time - f.ignition_time
        for f in fires
        if f.detected and f.first_detection_time is not None
    ]
    uavs = env.uavs
    n_uav = len(uavs)
    batteries = [u.battery_fraction() for u in uavs] if uavs else []
    distances = [u.body.distance_m for u in uavs]
    airborne = [u.body.airborne_s for u in uavs]
    total_dist = float(sum(distances))
    total_air = float(sum(airborne))
    reports = env.metrics.true_detections + env.metrics.false_detections
    mean_r = float(np.mean(env.metrics.step_rewards)) if env.metrics.step_rewards else None
    var_r = float(np.var(env.metrics.step_rewards)) if len(env.metrics.step_rewards) > 1 else None

    utilization = None
    if uavs and total_air > 0:
        idle = sum(u.body.idle_s for u in uavs)
        utilization = 1.0 - idle / total_air

    workload = None
    if n_uav >= 2:
        cells = [u.covered_cells for u in uavs]
        mean_c = float(np.mean(cells))
        if mean_c > 0:
            workload = float(np.std(cells) / mean_c)

    payload = {
        "sim_time_s": env.time_s,
        "active_fires": sum(1 for f in fires if f.active),
        "total_fires": n_fires,
        "detected_fires": n_detected,
        "extinguished_fires": sum(1 for f in fires if f.suppressed),
        "extinguished_area_m2": env.fire.extinguished_area_m2(),
        "suppression_rate": _div(sum(1 for f in fires if f.suppressed), n_fires),
        "detection_rate": _div(n_detected, n_fires),
        "miss_rate": _div(n_fires - n_detected, n_fires),
        "false_positive_rate": _div(env.metrics.false_detections, reports),
        "time_to_first_detection_s": env.metrics.time_to_first_detection_s,
        "mean_detection_delay_s": float(np.mean(delays)) if delays else None,
        "coverage_fraction": env.coverage.coverage_fraction(),
        "explored_area_m2": env.coverage.area_m2(env.coverage.explored_cells()),
        "eligible_area_m2": env.coverage.area_m2(env.coverage.eligible_cells()),
        "inaccessible_area_m2": env.coverage.area_m2(int(np.sum(env.coverage.inaccessible))),
        "burned_area_m2": env.fire.burned_area_m2(),
        "active_fire_area_m2": env.fire.active_area_m2(),
        "total_distance_m": total_dist,
        "mean_speed_mps": _div(total_dist, total_air),
        "mean_battery_fraction": float(np.mean(batteries)) if batteries else None,
        "energy_used_fraction": (1.0 - float(np.mean(batteries))) if batteries else None,
        "collisions": env.collisions.collisions,
        "near_collisions": env.collisions.near_events,
        "min_inter_agent_distance_m": env.collisions.min_separation_observed_m,
        "communication_link_fraction": env.comms.connected_fraction(n_uav),
        "messages_sent": env.comms.sent,
        "messages_received": env.comms.received,
        "communication_failures": env.comms.failures,
        "search_overlap_fraction": env.coverage.overlap_fraction(),
        "mean_pairwise_jaccard": env.coverage.pairwise_overlap_fraction(),
        "episode_reward": env.metrics.episode_reward,
        "mean_step_reward": mean_r,
        "reward_variance": var_r,
        "episode_length": env.step_count,
        "true_detections": env.metrics.true_detections,
        "false_detections": env.metrics.false_detections,
        "agent_utilization": utilization,
        "workload_cv": workload,
        "uavs_total": n_uav,
        "uavs_active": sum(1 for u in uavs if u.body.battery_j > 0 and u.status.value not in ("landed", "collided", "battery_depleted")),
        "policy_entropy": None,
        "value_loss": None,
        "policy_loss": None,
        "success_rate": None,
    }
    return payload


def per_uav_metrics(env) -> list[dict]:
    out = []
    for u in env.uavs:
        out.append({
            "uav_id": u.uav_id,
            "distance_m": u.body.distance_m,
            "speed_mps": u.body.speed,
            "battery_fraction": u.battery_fraction(),
            "airborne_s": u.body.airborne_s,
            "idle_s": u.body.idle_s,
            "covered_cells": u.covered_cells,
            "covered_area_m2": u.covered_cells * env.coverage.cell_size_m * env.coverage.cell_size_m,
            "true_detections": u.detections_true,
            "false_detections": u.detections_false,
            "actions_taken": u.actions_taken,
            "messages_sent": u.messages_sent,
            "messages_received": u.messages_received,
            "near_collisions": u.near_collisions,
            "collisions": u.collisions,
            "cells_extinguished": u.cells_extinguished,
            "fires_suppressed": u.fires_suppressed,
            "status": u.status.value,
        })
    return out
