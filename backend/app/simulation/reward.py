"""Reward function. The implementation MUST match the documented equation.

    R_t = α * newly_detected_fires
        + ζ * cells_extinguished
        + η * fires_fully_suppressed
        + β * coverage_gain_norm
        - γ * energy_used_norm
        - δ * collision_count
        - ε * new_overlap_norm

Detection does not count as suppression. Cells are extinguished only
when a UAV is inside suppression_range of a BURNING cell.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RewardWeights:
    alpha_detection: float = 8.0
    beta_coverage: float = 0.4
    gamma_energy: float = 0.04
    delta_collision: float = 8.0
    epsilon_overlap: float = 0.3
    zeta_extinguish_cell: float = 0.35
    eta_suppress_fire: float = 16.0


def compute_reward(
    newly_detected: int,
    coverage_gain_fraction: float,
    energy_used_fraction: float,
    collisions: int,
    overlap_increment: float,
    weights: RewardWeights,
    cells_extinguished: int = 0,
    fires_suppressed: int = 0,
) -> float:
    return (
        weights.alpha_detection * newly_detected
        + weights.zeta_extinguish_cell * cells_extinguished
        + weights.eta_suppress_fire * fires_suppressed
        + weights.beta_coverage * coverage_gain_fraction
        - weights.gamma_energy * energy_used_fraction
        - weights.delta_collision * collisions
        - weights.epsilon_overlap * overlap_increment
    )


def reward_equation_latex(w: RewardWeights) -> str:
    return (
        f"R = {w.alpha_detection:g}·N_detected"
        f" + {w.zeta_extinguish_cell:g}·N_cells_out"
        f" + {w.eta_suppress_fire:g}·N_fires_out"
        f" + {w.beta_coverage:g}·Δcoverage"
        f" − {w.gamma_energy:g}·Δenergy"
        f" − {w.delta_collision:g}·N_collisions"
        f" − {w.epsilon_overlap:g}·Δoverlap"
    )
