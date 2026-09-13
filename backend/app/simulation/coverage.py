"""Search coverage on a regular grid in the local projected frame.

Definitions (eligible search area):

    total_geographic_area  = AOI width × height
    inaccessible_area      = cells marked inaccessible (water / out of AOI)
    eligible_search_area   = total - inaccessible
    explored_area          = eligible cells observed by ≥1 UAV sensor
    burned_area            = cells that are BURNING or BURNED
    coverage_percentage    = explored_area / eligible_search_area * 100

Coverage does not count inaccessible cells as successfully searched.
A cell is explored when the planimetric distance from a live UAV to the
cell centre is ≤ that UAV's sensor range.

The coverage grid may be coarser than the fire grid for CPU efficiency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.simulation.uav import UAVAgent


@dataclass
class CoverageModel:
    height: int
    width: int
    cell_size_m: float
    origin_x: float
    origin_y: float
    extent_x: float
    extent_y: float
    explored: np.ndarray  # bool
    inaccessible: np.ndarray  # bool
    first_explored_s: np.ndarray  # -1 if never
    explorer_id: np.ndarray  # int8 agent index, -1 none
    per_agent: np.ndarray  # (n_agents, H, W) bool — may be omitted if n large

    @classmethod
    def create(
        cls,
        origin_x: float,
        origin_y: float,
        extent_x: float,
        extent_y: float,
        cell_size_m: float,
        n_agents: int,
        inaccessible: np.ndarray | None = None,
    ) -> "CoverageModel":
        width = max(8, int(np.ceil(extent_x / cell_size_m)))
        height = max(8, int(np.ceil(extent_y / cell_size_m)))
        inacc = (
            inaccessible.astype(bool)
            if inaccessible is not None
            else np.zeros((height, width), dtype=bool)
        )
        return cls(
            height=height,
            width=width,
            cell_size_m=cell_size_m,
            origin_x=origin_x,
            origin_y=origin_y,
            extent_x=extent_x,
            extent_y=extent_y,
            explored=np.zeros((height, width), dtype=bool),
            inaccessible=inacc,
            first_explored_s=np.full((height, width), -1.0, dtype=np.float32),
            explorer_id=np.full((height, width), -1, dtype=np.int16),
            per_agent=np.zeros((n_agents, height, width), dtype=bool),
        )

    def cell_centres(self) -> tuple[np.ndarray, np.ndarray]:
        cols = np.arange(self.width)
        rows = np.arange(self.height)
        xs = self.origin_x + (cols + 0.5) * self.cell_size_m
        ys = self.origin_y + self.extent_y - (rows + 0.5) * self.cell_size_m
        return np.meshgrid(xs, ys)

    def update(self, uavs: list[UAVAgent], time_s: float) -> float:
        """Mark newly observed eligible cells. Returns newly explored area (m²)."""
        xx, yy = self.cell_centres()
        newly = np.zeros_like(self.explored)
        for i, uav in enumerate(uavs):
            if uav.body.battery_j <= 0:
                continue
            d = np.hypot(xx - uav.body.x, yy - uav.body.y)
            seen = d <= uav.config.sensor_range_m
            if i < self.per_agent.shape[0]:
                self.per_agent[i] |= seen
            gain = seen & ~self.explored & ~self.inaccessible
            newly |= gain
            first = gain & (self.explorer_id < 0)
            self.explorer_id[first] = i
            self.first_explored_s[gain] = time_s
            uav.covered_cells = int(np.sum(self.per_agent[i] & ~self.inaccessible)) if i < self.per_agent.shape[0] else uav.covered_cells
        self.explored |= newly
        return float(np.sum(newly) * self.cell_size_m * self.cell_size_m)

    def eligible_cells(self) -> int:
        return int(np.sum(~self.inaccessible))

    def explored_cells(self) -> int:
        return int(np.sum(self.explored & ~self.inaccessible))

    def coverage_fraction(self) -> float:
        elig = self.eligible_cells()
        if elig == 0:
            return 0.0
        return self.explored_cells() / elig

    def overlap_fraction(self) -> float | None:
        """Fraction of eligible cells observed by ≥2 agents.

        Requires per-agent maps. Returns None if fewer than 2 agents.
        """
        if self.per_agent.shape[0] < 2:
            return None
        counts = np.sum(self.per_agent, axis=0)
        elig = ~self.inaccessible
        denom = int(np.sum(elig))
        if denom == 0:
            return None
        return float(np.sum((counts >= 2) & elig) / denom)

    def pairwise_overlap_fraction(self) -> float | None:
        """Mean Jaccard overlap across agent pairs on eligible cells."""
        n = self.per_agent.shape[0]
        if n < 2:
            return None
        elig = ~self.inaccessible
        jacs: list[float] = []
        for i in range(n):
            a = self.per_agent[i] & elig
            for j in range(i + 1, n):
                b = self.per_agent[j] & elig
                inter = int(np.sum(a & b))
                union = int(np.sum(a | b))
                if union > 0:
                    jacs.append(inter / union)
        if not jacs:
            return None
        return float(np.mean(jacs))

    def area_m2(self, n_cells: int) -> float:
        return n_cells * self.cell_size_m * self.cell_size_m

    def downsample_for_display(self, max_side: int = 48) -> list[list[float]]:
        """Coarse coverage heatmap in [0, 1] row-major (north at top)."""
        h, w = self.height, self.width
        step_r = max(1, h // max_side)
        step_c = max(1, w // max_side)
        grid: list[list[float]] = []
        for r in range(0, h, step_r):
            row: list[float] = []
            for c in range(0, w, step_c):
                block = self.explored[r : r + step_r, c : c + step_c]
                inacc = self.inaccessible[r : r + step_r, c : c + step_c]
                valid = ~inacc
                if not np.any(valid):
                    row.append(0.0)
                else:
                    row.append(float(np.mean(block[valid])))
            grid.append(row)
        return grid
