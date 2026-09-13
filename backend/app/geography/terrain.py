"""Terrain interface.

When no DEM is available, slope is identically zero and is labeled as such.
Do not invent elevation.
"""

from __future__ import annotations

import numpy as np


class TerrainModel:
    """Optional slope field in the local projected frame."""

    def __init__(self, slope_rad: np.ndarray | None, available: bool) -> None:
        self.slope_rad = slope_rad
        self.available = available

    @classmethod
    def unavailable(cls, height: int, width: int) -> "TerrainModel":
        return cls(slope_rad=np.zeros((height, width), dtype=np.float32), available=False)

    def slope_at(self, row: int, col: int) -> float:
        if self.slope_rad is None:
            return 0.0
        return float(self.slope_rad[row, col])
