"""Projected coordinate transforms for local Euclidean calculations.

Never use latitude/longitude directly for Euclidean distances.

Default local CRS: Azimuthal Equidistant (AEQD) on WGS84, centred on the
area of interest. Distances and headings in this frame are in metres and
radians and are valid near the projection centre (typical AOI ≤ 50 km).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyproj import CRS, Transformer


@dataclass(frozen=True)
class GeoBounds:
    """Axis-aligned geographic bounding box (WGS84)."""

    south: float
    west: float
    north: float
    east: float

    def contains(self, lat: float, lon: float) -> bool:
        return self.south <= lat <= self.north and self.west <= lon <= self.east

    def centroid(self) -> tuple[float, float]:
        return ((self.south + self.north) / 2.0, (self.west + self.east) / 2.0)

    def width_deg(self) -> float:
        return self.east - self.west

    def height_deg(self) -> float:
        return self.north - self.south


class CoordinateTransform:
    """Bidirectional WGS84 ↔ local metres (east, north).

    Local frame:
        x = easting (metres)
        y = northing (metres)
        origin = (0, 0) at the AOI centre
    """

    def __init__(self, center_lat: float, center_lon: float) -> None:
        if not (-90.0 <= center_lat <= 90.0):
            raise ValueError(f"center_lat out of range: {center_lat}")
        if not (-180.0 <= center_lon <= 180.0):
            raise ValueError(f"center_lon out of range: {center_lon}")
        self.center_lat = float(center_lat)
        self.center_lon = float(center_lon)
        self.crs_wgs84 = CRS.from_epsg(4326)
        self.crs_local = CRS.from_proj4(
            f"+proj=aeqd +lat_0={self.center_lat} +lon_0={self.center_lon} "
            f"+datum=WGS84 +units=m +no_defs"
        )
        self._to_local = Transformer.from_crs(
            self.crs_wgs84, self.crs_local, always_xy=True
        )
        self._to_wgs84 = Transformer.from_crs(
            self.crs_local, self.crs_wgs84, always_xy=True
        )

    @classmethod
    def from_bounds(cls, bounds: GeoBounds) -> "CoordinateTransform":
        lat, lon = bounds.centroid()
        return cls(lat, lon)

    def to_local(self, lat: float, lon: float) -> tuple[float, float]:
        """(lat, lon) → (x_east_m, y_north_m)."""
        x, y = self._to_local.transform(lon, lat)
        return float(x), float(y)

    def to_wgs84(self, x: float, y: float) -> tuple[float, float]:
        """(x_east_m, y_north_m) → (lat, lon)."""
        lon, lat = self._to_wgs84.transform(x, y)
        return float(lat), float(lon)

    def to_local_vec(
        self, lats: np.ndarray, lons: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = self._to_local.transform(lons, lats)
        return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)

    def to_wgs84_vec(
        self, xs: np.ndarray, ys: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        lons, lats = self._to_wgs84.transform(xs, ys)
        return np.asarray(lats, dtype=np.float64), np.asarray(lons, dtype=np.float64)

    def bounds_local(self, bounds: GeoBounds) -> tuple[float, float, float, float]:
        """Return (x_min, y_min, x_max, y_max) of the four geographic corners."""
        corners = [
            self.to_local(bounds.south, bounds.west),
            self.to_local(bounds.south, bounds.east),
            self.to_local(bounds.north, bounds.west),
            self.to_local(bounds.north, bounds.east),
        ]
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        return min(xs), min(ys), max(xs), max(ys)


def heading_components(heading_rad: float) -> tuple[float, float]:
    """Unit vector (east, north) for a navigation heading.

    Heading convention (used everywhere in this simulator):
        0 rad = North, increasing clockwise.
    """
    return float(np.sin(heading_rad)), float(np.cos(heading_rad))


def heading_from_vector(dx_east: float, dy_north: float) -> float:
    """Navigation heading [0, 2π) from an ENU displacement."""
    return float(np.mod(np.arctan2(dx_east, dy_north), 2.0 * np.pi))


def wrap_angle(angle: float) -> float:
    """Wrap radians to (-π, π]."""
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def compass_deg_to_rad(deg: float) -> float:
    """Compass degrees (0=N, clockwise) → radians."""
    return float(np.deg2rad(deg % 360.0))


def rad_to_compass_deg(rad: float) -> float:
    return float(np.rad2deg(rad) % 360.0)
