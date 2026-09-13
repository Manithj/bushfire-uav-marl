"""World geometry: AOI in geographic and local frames."""

from __future__ import annotations

from dataclasses import dataclass

from app.geography.coordinate_transform import CoordinateTransform, GeoBounds
from app.geography.regions import DEFAULT_REGION_ID, get_region
from app.schemas.scenario import LocationConfig


@dataclass
class World:
    region_id: str
    bounds: GeoBounds
    transform: CoordinateTransform
    origin_x: float
    origin_y: float
    extent_x: float
    extent_y: float

    @property
    def x_min(self) -> float:
        return self.origin_x

    @property
    def y_min(self) -> float:
        return self.origin_y

    @property
    def x_max(self) -> float:
        return self.origin_x + self.extent_x

    @property
    def y_max(self) -> float:
        return self.origin_y + self.extent_y


def world_from_location(location: LocationConfig) -> World:
    region = get_region(location.region_id) if location.region_id else get_region(DEFAULT_REGION_ID)
    if all(v is not None for v in (location.south, location.west, location.north, location.east)):
        bounds = GeoBounds(
            south=float(location.south),
            west=float(location.west),
            north=float(location.north),
            east=float(location.east),
        )
    else:
        bounds = region.bounds
    transform = CoordinateTransform.from_bounds(bounds)
    x_min, y_min, x_max, y_max = transform.bounds_local(bounds)
    # Use a slightly inset rectangle so corners stay inside the geographic box.
    pad = 0.0
    return World(
        region_id=region.id,
        bounds=bounds,
        transform=transform,
        origin_x=x_min + pad,
        origin_y=y_min + pad,
        extent_x=max(500.0, x_max - x_min - 2 * pad),
        extent_y=max(500.0, y_max - y_min - 2 * pad),
    )
