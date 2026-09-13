"""Geographic dataset loader.

Distinguishes real published data from simulation-generated layers.
If a dataset file is absent, returns a clearly labeled placeholder.
Does not invent roads, terrain, or public locations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.geography.regions import DEFAULT_REGION_ID, NamedRegion, get_region

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "geo"


@dataclass
class GeoLayer:
    id: str
    name: str
    kind: str
    source: str
    is_real_geographic_data: bool
    available: bool
    geometry: dict[str, Any] | None = None
    notes: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class GeographicContext:
    region: NamedRegion
    layers: list[GeoLayer]

    def layer(self, layer_id: str) -> GeoLayer | None:
        for layer in self.layers:
            if layer.id == layer_id:
                return layer
        return None


def _placeholder(layer_id: str, name: str, kind: str, expected: str) -> GeoLayer:
    return GeoLayer(
        id=layer_id,
        name=name,
        kind=kind,
        source="unavailable",
        is_real_geographic_data=False,
        available=False,
        geometry=None,
        notes=(
            f"PLACEHOLDER: no published {expected} dataset is bundled. "
            f"This layer is not simulated as real geography. "
            f"Place a GeoJSON at data/geo/{layer_id}.geojson to enable it."
        ),
    )


def _load_optional_geojson(layer_id: str, name: str, kind: str, source: str) -> GeoLayer:
    path = DATA_DIR / f"{layer_id}.geojson"
    if not path.exists():
        return _placeholder(layer_id, name, kind, name.lower())
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return GeoLayer(
            id=layer_id,
            name=name,
            kind=kind,
            source=str(path),
            is_real_geographic_data=False,
            available=False,
            notes=f"Failed to parse {path.name}: {exc}",
        )
    return GeoLayer(
        id=layer_id,
        name=name,
        kind=kind,
        source=source,
        is_real_geographic_data=True,
        available=True,
        geometry=data,
        notes=f"Loaded published dataset from {path.name}.",
    )


def load_geographic_context(region_id: str = DEFAULT_REGION_ID) -> GeographicContext:
    region = get_region(region_id)
    layers = [
        GeoLayer(
            id="aoi_envelope",
            name="Study-area envelope",
            kind="boundary",
            source="Named WGS84 bounding box (public place extent)",
            is_real_geographic_data=True,
            available=True,
            geometry={
                "type": "Polygon",
                "coordinates": [
                    [
                        [region.bounds.west, region.bounds.south],
                        [region.bounds.east, region.bounds.south],
                        [region.bounds.east, region.bounds.north],
                        [region.bounds.west, region.bounds.north],
                        [region.bounds.west, region.bounds.south],
                    ]
                ],
            },
            notes=(
                "Real geographic coordinates of an axis-aligned research AOI. "
                "Not a legal administrative or park boundary."
            ),
            attributes={"region_id": region.id},
        ),
        _load_optional_geojson(
            "victoria_boundary",
            "Victoria state boundary",
            "boundary",
            "User-supplied GeoJSON (e.g. ABS / data.gov.au)",
        ),
        _load_optional_geojson(
            "roads",
            "Roads",
            "transport",
            "User-supplied GeoJSON (e.g. OSM extract)",
        ),
        _load_optional_geojson(
            "water",
            "Water bodies",
            "hydrography",
            "User-supplied GeoJSON",
        ),
        _load_optional_geojson(
            "vegetation",
            "Vegetation / land cover",
            "landcover",
            "User-supplied GeoJSON or raster",
        ),
        _load_optional_geojson(
            "protected_areas",
            "Protected areas",
            "boundary",
            "User-supplied GeoJSON",
        ),
        _load_optional_geojson(
            "settlements",
            "Settlements",
            "populated_place",
            "User-supplied GeoJSON",
        ),
        GeoLayer(
            id="elevation",
            name="Elevation / terrain",
            kind="terrain",
            source="unavailable",
            is_real_geographic_data=False,
            available=False,
            notes=(
                "PLACEHOLDER: no DEM is bundled. Slope contribution to fire "
                "spread is set to 0. This is a documented model assumption, "
                "not real terrain."
            ),
        ),
        GeoLayer(
            id="basemap",
            name="Esri World Imagery basemap tiles",
            kind="basemap",
            source="Esri / Maxar / Earthstar Geographics public tile service",
            is_real_geographic_data=True,
            available=True,
            notes=(
                "Tiles are published satellite imagery. They are a visualization "
                "backdrop and are not used as simulation geometry unless a "
                "matching vector layer is loaded."
            ),
        ),
    ]
    return GeographicContext(region=region, layers=layers)
