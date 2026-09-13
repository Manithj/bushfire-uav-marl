"""Named study regions in Victoria, Australia.

Bounding boxes are real WGS84 extents of publicly known places.
They are not cadastral boundaries. State/park polygons are loaded
separately when a dataset is present; otherwise the UI must show a
labeled placeholder rather than a fabricated outline.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.geography.coordinate_transform import GeoBounds


@dataclass(frozen=True)
class NamedRegion:
    id: str
    name: str
    state: str
    country: str
    bounds: GeoBounds
    description: str
    data_status: str


# Victoria state envelope (Geoscience Australia / ABS approximate extent).
# This is an axis-aligned envelope, not the legal state boundary polygon.
VICTORIA_ENVELOPE = GeoBounds(
    south=-39.20,
    west=140.96,
    north=-33.98,
    east=150.04,
)

REGIONS: dict[str, NamedRegion] = {
    "victoria": NamedRegion(
        id="victoria",
        name="Victoria (statewide envelope)",
        state="Victoria",
        country="Australia",
        bounds=VICTORIA_ENVELOPE,
        description=(
            "Axis-aligned envelope of Victoria. Too large for a high-resolution "
            "fire grid; select a study area below for simulation."
        ),
        data_status="real_envelope",
    ),
    "kinglake": NamedRegion(
        id="kinglake",
        name="Kinglake / Central Highlands",
        state="Victoria",
        country="Australia",
        bounds=GeoBounds(south=-37.545, west=145.265, north=-37.495, east=145.355),
        description=(
            "Research AOI (~5.5 × 8 km) around Kinglake. "
            "Historically affected by the 2009 Black Saturday fires. "
            "Box is a searchable study envelope, not the park legal boundary."
        ),
        data_status="real_envelope",
    ),
    "grampians": NamedRegion(
        id="grampians",
        name="Grampians (Gariwerd)",
        state="Victoria",
        country="Australia",
        bounds=GeoBounds(south=-37.18, west=142.46, north=-37.12, east=142.56),
        description=(
            "Study box around Halls Gap / northern Grampians National Park. "
            "Box is a research AOI, not the park legal boundary."
        ),
        data_status="real_envelope",
    ),
    "east_gippsland": NamedRegion(
        id="east_gippsland",
        name="East Gippsland / Mallacoota hinterland",
        state="Victoria",
        country="Australia",
        bounds=GeoBounds(south=-37.58, west=149.68, north=-37.52, east=149.78),
        description=(
            "Study box west of Mallacoota, East Gippsland. "
            "Region affected by the 2019–20 Black Summer fires. "
            "Box is a research AOI, not a cadastral boundary."
        ),
        data_status="real_envelope",
    ),
    "otways": NamedRegion(
        id="otways",
        name="Otway Ranges",
        state="Victoria",
        country="Australia",
        bounds=GeoBounds(south=-38.73, west=143.54, north=-38.67, east=143.64),
        description=(
            "Study box in the Otway Ranges inland of Apollo Bay. "
            "Box is a research AOI, not the park legal boundary."
        ),
        data_status="real_envelope",
    ),
    "alpine": NamedRegion(
        id="alpine",
        name="Alpine / Mount Buffalo foothills",
        state="Victoria",
        country="Australia",
        bounds=GeoBounds(south=-36.75, west=146.76, north=-36.69, east=146.86),
        description=(
            "Study box in the Victorian Alps near Mount Buffalo. "
            "Box is a research AOI, not the park legal boundary."
        ),
        data_status="real_envelope",
    ),
}

DEFAULT_REGION_ID = "kinglake"


def get_region(region_id: str) -> NamedRegion:
    if region_id not in REGIONS:
        known = ", ".join(sorted(REGIONS))
        raise KeyError(f"Unknown region '{region_id}'. Known: {known}")
    return REGIONS[region_id]


def list_regions() -> list[NamedRegion]:
    return [REGIONS[k] for k in REGIONS]
