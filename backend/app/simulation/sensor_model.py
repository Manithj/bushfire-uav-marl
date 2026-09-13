"""Probabilistic UAV fire-detection sensor.

Detection is never magically perfect.

A UAV reports a true detection of fire source F when at least one
BURNING cell belonging to F lies inside the sensor radius and a
Bernoulli trial succeeds:

    p_true = p_det * (1 - p_fn) * range_factor * intensity_factor

    range_factor = 1                          if d <= R
                   clip(1 - (d-R)/R_soft, 0)  otherwise (hard cutoff at R)
    intensity_factor = clip(0.25 + 0.75 * I, 0, 1)

False positives: each timestep, if no true fire is in range, a false
positive is emitted with probability p_fp (independent).

Line-of-sight is not applied unless a DEM is available (currently not).
This assumption is exposed in the research panel.

All random draws use the simulation RNG (reproducible with seed).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.simulation.fire_model import BURNING, FireModel
from app.simulation.uav import UAVAgent


@dataclass(frozen=True)
class DetectionEvent:
    time_s: float
    uav_id: str
    is_true: bool
    fire_id: str | None
    x: float
    y: float
    lat: float
    lon: float
    intensity: float
    distance_m: float
    probability_used: float


@dataclass(frozen=True)
class SensorAssumptions:
    line_of_sight_applied: bool
    hard_range_cutoff: bool
    terrain_occlusion: bool
    weather_attenuation: bool
    notes: str


def sensor_assumptions() -> SensorAssumptions:
    return SensorAssumptions(
        line_of_sight_applied=False,
        hard_range_cutoff=True,
        terrain_occlusion=False,
        weather_attenuation=False,
        notes=(
            "Detection uses planimetric distance in the local AEQD frame. "
            "No DEM is loaded, so terrain occlusion and line-of-sight are "
            "not applied. Weather does not attenuate the sensor. "
            "These are research simplifications. "
            "A detection report does not extinguish the fire; "
            "the UAV must overfly a BURNING cell within suppression range."
        ),
    )


def _intensity_factor(intensity: float) -> float:
    return float(np.clip(0.25 + 0.75 * intensity, 0.0, 1.0))


def sense_fires(
    uav: UAVAgent,
    fire: FireModel,
    rng: np.random.Generator,
    time_s: float,
    to_wgs84,
) -> list[DetectionEvent]:
    """Run one sensor scan for a single UAV."""
    events: list[DetectionEvent] = []
    rng_uav = uav.config
    r_max = rng_uav.sensor_range_m
    if r_max <= 0 or uav.body.battery_j <= 0:
        return events

    p_det = float(np.clip(rng_uav.detection_probability, 0.0, 1.0))
    p_fn = float(np.clip(rng_uav.false_negative_probability, 0.0, 1.0))
    p_fp = float(np.clip(rng_uav.false_positive_probability, 0.0, 1.0))
    noise = max(0.0, rng_uav.observation_noise_m)

    burning = fire.state == BURNING
    if not np.any(burning):
        if rng.random() < p_fp:
            events.append(_false_positive(uav, rng, time_s, noise, to_wgs84, p_fp))
        return events

    rows, cols = np.nonzero(burning)
    xs = fire.origin_x + (cols + 0.5) * fire.cell_size_m
    ys = fire.origin_y + fire.extent_y - (rows + 0.5) * fire.cell_size_m
    dist = np.hypot(xs - uav.body.x, ys - uav.body.y)
    in_range = dist <= r_max
    if not np.any(in_range):
        if rng.random() < p_fp:
            events.append(_false_positive(uav, rng, time_s, noise, to_wgs84, p_fp))
        return events

    # Best (closest, then brightest) cell per fire_id
    best: dict[int, tuple[float, float, float, float, int, int]] = {}
    for i in np.nonzero(in_range)[0]:
        fid = int(fire.fire_id_grid[rows[i], cols[i]])
        intensity = float(fire.intensity[rows[i], cols[i]])
        d = float(dist[i])
        prev = best.get(fid)
        if prev is None or d < prev[0] or (d == prev[0] and intensity > prev[1]):
            best[fid] = (d, intensity, float(xs[i]), float(ys[i]), int(rows[i]), int(cols[i]))

    any_true_attempt = False
    for fid, (d, intensity, x, y, _r, _c) in best.items():
        if fid < 0 or fid >= len(fire.sources):
            continue
        source = fire.sources[fid]
        p = p_det * (1.0 - p_fn) * _intensity_factor(intensity)
        p = float(np.clip(p, 0.0, 1.0))
        any_true_attempt = True
        if rng.random() >= p:
            continue
        nx = x + float(rng.normal(0.0, noise))
        ny = y + float(rng.normal(0.0, noise))
        lat, lon = to_wgs84(nx, ny)
        events.append(
            DetectionEvent(
                time_s=time_s,
                uav_id=uav.uav_id,
                is_true=True,
                fire_id=source.fire_id,
                x=nx,
                y=ny,
                lat=lat,
                lon=lon,
                intensity=intensity,
                distance_m=d,
                probability_used=p,
            )
        )

    if not any_true_attempt and rng.random() < p_fp:
        events.append(_false_positive(uav, rng, time_s, noise, to_wgs84, p_fp))
    return events


def _false_positive(
    uav: UAVAgent,
    rng: np.random.Generator,
    time_s: float,
    noise: float,
    to_wgs84,
    p_fp: float,
) -> DetectionEvent:
    ang = float(rng.uniform(0.0, 2.0 * np.pi))
    rad = float(rng.uniform(0.0, uav.config.sensor_range_m))
    x = uav.body.x + rad * np.sin(ang)
    y = uav.body.y + rad * np.cos(ang)
    x += float(rng.normal(0.0, noise))
    y += float(rng.normal(0.0, noise))
    lat, lon = to_wgs84(x, y)
    return DetectionEvent(
        time_s=time_s,
        uav_id=uav.uav_id,
        is_true=False,
        fire_id=None,
        x=x,
        y=y,
        lat=lat,
        lon=lon,
        intensity=float(rng.uniform(0.1, 0.6)),
        distance_m=rad,
        probability_used=p_fp,
    )
