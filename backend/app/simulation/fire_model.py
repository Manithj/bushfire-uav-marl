"""Simplified grid-based wildfire propagation model.

Research simulation / simplified wildfire propagation model.
This is NOT an operational wildfire prediction system.

Model (Alexandridis-style cellular automaton + elliptical ROS):

    Cell states: UNBURNED=0, BURNING=1, BURNED=2, UNBURNABLE=3, EXTINGUISHED=4

    EXTINGUISHED is UAV suppression only. Detection does not change cell
    state. Natural burnout is BURNED, never EXTINGUISHED.

    For each burning cell, each of 8 neighbours ignites with probability

        p = 1 - exp(-R(φ) / d * dt)

    where d is centre-to-centre distance and R(φ) is the elliptical
    rate of spread (m/s) toward the neighbour:

        R(φ) = R_head * (1 - e) / (1 - e * cos(φ))
        R_head = R0 * fuel * (1 + a * U^b)
        e = clip((LB - 1) / (LB + 1), 0, 0.95)
        LB = 1 + c * U          # length-to-breadth vs wind speed U (m/s)

    φ is the angle between the downwind direction and the neighbour vector.
    Slope is multiplied as (1 + k_slope * tan(slope)) when a DEM exists;
    otherwise slope = 0.

    Fuel is a [0, 1] field. If no vegetation layer is loaded, fuel is
    uniform and labeled synthetic.

    Stochastic draws use numpy.random.Generator(seed). The same
    (grid, wind, fuel, ignitions, seed, dt) produces the same fire history.

    When stochastic=False, each unburned cell accumulates travel-time
    exposure Σ (R(φ)/d)·dt and ignites when exposure ≥ 1 (deterministic
    Huygens-style arrival). Do not use a 0.5 probability cutoff.

    Burning duration of a cell: T_burn = T0 / max(fuel, ε). After T_burn
    the cell becomes BURNED and no longer ignites neighbours.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

UNBURNED = np.int8(0)
BURNING = np.int8(1)
BURNED = np.int8(2)
UNBURNABLE = np.int8(3)
EXTINGUISHED = np.int8(4)

# 8-neighbour offsets (drow, dcol) in grid space.
# Grid: row increases south (decreasing northing), col increases east.
_NEIGHBOURS = (
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
)


@dataclass
class FireSource:
    fire_id: str
    x: float
    y: float
    lat: float
    lon: float
    initial_intensity: float
    ignition_time: float
    radius_m: float
    active: bool = True
    detected: bool = False
    first_detection_time: float | None = None
    detected_by: str | None = None
    suppressed: bool = False
    extinguished_time: float | None = None
    suppressed_by: str | None = None


@dataclass
class FireModelConfig:
    cell_size_m: float = 50.0
    base_ros_mps: float = 0.35
    wind_ros_coeff: float = 0.18
    wind_ros_exponent: float = 1.15
    lb_wind_coeff: float = 0.25
    burnout_s: float = 180.0
    slope_coeff: float = 1.0
    stochastic: bool = True
    intensity_peak: float = 1.0
    intensity_decay: float = 0.004
    suppress_rate: float = 0.7


@dataclass
class FireModel:
    """Raster fire state in the local projected frame."""

    height: int
    width: int
    cell_size_m: float
    origin_x: float
    origin_y: float
    extent_x: float
    extent_y: float
    config: FireModelConfig
    rng: np.random.Generator
    state: np.ndarray
    intensity: np.ndarray
    fuel: np.ndarray
    time_burning: np.ndarray
    fire_id_grid: np.ndarray
    slope: np.ndarray
    exposure: np.ndarray
    fuel_is_synthetic: bool = True
    slope_is_real: bool = False
    sources: list[FireSource] = field(default_factory=list)
    _id_to_index: dict[str, int] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        origin_x: float,
        origin_y: float,
        extent_x: float,
        extent_y: float,
        config: FireModelConfig,
        rng: np.random.Generator,
        fuel: np.ndarray | None = None,
        slope: np.ndarray | None = None,
        fuel_is_synthetic: bool = True,
        slope_is_real: bool = False,
    ) -> "FireModel":
        cell = config.cell_size_m
        width = max(8, int(np.ceil(extent_x / cell)))
        height = max(8, int(np.ceil(extent_y / cell)))
        if fuel is None:
            fuel = np.full((height, width), 0.85, dtype=np.float32)
        if slope is None:
            slope = np.zeros((height, width), dtype=np.float32)
        return cls(
            height=height,
            width=width,
            cell_size_m=cell,
            origin_x=origin_x,
            origin_y=origin_y,
            extent_x=extent_x,
            extent_y=extent_y,
            config=config,
            rng=rng,
            state=np.zeros((height, width), dtype=np.int8),
            intensity=np.zeros((height, width), dtype=np.float32),
            fuel=fuel.astype(np.float32),
            time_burning=np.zeros((height, width), dtype=np.float32),
            fire_id_grid=np.full((height, width), -1, dtype=np.int16),
            slope=slope.astype(np.float32),
            exposure=np.zeros((height, width), dtype=np.float32),
            fuel_is_synthetic=fuel_is_synthetic,
            slope_is_real=slope_is_real,
        )

    def cell_of(self, x: float, y: float) -> tuple[int, int] | None:
        col = int((x - self.origin_x) / self.cell_size_m)
        row = int((self.origin_y + self.extent_y - y) / self.cell_size_m)
        if 0 <= row < self.height and 0 <= col < self.width:
            return row, col
        return None

    def cell_centre(self, row: int, col: int) -> tuple[float, float]:
        x = self.origin_x + (col + 0.5) * self.cell_size_m
        y = self.origin_y + self.extent_y - (row + 0.5) * self.cell_size_m
        return x, y

    def ignite(
        self,
        fire_id: str,
        x: float,
        y: float,
        lat: float,
        lon: float,
        intensity: float,
        ignition_time: float,
        radius_m: float,
    ) -> FireSource:
        source = FireSource(
            fire_id=fire_id,
            x=x,
            y=y,
            lat=lat,
            lon=lon,
            initial_intensity=float(np.clip(intensity, 0.05, 1.0)),
            ignition_time=ignition_time,
            radius_m=max(self.cell_size_m, radius_m),
        )
        idx = len(self.sources)
        self.sources.append(source)
        self._id_to_index[fire_id] = idx

        radius_cells = max(1, int(np.ceil(source.radius_m / self.cell_size_m)))
        centre = self.cell_of(x, y)
        if centre is None:
            source.active = False
            return source
        cr, cc = centre
        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):
                r, c = cr + dr, cc + dc
                if not (0 <= r < self.height and 0 <= c < self.width):
                    continue
                if self.state[r, c] == UNBURNABLE:
                    continue
                cx, cy = self.cell_centre(r, c)
                if (cx - x) ** 2 + (cy - y) ** 2 <= source.radius_m**2:
                    if self.fuel[r, c] <= 0.02:
                        continue
                    self.state[r, c] = BURNING
                    self.intensity[r, c] = source.initial_intensity
                    self.fire_id_grid[r, c] = idx
                    self.time_burning[r, c] = 0.0
        return source

    def mark_detected(self, fire_id: str, time_s: float, uav_id: str) -> bool:
        src = self.source_by_id(fire_id)
        if src is None or src.detected:
            return False
        src.detected = True
        src.first_detection_time = time_s
        src.detected_by = uav_id
        return True

    def suppress(
        self,
        x: float,
        y: float,
        range_m: float,
        dt: float,
        uav_id: str,
        time_s: float,
    ) -> tuple[int, list[str]]:
        """Put out a fire only when the UAV overlaps it.

        Detection does not extinguish. The UAV must be within range_m of
        the ignition point or a BURNING cell. That fire is then fully
        EXTINGUISHED: every remaining BURNING cell is cleared and it
        stops spreading. Natural burnout is still BURNED, not this path.
        """
        del dt
        if range_m <= 0:
            return 0, []
        hit: set[int] = set()
        for i, src in enumerate(self.sources):
            if src.active and float(np.hypot(src.x - x, src.y - y)) <= range_m:
                hit.add(i)
        burning = self.state == BURNING
        if np.any(burning):
            rows, cols = np.nonzero(burning)
            xs = self.origin_x + (cols + 0.5) * self.cell_size_m
            ys = self.origin_y + self.extent_y - (rows + 0.5) * self.cell_size_m
            in_range = np.hypot(xs - x, ys - y) <= range_m
            for idx in self.fire_id_grid[rows[in_range], cols[in_range]]:
                if int(idx) >= 0:
                    hit.add(int(idx))
        if not hit:
            return 0, []
        n = 0
        newly: list[str] = []
        for i in hit:
            mask = (self.fire_id_grid == i) & (self.state == BURNING)
            n += int(np.sum(mask))
            self.state[mask] = EXTINGUISHED
            self.intensity[mask] = 0.0
            src = self.sources[i]
            src.active = False
            if not src.suppressed:
                src.suppressed = True
                src.extinguished_time = time_s
                src.suppressed_by = uav_id
                newly.append(src.fire_id)
        return n, newly

    def extinguished_area_m2(self) -> float:
        n = int(np.sum(self.state == EXTINGUISHED))
        return n * self.cell_size_m * self.cell_size_m

    def nearest_burning(
        self,
        x: float,
        y: float,
        fire_id: str | None = None,
    ) -> tuple[float, float, float] | None:
        """Nearest BURNING cell centre to (x, y), optionally of one source."""
        mask = self.state == BURNING
        if fire_id is not None:
            idx = self._id_to_index.get(fire_id)
            if idx is None:
                return None
            mask = mask & (self.fire_id_grid == idx)
        if not np.any(mask):
            return None
        rows, cols = np.nonzero(mask)
        xs = self.origin_x + (cols + 0.5) * self.cell_size_m
        ys = self.origin_y + self.extent_y - (rows + 0.5) * self.cell_size_m
        d2 = (xs - x) ** 2 + (ys - y) ** 2
        k = int(np.argmin(d2))
        return float(xs[k]), float(ys[k]), float(self.intensity[rows[k], cols[k]])

    def source_by_id(self, fire_id: str) -> FireSource | None:
        idx = self._id_to_index.get(fire_id)
        if idx is None:
            return None
        return self.sources[idx]

    def source_id_at(self, row: int, col: int) -> str | None:
        idx = int(self.fire_id_grid[row, col])
        if idx < 0 or idx >= len(self.sources):
            return None
        return self.sources[idx].fire_id

    def head_ros(self, wind_speed_mps: float, fuel: np.ndarray) -> np.ndarray:
        cfg = self.config
        return (
            cfg.base_ros_mps
            * fuel
            * (1.0 + cfg.wind_ros_coeff * np.power(max(wind_speed_mps, 0.0), cfg.wind_ros_exponent))
        ).astype(np.float32)

    def eccentricity(self, wind_speed_mps: float) -> float:
        lb = 1.0 + self.config.lb_wind_coeff * max(wind_speed_mps, 0.0)
        e = (lb - 1.0) / (lb + 1.0)
        return float(np.clip(e, 0.0, 0.95))

    def step(self, dt: float, wind_speed_mps: float, wind_from_deg: float, time_s: float) -> None:
        """Advance fire one timestep.

        wind_from_deg: meteorological wind direction, degrees FROM which
        the wind blows, 0=North, clockwise. Downwind heading is +180°.
        """
        del time_s
        if dt <= 0:
            return
        burning = self.state == BURNING
        if not np.any(burning):
            for src in self.sources:
                src.active = False
            return

        downwind_deg = (wind_from_deg + 180.0) % 360.0
        downwind_rad = np.deg2rad(downwind_deg)
        ecc = self.eccentricity(wind_speed_mps)
        head = self.head_ros(wind_speed_mps, self.fuel)

        new_ignitions = np.zeros_like(self.state, dtype=bool)
        new_ids = np.full_like(self.fire_id_grid, -1)
        new_intensity = np.zeros_like(self.intensity)
        exposure_delta = np.zeros_like(self.exposure)

        for drow, dcol in _NEIGHBOURS:
            # Neighbour vector in ENU: col→east, -row→north
            dx = float(dcol) * self.cell_size_m
            dy = float(-drow) * self.cell_size_m
            dist = float(np.hypot(dx, dy))
            neighbour_heading = float(np.arctan2(dx, dy))
            phi = neighbour_heading - downwind_rad
            cos_phi = float(np.cos(phi))
            ros_scale = (1.0 - ecc) / max(1.0 - ecc * cos_phi, 1e-6)

            src_state = np.roll(np.roll(burning, drow, axis=0), dcol, axis=1)
            src_id = np.roll(np.roll(self.fire_id_grid, drow, axis=0), dcol, axis=1)
            src_int = np.roll(np.roll(self.intensity, drow, axis=0), dcol, axis=1)
            src_head = np.roll(np.roll(head, drow, axis=0), dcol, axis=1)
            # Invalidate wraps
            if drow > 0:
                src_state[:drow, :] = False
            elif drow < 0:
                src_state[drow:, :] = False
            if dcol > 0:
                src_state[:, :dcol] = False
            elif dcol < 0:
                src_state[:, dcol:] = False

            candidates = (self.state == UNBURNED) & src_state & (self.fuel > 0.02)
            if not np.any(candidates):
                continue

            ros = src_head * ros_scale
            if self.slope_is_real:
                ros = ros * (1.0 + self.config.slope_coeff * np.tan(self.slope))
            increment = np.maximum(ros, 0.0) / dist * dt
            p = 1.0 - np.exp(-increment)
            p = np.clip(p, 0.0, 1.0)
            if self.config.stochastic:
                draw = self.rng.random(self.state.shape) < p
                ignite = candidates & draw
            else:
                exposure_delta = np.where(candidates, exposure_delta + increment.astype(np.float32), exposure_delta)
                ignite = np.zeros_like(candidates)

            new_ignitions |= ignite
            assign = ignite & (new_ids < 0)
            new_ids = np.where(assign, src_id, new_ids)
            new_intensity = np.where(assign, np.clip(src_int * 0.95, 0.15, 1.0), new_intensity)

        if not self.config.stochastic:
            self.exposure += exposure_delta
            det_ignite = (self.state == UNBURNED) & (self.exposure >= 1.0) & (self.fuel > 0.02)
            new_ignitions |= det_ignite
            for drow, dcol in _NEIGHBOURS:
                src_id = np.roll(np.roll(self.fire_id_grid, drow, axis=0), dcol, axis=1)
                src_state = np.roll(np.roll(burning, drow, axis=0), dcol, axis=1)
                src_int = np.roll(np.roll(self.intensity, drow, axis=0), dcol, axis=1)
                if drow > 0:
                    src_state[:drow, :] = False
                elif drow < 0:
                    src_state[drow:, :] = False
                if dcol > 0:
                    src_state[:, :dcol] = False
                elif dcol < 0:
                    src_state[:, dcol:] = False
                assign = det_ignite & src_state & (new_ids < 0)
                new_ids = np.where(assign, src_id, new_ids)
                new_intensity = np.where(assign, np.clip(src_int * 0.95, 0.15, 1.0), new_intensity)
            self.exposure[det_ignite] = 0.0

        self.state[new_ignitions] = BURNING
        self.fire_id_grid[new_ignitions] = new_ids[new_ignitions]
        self.intensity[new_ignitions] = new_intensity[new_ignitions]
        self.time_burning[new_ignitions] = 0.0

        self.time_burning[burning] += dt
        burnout = self.config.burnout_s / np.maximum(self.fuel, 0.15)
        expired = burning & (self.time_burning >= burnout)
        self.state[expired] = BURNED
        self.intensity[expired] = 0.0

        still = self.state == BURNING
        self.intensity[still] = np.clip(
            self.intensity[still]
            * (1.0 - self.config.intensity_decay * dt)
            + 0.05 * self.fuel[still],
            0.08,
            self.config.intensity_peak,
        )

        for i, src in enumerate(self.sources):
            src.active = bool(np.any((self.state == BURNING) & (self.fire_id_grid == i)))

    def burned_area_m2(self) -> float:
        n = int(np.sum((self.state == BURNED) | (self.state == BURNING)))
        return n * self.cell_size_m * self.cell_size_m

    def active_area_m2(self) -> float:
        n = int(np.sum(self.state == BURNING))
        return n * self.cell_size_m * self.cell_size_m

    def burning_cells(self) -> list[tuple[int, int]]:
        rows, cols = np.nonzero(self.state == BURNING)
        return list(zip(rows.tolist(), cols.tolist(), strict=True))

    def perimeter_points_local(self, max_points: int = 180) -> list[list[tuple[float, float]]]:
        """Convex hull of affected cells (burning, burned, extinguished)."""
        from scipy.spatial import ConvexHull

        rings: list[list[tuple[float, float]]] = []
        for i, src in enumerate(self.sources):
            mask = (self.fire_id_grid == i) & (
                (self.state == BURNING) | (self.state == BURNED) | (self.state == EXTINGUISHED)
            )
            if not np.any(mask):
                rings.append([(src.x, src.y)])
                continue
            rows, cols = np.nonzero(mask)
            xs = self.origin_x + (cols + 0.5) * self.cell_size_m
            ys = self.origin_y + self.extent_y - (rows + 0.5) * self.cell_size_m
            pts_xy = np.column_stack([xs, ys])
            if len(pts_xy) < 3:
                rings.append([(float(xs[0]), float(ys[0])), (src.x, src.y)])
                continue
            try:
                hull = ConvexHull(pts_xy)
                verts = pts_xy[hull.vertices]
                if len(verts) > max_points:
                    step = int(np.ceil(len(verts) / max_points))
                    verts = verts[::step]
                pts = [(float(x), float(y)) for x, y in verts]
                if pts[0] != pts[-1]:
                    pts.append(pts[0])
                rings.append(pts)
            except Exception:
                rings.append([(src.x, src.y)])
        return rings

    def snapshot_public(self) -> dict:
        burned = int(np.sum(self.state == BURNED))
        burning = int(np.sum(self.state == BURNING))
        extinguished = int(np.sum(self.state == EXTINGUISHED))
        return {
            "grid_height": self.height,
            "grid_width": self.width,
            "cell_size_m": self.cell_size_m,
            "burned_cells": burned,
            "burning_cells": burning,
            "extinguished_cells": extinguished,
            "burned_area_m2": self.burned_area_m2(),
            "active_area_m2": self.active_area_m2(),
            "extinguished_area_m2": self.extinguished_area_m2(),
            "fuel_is_synthetic": self.fuel_is_synthetic,
            "slope_is_real": self.slope_is_real,
            "model_label": "Research simulation / simplified wildfire propagation model",
            "suppression_note": (
                "EXTINGUISHED cells were put out by a UAV within suppression range. "
                "Detection alone does not extinguish."
            ),
        }
