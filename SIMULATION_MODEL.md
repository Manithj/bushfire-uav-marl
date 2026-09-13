# Simulation model

**Label:** research simulation / simplified wildfire propagation model.

## Fire

Grid cellular automaton with 8-neighbour spread.

Cell states: `UNBURNED`, `BURNING`, `BURNED`, `UNBURNABLE`, `EXTINGUISHED`.

`EXTINGUISHED` is UAV suppression only. Detection does not change cell state. Natural burnout is `BURNED`.

Head rate of spread (m/s):

```
R_head = R0 · fuel · (1 + a · U^b)
```

Elliptical scale toward a neighbour at heading φ relative to the downwind direction:

```
R(φ) = R_head · (1 − e) / (1 − e · cos φ)
e = clip((LB − 1)/(LB + 1), 0, 0.95)
LB = 1 + c · U
```

`U` is wind speed (m/s). Wind direction is meteorological **FROM** (0° = North, clockwise). Downwind = FROM + 180°.

**Stochastic mode:** neighbour ignites with `p = 1 − exp(−R(φ)/d · dt)` using `numpy.random.Generator(seed)`.

**Deterministic mode:** each unburned cell accumulates exposure `Σ R(φ)/d · dt` and ignites at 1.0 (Huygens-style arrival).

Slope multiplies ROS only when a DEM is loaded (`slope_is_real`). Bundled runs have slope = 0.

Fuel is uniform synthetic 0.85 unless a vegetation layer is supplied.

Burnout time of a cell: `T_burn = T0 / max(fuel, 0.15)`.

## UAV dynamics

Point-mass kinematics in the local ENU frame.

- Heading 0 = North, clockwise.
- Speed, climb rate, and turn rate are clipped to configured maxima.
- Position is clipped to the AOI.
- Power (W): `P = P_hover + k · v²`. Energy `P·dt` is subtracted from `battery_j`.
- Return-to-base when `battery_j / capacity ≤ threshold`.

Wind does **not** advect the airframe (documented simplification).

## Sensor

A true detection of fire source `F` occurs if a BURNING cell of `F` is inside `sensor_range_m` and a Bernoulli trial succeeds:

```
p_true = p_det · (1 − p_fn) · clip(0.25 + 0.75·I, 0, 1)
```

False positives: if no in-range fire, emit a spurious report with probability `p_fp`.

No line-of-sight or weather attenuation (no DEM / no atmospheric model).

A detection report does **not** extinguish the fire. The map pin stays orange until a UAV overlaps the fire point.

## Suppression

A UAV extinguishes a fire only when it **overlaps** the fire point (planimetric distance ≤ `suppression_range_m`, default 120 m). Detection range is separate and does not change cell state.

On overlap, every remaining `BURNING` cell of that source becomes `EXTINGUISHED`. The source is marked suppressed, turns green on the map, and **stops spreading**.

## Communication

Undirected link if planimetric distance ≤ min of the two ranges and both batteries are positive. Detection broadcasts are one-hop (multi-hop optional). Drops use `failure_probability`.

## Coverage

```
coverage = explored_eligible_cells / eligible_search_cells
```

A cell is explored when any live UAV is within its sensor range of the cell centre. Inaccessible cells are never counted as searched.

## Collisions

From geometry: near-miss if horizontal distance < `min_separation` and `|Δz|` < 15 m. Collision if horizontal distance < 8 m and `|Δz|` < 15 m.
