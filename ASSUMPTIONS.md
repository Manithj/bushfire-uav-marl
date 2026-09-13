# Assumptions and limitations

The UI **Assumptions** panel repeats this list. The simulator must never be described as operational.

## Model

- Fire: simplified elliptical/cellular spread, not Phoenix, SPARK, Prometheus, or FARSITE.
- UAV: kinematic point mass; no wind drift, no rotor dynamics, no airspace regulation.
- Sensor: planimetric range + Bernoulli; no terrain occlusion, smoke attenuation, or camera ISP. Detection reports do not extinguish fires.
- Suppression: UAV must be inside `suppression_range_m` of a BURNING cell. Intensity is reduced at a fixed rate. This is not a real retardant / water-drop model.
- Communication: geometric range + optional i.i.d. drop; not a radio-channel model.
- Fuel: synthetic uniform field unless the user adds `backend/data/geo/vegetation.geojson`.
- Terrain: slope = 0 unless a DEM is provided. 3D mode uses an **ellipsoid**, not a real DEM.
- Weather: constant wind speed/direction for the episode.

## Geography

| Layer | Status |
| --- | --- |
| Study-area envelope | Real WGS84 box of a named place; **not** a legal boundary |
| Esri World Imagery | Published satellite tiles; visualization only |
| Victoria polygon, roads, water, vegetation, settlements, protected areas | Loaded only if the user places GeoJSON in `backend/data/geo/`. Otherwise a labeled **placeholder** |
| Elevation | Placeholder |

Do not invent roads, public locations, or terrain.

## Stochasticity

Reproducibility holds for the same scenario JSON + seed + policy implementation. Changing NumPy/PyTorch versions can change bit-level floats; tests check identity on this environment.
