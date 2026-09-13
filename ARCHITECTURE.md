# Architecture

```
                    Scenario JSON + seed
                            │
                            ▼
                   WildfireEnv.step()
                   (Python, NumPy)
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
     Training (headless)          Deployment API
     scripts/train_ippo.py        FastAPI + WebSocket
              │                           │
              ▼                           ├─ 2D MapLibre
     checkpoint .pt                       └─ optional Cesium 3D
```

Visualization is never required for training. Both paths call the same `WildfireEnv.step`.

## Backend packages

| Package | Role |
| --- | --- |
| `app/simulation` | Fire, UAV kinematics, sensor, coverage, comms, collisions, metrics, replay |
| `app/marl` | Observation/action spaces, `Policy.act`, checkpoint load |
| `app/baselines` | random, lawnmower, greedy, frontier, independent |
| `app/geography` | AEQD transform, Victoria regions, layer loader |
| `app/experiments` | Multi-seed runner, mean/SD/CI |
| `app/api` | REST + `/ws/simulations/{id}` |
| `app/schemas` | Pydantic scenario (JSON-serializable) |

## Frontend

React + TypeScript + Vite + Tailwind. MapLibre is the default 2D view. Cesium is lazy-loaded and **must** consume the same `SimState` payload.

## Coordinate frames

- Geographic: WGS84 (EPSG:4326) for storage and map display.
- Local: azimuthal equidistant, metres, origin at AOI centroid. All Euclidean distances, headings, and grids use this frame.

## Clock

Fixed simulation timestep `dt` (default 1 s). The UI speed multiplier (`×0.5…×5`) only changes wall-clock sleep, not `dt`. The displayed clock is `environment.time_s`.
