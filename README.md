# Bushfire UAV MARL Laboratory

Research-grade interactive simulator for **multi-agent reinforcement learning** on cooperative wildfire *detection* with autonomous UAVs.

This is a **research simulation / simplified wildfire propagation model**. It is **not** an operational wildfire prediction, fire-behaviour, or UAV command-and-control system.

## What it does

A supervisor can select a real WGS84 study box in Victoria, place ignition points, configure weather and UAV/sensor parameters, run a seeded simulation under a baseline or loaded MARL policy, inspect metrics that are computed from simulation state, compare policies across seeds, and export JSON/CSV.

## Run

Backend (Python 3.11+, [uv](https://github.com/astral-sh/uv)):

```bash
cd backend
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
uv run pytest
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The Vite dev server proxies `/api` and `/ws` to the backend.

Headless deterministic check (no UI):

```bash
cd backend
uv run python scripts/verify_deterministic.py
```

Train MAPPO and MADDPG to overfly and extinguish (CPU, no visualization):

```bash
cd backend
uv run python scripts/train_mappo_maddpg.py --outdir data/checkpoints
```

Then choose policy `dqn`, `mappo`, or `maddpg`. An absent checkpoint is labeled **untrained** and must not be read as a MARL result.

## Documentation

| File | Contents |
| --- | --- |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Modules, training vs deployment |
| [SIMULATION_MODEL.md](SIMULATION_MODEL.md) | Fire, UAV, sensor, comms mathematics |
| [MARL.md](MARL.md) | Observation, action, reward, policy interface |
| [METRICS.md](METRICS.md) | Metric definitions and sources |
| [ASSUMPTIONS.md](ASSUMPTIONS.md) | Limitations and data provenance |
| [EXPERIMENTS.md](EXPERIMENTS.md) | Multi-seed runner and reporting |
| [API.md](API.md) | REST and WebSocket |

## Scientific rules used in this repo

- Metrics come from simulation state. If undefined, the API returns `null` (UI: **N/A**).
- Distances are computed in a local azimuthal-equidistant metre frame, never in raw lat/lon.
- `seed` plus scenario plus policy determines the trajectory (see tests).
- Real geographic data (AOI envelopes, OSM/CARTO tiles) is labeled separately from simulated fire/UAV/coverage layers.
- Missing datasets are placeholders, not invented roads or terrain.
