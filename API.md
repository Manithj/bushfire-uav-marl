# API

Base URL: `http://localhost:8000`.

## REST

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/health` | Liveness |
| GET | `/api/regions` | Victoria study areas |
| GET | `/api/geography/{region_id}` | Layers with real vs placeholder flags |
| GET | `/api/policies` | Policy names |
| GET | `/api/metrics/catalog` | Metric definitions |
| GET | `/api/research/model` | Spaces, reward, sensor assumptions |
| POST | `/api/scenarios` | Persist scenario JSON |
| GET | `/api/scenarios` | List saved |
| POST | `/api/simulations` | Create env from scenario, `reset(seed)` |
| GET | `/api/simulations/{id}` | Full state + metrics |
| POST | `/api/simulations/{id}/start` | Run loop |
| POST | `/api/simulations/{id}/pause` | Pause |
| POST | `/api/simulations/{id}/resume` | Resume |
| POST | `/api/simulations/{id}/stop` | Stop |
| POST | `/api/simulations/{id}/reset` | Reset same seed |
| POST | `/api/simulations/{id}/step?n=1` | Single/multi step |
| POST | `/api/simulations/{id}/speed?speed=2` | 0.5, 1, 2, 5 |
| GET | `/api/simulations/{id}/metrics` | Metrics only |
| GET | `/api/simulations/{id}/events` | Engine event log |
| GET | `/api/simulations/{id}/debug/{uav_id}` | Observation/action dump |
| GET | `/api/simulations/{id}/replay` | Replay log |
| GET | `/api/simulations/{id}/export.json` | Download |
| GET | `/api/simulations/{id}/export.csv` | Download |
| POST | `/api/experiments/run` | Multi-seed comparison |
| GET | `/api/experiments/{id}` | Stored result |
| GET | `/api/experiments/{id}/report` | Text summary |
| GET | `/api/experiments/{id}/export.csv` | Trial table |

## WebSocket

```
/ws/simulations/{simulation_id}
```

Messages: `{ "type": "snapshot"|"tick"|"status", "state": { ... } }`.

`state.metrics` uses `null` for undefined values. `state.fire_grid.model_label` is the fire-model disclaimer.
