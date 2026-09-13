"""REST API routes."""

from __future__ import annotations

import csv
import io
import json
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse, Response

from app.baselines.registry import POLICY_NAMES
from app.experiments.config import ExperimentConfig
from app.experiments.runner import run_experiment
from app.geography.loader import load_geographic_context
from app.geography.regions import list_regions
from app.marl.action import ACTION_NAMES
from app.marl.observation import observation_spec
from app.schemas.scenario import ScenarioConfig
from app.simulation.manager import MANAGER
from app.simulation.metrics import METRIC_SPECS
from app.simulation.reward import RewardWeights, reward_equation_latex
from app.simulation.sensor_model import sensor_assumptions

router = APIRouter()

_EXPERIMENTS: dict[str, dict[str, Any]] = {}


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/regions")
def regions() -> list[dict[str, Any]]:
    return [
        {
            "id": r.id,
            "name": r.name,
            "state": r.state,
            "country": r.country,
            "bounds": {
                "south": r.bounds.south,
                "west": r.bounds.west,
                "north": r.bounds.north,
                "east": r.bounds.east,
            },
            "description": r.description,
            "data_status": r.data_status,
        }
        for r in list_regions()
    ]


@router.get("/geography/{region_id}")
def geography(region_id: str) -> dict[str, Any]:
    try:
        ctx = load_geographic_context(region_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {
        "region": {
            "id": ctx.region.id,
            "name": ctx.region.name,
            "bounds": ctx.region.bounds.__dict__,
            "description": ctx.region.description,
            "data_status": ctx.region.data_status,
        },
        "layers": [
            {
                "id": L.id,
                "name": L.name,
                "kind": L.kind,
                "source": L.source,
                "is_real_geographic_data": L.is_real_geographic_data,
                "available": L.available,
                "notes": L.notes,
                "geometry": L.geometry,
            }
            for L in ctx.layers
        ],
    }


@router.get("/policies")
def policies() -> list[dict[str, Any]]:
    return [
        {"name": n, "is_marl": n in {"dqn", "mappo", "maddpg"}}
        for n in POLICY_NAMES
    ]


@router.get("/metrics/catalog")
def metrics_catalog() -> list[dict[str, str]]:
    return [s.__dict__ for s in METRIC_SPECS]


@router.get("/research/model")
def research_model() -> dict[str, Any]:
    w = RewardWeights()
    return {
        "fire_model_label": "Research simulation / simplified wildfire propagation model",
        "action_space": list(ACTION_NAMES),
        "observation": observation_spec(False).__dict__,
        "reward_equation": reward_equation_latex(w),
        "reward_weights": w.__dict__,
        "sensor_assumptions": sensor_assumptions().__dict__,
        "training_vs_execution": {
            "training": "Python MARL environment (WildfireEnv) without visualization.",
            "execution": "Same WildfireEnv.step powered by a Policy.act interface.",
            "global_state": "Agents do not receive global fire locations unless policy.allow_global_state is True.",
        },
    }


@router.post("/scenarios")
def create_scenario(scenario: ScenarioConfig) -> dict[str, Any]:
    sid = MANAGER.save_scenario(scenario)
    return {"scenario_id": sid, "scenario": scenario.model_dump()}


@router.get("/scenarios")
def list_scenarios() -> list[dict[str, Any]]:
    return MANAGER.list_scenarios()


@router.get("/scenarios/{scenario_id}")
def get_scenario(scenario_id: str) -> dict[str, Any]:
    sc = MANAGER.get_scenario(scenario_id)
    if sc is None:
        raise HTTPException(404, "scenario not found")
    return sc.model_dump()


@router.post("/simulations")
def create_simulation(scenario: ScenarioConfig) -> dict[str, Any]:
    h = MANAGER.create(scenario)
    return {"simulation_id": h.simulation_id, "state": MANAGER.public(h)}


@router.get("/simulations")
def list_simulations() -> list[dict[str, Any]]:
    return MANAGER.list_sims()


@router.get("/simulations/{simulation_id}")
def get_simulation(simulation_id: str) -> dict[str, Any]:
    h = MANAGER.get(simulation_id)
    if h is None:
        raise HTTPException(404, "simulation not found")
    return MANAGER.public(h)


@router.post("/simulations/{simulation_id}/start")
async def start_simulation(simulation_id: str) -> dict[str, Any]:
    try:
        h = await MANAGER.start(simulation_id)
    except KeyError:
        raise HTTPException(404, "simulation not found") from None
    return {"status": h.status, "simulation_id": h.simulation_id}


@router.post("/simulations/{simulation_id}/pause")
async def pause_simulation(simulation_id: str) -> dict[str, Any]:
    try:
        h = await MANAGER.pause(simulation_id)
    except KeyError:
        raise HTTPException(404, "simulation not found") from None
    return {"status": h.status}


@router.post("/simulations/{simulation_id}/resume")
async def resume_simulation(simulation_id: str) -> dict[str, Any]:
    try:
        h = await MANAGER.resume(simulation_id)
    except KeyError:
        raise HTTPException(404, "simulation not found") from None
    return {"status": h.status}


@router.post("/simulations/{simulation_id}/stop")
async def stop_simulation(simulation_id: str) -> dict[str, Any]:
    try:
        h = await MANAGER.stop(simulation_id)
    except KeyError:
        raise HTTPException(404, "simulation not found") from None
    return {"status": h.status}


@router.post("/simulations/{simulation_id}/reset")
async def reset_simulation(simulation_id: str) -> dict[str, Any]:
    try:
        h = await MANAGER.reset(simulation_id)
    except KeyError:
        raise HTTPException(404, "simulation not found") from None
    return {"status": h.status, "state": MANAGER.public(h)}


@router.post("/simulations/{simulation_id}/step")
async def step_simulation(simulation_id: str, n: int = 1) -> dict[str, Any]:
    try:
        h = await MANAGER.step_once(simulation_id, n=n)
    except KeyError:
        raise HTTPException(404, "simulation not found") from None
    return {"status": h.status, "state": MANAGER.public(h)}


@router.post("/simulations/{simulation_id}/speed")
def set_speed(simulation_id: str, speed: float) -> dict[str, Any]:
    try:
        h = MANAGER.set_speed(simulation_id, speed)
    except KeyError:
        raise HTTPException(404, "simulation not found") from None
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"speed": h.speed}


@router.get("/simulations/{simulation_id}/metrics")
def get_metrics(simulation_id: str) -> dict[str, Any]:
    h = MANAGER.get(simulation_id)
    if h is None:
        raise HTTPException(404, "simulation not found")
    return {"metrics": h.env.public_state()["metrics"], "per_uav": h.env.public_state()["per_uav"]}


@router.get("/simulations/{simulation_id}/events")
def get_events(simulation_id: str) -> list[dict[str, Any]]:
    h = MANAGER.get(simulation_id)
    if h is None:
        raise HTTPException(404, "simulation not found")
    return h.env.events.as_dicts()


@router.get("/simulations/{simulation_id}/debug/{uav_id}")
def debug_uav(simulation_id: str, uav_id: str) -> dict[str, Any]:
    h = MANAGER.get(simulation_id)
    if h is None:
        raise HTTPException(404, "simulation not found")
    payload = h.env.debug_agent(uav_id)
    if payload is None:
        raise HTTPException(404, "uav not found")
    return payload


@router.get("/simulations/{simulation_id}/replay")
def get_replay(simulation_id: str) -> dict[str, Any]:
    h = MANAGER.get(simulation_id)
    if h is None or h.replay is None:
        raise HTTPException(404, "replay not found")
    h.replay.events = h.env.events.as_dicts()
    return h.replay.to_dict()


@router.get("/simulations/{simulation_id}/export.json")
def export_json(simulation_id: str) -> Response:
    h = MANAGER.get(simulation_id)
    if h is None:
        raise HTTPException(404, "simulation not found")
    state = h.env.public_state()
    payload = {
        "simulation_id": simulation_id,
        "scenario": h.scenario.model_dump(),
        "metrics": state["metrics"],
        "per_uav": state["per_uav"],
        "events": h.env.events.as_dicts(),
        "policy": state["policy"],
    }
    return Response(
        content=json.dumps(payload, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=sim_{simulation_id[:8]}.json"},
    )


@router.get("/simulations/{simulation_id}/export.csv")
def export_csv(simulation_id: str) -> Response:
    h = MANAGER.get(simulation_id)
    if h is None:
        raise HTTPException(404, "simulation not found")
    metrics = h.env.public_state()["metrics"]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["metric", "value"])
    for k, v in metrics.items():
        w.writerow([k, "" if v is None else v])
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=sim_{simulation_id[:8]}.csv"},
    )


@router.post("/experiments/run")
def experiments_run(cfg: ExperimentConfig) -> dict[str, Any]:
    result = run_experiment(cfg)
    _EXPERIMENTS[result["experiment_id"]] = result
    return result


@router.get("/experiments")
def experiments_list() -> list[dict[str, Any]]:
    return [
        {
            "experiment_id": r["experiment_id"],
            "name": r["name"],
            "timestamp": r["timestamp"],
            "policies": r["policies"],
            "seeds": r["seeds"],
        }
        for r in _EXPERIMENTS.values()
    ]


@router.get("/experiments/{experiment_id}")
def experiment_get(experiment_id: str) -> dict[str, Any]:
    r = _EXPERIMENTS.get(experiment_id)
    if r is None:
        raise HTTPException(404, "experiment not found")
    return r


@router.get("/experiments/{experiment_id}/report")
def experiment_report(experiment_id: str) -> PlainTextResponse:
    r = _EXPERIMENTS.get(experiment_id)
    if r is None:
        raise HTTPException(404, "experiment not found")
    return PlainTextResponse(r["report_text"])


@router.get("/experiments/{experiment_id}/export.csv")
def experiment_csv(experiment_id: str) -> Response:
    r = _EXPERIMENTS.get(experiment_id)
    if r is None:
        raise HTTPException(404, "experiment not found")
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["policy", "seed", "metric", "value"])
    for policy, trials in r["trials"].items():
        for trial in trials:
            seed = trial.get("seed")
            for k, v in trial.items():
                if k in {"policy", "seed", "trained_policy"}:
                    continue
                w.writerow([policy, seed, k, "" if v is None else v])
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=exp_{experiment_id[:8]}.csv"},
    )
