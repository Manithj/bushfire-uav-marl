import type {
  ExperimentResult,
  GeoLayer,
  MetricSpec,
  Region,
  ScenarioConfig,
  SimState,
} from "./types";

const json = async <T>(res: Response | Promise<Response>): Promise<T> => {
  const resolved = await res;
  if (!resolved.ok) {
    const text = await resolved.text();
    throw new Error(text || resolved.statusText);
  }
  return resolved.json() as Promise<T>;
};

export const api = {
  regions: () => json<Region[]>(fetch("/api/regions")),
  geography: (id: string) =>
    json<{ region: Region; layers: GeoLayer[] }>(fetch(`/api/geography/${id}`)),
  policies: () => json<{ name: string; is_marl: boolean }[]>(fetch("/api/policies")),
  metricsCatalog: () => json<MetricSpec[]>(fetch("/api/metrics/catalog")),
  researchModel: () => json<Record<string, unknown>>(fetch("/api/research/model")),
  createSimulation: (scenario: ScenarioConfig) =>
    json<{ simulation_id: string; state: SimState }>(
      fetch("/api/simulations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(scenario),
      }),
    ),
  getSimulation: (id: string) => json<SimState>(fetch(`/api/simulations/${id}`)),
  start: (id: string) => json<{ status: string }>(fetch(`/api/simulations/${id}/start`, { method: "POST" })),
  pause: (id: string) => json<{ status: string }>(fetch(`/api/simulations/${id}/pause`, { method: "POST" })),
  resume: (id: string) => json<{ status: string }>(fetch(`/api/simulations/${id}/resume`, { method: "POST" })),
  stop: (id: string) => json<{ status: string }>(fetch(`/api/simulations/${id}/stop`, { method: "POST" })),
  reset: (id: string) =>
    json<{ status: string; state: SimState }>(fetch(`/api/simulations/${id}/reset`, { method: "POST" })),
  step: (id: string, n = 1) =>
    json<{ status: string; state: SimState }>(
      fetch(`/api/simulations/${id}/step?n=${n}`, { method: "POST" }),
    ),
  setSpeed: (id: string, speed: number) =>
    json<{ speed: number }>(fetch(`/api/simulations/${id}/speed?speed=${speed}`, { method: "POST" })),
  events: (id: string) => json<SimState["events"]>(fetch(`/api/simulations/${id}/events`)),
  debug: (id: string, uavId: string) => json<Record<string, unknown>>(fetch(`/api/simulations/${id}/debug/${uavId}`)),
  replay: (id: string) => json<Record<string, unknown>>(fetch(`/api/simulations/${id}/replay`)),
  saveScenario: (scenario: ScenarioConfig) =>
    json<{ scenario_id: string }>(
      fetch("/api/scenarios", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(scenario),
      }),
    ),
  runExperiment: (body: unknown) =>
    json<ExperimentResult>(
      fetch("/api/experiments/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    ),
};

export function simulationSocket(id: string, onMsg: (data: { type: string; state?: SimState; status?: string }) => void): WebSocket {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${window.location.host}/ws/simulations/${id}`);
  ws.onmessage = (ev) => {
    onMsg(JSON.parse(ev.data) as { type: string; state?: SimState; status?: string });
  };
  return ws;
}

export function exportUrl(simId: string, kind: "json" | "csv"): string {
  return `/api/simulations/${simId}/export.${kind}`;
}

export function experimentExportUrl(id: string): string {
  return `/api/experiments/${id}/export.csv`;
}
