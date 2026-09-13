import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip as RTooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, experimentExportUrl, exportUrl, simulationSocket } from "./api";
import { MapView } from "./components/MapView";

const Map3D = lazy(() => import("./components/Map3D").then((m) => ({ default: m.Map3D })));
import { Tip } from "./components/Tooltip";
import { defaultScenario, formatMetric, formatTime, POLICY_HELP, WIND_LABELS, randomKinglakeFires } from "./lib";
import type {
  ExperimentResult,
  GeoLayer,
  MetricSpec,
  PlaceMode,
  Region,
  ScenarioConfig,
  SimEvent,
  SimState,
} from "./types";

type Tab = "scenario" | "metrics" | "events" | "compare" | "research" | "debug" | "assumptions";

const DEMO_STEPS = [
  "Select Victoria region",
  "Inspect geographic area",
  "Place wildfire(s)",
  "Configure environment",
  "Deploy UAV fleet",
  "Select policy",
  "Start simulation",
  "Observe coordination",
  "Inspect detection",
  "Inspect metrics",
  "Compare baselines",
  "Export results",
];

export default function App() {
  const [scenario, setScenario] = useState<ScenarioConfig>(defaultScenario);
  const [regions, setRegions] = useState<Region[]>([]);
  const [layers, setLayers] = useState<GeoLayer[]>([]);
  const [catalog, setCatalog] = useState<MetricSpec[]>([]);
  const [simId, setSimId] = useState<string | null>(null);
  const [state, setState] = useState<SimState | null>(null);
  const [events, setEvents] = useState<SimEvent[]>([]);
  const [tab, setTab] = useState<Tab>("scenario");
  const [placeMode, setPlaceMode] = useState<PlaceMode>("pan");
  const [hover, setHover] = useState<{ lat: number; lon: number } | null>(null);
  const [view3d, setView3d] = useState(false);
  const [showCoverage, setShowCoverage] = useState(true);
  const [showSensors, setShowSensors] = useState(true);
  const [showComms, setShowComms] = useState(true);
  const [showTrails, setShowTrails] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [demoStep, setDemoStep] = useState(0);
  const [series, setSeries] = useState<{ t: number; coverage: number; reward: number; detected: number }[]>([]);
  const [experiment, setExperiment] = useState<ExperimentResult | null>(null);
  const [expBusy, setExpBusy] = useState(false);
  const [debugId, setDebugId] = useState<string>("uav_00");
  const [debug, setDebug] = useState<Record<string, unknown> | null>(null);
  const [replayOpen, setReplayOpen] = useState(false);
  const [replayIdx, setReplayIdx] = useState(0);
  const [replayFrames, setReplayFrames] = useState<SimState[]>([]);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    void api.regions().then(setRegions);
    void api.metricsCatalog().then(setCatalog);
    void api.geography(scenario.location.region_id).then((g) => setLayers(g.layers));
  }, []);

  const applyIncoming = useCallback((s: SimState) => {
    setState(s);
    if (s.events) setEvents(s.events);
    setSeries((prev) => {
      const next = [
        ...prev,
        {
          t: s.time_s,
          coverage: s.metrics.coverage_fraction ?? 0,
          reward: s.metrics.episode_reward ?? 0,
          detected: s.metrics.detected_fires ?? 0,
        },
      ];
      return next.length > 800 ? next.slice(-800) : next;
    });
  }, []);

  const connectWs = (id: string) => {
    wsRef.current?.close();
    wsRef.current = simulationSocket(id, (msg) => {
      if (msg.state) applyIncoming(msg.state);
    });
  };

  const createSim = async () => {
    setBusy(true);
    setError(null);
    try {
      const res = await api.createSimulation(scenario);
      setSimId(res.simulation_id);
      setSeries([]);
      setEvents(res.state.events ?? []);
      applyIncoming(res.state);
      connectWs(res.simulation_id);
      setDemoStep(6);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const cmd = async (fn: () => Promise<unknown>) => {
    setError(null);
    try {
      await fn();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const onMapClick = (lat: number, lon: number) => {
    if (placeMode === "fire") {
      const n = scenario.fires.length + 1;
      setScenario({
        ...scenario,
        fires: [
          ...scenario.fires,
          {
            fire_id: `fire_${String(n).padStart(2, "0")}`,
            latitude: lat,
            longitude: lon,
            initial_intensity: 0.8,
            ignition_time: 0,
            radius_m: 80,
          },
        ],
      });
      setDemoStep((s) => Math.max(s, 3));
    } else if (placeMode === "uav") {
      setScenario({
        ...scenario,
        uavs: {
          ...scenario.uavs,
          placements: [
            ...scenario.uavs.placements,
            { latitude: lat, longitude: lon, heading_deg: 0, altitude_m: scenario.uavs.altitude_m },
          ],
          count: Math.max(scenario.uavs.count, scenario.uavs.placements.length + 1),
        },
      });
      setDemoStep((s) => Math.max(s, 5));
    }
  };

  const changeRegion = async (id: string) => {
    setScenario({ ...scenario, location: { region_id: id }, fires: [], uavs: { ...scenario.uavs, placements: [] } });
    const g = await api.geography(id);
    setLayers(g.layers);
    setDemoStep(1);
  };

  const runCompare = async () => {
    setExpBusy(true);
    setError(null);
    try {
      const short = {
        ...scenario,
        environment: { ...scenario.environment, duration_s: Math.min(scenario.environment.duration_s, 180) },
      };
      const res = await api.runExperiment({
        name: `${scenario.name} comparison`,
        scenario: short,
        policies: ["dqn", "mappo", "maddpg"],
        seeds: [1, 2, 3],
        checkpoint_path: scenario.policy.checkpoint_path,
        max_steps: 180,
      });
      setExperiment(res);
      setTab("compare");
      setDemoStep(11);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setExpBusy(false);
    }
  };

  const loadReplay = async () => {
    if (!simId) return;
    const raw = await api.replay(simId);
    const frames = (raw.frames as SimState[]) ?? [];
    setReplayFrames(frames);
    setReplayIdx(0);
    setReplayOpen(true);
  };

  const m = state?.metrics;
  const region = regions.find((r) => r.id === scenario.location.region_id);

  const specById = useMemo(() => Object.fromEntries(catalog.map((c) => [c.id, c])), [catalog]);

  return (
    <div className="flex h-full flex-col">
      <header className="flex items-center justify-between border-b border-ink-600 bg-ink-900 px-4 py-2">
        <div>
          <div className="text-[11px] uppercase tracking-[0.18em] text-mute">Research laboratory</div>
          <h1 className="text-base font-semibold">Multi-agent UAV wildfire detection — Victoria</h1>
        </div>
        <div className="text-right text-[11px] text-mute">
          Simplified wildfire propagation model · not an operational predictor
          <div>Simulation clock {state ? formatTime(state.time_s) : "—"} · seed {scenario.seed}</div>
        </div>
      </header>

      <div className="flex gap-1 overflow-x-auto border-b border-ink-600 bg-ink-800 px-3 py-1 text-[11px]">
        {DEMO_STEPS.map((label, i) => (
          <button
            key={label}
            className={`whitespace-nowrap rounded px-2 py-1 ${i <= demoStep ? "bg-ink-600 text-paper" : "text-mute"}`}
            onClick={() => setDemoStep(i)}
            type="button"
          >
            {i + 1}. {label}
          </button>
        ))}
      </div>

      <div className="grid min-h-0 flex-1 grid-cols-[minmax(200px,260px)_minmax(0,1fr)_minmax(200px,300px)]">
        <aside className="panel-scroll overflow-y-auto border-r border-ink-600 bg-ink-900 p-3 text-[12px]">
          <Section title="Scenario">
            <label className="block text-mute">Location</label>
            <select
              className="mb-2 w-full bg-ink-800 p-1"
              value={scenario.location.region_id}
              onChange={(e) => void changeRegion(e.target.value)}
            >
              {regions.filter((r) => r.id !== "victoria").map((r) => (
                <option key={r.id} value={r.id}>{r.name}</option>
              ))}
            </select>
            <p className="mb-2 text-[11px] text-mute">{region?.description}</p>
            <Row label="Wind speed (km/h)">
              <input type="number" className="w-20 bg-ink-800 px-1" value={scenario.environment.wind_speed_kmh}
                onChange={(e) => setScenario({ ...scenario, environment: { ...scenario.environment, wind_speed_kmh: Number(e.target.value) } })} />
            </Row>
            <Row label="Wind from">
              <select className="bg-ink-800 px-1" value={scenario.environment.wind_from}
                onChange={(e) => setScenario({ ...scenario, environment: { ...scenario.environment, wind_from: e.target.value } })}>
                {WIND_LABELS.map((w) => <option key={w}>{w}</option>)}
              </select>
            </Row>
            <Row label="Duration (min)">
              <input type="number" className="w-20 bg-ink-800 px-1" value={scenario.environment.duration_s / 60}
                onChange={(e) => setScenario({ ...scenario, environment: { ...scenario.environment, duration_s: Number(e.target.value) * 60 } })} />
            </Row>
            <Row label="UAV count">
              <input type="number" className="w-20 bg-ink-800 px-1" min={1} max={16} value={scenario.uavs.count}
                onChange={(e) => setScenario({ ...scenario, uavs: { ...scenario.uavs, count: Number(e.target.value) } })} />
            </Row>
            <Row label="Altitude (m)">
              <input type="number" className="w-20 bg-ink-800 px-1" value={scenario.uavs.altitude_m}
                onChange={(e) => setScenario({ ...scenario, uavs: { ...scenario.uavs, altitude_m: Number(e.target.value) } })} />
            </Row>
            <Row label="Detect range (m)">
              <input type="number" className="w-20 bg-ink-800 px-1" value={scenario.sensor.sensor_range_m}
                onChange={(e) => setScenario({ ...scenario, sensor: { ...scenario.sensor, sensor_range_m: Number(e.target.value) } })} />
            </Row>
            <Row label="Suppress range (m)">
              <input type="number" className="w-20 bg-ink-800 px-1" value={scenario.sensor.suppression_range_m}
                onChange={(e) => setScenario({ ...scenario, sensor: { ...scenario.sensor, suppression_range_m: Number(e.target.value) } })} />
            </Row>
            <Row label="P(detect)">
              <input type="number" step="0.01" className="w-20 bg-ink-800 px-1" value={scenario.sensor.detection_probability}
                onChange={(e) => setScenario({ ...scenario, sensor: { ...scenario.sensor, detection_probability: Number(e.target.value) } })} />
            </Row>
            <Row label="P(false +)">
              <input type="number" step="0.001" className="w-20 bg-ink-800 px-1" value={scenario.sensor.false_positive_probability}
                onChange={(e) => setScenario({ ...scenario, sensor: { ...scenario.sensor, false_positive_probability: Number(e.target.value) } })} />
            </Row>
            <Row label="Comm range (m)">
              <input type="number" className="w-20 bg-ink-800 px-1" value={scenario.communication.range_m}
                onChange={(e) => setScenario({ ...scenario, communication: { ...scenario.communication, range_m: Number(e.target.value) } })} />
            </Row>
            <Row label="Seed">
              <input type="number" className="w-20 bg-ink-800 px-1" value={scenario.seed}
                onChange={(e) => setScenario({ ...scenario, seed: Number(e.target.value) })} />
            </Row>
            <label className="mt-2 block text-mute">Policy</label>
            <select className="mb-1 w-full bg-ink-800 p-1" value={scenario.policy.name}
              onChange={(e) => setScenario({ ...scenario, policy: { ...scenario.policy, name: e.target.value } })}>
              {Object.keys(POLICY_HELP).map((p) => <option key={p}>{p}</option>)}
            </select>
            <p className="mb-2 text-[11px] text-mute">{POLICY_HELP[scenario.policy.name]}</p>
            <label className="mb-2 flex items-start gap-2 text-[11px] text-mute">
              <input type="checkbox" className="mt-0.5" checked={scenario.policy.allow_global_state}
                onChange={(e) => setScenario({ ...scenario, policy: { ...scenario.policy, allow_global_state: e.target.checked } })} />
              Know placed ignition locations (otherwise UAVs only go after a detection)
            </label>
            {["dqn", "mappo", "maddpg"].includes(scenario.policy.name) && (
              <input className="mb-2 w-full bg-ink-800 px-1" placeholder="Checkpoint path (optional)"
                value={scenario.policy.checkpoint_path ?? ""}
                onChange={(e) => setScenario({ ...scenario, policy: { ...scenario.policy, checkpoint_path: e.target.value || null } })} />
            )}
            <div className="mb-2 flex gap-1">
              {(["pan", "fire", "uav"] as PlaceMode[]).map((m) => (
                <button key={m} type="button" onClick={() => setPlaceMode(m)}
                  className={`flex-1 border px-1 py-1 ${placeMode === m ? "border-uav bg-ink-700" : "border-ink-600"}`}>
                  {m}
                </button>
              ))}
            </div>
            <div className="mb-2 text-[11px] text-mute">
              Fires: {scenario.fires.length} · UAV placements: {scenario.uavs.placements.length || "default grid"}
            </div>
            <button type="button" className="mb-1 w-full border border-ink-600 px-2 py-1"
              onClick={() => setScenario({ ...scenario, fires: randomKinglakeFires(4) })}>
              Random 4 fire points
            </button>
            <button type="button" className="mb-1 w-full bg-uav px-2 py-1 text-ink-950" disabled={busy} onClick={() => void createSim()}>
              {busy ? "Creating…" : "Create simulation from scenario"}
            </button>
            <button type="button" className="w-full border border-ink-600 px-2 py-1"
              onClick={() => void api.saveScenario(scenario).then((r) => alert(`Saved ${r.scenario_id}`))}>
              Save scenario JSON
            </button>
          </Section>
        </aside>

        <main className="relative min-h-0 min-w-0 overflow-hidden">
          {view3d ? (
            <Suspense fallback={<div className="flex h-full items-center justify-center text-mute">Loading 3D view…</div>}>
              <Map3D state={replayOpen && replayFrames[replayIdx] ? replayFrames[replayIdx] : state} />
            </Suspense>
          ) : (
            <MapView
              state={replayOpen && replayFrames[replayIdx] ? replayFrames[replayIdx] : state}
              layers={layers}
              placeMode={placeMode}
              showCoverage={showCoverage}
              showSensors={showSensors}
              showComms={showComms}
              showTrails={showTrails}
              onMapClick={onMapClick}
              onHover={(lat, lon) => setHover({ lat, lon })}
            />
          )}
          <div className="absolute left-3 top-3 space-y-2">
            <div className="bg-ink-900/90 p-2 text-[11px]">
              <div className="mb-1 font-medium">Layers</div>
              <label className="flex gap-2"><input type="checkbox" checked={showCoverage} onChange={(e) => setShowCoverage(e.target.checked)} /> Coverage heatmap (simulated)</label>
              <label className="flex gap-2"><input type="checkbox" checked={showSensors} onChange={(e) => setShowSensors(e.target.checked)} /> Detect (blue) / suppress (teal) range</label>
              <label className="flex gap-2"><input type="checkbox" checked={showComms} onChange={(e) => setShowComms(e.target.checked)} /> Communication links (simulated)</label>
              <label className="flex gap-2"><input type="checkbox" checked={showTrails} onChange={(e) => setShowTrails(e.target.checked)} /> Flight paths (simulated)</label>
              <div className="mt-2 border-t border-ink-600 pt-1 text-mute">
                Basemap: published Esri/Maxar imagery. Fire, UAV, coverage: simulation-generated.
              </div>
            </div>
            <LegendBox />
          </div>
          <div className="absolute right-3 top-3 bg-ink-900/90 p-2 text-[11px]">
            <div>N</div>
            <div className="mx-auto h-8 w-px bg-paper" />
            <div className="text-mute">north-up</div>
          </div>
          <div className="absolute bottom-3 left-3 right-3 flex flex-wrap items-center gap-2 bg-ink-900/92 p-2 text-[12px]">
            <button className="border border-ink-600 bg-ink-800 px-2 py-1 disabled:opacity-40" type="button" disabled={!simId} onClick={() => void cmd(() => api.start(simId!))}>Start</button>
            <button className="border border-ink-600 bg-ink-800 px-2 py-1 disabled:opacity-40" type="button" disabled={!simId} onClick={() => void cmd(() => api.pause(simId!))}>Pause</button>
            <button className="border border-ink-600 bg-ink-800 px-2 py-1 disabled:opacity-40" type="button" disabled={!simId} onClick={() => void cmd(() => api.resume(simId!))}>Resume</button>
            <button className="border border-ink-600 bg-ink-800 px-2 py-1 disabled:opacity-40" type="button" disabled={!simId} onClick={() => void cmd(() => api.stop(simId!))}>Stop</button>
            <button className="border border-ink-600 bg-ink-800 px-2 py-1 disabled:opacity-40" type="button" disabled={!simId} onClick={() => void cmd(async () => { const r = await api.reset(simId!); applyIncoming(r.state); setSeries([]); })}>Reset</button>
            <button className="border border-ink-600 bg-ink-800 px-2 py-1 disabled:opacity-40" type="button" disabled={!simId} onClick={() => void cmd(() => api.step(simId!, 1))}>Step</button>
            {[0.5, 1, 2, 5].map((sp) => (
              <button className="border border-ink-600 bg-ink-800 px-2 py-1 disabled:opacity-40" key={sp} type="button" disabled={!simId} onClick={() => void cmd(() => api.setSpeed(simId!, sp))}>×{sp}</button>
            ))}
            <button className="border border-ink-600 bg-ink-800 px-2 py-1 disabled:opacity-40" type="button" disabled={!simId} onClick={() => void loadReplay()}>Replay</button>
            <button className={`border border-ink-600 px-2 py-1 ${view3d ? "bg-ink-600" : "bg-ink-800"}`} type="button" onClick={() => setView3d((v) => !v)}>
              {view3d ? "2D map" : "3D (optional)"}
            </button>
            <span className="ml-auto font-mono text-[11px] text-mute">
              {hover ? `${hover.lat.toFixed(5)}, ${hover.lon.toFixed(5)}` : "lat, lon"}
              {state ? ` · ${state.status ?? ""} · ${state.policy.name}` : ""}
            </span>
          </div>
          {replayOpen && (
            <div className="absolute bottom-14 left-3 right-3 bg-ink-800 p-2 text-[12px]">
              Replay frame {replayIdx + 1}/{replayFrames.length}
              <input type="range" className="w-full" min={0} max={Math.max(0, replayFrames.length - 1)} value={replayIdx}
                onChange={(e) => setReplayIdx(Number(e.target.value))} />
              <button type="button" onClick={() => setReplayOpen(false)}>Close replay</button>
            </div>
          )}
        </main>

        <aside className="flex min-h-0 flex-col border-l border-ink-600 bg-ink-900">
          <nav className="flex flex-wrap gap-1 border-b border-ink-600 p-2 text-[11px]">
            {(["scenario", "metrics", "events", "compare", "research", "debug", "assumptions"] as Tab[]).map((t) => (
              <button key={t} type="button" className={`px-2 py-1 ${tab === t ? "bg-ink-600" : "text-mute"}`} onClick={() => setTab(t)}>
                {t}
              </button>
            ))}
          </nav>
          <div className="panel-scroll min-h-0 flex-1 overflow-y-auto p-3 text-[12px]">
            {error && <div className="mb-2 border border-fire p-2 text-fire">{error}</div>}
            {tab === "scenario" && (
              <div>
                <h3 className="mb-2 font-medium">Live status</h3>
                <KV label="Status" value={state?.status?.toUpperCase() ?? "NO SIM"} />
                <KV label="Time" value={state ? formatTime(state.time_s) : "N/A"} tip={specById.sim_time_s?.definition} />
                <KV label="Active fires" value={m ? `${m.active_fires ?? "N/A"} / ${m.total_fires ?? "N/A"}` : "N/A"} />
                <KV label="Detected fires" value={m ? `${m.detected_fires ?? "N/A"} / ${m.total_fires ?? "N/A"}` : "N/A"}
                  tip="Seen by a sensor. Still orange on the map until a UAV overlaps the fire point." />
                <KV label="Extinguished" value={m ? `${m.extinguished_fires ?? "N/A"} / ${m.total_fires ?? "N/A"}` : "N/A"}
                  tip="UAV overlapped the fire point. That source turns green and stops spreading." />
                <KV label="Coverage" value={formatMetric(m?.coverage_fraction ?? null, "percent")}
                  tip={specById.coverage_fraction?.definition} />
                <KV label="UAVs" value={m ? `${m.uavs_active ?? "N/A"} / ${m.uavs_total ?? "N/A"}` : "N/A"} />
                <KV label="Avg battery" value={formatMetric(m?.mean_battery_fraction ?? null, "percent")}
                  tip={specById.mean_battery_fraction?.definition} />
                <KV label="Collisions" value={formatMetric(m?.collisions ?? null, "count")}
                  tip={specById.collisions?.definition} />
                <KV label="Comm. links" value={formatMetric(m?.communication_link_fraction ?? null, "percent")}
                  tip={specById.communication_link_fraction?.definition} />
                <KV label="Detection time" value={formatMetric(m?.mean_detection_delay_s ?? null, "time")}
                  tip={specById.mean_detection_delay_s?.definition} />
                <p className="mt-3 text-[11px] text-mute">
                  Policy: {state?.policy.name ?? scenario.policy.name}
                  {state?.policy.trained === false && state?.policy.name === "marl_untrained" ? " — untrained (random). Not a MARL result." : ""}
                </p>
              </div>
            )}
            {tab === "metrics" && m && (
              <MetricsPanel metrics={m} catalog={catalog} series={series} simId={simId} />
            )}
            {tab === "events" && <EventPanel events={events} />}
            {tab === "compare" && (
              <ComparePanel experiment={experiment} busy={expBusy} onRun={() => void runCompare()} />
            )}
            {tab === "research" && <ResearchPanel scenario={scenario} state={state} />}
            {tab === "debug" && (
              <DebugPanel
                simId={simId}
                debugId={debugId}
                setDebugId={setDebugId}
                debug={debug}
                uavs={state?.uavs.map((u) => u.uav_id) ?? []}
                onLoad={() => simId && api.debug(simId, debugId).then(setDebug)}
              />
            )}
            {tab === "assumptions" && <AssumptionsPanel layers={layers} state={state} />}
          </div>
        </aside>
      </div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="mb-3">
      <h2 className="mb-2 text-[11px] uppercase tracking-wider text-mute">{title}</h2>
      {children}
    </section>
  );
}

function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="mb-1 flex items-center justify-between gap-2">
      <span className="text-mute">{label}</span>
      {children}
    </div>
  );
}

function KV({ label, value, tip }: { label: string; value: string; tip?: string }) {
  return (
    <div className="flex justify-between border-b border-ink-700 py-1">
      <Tip label={tip ?? "See METRICS.md / metric catalog for the formal definition."}>
        <span className="text-mute">{label}</span>
      </Tip>
      <span className="font-mono">{value}</span>
    </div>
  );
}

function LegendBox() {
  return (
    <div className="bg-ink-900/90 p-2 text-[11px]">
      <div className="mb-1 font-medium">Legend</div>
      <div><span className="mr-1 inline-block h-2 w-2 bg-fire" /> Burning (orange spread)</div>
      <div><span className="mr-1 inline-block h-2 w-2 bg-ok" /> Extinguished (UAV overlapped point)</div>
      <div><span className="mr-1 inline-block h-2 w-2 bg-uav" /> UAV</div>
      <div><span className="mr-1 inline-block h-2 w-2 bg-warn" /> Communication link</div>
      <div><span className="mr-1 inline-block h-2 w-2 bg-mute" /> AOI envelope (real coords)</div>
    </div>
  );
}

function MetricsPanel({
  metrics,
  catalog,
  series,
  simId,
}: {
  metrics: Record<string, number | null>;
  catalog: MetricSpec[];
  series: { t: number; coverage: number; reward: number; detected: number }[];
  simId: string | null;
}) {
  return (
    <div>
      <h3 className="mb-2 font-medium">Deployment metrics</h3>
      <p className="mb-2 text-[11px] text-mute">
        RL reward is an optimisation signal, not classification accuracy.
      </p>
      {catalog.map((c) => (
        <div key={c.id} className="mb-1">
          <KV
            label={`${c.name}${c.unit && c.unit !== "1" && c.unit !== "fraction" ? ` (${c.unit})` : ""}`}
            value={
              c.unit === "s" ? formatMetric(metrics[c.id] ?? null, "time")
                : c.unit === "fraction" ? formatMetric(metrics[c.id] ?? null, "percent")
                  : formatMetric(metrics[c.id] ?? null, c.unit === "count" ? "count" : "number")
            }
            tip={`${c.definition} Source: ${c.source}.`}
          />
        </div>
      ))}
      <div className="mt-3 h-40">
        <ResponsiveContainer>
          <LineChart data={series}>
            <CartesianGrid stroke="#2a3544" />
            <XAxis dataKey="t" stroke="#8b98a5" tick={{ fontSize: 10 }} />
            <YAxis stroke="#8b98a5" tick={{ fontSize: 10 }} />
            <RTooltip contentStyle={{ background: "#10161e", border: "1px solid #2a3544" }} />
            <Legend />
            <Line type="monotone" dataKey="coverage" stroke="#3d8bfd" dot={false} name="Coverage" />
            <Line type="monotone" dataKey="detected" stroke="#2f9e6c" dot={false} name="Detected fires" />
          </LineChart>
        </ResponsiveContainer>
      </div>
      {simId && (
        <div className="mt-2 flex gap-2">
          <a className="underline" href={exportUrl(simId, "json")}>Export JSON</a>
          <a className="underline" href={exportUrl(simId, "csv")}>Export CSV</a>
        </div>
      )}
    </div>
  );
}

function EventPanel({ events }: { events: SimEvent[] }) {
  return (
    <ol className="space-y-1 font-mono text-[11px]">
      {events.length === 0 && <li className="text-mute">No events yet.</li>}
      {events.map((e, i) => (
        <li key={`${e.time_s}-${i}`}>
          <span className="text-mute">{formatTime(e.time_s)}</span> {e.message}
        </li>
      ))}
    </ol>
  );
}

function ComparePanel({
  experiment,
  busy,
  onRun,
}: {
  experiment: ExperimentResult | null;
  busy: boolean;
  onRun: () => void;
}) {
  const keys = [
    ["detection_rate", "Detection rate"],
    ["mean_detection_delay_s", "Detection time"],
    ["coverage_fraction", "Coverage"],
    ["total_distance_m", "Distance"],
    ["energy_used_fraction", "Energy"],
    ["collisions", "Collisions"],
    ["messages_sent", "Communication"],
  ] as const;
  return (
    <div>
      <p className="mb-2 text-[11px] text-mute">
        Same scenario, seeds 1–3, 180 s headless. Interactive default is short for this laptop.
        Increase seeds/duration for publication-quality estimates. No significance test is run.
      </p>
      <button type="button" className="mb-3 w-full bg-ink-600 py-1" disabled={busy} onClick={onRun}>
        {busy ? "Running trials…" : "Run MARL vs baselines"}
      </button>
      {experiment && (
        <>
          <table className="w-full text-left text-[11px]">
            <thead>
              <tr>
                <th>Metric</th>
                {experiment.policies.map((p) => <th key={p} className="text-right">{p}</th>)}
              </tr>
            </thead>
            <tbody>
              {keys.map(([k, label]) => (
                <tr key={k} className="border-t border-ink-700">
                  <td className="py-1">{label}</td>
                  {experiment.policies.map((p) => {
                    const s = experiment.summary[p][k];
                    const txt = s.mean == null ? "N/A" : s.n > 1 ? `${s.mean.toPrecision(3)} ± ${s.std?.toPrecision(2)}` : String(s.mean);
                    return <td key={p} className="text-right font-mono">{txt}</td>;
                  })}
                </tr>
              ))}
            </tbody>
          </table>
          <pre className="mt-3 whitespace-pre-wrap text-[10px] text-mute">{experiment.report_text}</pre>
          <a className="underline" href={experimentExportUrl(experiment.experiment_id)}>Export trials CSV</a>
        </>
      )}
    </div>
  );
}

function ResearchPanel({ scenario, state }: { scenario: ScenarioConfig; state: SimState | null }) {
  const w = scenario.reward;
  return (
    <div className="space-y-3 text-[12px]">
      <h3 className="font-medium">Environment</h3>
      <p>Timestep {scenario.environment.timestep_s} s. Action space: discrete-9 (N…NW, HOVER). Observation dim {state?.observation_spec.dim ?? 57}.</p>
      <p className="text-mute">{state?.observation_spec.communication_assumption}</p>
      <p className="text-mute">{state?.observation_spec.normalization}</p>
      <h3 className="font-medium">Reward (implementation = this equation)</h3>
      <p className="font-mono text-[11px]">
        R = {w.alpha_detection}·N_detected + {w.zeta_extinguish_cell}·N_cells_out + {w.eta_suppress_fire}·N_fires_out + {w.beta_coverage}·Δcoverage − {w.gamma_energy}·Δenergy − {w.delta_collision}·N_collisions − {w.epsilon_overlap}·Δoverlap
      </p>
      <p className="text-mute">Detection does not extinguish. A UAV must overfly a BURNING cell within suppression range ({scenario.sensor.suppression_range_m} m). Reward is not classification accuracy.</p>
      <h3 className="font-medium">MARL</h3>
      <p>Interface: policy.act(observation, agent_id, state). Algorithm is not hardcoded. Checkpoint: {scenario.policy.checkpoint_path || "none"}.</p>
      <p>Training uses WildfireEnv without visualization. Execution uses the same step().</p>
      <p>allow_global_state = {String(scenario.policy.allow_global_state)} (training privilege if true).</p>
      <h3 className="font-medium">Sensor</h3>
      <p>Detection (blue): reports a fire inside {scenario.sensor.sensor_range_m} m — does not put it out. Extinguish: UAV must overlap the fire point within {scenario.sensor.suppression_range_m} m. That source turns green and all of its orange spread stops.</p>
      <h3 className="font-medium">Fire</h3>
      <p>{state?.fire_grid.model_label}. Fuel synthetic: {String(state?.fire_grid.fuel_is_synthetic ?? true)}. Slope real: {String(state?.fire_grid.slope_is_real ?? false)}.</p>
    </div>
  );
}

function DebugPanel({
  simId,
  debugId,
  setDebugId,
  debug,
  uavs,
  onLoad,
}: {
  simId: string | null;
  debugId: string;
  setDebugId: (s: string) => void;
  debug: Record<string, unknown> | null;
  uavs: string[];
  onLoad: () => void;
}) {
  return (
    <div>
      <select className="mb-2 w-full bg-ink-800 p-1" value={debugId} onChange={(e) => setDebugId(e.target.value)}>
        {(uavs.length ? uavs : ["uav_00"]).map((id) => <option key={id}>{id}</option>)}
      </select>
      <button type="button" className="mb-2 w-full bg-ink-600 py-1" disabled={!simId} onClick={onLoad}>Inspect agent</button>
      <pre className="whitespace-pre-wrap break-all text-[10px] text-mute">{debug ? JSON.stringify(debug, null, 2) : "Select a UAV."}</pre>
    </div>
  );
}

function AssumptionsPanel({ layers, state }: { layers: GeoLayer[]; state: SimState | null }) {
  return (
    <div className="space-y-2 text-[12px]">
      <p>This application is a research simulation platform, not an operational wildfire prediction or UAV command system.</p>
      <ul className="list-disc space-y-1 pl-4">
        <li>Fire spread is a simplified elliptical/cellular model, not Phoenix/Spark/FARSITE.</li>
        <li>UAV motion is kinematic (bounded speed, turn, climb). No wind drift on airframes.</li>
        <li>Sensor detections are probabilistic; LOS/terrain occlusion are not applied (no DEM). A detection report does not extinguish the fire.</li>
        <li>Suppression requires a UAV inside suppression range of a BURNING cell. Intensity drops until the cell is EXTINGUISHED.</li>
        <li>Communication is simulated range + optional drop probability, one-hop unless enabled.</li>
        <li>Fuel is synthetic-uniform unless a vegetation dataset is loaded.</li>
        <li>AOI boxes are real WGS84 envelopes, not legal park boundaries.</li>
      </ul>
      <h3 className="pt-2 font-medium">Geographic layers</h3>
      {layers.map((l) => (
        <div key={l.id} className="border-b border-ink-700 py-1">
          <div>{l.name} · {l.available ? "available" : "placeholder"} · {l.is_real_geographic_data ? "real data" : "not real geography"}</div>
          <div className="text-[11px] text-mute">{l.notes}</div>
        </div>
      ))}
      {state && (
        <p className="text-mute">Sensor notes: {String(state.sensor_assumptions.notes)}</p>
      )}
    </div>
  );
}
