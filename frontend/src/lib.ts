import type { MetricValue, ScenarioConfig } from "./types";

export function formatTime(s: number | null | undefined): string {
  if (s == null || Number.isNaN(s)) return "N/A";
  const t = Math.max(0, Math.floor(s));
  const hh = String(Math.floor(t / 3600)).padStart(2, "0");
  const mm = String(Math.floor((t % 3600) / 60)).padStart(2, "0");
  const ss = String(t % 60).padStart(2, "0");
  return `${hh}:${mm}:${ss}`;
}

export function formatMetric(value: MetricValue, kind: "fraction" | "percent" | "number" | "time" | "count" = "number"): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "N/A";
  if (kind === "time") return formatTime(value);
  if (kind === "percent") return `${(value * 100).toFixed(1)}%`;
  if (kind === "fraction") return value.toFixed(3);
  if (kind === "count") return String(Math.round(value));
  if (Math.abs(value) >= 1000) return value.toFixed(0);
  return Number.isInteger(value) ? String(value) : value.toFixed(2);
}

export function randomKinglakeFires(count = 4): ScenarioConfig["fires"] {
  const fires = [];
  for (let i = 0; i < count; i += 1) {
    fires.push({
      fire_id: `fire_${String(i + 1).padStart(2, "0")}`,
      latitude: -37.538 + Math.random() * 0.036,
      longitude: 145.275 + Math.random() * 0.07,
      initial_intensity: 0.7 + Math.random() * 0.25,
      ignition_time: i < 2 ? 0 : Math.floor(Math.random() * 15),
      radius_m: 70,
    });
  }
  return fires;
}

export function defaultScenario(): ScenarioConfig {
  return {
    name: "Kinglake research scenario A",
    location: { region_id: "kinglake" },
    fires: [
      {
        fire_id: "fire_01",
        latitude: -37.505,
        longitude: 145.28,
        initial_intensity: 0.9,
        ignition_time: 0,
        radius_m: 80,
      },
      {
        fire_id: "fire_02",
        latitude: -37.52,
        longitude: 145.34,
        initial_intensity: 0.75,
        ignition_time: 20,
        radius_m: 70,
      },
    ],
    environment: {
      wind_speed_kmh: 25,
      wind_from: "NW",
      duration_s: 1200,
      timestep_s: 1,
      fire_cell_size_m: 60,
      coverage_cell_size_m: 100,
      stochastic_fire: true,
      base_ros_mps: 0.35,
    },
    uavs: {
      count: 6,
      altitude_m: 120,
      max_speed_mps: 18,
      cruise_speed_mps: 12,
      battery_capacity_j: 180000,
      return_to_base_fraction: 0.3,
      min_separation_m: 40,
      placements: [],
    },
    sensor: {
      sensor_range_m: 800,
      field_of_view_deg: 360,
      detection_probability: 0.9,
      false_positive_probability: 0.002,
      false_negative_probability: 0.1,
      observation_noise_m: 15,
      suppression_range_m: 120,
    },
    communication: {
      range_m: 4500,
      failure_probability: 0,
      multi_hop: false,
    },
    reward: {
      alpha_detection: 8,
      beta_coverage: 0.4,
      gamma_energy: 0.04,
      delta_collision: 8,
      epsilon_overlap: 0.3,
      zeta_extinguish_cell: 0.35,
      eta_suppress_fire: 16,
    },
    policy: {
      name: "mappo",
      checkpoint_path: null,
      allow_global_state: true,
      algorithm: null,
    },
    seed: 42,
    action_space: "discrete9",
  };
}

export const WIND_LABELS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"] as const;

export const POLICY_HELP: Record<string, string> = {
  dqn: "Independent DQN actor (data/checkpoints/dqn.pt). Warm-started from fire-seeking greedy. Untrained random if missing.",
  mappo: "MAPPO trained to fly to known fires and overfly them to extinguish (data/checkpoints/mappo.pt). Detection alone does not put a fire out.",
  maddpg: "MADDPG (Gumbel-Softmax) trained to fly to known fires and overfly them to extinguish (data/checkpoints/maddpg.pt).",
};
