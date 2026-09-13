export type PlaceMode = "pan" | "fire" | "uav";

export interface Bounds {
  south: number;
  west: number;
  north: number;
  east: number;
}

export interface Region {
  id: string;
  name: string;
  state: string;
  country: string;
  bounds: Bounds;
  description: string;
  data_status: string;
}

export interface GeoLayer {
  id: string;
  name: string;
  kind: string;
  source: string;
  is_real_geographic_data: boolean;
  available: boolean;
  notes: string;
  geometry: GeoJSON.Geometry | null;
}

export interface FireSourceConfig {
  fire_id?: string | null;
  latitude: number;
  longitude: number;
  initial_intensity: number;
  ignition_time: number;
  radius_m: number;
}

export interface UAVPlacement {
  uav_id?: string | null;
  latitude: number;
  longitude: number;
  heading_deg: number;
  altitude_m?: number | null;
}

export interface ScenarioConfig {
  name: string;
  location: {
    region_id: string;
    south?: number | null;
    west?: number | null;
    north?: number | null;
    east?: number | null;
  };
  fires: FireSourceConfig[];
  environment: {
    wind_speed_kmh: number;
    wind_from: string;
    wind_from_deg?: number | null;
    duration_s: number;
    timestep_s: number;
    fire_cell_size_m: number;
    coverage_cell_size_m: number;
    stochastic_fire: boolean;
    base_ros_mps: number;
  };
  uavs: {
    count: number;
    altitude_m: number;
    max_speed_mps: number;
    cruise_speed_mps: number;
    battery_capacity_j: number;
    return_to_base_fraction: number;
    min_separation_m: number;
    placements: UAVPlacement[];
  };
  sensor: {
    sensor_range_m: number;
    field_of_view_deg: number;
    detection_probability: number;
    false_positive_probability: number;
    false_negative_probability: number;
    observation_noise_m: number;
    suppression_range_m: number;
  };
  communication: {
    range_m: number;
    failure_probability: number;
    multi_hop: boolean;
  };
  reward: {
    alpha_detection: number;
    beta_coverage: number;
    gamma_energy: number;
    delta_collision: number;
    epsilon_overlap: number;
    zeta_extinguish_cell: number;
    eta_suppress_fire: number;
  };
  policy: {
    name: string;
    checkpoint_path?: string | null;
    allow_global_state: boolean;
    algorithm?: string | null;
  };
  seed: number;
  action_space: "discrete9";
}

export interface SimEvent {
  time_s: number;
  kind: string;
  message: string;
  agent_id?: string | null;
  fire_id?: string | null;
}

export interface UAVState {
  uav_id: string;
  latitude: number;
  longitude: number;
  altitude_m: number;
  heading_deg: number;
  speed_mps: number;
  battery_fraction: number;
  status: string;
  sensor_range_m: number;
  suppression_range_m?: number;
  comm_range_m: number;
  known_fires: string[];
  last_action: number;
  last_reward: number;
}

export interface FireState {
  fire_id: string;
  latitude: number;
  longitude: number;
  intensity: number;
  ignition_time: number;
  radius_m: number;
  active: boolean;
  detected: boolean;
  first_detection_time: number | null;
  detected_by: string | null;
  suppressed: boolean;
  extinguished_time: number | null;
  suppressed_by: string | null;
  perimeter: { latitude: number; longitude: number }[];
}

export type MetricValue = number | null;

export type Metrics = Record<string, MetricValue>;

export interface SimState {
  time_s: number;
  step: number;
  terminated: boolean;
  status?: string;
  speed?: number;
  simulation_id?: string;
  region_id: string;
  bounds: Bounds;
  wind: { speed_kmh: number; from_deg: number; from: string };
  uavs: UAVState[];
  fires: FireState[];
  communication_links: { a: string; b: string; distance_m: number }[];
  coverage_heatmap: number[][];
  coverage_fraction: number;
  fire_grid: {
    burned_area_m2: number;
    burning_cells: number;
    burned_cells: number;
    extinguished_cells?: number;
    extinguished_area_m2?: number;
    suppression_note?: string;
    fuel_is_synthetic: boolean;
    slope_is_real: boolean;
    model_label: string;
  };
  metrics: Metrics;
  per_uav: Record<string, MetricValue | string>[];
  policy: {
    name: string | null;
    kind: string | null;
    trained: boolean;
    algorithm: string | null;
    checkpoint_path: string | null;
    notes: string | null;
  };
  reward_equation: string;
  sensor_assumptions: Record<string, boolean | string>;
  observation_spec: {
    dim: number;
    layout: string[];
    partial: boolean;
    allow_global_state: boolean;
    communication_assumption: string;
    normalization: string;
  };
  seed: number;
  trails?: Record<string, { latitude: number; longitude: number }[]>;
  events?: SimEvent[];
}

export interface MetricSpec {
  id: string;
  name: string;
  definition: string;
  unit: string;
  source: string;
  group: string;
}

export interface ExperimentResult {
  experiment_id: string;
  name: string;
  timestamp: string;
  policies: string[];
  seeds: number[];
  summary: Record<string, Record<string, StatSummary>>;
  report_text: string;
  significance_tested: boolean;
  notes: string;
  trials: Record<string, Metrics[]>;
}

export interface StatSummary {
  n: number;
  mean: number | null;
  std: number | null;
  median: number | null;
  min: number | null;
  max: number | null;
  ci95_low: number | null;
  ci95_high: number | null;
}
