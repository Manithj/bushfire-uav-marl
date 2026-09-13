"""Multi-agent wildfire search environment.

Gymnasium / PettingZoo-compatible. Visualization is not required.
The same transition logic powers training and the interactive simulator.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

from app.baselines.registry import make_policy
from app.geography.coordinate_transform import compass_deg_to_rad
from app.marl.action import N_DISCRETE_ACTIONS, decode_discrete, discrete_space, validate_action
from app.marl.observation import (
    OBS_DIM,
    build_observation,
    local_coverage_patch,
    observation_space,
    observation_spec,
)
from app.marl.policy import Policy
from app.schemas.scenario import ScenarioConfig, resolve_wind_from_deg
from app.simulation.collision import CollisionMonitor
from app.simulation.communication import CommunicationModel
from app.simulation.coverage import CoverageModel
from app.simulation.events import EventLog
from app.simulation.fire_model import BURNING, FireModel, FireModelConfig
from app.simulation.metrics import MetricsAccumulator, compute_public_metrics, per_uav_metrics
from app.simulation.reward import RewardWeights, compute_reward, reward_equation_latex
from app.simulation.sensor_model import DetectionEvent, sense_fires, sensor_assumptions
from app.simulation.uav import UAVAgent, UAVConfig, UAVStatus
from app.simulation.uav_dynamics import UAVBodyState, UAVDynamicsConfig, apply_action
from app.simulation.world import World, world_from_location


class WildfireEnv(ParallelEnv):
    metadata = {"name": "wildfire_uav_v0", "is_parallel": True, "render_modes": []}

    def __init__(self, scenario: ScenarioConfig) -> None:
        super().__init__()
        self.scenario = scenario
        self.world: World = world_from_location(scenario.location)
        self.possible_agents = [f"uav_{i:02d}" for i in range(scenario.uavs.count)]
        self.agents: list[str] = []
        self._obs_space = observation_space()
        self._act_space = discrete_space()
        self.observation_spaces = {a: self._obs_space for a in self.possible_agents}
        self.action_spaces = {a: self._act_space for a in self.possible_agents}

        self.rng = np.random.default_rng(scenario.seed)
        self.fire: FireModel | None = None
        self.coverage: CoverageModel | None = None
        self.uavs: list[UAVAgent] = []
        self.comms = CommunicationModel(
            failure_probability=scenario.communication.failure_probability,
            multi_hop=scenario.communication.multi_hop,
        )
        self.collisions = CollisionMonitor()
        self.events = EventLog()
        self.metrics = MetricsAccumulator()
        self.reward_weights = RewardWeights(
            alpha_detection=scenario.reward.alpha_detection,
            beta_coverage=scenario.reward.beta_coverage,
            gamma_energy=scenario.reward.gamma_energy,
            delta_collision=scenario.reward.delta_collision,
            epsilon_overlap=scenario.reward.epsilon_overlap,
            zeta_extinguish_cell=scenario.reward.zeta_extinguish_cell,
            eta_suppress_fire=scenario.reward.eta_suppress_fire,
        )
        self.time_s = 0.0
        self.step_count = 0
        self.dt = scenario.environment.timestep_s
        self.duration_s = scenario.environment.duration_s
        self.wind_speed_mps = scenario.environment.wind_speed_kmh / 3.6
        self.wind_from_deg = resolve_wind_from_deg(scenario.environment)
        self.policy: Policy | None = None
        self.pending_ignitions: list = []
        self.detections: list[DetectionEvent] = []
        self._terminated = False
        self._base_xy: tuple[float, float] = (0.0, 0.0)

    # --- PettingZoo spaces (shared) ---
    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        del options
        if seed is not None:
            self.scenario.seed = int(seed)
        self.rng = np.random.default_rng(self.scenario.seed)
        self.world = world_from_location(self.scenario.location)
        self.time_s = 0.0
        self.step_count = 0
        self._terminated = False
        self.detections = []
        self.events = EventLog()
        self.metrics = MetricsAccumulator()
        self.collisions = CollisionMonitor()
        self.comms = CommunicationModel(
            failure_probability=self.scenario.communication.failure_probability,
            multi_hop=self.scenario.communication.multi_hop,
        )
        self.wind_speed_mps = self.scenario.environment.wind_speed_kmh / 3.6
        self.wind_from_deg = resolve_wind_from_deg(self.scenario.environment)

        fire_rng = np.random.default_rng(self.scenario.seed)
        cfg = FireModelConfig(
            cell_size_m=self.scenario.environment.fire_cell_size_m,
            base_ros_mps=self.scenario.environment.base_ros_mps,
            stochastic=self.scenario.environment.stochastic_fire,
        )
        self.fire = FireModel.create(
            origin_x=self.world.origin_x,
            origin_y=self.world.origin_y,
            extent_x=self.world.extent_x,
            extent_y=self.world.extent_y,
            config=cfg,
            rng=fire_rng,
            fuel_is_synthetic=True,
            slope_is_real=False,
        )
        n = self.scenario.uavs.count
        self.coverage = CoverageModel.create(
            origin_x=self.world.origin_x,
            origin_y=self.world.origin_y,
            extent_x=self.world.extent_x,
            extent_y=self.world.extent_y,
            cell_size_m=self.scenario.environment.coverage_cell_size_m,
            n_agents=n,
        )
        self.uavs = self._spawn_uavs()
        self.agents = [u.uav_id for u in self.uavs]
        self.possible_agents = list(self.agents)
        self.observation_spaces = {a: self._obs_space for a in self.agents}
        self.action_spaces = {a: self._act_space for a in self.agents}
        self._base_xy = (self.uavs[0].body.x, self.uavs[0].body.y) if self.uavs else (0.0, 0.0)

        self.pending_ignitions = list(self.scenario.fires)
        if not self.pending_ignitions:
            self.pending_ignitions = self._default_fires()
        self._apply_due_ignitions()

        self.events.add(0.0, "sim", "Simulation reset")
        for u in self.uavs:
            self.events.add(0.0, "uav", f"{u.uav_id} launched", agent_id=u.uav_id)

        self.policy = make_policy(
            self.scenario.policy.name,
            seed=self.scenario.seed,
            checkpoint_path=self.scenario.policy.checkpoint_path,
        )
        self.policy.reset(self.scenario.seed)

        self.coverage.update(self.uavs, 0.0)
        self.comms.update_links(self.uavs)
        return self._observe_all(), self._infos()

    def step(self, actions: dict[str, int]) -> tuple[dict, dict, dict, dict, dict]:
        if self.fire is None or self.coverage is None:
            raise RuntimeError("Call reset() before step().")
        if self._terminated:
            obs = self._observe_all()
            z = {a: 0.0 for a in self.agents}
            term = {a: True for a in self.agents}
            trunc = {a: False for a in self.agents}
            return obs, z, term, trunc, self._infos()

        dt = self.dt
        prev_energy = float(np.mean([u.battery_fraction() for u in self.uavs])) if self.uavs else 0.0
        prev_overlap = self.coverage.overlap_fraction() or 0.0
        prev_cov = self.coverage.coverage_fraction()

        for u in self.uavs:
            raw = actions.get(u.uav_id, 8)
            action = validate_action(int(raw))
            u.last_action = action
            u.actions_taken += 1
            if u.status == UAVStatus.COLLIDED:
                continue
            if u.body.battery_j <= 0:
                u.status = UAVStatus.BATTERY_DEPLETED
                continue
            if u.should_return_to_base() and u.status != UAVStatus.RETURNING:
                u.status = UAVStatus.RETURNING
                self.events.add(
                    self.time_s,
                    "uav",
                    f"{u.uav_id} battery reached {u.battery_fraction()*100:.0f}% — returning to base",
                    agent_id=u.uav_id,
                )
            if u.status == UAVStatus.RETURNING:
                cmd_heading = _heading_to(u.body.x, u.body.y, self._base_xy[0], self._base_xy[1])
                u.body = apply_action(
                    u.body,
                    cmd_heading,
                    u.config.dynamics.cruise_speed_mps,
                    u.config.dynamics.min_altitude_m,
                    u.config.dynamics,
                    dt,
                    self.world.x_min,
                    self.world.x_max,
                    self.world.y_min,
                    self.world.y_max,
                )
                if np.hypot(u.body.x - self._base_xy[0], u.body.y - self._base_xy[1]) < 40.0:
                    u.status = UAVStatus.LANDED
                    u.body.speed = 0.0
                    self.events.add(self.time_s + dt, "uav", f"{u.uav_id} landed", agent_id=u.uav_id)
            else:
                cmd = decode_discrete(action, u.config.start_altitude_m, u.config.dynamics, u.body.heading)
                u.body = apply_action(
                    u.body,
                    cmd.heading,
                    cmd.speed,
                    cmd.altitude,
                    u.config.dynamics,
                    dt,
                    self.world.x_min,
                    self.world.x_max,
                    self.world.y_min,
                    self.world.y_max,
                )
                if u.status not in (UAVStatus.INVESTIGATING, UAVStatus.SUPPRESSING):
                    u.status = UAVStatus.SEARCHING if cmd.speed > 0.4 else UAVStatus.IDLE
            lat, lon = self.world.transform.to_wgs84(u.body.x, u.body.y)
            u.lat, u.lon = lat, lon

        sep_events = self.collisions.check(self.uavs, self.time_s + dt)
        for ev in sep_events:
            kind = "collision" if ev.is_collision else "near_collision"
            self.events.add(
                ev.time_s,
                kind,
                f"{'Collision' if ev.is_collision else 'Near-collision'} {ev.a}–{ev.b} at {ev.distance_m:.1f} m",
                agent_id=ev.a,
                data={"other": ev.b, "distance_m": ev.distance_m},
            )

        cov_gain_m2 = self.coverage.update(self.uavs, self.time_s + dt)
        eligible = max(self.coverage.area_m2(self.coverage.eligible_cells()), 1.0)
        cov_gain_frac = cov_gain_m2 / eligible

        self.comms.update_links(self.uavs)
        new_true = 0
        to_wgs = self.world.transform.to_wgs84
        for u in self.uavs:
            if u.body.battery_j <= 0 or u.status in (UAVStatus.LANDED, UAVStatus.COLLIDED):
                continue
            found = sense_fires(u, self.fire, self.rng, self.time_s + dt, to_wgs)
            for det in found:
                self.detections.append(det)
                self.metrics.record_detection(det.is_true, det.time_s)
                if det.is_true:
                    u.detections_true += 1
                    u.known_fire_ids.add(det.fire_id)  # type: ignore[arg-type]
                    u.status = UAVStatus.INVESTIGATING
                    first = self.fire.mark_detected(det.fire_id, det.time_s, u.uav_id)  # type: ignore[arg-type]
                    if first:
                        new_true += 1
                        self.events.add(
                            det.time_s,
                            "detection",
                            f"{u.uav_id} detected {det.fire_id}",
                            agent_id=u.uav_id,
                            fire_id=det.fire_id,
                        )
                else:
                    u.detections_false += 1
                    self.events.add(
                        det.time_s,
                        "false_positive",
                        f"{u.uav_id} false-positive detection",
                        agent_id=u.uav_id,
                    )
                msgs = self.comms.broadcast_detection(det, self.uavs, self.rng)
                for m in msgs:
                    if m.delivered and det.is_true:
                        self.events.add(
                            det.time_s,
                            "comm",
                            f"Detection of {det.fire_id} broadcast {m.sender} → {m.receiver}",
                            agent_id=m.sender,
                            fire_id=det.fire_id,
                        )

        self.time_s += dt
        self.step_count += 1
        self._apply_due_ignitions()
        cells_out, fires_out = self._apply_suppression(dt)
        self.fire.step(dt, self.wind_speed_mps, self.wind_from_deg, self.time_s)

        energy_now = float(np.mean([u.battery_fraction() for u in self.uavs])) if self.uavs else 0.0
        energy_used = max(0.0, prev_energy - energy_now)
        overlap_now = self.coverage.overlap_fraction() or 0.0
        overlap_inc = max(0.0, overlap_now - prev_overlap)
        n_col = sum(1 for e in sep_events if e.is_collision)
        team_r = compute_reward(
            newly_detected=new_true,
            coverage_gain_fraction=cov_gain_frac,
            energy_used_fraction=energy_used,
            collisions=n_col,
            overlap_increment=overlap_inc,
            weights=self.reward_weights,
            cells_extinguished=cells_out,
            fires_suppressed=fires_out,
        )
        self.metrics.add_reward(team_r)
        self.metrics.reward_components["detection"] += self.reward_weights.alpha_detection * new_true
        self.metrics.reward_components["coverage"] += self.reward_weights.beta_coverage * cov_gain_frac
        self.metrics.reward_components["energy"] += self.reward_weights.gamma_energy * energy_used
        self.metrics.reward_components["collision"] += self.reward_weights.delta_collision * n_col
        self.metrics.reward_components["overlap"] += self.reward_weights.epsilon_overlap * overlap_inc
        self.metrics.reward_components["extinguish"] += self.reward_weights.zeta_extinguish_cell * cells_out
        self.metrics.reward_components["suppress"] += self.reward_weights.eta_suppress_fire * fires_out
        for u in self.uavs:
            u.last_reward = team_r

        del prev_cov
        terminated = self.time_s >= self.duration_s
        if all(u.body.battery_j <= 0 or u.status in (UAVStatus.LANDED, UAVStatus.COLLIDED, UAVStatus.BATTERY_DEPLETED) for u in self.uavs):
            terminated = True
        if terminated and not self._terminated:
            self._terminated = True
            self.events.add(self.time_s, "sim", "Simulation ended")

        agents = [u.uav_id for u in self.uavs]
        self.agents = [] if terminated else list(agents)
        rewards = {a: team_r for a in agents}
        term = {a: terminated for a in agents}
        trunc = {a: False for a in agents}
        if terminated:
            rewards["__all__"] = team_r
            term["__all__"] = True
            trunc["__all__"] = False
        return self._observe_all(), rewards, term, trunc, self._infos()

    def step_policy(self) -> tuple[dict, dict, dict, dict, dict]:
        obs = self._observe_all()
        assert self.policy is not None
        states = {
            u.uav_id: {"agent_index": i, "n_agents": len(self.uavs)}
            for i, u in enumerate(self.uavs)
        }
        actions = self.policy.act_many(obs, states)
        return self.step(actions)

    def _observe_all(self) -> dict[str, np.ndarray]:
        assert self.fire is not None and self.coverage is not None
        downwind = compass_deg_to_rad((self.wind_from_deg + 180.0) % 360.0)
        t_norm = self.time_s / max(self.duration_s, 1.0)
        allow_global = self.scenario.policy.allow_global_state
        out: dict[str, np.ndarray] = {}
        for u in self.uavs:
            known: list[tuple[float, float, float]] = []
            if allow_global:
                ids = [src.fire_id for src in self.fire.sources if src.active]
            else:
                ids = [fid for fid in u.known_fire_ids if (s := self.fire.source_by_id(fid)) is not None and s.active]
            for fid in ids:
                src = self.fire.source_by_id(fid)
                if src is None or not src.active:
                    continue
                # Aim at the fire point the map shows. Overlap that point to extinguish.
                known.append((src.x, src.y, src.initial_intensity))
            patch = local_coverage_patch(self.coverage, u)
            out[u.uav_id] = build_observation(
                agent=u,
                all_uavs=self.uavs,
                known_fires=known,
                coverage_patch=patch,
                wind_speed_mps=self.wind_speed_mps,
                downwind_rad=downwind,
                time_norm=t_norm,
                extent_x=self.world.extent_x,
                extent_y=self.world.extent_y,
                max_altitude=u.config.dynamics.max_altitude_m,
                max_speed=u.config.dynamics.max_speed_mps,
                origin_x=self.world.origin_x,
                origin_y=self.world.origin_y,
            )
        return out

    def _infos(self) -> dict[str, dict[str, Any]]:
        return {u.uav_id: {"status": u.status.value, "battery": u.battery_fraction()} for u in self.uavs}

    def _apply_suppression(self, dt: float) -> tuple[int, int]:
        """Extinguish a fire only if the UAV overlaps it.

        A green map pin is extinguished, not merely detected. Overlap
        puts the whole fire out so orange spread from that source stops.
        """
        assert self.fire is not None
        cells_out = 0
        fires_out = 0
        for u in self.uavs:
            if u.body.battery_j <= 0 or u.status in (
                UAVStatus.LANDED,
                UAVStatus.COLLIDED,
                UAVStatus.BATTERY_DEPLETED,
                UAVStatus.RETURNING,
            ):
                continue
            n, newly = self.fire.suppress(
                u.body.x,
                u.body.y,
                u.config.suppression_range_m,
                dt,
                u.uav_id,
                self.time_s,
            )
            if n:
                u.cells_extinguished += n
                cells_out += n
                u.status = UAVStatus.SUPPRESSING
            if newly:
                u.fires_suppressed += len(newly)
                fires_out += len(newly)
                for fid in newly:
                    self.events.add(
                        self.time_s,
                        "suppression",
                        f"{u.uav_id} extinguished {fid} (overflew within {u.config.suppression_range_m:.0f} m)",
                        agent_id=u.uav_id,
                        fire_id=fid,
                    )
        return cells_out, fires_out

    def _spawn_uavs(self) -> list[UAVAgent]:
        sc = self.scenario
        dyn = UAVDynamicsConfig(
            max_speed_mps=sc.uavs.max_speed_mps,
            cruise_speed_mps=sc.uavs.cruise_speed_mps,
            battery_capacity_j=sc.uavs.battery_capacity_j,
            return_to_base_fraction=sc.uavs.return_to_base_fraction,
        )
        agents: list[UAVAgent] = []
        n = sc.uavs.count
        placements = {i: p for i, p in enumerate(sc.uavs.placements)}
        for i in range(n):
            uid = f"uav_{i:02d}"
            if i in placements:
                plat, plon = placements[i].latitude, placements[i].longitude
                heading = np.deg2rad(placements[i].heading_deg)
                alt = placements[i].altitude_m or sc.uavs.altitude_m
                if placements[i].uav_id:
                    uid = placements[i].uav_id
            else:
                # Grid across the AOI so the fleet can search immediately.
                ncols = int(np.ceil(np.sqrt(n)))
                nrows = int(np.ceil(n / ncols))
                row, col = divmod(i, ncols)
                x = self.world.origin_x + (col + 0.5) / ncols * self.world.extent_x
                y = self.world.origin_y + (row + 0.5) / nrows * self.world.extent_y
                plat, plon = self.world.transform.to_wgs84(x, y)
                heading = 0.0
                alt = sc.uavs.altitude_m
            x, y = self.world.transform.to_local(plat, plon)
            x = float(np.clip(x, self.world.x_min, self.world.x_max))
            y = float(np.clip(y, self.world.y_min, self.world.y_max))
            cfg = UAVConfig(
                uav_id=uid,
                dynamics=dyn,
                sensor_range_m=sc.sensor.sensor_range_m,
                field_of_view_deg=sc.sensor.field_of_view_deg,
                detection_probability=sc.sensor.detection_probability,
                false_positive_probability=sc.sensor.false_positive_probability,
                false_negative_probability=sc.sensor.false_negative_probability,
                observation_noise_m=sc.sensor.observation_noise_m,
                communication_range_m=sc.communication.range_m,
                min_separation_m=sc.uavs.min_separation_m,
                start_lat=plat,
                start_lon=plon,
                start_altitude_m=alt,
                start_heading_deg=float(np.rad2deg(heading)),
                suppression_range_m=sc.sensor.suppression_range_m,
            )
            body = UAVBodyState(
                x=x,
                y=y,
                z=alt,
                heading=heading,
                speed=0.0,
                battery_j=dyn.battery_capacity_j,
            )
            agents.append(UAVAgent(config=cfg, body=body, lat=plat, lon=plon))
        return agents

    def _default_fires(self) -> list:
        from app.schemas.scenario import FireSourceConfig

        rng = np.random.default_rng(self.scenario.seed + 17)
        fires = []
        for i in range(2):
            x = self.world.origin_x + float(rng.uniform(0.25, 0.75)) * self.world.extent_x
            y = self.world.origin_y + float(rng.uniform(0.35, 0.80)) * self.world.extent_y
            lat, lon = self.world.transform.to_wgs84(x, y)
            fires.append(
                FireSourceConfig(
                    fire_id=f"fire_{i+1:02d}",
                    latitude=lat,
                    longitude=lon,
                    initial_intensity=0.85,
                    ignition_time=0.0 if i == 0 else 30.0,
                    radius_m=80.0,
                )
            )
        return fires

    def _apply_due_ignitions(self) -> None:
        assert self.fire is not None
        remain = []
        for i, fc in enumerate(self.pending_ignitions):
            if self.time_s + 1e-9 < fc.ignition_time:
                remain.append(fc)
                continue
            fid = fc.fire_id or f"fire_{len(self.fire.sources)+1:02d}"
            x, y = self.world.transform.to_local(fc.latitude, fc.longitude)
            self.fire.ignite(
                fire_id=fid,
                x=x,
                y=y,
                lat=fc.latitude,
                lon=fc.longitude,
                intensity=fc.initial_intensity,
                ignition_time=fc.ignition_time,
                radius_m=fc.radius_m,
            )
            self.events.add(self.time_s, "fire", f"{fid} ignited", fire_id=fid)
        self.pending_ignitions = remain

    def public_state(self) -> dict[str, Any]:
        assert self.fire is not None and self.coverage is not None
        to_wgs = self.world.transform.to_wgs84
        uav_payload = []
        trail_step = max(1, int(5 / self.dt))
        for u in self.uavs:
            uav_payload.append({
                "uav_id": u.uav_id,
                "latitude": u.lat,
                "longitude": u.lon,
                "x": u.body.x,
                "y": u.body.y,
                "altitude_m": u.body.z,
                "heading_deg": float(np.rad2deg(u.body.heading) % 360.0),
                "speed_mps": u.body.speed,
                "battery_fraction": u.battery_fraction(),
                "status": u.status.value,
                "sensor_range_m": u.config.sensor_range_m,
                "comm_range_m": u.config.communication_range_m,
                "known_fires": sorted(u.known_fire_ids),
                "last_action": u.last_action,
                "last_reward": u.last_reward,
                "detections_true": u.detections_true,
                "detections_false": u.detections_false,
                "suppression_range_m": u.config.suppression_range_m,
                "cells_extinguished": u.cells_extinguished,
                "fires_suppressed": u.fires_suppressed,
            })
        fire_payload = []
        rings = self.fire.perimeter_points_local()
        for src, ring in zip(self.fire.sources, rings, strict=False):
            ring_ll = [to_wgs(x, y) for x, y in ring]
            fire_payload.append({
                "fire_id": src.fire_id,
                "latitude": src.lat,
                "longitude": src.lon,
                "intensity": src.initial_intensity,
                "ignition_time": src.ignition_time,
                "radius_m": src.radius_m,
                "active": src.active,
                "detected": src.detected,
                "first_detection_time": src.first_detection_time,
                "detected_by": src.detected_by,
                "suppressed": src.suppressed,
                "extinguished_time": src.extinguished_time,
                "suppressed_by": src.suppressed_by,
                "perimeter": [{"latitude": la, "longitude": lo} for la, lo in ring_ll],
            })
        links = [{"a": l.a, "b": l.b, "distance_m": l.distance_m} for l in self.comms.links]
        return {
            "time_s": self.time_s,
            "step": self.step_count,
            "terminated": self._terminated,
            "region_id": self.world.region_id,
            "bounds": {
                "south": self.world.bounds.south,
                "west": self.world.bounds.west,
                "north": self.world.bounds.north,
                "east": self.world.bounds.east,
            },
            "wind": {
                "speed_kmh": self.scenario.environment.wind_speed_kmh,
                "from_deg": self.wind_from_deg,
                "from": self.scenario.environment.wind_from,
            },
            "uavs": uav_payload,
            "fires": fire_payload,
            "communication_links": links,
            "coverage_heatmap": self.coverage.downsample_for_display(40),
            "coverage_fraction": self.coverage.coverage_fraction(),
            "fire_grid": self.fire.snapshot_public(),
            "metrics": compute_public_metrics(self),
            "per_uav": per_uav_metrics(self),
            "policy": {
                "name": self.policy.info.name if self.policy else None,
                "kind": self.policy.info.kind if self.policy else None,
                "trained": self.policy.info.trained if self.policy else False,
                "algorithm": self.policy.info.algorithm if self.policy else None,
                "checkpoint_path": self.policy.info.checkpoint_path if self.policy else None,
                "notes": self.policy.info.notes if self.policy else None,
            },
            "reward_equation": reward_equation_latex(self.reward_weights),
            "sensor_assumptions": sensor_assumptions().__dict__,
            "observation_spec": observation_spec(self.scenario.policy.allow_global_state).__dict__,
            "seed": self.scenario.seed,
            "trail_hint_step": trail_step,
        }

    def debug_agent(self, uav_id: str) -> dict[str, Any] | None:
        obs = self._observe_all()
        for u in self.uavs:
            if u.uav_id != uav_id:
                continue
            nearest = None
            if self.fire:
                best = None
                for src in self.fire.sources:
                    d = float(np.hypot(src.x - u.body.x, src.y - u.body.y))
                    if best is None or d < best[0]:
                        best = (d, src.fire_id)
                nearest = {"fire_id": best[1], "distance_m": best[0]} if best else None
            return {
                "uav_id": u.uav_id,
                "observation": obs[u.uav_id].tolist(),
                "action": u.last_action,
                "reward": u.last_reward,
                "battery_fraction": u.battery_fraction(),
                "position": {"lat": u.lat, "lon": u.lon, "alt_m": u.body.z, "x": u.body.x, "y": u.body.y},
                "heading_deg": float(np.rad2deg(u.body.heading) % 360.0),
                "nearest_fire": nearest,
                "sensor": {
                    "range_m": u.config.sensor_range_m,
                    "detection_probability": u.config.detection_probability,
                    "false_positive_probability": u.config.false_positive_probability,
                    "false_negative_probability": u.config.false_negative_probability,
                },
                "communication": {
                    "range_m": u.config.communication_range_m,
                    "neighbours": self.comms.neighbours(u.uav_id),
                    "known_fires": sorted(u.known_fire_ids),
                    "messages_sent": u.messages_sent,
                    "messages_received": u.messages_received,
                },
                "status": u.status.value,
            }
        return None


def _heading_to(x0: float, y0: float, x1: float, y1: float) -> float:
    from app.geography.coordinate_transform import heading_from_vector

    return heading_from_vector(x1 - x0, y1 - y0)


# Silence unused import if type checkers complain about spaces.Box usage
_ = (OBS_DIM, N_DISCRETE_ACTIONS)
