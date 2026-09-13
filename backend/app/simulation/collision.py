"""UAV separation and collision detection from simulation geometry.

A near-collision is recorded when horizontal distance < min_separation
and |Δz| < vertical_threshold.

A collision is recorded when horizontal distance < collision_radius
and |Δz| < vertical_threshold.

Events are generated from positions, not from UI animations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.simulation.uav import UAVAgent, UAVStatus


@dataclass(frozen=True)
class SeparationEvent:
    time_s: float
    a: str
    b: str
    distance_m: float
    delta_z_m: float
    is_collision: bool


@dataclass
class CollisionMonitor:
    collision_radius_m: float = 8.0
    vertical_threshold_m: float = 15.0
    near_events: int = 0
    collisions: int = 0
    min_separation_observed_m: float | None = None

    def check(self, uavs: list[UAVAgent], time_s: float) -> list[SeparationEvent]:
        events: list[SeparationEvent] = []
        n = len(uavs)
        for i in range(n):
            a = uavs[i]
            if a.status in (UAVStatus.LANDED, UAVStatus.BATTERY_DEPLETED):
                continue
            for j in range(i + 1, n):
                b = uavs[j]
                if b.status in (UAVStatus.LANDED, UAVStatus.BATTERY_DEPLETED):
                    continue
                d = float(np.hypot(a.body.x - b.body.x, a.body.y - b.body.y))
                dz = abs(a.body.z - b.body.z)
                if self.min_separation_observed_m is None:
                    self.min_separation_observed_m = d
                else:
                    self.min_separation_observed_m = min(self.min_separation_observed_m, d)
                if dz >= self.vertical_threshold_m:
                    continue
                sep = min(a.config.min_separation_m, b.config.min_separation_m)
                if d < self.collision_radius_m:
                    self.collisions += 1
                    a.collisions += 1
                    b.collisions += 1
                    a.status = UAVStatus.COLLIDED
                    b.status = UAVStatus.COLLIDED
                    events.append(
                        SeparationEvent(
                            time_s=time_s,
                            a=a.uav_id,
                            b=b.uav_id,
                            distance_m=d,
                            delta_z_m=dz,
                            is_collision=True,
                        )
                    )
                elif d < sep:
                    self.near_events += 1
                    a.near_collisions += 1
                    b.near_collisions += 1
                    events.append(
                        SeparationEvent(
                            time_s=time_s,
                            a=a.uav_id,
                            b=b.uav_id,
                            distance_m=d,
                            delta_z_m=dz,
                            is_collision=False,
                        )
                    )
        return events
