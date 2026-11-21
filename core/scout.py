"""
Drone Scout Module for ASTRA-SCOUT.
Models flight trajectory generation, camera field-of-view (FOV) projection,
and aerial reconnaissance missions.
"""

from enum import Enum
from typing import List, Tuple, Dict
import numpy as np
from .energy_model import DroneEnergyModel
from .sensor_fusion import SensorFusionGrid


class FlightPattern(str, Enum):
    CORRIDOR = "Corridor Reconnaissance"
    BOUSTROPHEDON = "Lawnmower Area Survey"
    TARGETED_CHOKEPOINTS = "Chokepoint Inspection"


class DroneScout:
    """Autonomous planetary aerial scout drone."""

    def __init__(
        self,
        flight_altitude_m: float = 12.0,
        camera_fov_deg: float = 60.0,
        sensor_confidence: float = 0.88,
        energy_model: DroneEnergyModel = None
    ):
        self.flight_altitude_m = flight_altitude_m
        self.camera_fov_deg = camera_fov_deg
        self.sensor_confidence = sensor_confidence
        self.energy_model = energy_model or DroneEnergyModel()
        
        # Camera ground footprint radius in grid cells (at 1m/cell)
        half_fov_rad = np.radians(camera_fov_deg / 2.0)
        self.footprint_radius = max(2, int(flight_altitude_m * np.tan(half_fov_rad)))

    def generate_flight_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        grid_size: int,
        pattern: FlightPattern = FlightPattern.CORRIDOR,
        sweep_width: int = 14
    ) -> List[Tuple[int, int]]:
        """Generates waypoint trajectory for the drone based on chosen survey strategy."""
        waypoints: List[Tuple[int, int]] = []
        sy, sx = start
        gy, gx = goal

        if pattern == FlightPattern.CORRIDOR:
            # Direct survey corridor from start to goal with lateral scouting splay
            steps = 8
            for i in range(steps + 1):
                t = i / steps
                wy = int(np.clip(sy + t * (gy - sy), 0, grid_size - 1))
                wx = int(np.clip(sx + t * (gx - sx), 0, grid_size - 1))
                # Add slight lateral offset to widen coverage
                offset = int(sweep_width * 0.35 * np.sin(t * np.pi))
                wy = int(np.clip(wy + offset, 0, grid_size - 1))
                waypoints.append((wy, wx))

        elif pattern == FlightPattern.BOUSTROPHEDON:
            # Lawnmower pattern across the mission sector
            min_x = max(2, min(sx, gx) - 4)
            max_x = min(grid_size - 3, max(sx, gx) + 4)
            min_y = max(2, min(sy, gy) - sweep_width)
            max_y = min(grid_size - 3, max(sy, gy) + sweep_width)

            x_steps = list(range(min_x, max_x + 1, max(3, self.footprint_radius * 2)))
            toggle = True
            for x_val in x_steps:
                if toggle:
                    waypoints.append((min_y, x_val))
                    waypoints.append((max_y, x_val))
                else:
                    waypoints.append((max_y, x_val))
                    waypoints.append((min_y, x_val))
                toggle = not toggle

        else:  # TARGETED_CHOKEPOINTS
            # Focuses on midpoint and quarter-point choke corridors
            mid_y, mid_x = (sy + gy) // 2, (sx + gx) // 2
            waypoints = [
                start,
                ((sy * 3 + gy) // 4, (sx * 3 + gx) // 4 + sweep_width // 2),
                (mid_y, mid_x),
                ((sy + gy * 3) // 4, (sx + gx * 3) // 4 - sweep_width // 2),
                goal,
            ]

        return waypoints

    def execute_survey(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        environment,
        fusion_grid: SensorFusionGrid,
        pattern: FlightPattern = FlightPattern.CORRIDOR,
        sweep_width: int = 14
    ) -> Dict:
        """
        Executes aerial survey mission: computes coverage footprint,
        updates multi-tier sensor fusion grid, and calculates flight energy.
        """
        waypoints = self.generate_flight_path(start, goal, environment.size, pattern, sweep_width)
        size = environment.size
        coverage_mask = np.zeros((size, size), dtype=bool)
        y_grid, x_grid = np.mgrid[0:size, 0:size]

        # Project sensor FOV footprint along the flight path
        total_flight_dist = 0.0
        for i in range(len(waypoints) - 1):
            w1, w2 = waypoints[i], waypoints[i + 1]
            seg_dist = np.hypot(w2[0] - w1[0], w2[1] - w1[1]) * environment.resolution_m
            total_flight_dist += seg_dist

            # Interpolate footprint stamps along segment
            steps = max(2, int(seg_dist))
            for s in range(steps + 1):
                t = s / steps
                py = w1[0] + t * (w2[0] - w1[0])
                px = w1[1] + t * (w2[1] - w1[1])
                dist_sq = (y_grid - py)**2 + (x_grid - px)**2
                coverage_mask |= (dist_sq <= self.footprint_radius**2)

        # Update Sensor Fusion Grid with Aerial Scout observations
        from .terrain_analysis import compute_terrain_layers
        terrain = compute_terrain_layers(environment.elevation, environment.obstacles, environment.resolution_m)
        fusion_grid.apply_drone_observation(
            coverage_mask=coverage_mask,
            true_elevation=environment.elevation,
            true_hazard=terrain.hazard_index,
            true_obstacles=environment.obstacles,
            sensor_confidence=self.sensor_confidence
        )

        # Apply AI inference to surrounding unobserved periphery
        fusion_grid.apply_ai_inference(kernel_radius=6)

        # Calculate flight energy
        telemetry = self.energy_model.flight_energy(
            flight_distance_m=total_flight_dist,
            altitude_m=self.flight_altitude_m
        )
        telemetry["waypoints"] = waypoints
        telemetry["footprint_radius"] = self.footprint_radius
        telemetry["pattern"] = pattern.value

        return telemetry
