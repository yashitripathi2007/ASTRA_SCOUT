"""
Rover Agent Module for ASTRA-SCOUT.
Models the planetary rover entity, its dynamic local sensor perception,
step-by-step path traversal, obstacle encounter detection, and dynamic replanning triggers.
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
from .energy_model import RoverEnergyModel
from .sensor_fusion import SensorFusionGrid


class RoverAgent:
    """Represents the planetary exploration surface rover."""

    def __init__(
        self,
        start_pos: Tuple[int, int],
        sensor_radius: int = 5,
        battery_capacity_wh: float = 120.0,
        energy_model: Optional[RoverEnergyModel] = None
    ):
        self.start_pos = start_pos
        self.current_pos = start_pos
        self.sensor_radius = sensor_radius
        self.battery_capacity_wh = battery_capacity_wh
        self.battery_wh_remaining = battery_capacity_wh
        self.energy_model = energy_model or RoverEnergyModel()

        # Telemetry history
        self.traversed_path: List[Tuple[int, int]] = [start_pos]
        self.replanning_events: List[Tuple[int, int]] = []
        self.obstacles_discovered: List[Tuple[int, int]] = []
        self.total_energy_wh: float = 0.0
        self.total_distance_m: float = 0.0
        self.total_time_s: float = 0.0

    def get_local_sensor_mask(self, pos: Tuple[int, int], grid_size: int) -> np.ndarray:
        """Returns circular boolean mask of ground sensors around specified position."""
        y, x = np.mgrid[0:grid_size, 0:grid_size]
        return (y - pos[0])**2 + (x - pos[1])**2 <= self.sensor_radius**2

    def scan_surroundings(
        self,
        environment,
        fusion_grid: SensorFusionGrid,
        terrain_hazard: np.ndarray
    ) -> np.ndarray:
        """
        Executes local sensor sweep (LiDAR/stereo cameras) at rover's current location.
        Updates SensorFusionGrid with ROVER_GROUND ground-truth data.
        Returns the updated local mask.
        """
        mask = self.get_local_sensor_mask(self.current_pos, environment.size)
        fusion_grid.apply_rover_observation(
            rover_footprint=mask,
            true_elevation=environment.elevation,
            true_hazard=terrain_hazard,
            true_obstacles=environment.obstacles
        )
        return mask

    def check_path_collision(
        self,
        active_path: List[Tuple[int, int]],
        current_idx: int,
        fusion_grid: SensorFusionGrid,
        lookahead_steps: int = 6
    ) -> bool:
        """
        Checks if upcoming cells on the planned route are blocked by newly confirmed obstacles.
        """
        end_idx = min(len(active_path), current_idx + lookahead_steps + 1)
        for i in range(current_idx + 1, end_idx):
            cell = active_path[i]
            if fusion_grid.obstacle_belief[cell] or fusion_grid.hazard_belief[cell] >= 0.95:
                return True
        return False

    def step_to(
        self,
        next_pos: Tuple[int, int],
        environment,
        fusion_grid: SensorFusionGrid,
        terrain_roughness: np.ndarray,
        terrain_hazard: np.ndarray
    ) -> Dict:
        """
        Executes a single movement step from current_pos to next_pos.
        Consumes energy, updates sensor perception, and returns step telemetry.
        """
        p1 = self.current_pos
        p2 = next_pos
        z1 = environment.elevation[p1]
        z2 = environment.elevation[p2]
        rough = terrain_roughness[p2]
        haz = terrain_hazard[p2]

        step_telemetry = self.energy_model.step_energy(
            p1=p1,
            p2=p2,
            z1=z1,
            z2=z2,
            roughness=rough,
            hazard=haz,
            resolution_m=environment.resolution_m
        )

        # Update rover state
        self.current_pos = p2
        self.traversed_path.append(p2)
        self.total_energy_wh += step_telemetry["energy_wh"]
        self.total_distance_m += step_telemetry["distance_m"]
        self.total_time_s += step_telemetry["duration_s"]
        self.battery_wh_remaining = max(0.0, self.battery_wh_remaining - step_telemetry["energy_wh"])

        # Scan surroundings at new position
        self.scan_surroundings(environment, fusion_grid, terrain_hazard)

        return step_telemetry

    def get_mission_summary(self) -> Dict:
        """Returns comprehensive summary of the rover's mission execution."""
        return {
            "steps_taken": len(self.traversed_path) - 1,
            "total_distance_m": round(self.total_distance_m, 2),
            "total_energy_wh": round(self.total_energy_wh, 2),
            "total_time_s": round(self.total_time_s, 1),
            "replanning_events_count": len(self.replanning_events),
            "replanning_locations": self.replanning_events,
            "battery_soc_pct": round(100.0 * (self.battery_wh_remaining / self.battery_capacity_wh), 1),
        }
