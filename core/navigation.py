"""
Energy-Aware Navigation & Mission Simulation for ASTRA-SCOUT.
Implements multi-objective directional A* path planning and dynamic replanning
simulation comparing unassisted rover vs cooperative ASTRA-SCOUT system.
"""

import heapq
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np

from .sensor_fusion import SensorFusionGrid, ObservationSource
from .rover import RoverAgent
from .scout import DroneScout, FlightPattern
from .energy_model import RoverEnergyModel, DroneEnergyModel


NEIGHBORS_8 = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


@dataclass
class MissionResult:
    """Comprehensive comparison result of a rover-drone mission."""
    rover_only_path: List[Tuple[int, int]]
    cooperative_path: List[Tuple[int, int]]
    rover_only_telemetry: Dict
    cooperative_telemetry: Dict
    drone_telemetry: Optional[Dict]
    decision_assessment: Dict
    coverage_stats: Dict
    rover_only_replans: int
    cooperative_replans: int


class PathPlanner:
    """Direction-aware, multi-objective A* path planner."""

    def __init__(
        self,
        weight_distance: float = 1.0,
        weight_hazard: float = 3.5,
        weight_slope: float = 0.05,
        weight_uncertainty: float = 2.0
    ):
        self.w_dist = weight_distance
        self.w_hazard = weight_hazard
        self.w_slope = weight_slope
        self.w_unc = weight_uncertainty

    def plan(
        self,
        fusion_grid: SensorFusionGrid,
        elevation: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        caution_penalty: float = 2.5
    ) -> Tuple[Optional[List[Tuple[int, int]]], float]:
        """
        Executes multi-objective A* search respecting perceived hazards and uncertainty.
        If a cell is known to be an obstacle (hazard >= 0.95), it is strictly impassable.
        """
        h, w = fusion_grid.size, fusion_grid.size
        pq = [(0.0, start)]
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start: 0.0}
        closed = set()

        while pq:
            _, current = heapq.heappop(pq)
            if current in closed:
                continue
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path, g_score[goal]
            closed.add(current)

            cy, cx = current
            cz = elevation[cy, cx]

            for dy, dx in NEIGHBORS_8:
                ny, nx = cy + dy, cx + dx
                nxt = (ny, nx)
                if not (0 <= ny < h and 0 <= nx < w):
                    continue

                # Obstacle check in belief map
                if fusion_grid.obstacle_belief[ny, nx] or fusion_grid.hazard_belief[ny, nx] >= 0.95:
                    continue

                source = fusion_grid.source_map[ny, nx]
                dist_step = math.sqrt(2) if (dy and dx) else 1.0

                if source == ObservationSource.UNOBSERVED:
                    cell_hazard = 0.35
                    cell_slope = 10.0
                    unc_cost = caution_penalty * self.w_unc
                else:
                    cell_hazard = fusion_grid.hazard_belief[ny, nx]
                    nz = elevation[ny, nx]
                    incline_deg = abs(np.degrees(np.arctan2(nz - cz, dist_step)))
                    cell_slope = incline_deg
                    unc_cost = fusion_grid.uncertainty_map[ny, nx] * self.w_unc

                step_cost = dist_step * (
                    self.w_dist +
                    self.w_hazard * cell_hazard +
                    self.w_slope * (cell_slope / 10.0) +
                    unc_cost
                )

                tentative_g = g_score[current] + step_cost
                if tentative_g < g_score.get(nxt, float("inf")):
                    came_from[nxt] = current
                    g_score[nxt] = tentative_g
                    heuristic = math.hypot(goal[0] - ny, goal[1] - nx)
                    heapq.heappush(pq, (tentative_g + heuristic, nxt))

        return None, float("inf")


def plan_mission(
    environment,
    terrain,
    decision_engine,
    scout: Optional[DroneScout] = None,
    rover_radius: int = 5,
    scout_pattern: FlightPattern = FlightPattern.CORRIDOR,
    caution_penalty: float = 2.5
) -> MissionResult:
    """
    Executes full cooperative exploration mission simulation.
    Compares:
    1. Blind Rover-Only Baseline: traverses with only local perception, encounters obstacles,
       and must repeatedly replan.
    2. Cooperative ASTRA-SCOUT System: aerial scout maps corridor ahead, AI terrain intelligence
       fuses layers, and rover navigates smoothly with optimal energy efficiency.
    """
    planner = PathPlanner()

    # --- 1. Evaluate Decision Engine ---
    decision = decision_engine.evaluate_scout_tradeoff(
        environment, terrain, environment.start, environment.goal
    )

    # --- 2. Baseline: Rover-Only Mission (Dynamic Replanning) ---
    baseline_fusion = SensorFusionGrid(environment.size)
    baseline_rover = RoverAgent(
        start_pos=environment.start,
        sensor_radius=rover_radius,
        battery_capacity_wh=120.0
    )
    # Rover initial scan at start
    baseline_rover.scan_surroundings(environment, baseline_fusion, terrain.hazard_index)

    # Initial plan with limited local knowledge
    active_path, _ = planner.plan(
        baseline_fusion, environment.elevation, environment.start, environment.goal, caution_penalty
    )

    # Dynamic execution simulation
    max_steps = environment.size * 4
    step_count = 0
    while active_path and baseline_rover.current_pos != environment.goal and step_count < max_steps:
        step_count += 1
        # Find next waypoint
        try:
            curr_idx = active_path.index(baseline_rover.current_pos)
        except ValueError:
            curr_idx = 0

        if curr_idx + 1 >= len(active_path):
            break

        next_pos = active_path[curr_idx + 1]

        # Check if next_pos or immediately upcoming path is an impassable obstacle
        if environment.obstacles[next_pos]:
            # Rover detects obstacle upon approach!
            baseline_rover.replanning_events.append(baseline_rover.current_pos)
            baseline_rover.obstacles_discovered.append(next_pos)
            baseline_fusion.obstacle_belief[next_pos] = True
            baseline_fusion.hazard_belief[next_pos] = 1.0

            # Trigger Dynamic Replanning from current position
            new_path, _ = planner.plan(
                baseline_fusion, environment.elevation, baseline_rover.current_pos, environment.goal, caution_penalty
            )
            if new_path:
                active_path = new_path
                continue
            else:
                # Trapped
                break

        # Move to next_pos
        baseline_rover.step_to(
            next_pos=next_pos,
            environment=environment,
            fusion_grid=baseline_fusion,
            terrain_roughness=terrain.roughness,
            terrain_hazard=terrain.hazard_index
        )

        # Check lookahead on newly observed terrain
        if baseline_rover.check_path_collision(active_path, curr_idx + 1, baseline_fusion):
            baseline_rover.replanning_events.append(baseline_rover.current_pos)
            new_path, _ = planner.plan(
                baseline_fusion, environment.elevation, baseline_rover.current_pos, environment.goal, caution_penalty
            )
            if new_path:
                active_path = new_path

    rover_only_telemetry = baseline_rover.get_mission_summary()

    # --- 3. ASTRA-SCOUT Cooperative Mission ---
    coop_fusion = SensorFusionGrid(environment.size)
    coop_rover = RoverAgent(
        start_pos=environment.start,
        sensor_radius=rover_radius,
        battery_capacity_wh=120.0
    )
    # Initial rover scan
    coop_rover.scan_surroundings(environment, coop_fusion, terrain.hazard_index)

    # Aerial Drone Scout Execution
    scout_telemetry = None
    if scout is not None:
        scout_telemetry = scout.execute_survey(
            start=environment.start,
            goal=environment.goal,
            environment=environment,
            fusion_grid=coop_fusion,
            pattern=scout_pattern
        )

    # Cooperative A* planning on fused multi-tier belief map
    coop_path, _ = planner.plan(
        coop_fusion, environment.elevation, environment.start, environment.goal, caution_penalty
    )

    # Cooperative rover execution
    step_count = 0
    active_coop_path = coop_path
    while active_coop_path and coop_rover.current_pos != environment.goal and step_count < max_steps:
        step_count += 1
        try:
            curr_idx = active_coop_path.index(coop_rover.current_pos)
        except ValueError:
            curr_idx = 0

        if curr_idx + 1 >= len(active_coop_path):
            break

        next_pos = active_coop_path[curr_idx + 1]

        if environment.obstacles[next_pos]:
            # Rare obstacle discovery if drone missed it
            coop_rover.replanning_events.append(coop_rover.current_pos)
            coop_fusion.obstacle_belief[next_pos] = True
            coop_fusion.hazard_belief[next_pos] = 1.0
            new_path, _ = planner.plan(
                coop_fusion, environment.elevation, coop_rover.current_pos, environment.goal, caution_penalty
            )
            if new_path:
                active_coop_path = new_path
                continue
            else:
                break

        coop_rover.step_to(
            next_pos=next_pos,
            environment=environment,
            fusion_grid=coop_fusion,
            terrain_roughness=terrain.roughness,
            terrain_hazard=terrain.hazard_index
        )

    coop_telemetry = coop_rover.get_mission_summary()

    return MissionResult(
        rover_only_path=baseline_rover.traversed_path,
        cooperative_path=coop_rover.traversed_path,
        rover_only_telemetry=rover_only_telemetry,
        cooperative_telemetry=coop_telemetry,
        drone_telemetry=scout_telemetry,
        decision_assessment={
            "recommendation": decision.recommendation.value,
            "confidence_score": decision.confidence_score,
            "terrain_complexity": decision.terrain_complexity_score,
            "estimated_rover_saving_wh": decision.estimated_rover_saving_wh,
            "estimated_scout_cost_wh": decision.estimated_scout_cost_wh,
            "expected_net_benefit_wh": decision.expected_net_benefit_wh,
            "rationale": decision.rational
        },
        coverage_stats=coop_fusion.get_coverage_breakdown(),
        rover_only_replans=len(baseline_rover.replanning_events),
        cooperative_replans=len(coop_rover.replanning_events)
    )
