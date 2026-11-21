"""
Energy-Aware Navigation & Mission Simulation for ASTRA-SCOUT.
Implements multi-objective directional A* path planning and dynamic replanning
simulation comparing three strategies under identical conditions:
1. Rover-Only (No Scouting)
2. Always-Scout (Unconditional aerial reconnaissance)
3. Adaptive-Scout (Decision-theoretic scouting based on expected information benefit vs cost)
"""

import heapq
import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Dict, Optional, Any
import numpy as np

from .sensor_fusion import SensorFusionGrid, ObservationSource
from .rover import RoverAgent
from .scout import DroneScout, FlightPattern
from .energy_model import RoverEnergyModel, DroneEnergyModel
from .decision_engine import (
    AdaptiveDecisionEngine,
    DecisionRecommendation,
    MissionCostConfig,
    compute_mission_cost,
)


NEIGHBORS_8 = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


class StrategyMode(str, Enum):
    ROVER_ONLY = "Rover-Only (No Scouting)"
    ALWAYS_SCOUT = "Always-Scout"
    ADAPTIVE_SCOUT = "Adaptive-Scout"


@dataclass
class MissionResult:
    """Comprehensive comparison result of all three exploration strategies."""
    # Backward compatible fields
    rover_only_path: List[Tuple[int, int]]
    cooperative_path: List[Tuple[int, int]]
    rover_only_telemetry: Dict
    cooperative_telemetry: Dict
    drone_telemetry: Optional[Dict]
    decision_assessment: Dict
    coverage_stats: Dict
    rover_only_replans: int
    cooperative_replans: int

    # Adaptive strategy and 3-way benchmarking fields
    adaptive_path: Optional[List[Tuple[int, int]]] = None
    adaptive_telemetry: Optional[Dict] = None
    adaptive_drone_telemetry: Optional[Dict] = None
    adaptive_replans: int = 0
    adaptive_coverage_stats: Optional[Dict] = None
    baseline_coverage_stats: Optional[Dict] = None
    mission_costs: Optional[Dict[str, float]] = None
    route_risks: Optional[Dict[str, float]] = None
    cost_config: Optional[MissionCostConfig] = None
    strategy_results: Optional[Dict[str, Any]] = None
    confidence_history: Optional[List[Dict[str, Any]]] = None


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


def simulate_rover_traversal(
    environment,
    terrain,
    fusion_grid: SensorFusionGrid,
    planner: PathPlanner,
    rover_radius: int = 5,
    rover_battery_wh: float = 120.0,
    caution_penalty: float = 2.5,
    max_steps_mult: int = 4,
    cost_config: Optional[MissionCostConfig] = None
) -> RoverAgent:
    """
    Simulates dynamic step-by-step rover traversal with local perception,
    upcoming terrain confidence tracking, and dynamic replanning upon discovering unforeseen obstacles.
    """
    cfg = cost_config or MissionCostConfig()
    rover = RoverAgent(
        start_pos=environment.start,
        sensor_radius=rover_radius,
        battery_capacity_wh=rover_battery_wh
    )
    # Rover initial scan at start location
    rover.scan_surroundings(environment, fusion_grid, terrain.hazard_index)

    active_path, _ = planner.plan(
        fusion_grid, environment.elevation, environment.start, environment.goal, caution_penalty
    )

    # Initial confidence at mission origin
    if active_path:
        init_conf = rover.compute_upcoming_confidence(
            active_path=active_path,
            current_idx=0,
            fusion_grid=fusion_grid,
            lookahead_distance_m=cfg.confidence_lookahead_m,
            resolution_m=environment.resolution_m,
            weight_observed_ground=cfg.weight_observed_ground,
            weight_observed_aerial=cfg.weight_observed_aerial,
            weight_inferred=cfg.weight_inferred,
            weight_unobserved=cfg.weight_unobserved,
            hazard_penalty_weight=cfg.confidence_hazard_weight
        )
        dec_label = "Drive Direct" if init_conf >= cfg.confidence_threshold else "Low Confidence"
        rover.record_confidence_step(
            step=0,
            confidence=init_conf,
            threshold=cfg.confidence_threshold,
            decision=dec_label,
            rationale=f"Origin terrain confidence: {init_conf:.2f} (gate: {cfg.confidence_threshold:.2f})"
        )

    max_steps = environment.size * max_steps_mult
    step_count = 0

    while active_path and rover.current_pos != environment.goal and step_count < max_steps:
        step_count += 1
        try:
            curr_idx = active_path.index(rover.current_pos)
        except ValueError:
            curr_idx = 0

        if curr_idx + 1 >= len(active_path):
            break

        next_pos = active_path[curr_idx + 1]

        # Check if next_pos is an impassable obstacle
        if environment.obstacles[next_pos]:
            rover.replanning_events.append(rover.current_pos)
            rover.obstacles_discovered.append(next_pos)
            fusion_grid.obstacle_belief[next_pos] = True
            fusion_grid.hazard_belief[next_pos] = 1.0

            # Dynamic replanning from current position
            new_path, _ = planner.plan(
                fusion_grid, environment.elevation, rover.current_pos, environment.goal, caution_penalty
            )
            if new_path:
                active_path = new_path
                continue
            else:
                break

        # Move to next_pos
        rover.step_to(
            next_pos=next_pos,
            environment=environment,
            fusion_grid=fusion_grid,
            terrain_roughness=terrain.roughness,
            terrain_hazard=terrain.hazard_index
        )

        # Record confidence after step
        step_conf = rover.compute_upcoming_confidence(
            active_path=active_path,
            current_idx=curr_idx + 1,
            fusion_grid=fusion_grid,
            lookahead_distance_m=cfg.confidence_lookahead_m,
            resolution_m=environment.resolution_m,
            weight_observed_ground=cfg.weight_observed_ground,
            weight_observed_aerial=cfg.weight_observed_aerial,
            weight_inferred=cfg.weight_inferred,
            weight_unobserved=cfg.weight_unobserved,
            hazard_penalty_weight=cfg.confidence_hazard_weight
        )
        dec_label = "Drive Direct" if step_conf >= cfg.confidence_threshold else "Low Confidence"
        rover.record_confidence_step(
            step=step_count,
            confidence=step_conf,
            threshold=cfg.confidence_threshold,
            decision=dec_label,
            rationale=f"Step {step_count}: terrain confidence {step_conf:.2f} over next {cfg.confidence_lookahead_m:.1f}m"
        )

        # Check collision lookahead on newly observed terrain
        if rover.check_path_collision(active_path, curr_idx + 1, fusion_grid):
            rover.replanning_events.append(rover.current_pos)
            new_path, _ = planner.plan(
                fusion_grid, environment.elevation, rover.current_pos, environment.goal, caution_penalty
            )
            if new_path:
                active_path = new_path

    return rover


def plan_mission(
    environment,
    terrain,
    decision_engine: AdaptiveDecisionEngine,
    scout: Optional[DroneScout] = None,
    rover_radius: int = 5,
    scout_pattern: FlightPattern = FlightPattern.CORRIDOR,
    caution_penalty: float = 2.5,
    rover_battery_wh: float = 120.0,
    drone_battery_wh: float = 40.0,
    cost_config: Optional[MissionCostConfig] = None
) -> MissionResult:
    """
    Executes full multi-strategy planetary exploration mission simulation.
    Simulates three distinct strategies under identical conditions:
    1. Rover-Only: Local perception only, frequent dynamic replanning on obstacles.
    2. Always-Scout: Unconditional aerial drone reconnaissance ahead of the rover.
    3. Adaptive-Scout: Evaluates Expected Information Gain (EIG), net energy ROI,
       and battery feasibility to selectively deploy scouting.
    """
    planner = PathPlanner()
    cfg = cost_config or MissionCostConfig()

    # --- 1. Strategy 1: Rover-Only Baseline (Dynamic Replanning) ---
    baseline_fusion = SensorFusionGrid(environment.size)
    baseline_rover = simulate_rover_traversal(
        environment=environment,
        terrain=terrain,
        fusion_grid=baseline_fusion,
        planner=planner,
        rover_radius=rover_radius,
        rover_battery_wh=rover_battery_wh,
        caution_penalty=caution_penalty,
        cost_config=cfg
    )
    rover_only_telemetry = baseline_rover.get_mission_summary()
    rover_only_coverage = baseline_fusion.get_coverage_breakdown()
    rover_only_risk = float(np.mean([terrain.hazard_index[p] for p in baseline_rover.traversed_path])) if baseline_rover.traversed_path else 0.0
    rover_only_cost = compute_mission_cost(
        rover_only_telemetry["total_energy_wh"], 0.0, rover_only_risk, cfg
    )

    # --- 2. Strategy 2: Always-Scout (Unconditional Cooperative Reconnaissance) ---
    always_fusion = SensorFusionGrid(environment.size)
    always_rover_init = RoverAgent(environment.start, rover_radius, rover_battery_wh)
    always_rover_init.scan_surroundings(environment, always_fusion, terrain.hazard_index)

    always_drone_telemetry = None
    if scout is not None:
        always_drone_telemetry = scout.execute_survey(
            start=environment.start,
            goal=environment.goal,
            environment=environment,
            fusion_grid=always_fusion,
            pattern=scout_pattern
        )

    always_rover = simulate_rover_traversal(
        environment=environment,
        terrain=terrain,
        fusion_grid=always_fusion,
        planner=planner,
        rover_radius=rover_radius,
        rover_battery_wh=rover_battery_wh,
        caution_penalty=caution_penalty,
        cost_config=cfg
    )
    always_telemetry = always_rover.get_mission_summary()
    always_coverage = always_fusion.get_coverage_breakdown()
    always_risk = float(np.mean([terrain.hazard_index[p] for p in always_rover.traversed_path])) if always_rover.traversed_path else 0.0
    always_drone_wh = float(always_drone_telemetry["flight_energy_wh"]) if always_drone_telemetry else 0.0
    always_cost = compute_mission_cost(
        always_telemetry["total_energy_wh"], always_drone_wh, always_risk, cfg
    )

    # --- 3. Strategy 3: Adaptive-Scout (Selective Information-Theoretic Reconnaissance with Confidence Gating) ---
    adaptive_fusion = SensorFusionGrid(environment.size)
    adaptive_rover_init = RoverAgent(environment.start, rover_radius, rover_battery_wh)
    adaptive_rover_init.scan_surroundings(environment, adaptive_fusion, terrain.hazard_index)

    # Initial provisional path to evaluate upcoming corridor confidence before movement
    initial_path, _ = planner.plan(
        adaptive_fusion, environment.elevation, environment.start, environment.goal, caution_penalty
    )

    # Evaluate Gated Decision Engine
    decision = decision_engine.evaluate_scout_tradeoff(
        environment=environment,
        terrain=terrain,
        start=environment.start,
        goal=environment.goal,
        rover_battery_wh=rover_battery_wh,
        drone_battery_wh=drone_battery_wh,
        cost_config=cfg,
        fusion_grid=adaptive_fusion,
        planned_path=initial_path,
        current_idx=0
    )

    adaptive_drone_telemetry = None
    if scout is not None:
        if decision.recommendation == DecisionRecommendation.SCOUT_RECOMMENDED:
            adaptive_drone_telemetry = scout.execute_survey(
                start=environment.start,
                goal=environment.goal,
                environment=environment,
                fusion_grid=adaptive_fusion,
                pattern=scout_pattern
            )
        elif decision.recommendation == DecisionRecommendation.TARGETED_SCOUT:
            adaptive_drone_telemetry = scout.execute_survey(
                start=environment.start,
                goal=environment.goal,
                environment=environment,
                fusion_grid=adaptive_fusion,
                pattern=FlightPattern.TARGETED_CHOKEPOINTS
            )
        else:
            # DRIVE_DIRECT: Drone remains grounded, saves 100% of flight overhead
            adaptive_drone_telemetry = None

    adaptive_rover = simulate_rover_traversal(
        environment=environment,
        terrain=terrain,
        fusion_grid=adaptive_fusion,
        planner=planner,
        rover_radius=rover_radius,
        rover_battery_wh=rover_battery_wh,
        caution_penalty=caution_penalty,
        cost_config=cfg
    )
    adaptive_telemetry = adaptive_rover.get_mission_summary()
    adaptive_coverage = adaptive_fusion.get_coverage_breakdown()
    adaptive_risk = float(np.mean([terrain.hazard_index[p] for p in adaptive_rover.traversed_path])) if adaptive_rover.traversed_path else 0.0
    adaptive_drone_wh = float(adaptive_drone_telemetry["flight_energy_wh"]) if adaptive_drone_telemetry else 0.0
    adaptive_cost = compute_mission_cost(
        adaptive_telemetry["total_energy_wh"], adaptive_drone_wh, adaptive_risk, cfg
    )

    # Combined costs and risks dictionary
    mission_costs = {
        "rover_only": rover_only_cost,
        "always_scout": always_cost,
        "adaptive_scout": adaptive_cost
    }
    route_risks = {
        "rover_only": round(rover_only_risk, 3),
        "always_scout": round(always_risk, 3),
        "adaptive_scout": round(adaptive_risk, 3)
    }

    strategy_results = {
        StrategyMode.ROVER_ONLY.value: {
            "path": baseline_rover.traversed_path,
            "telemetry": rover_only_telemetry,
            "drone_telemetry": None,
            "replans": len(baseline_rover.replanning_events),
            "coverage": rover_only_coverage,
            "route_risk": rover_only_risk,
            "mission_cost": rover_only_cost,
        },
        StrategyMode.ALWAYS_SCOUT.value: {
            "path": always_rover.traversed_path,
            "telemetry": always_telemetry,
            "drone_telemetry": always_drone_telemetry,
            "replans": len(always_rover.replanning_events),
            "coverage": always_coverage,
            "route_risk": always_risk,
            "mission_cost": always_cost,
        },
        StrategyMode.ADAPTIVE_SCOUT.value: {
            "path": adaptive_rover.traversed_path,
            "telemetry": adaptive_telemetry,
            "drone_telemetry": adaptive_drone_telemetry,
            "replans": len(adaptive_rover.replanning_events),
            "coverage": adaptive_coverage,
            "route_risk": adaptive_risk,
            "mission_cost": adaptive_cost,
        }
    }

    return MissionResult(
        rover_only_path=baseline_rover.traversed_path,
        cooperative_path=always_rover.traversed_path,
        rover_only_telemetry=rover_only_telemetry,
        cooperative_telemetry=always_telemetry,
        drone_telemetry=always_drone_telemetry,
        decision_assessment={
            "recommendation": decision.recommendation.value,
            "confidence_score": decision.confidence_score,
            "terrain_confidence": decision.terrain_confidence_score,
            "confidence_threshold": decision.confidence_threshold,
            "terrain_complexity": decision.terrain_complexity_score,
            "path_uncertainty": decision.path_uncertainty_score,
            "estimated_rover_saving_wh": decision.estimated_rover_saving_wh,
            "estimated_scout_cost_wh": decision.estimated_scout_cost_wh,
            "expected_net_benefit_wh": decision.expected_net_benefit_wh,
            "rationale": decision.rational,
            "gating_bypassed": decision.gating_bypassed
        },
        coverage_stats=always_coverage,
        rover_only_replans=len(baseline_rover.replanning_events),
        cooperative_replans=len(always_rover.replanning_events),
        adaptive_path=adaptive_rover.traversed_path,
        adaptive_telemetry=adaptive_telemetry,
        adaptive_drone_telemetry=adaptive_drone_telemetry,
        adaptive_replans=len(adaptive_rover.replanning_events),
        adaptive_coverage_stats=adaptive_coverage,
        baseline_coverage_stats=rover_only_coverage,
        mission_costs=mission_costs,
        route_risks=route_risks,
        cost_config=cfg,
        strategy_results=strategy_results,
        confidence_history=adaptive_rover.confidence_history
    )
