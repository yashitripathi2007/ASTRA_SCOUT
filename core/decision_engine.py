"""
Adaptive Decision Engine for ASTRA-SCOUT.
Evaluates the core question: "Scout or Drive?"
Computes Expected Information Gain (EIG), Energy Trade-off ROI, and Battery Feasibility
to determine whether deploying the aerial scout drone pays off.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple, Optional, List, Any
import numpy as np
from .energy_model import DroneEnergyModel, RoverEnergyModel


class DecisionRecommendation(str, Enum):
    SCOUT_RECOMMENDED = "Deploy Scout (High Value)"
    DRIVE_DIRECT = "Drive Direct (Low Risk)"
    TARGETED_SCOUT = "Targeted Chokepoint Scout"


@dataclass
class MissionCostConfig:
    """Configurable weights and constraints for multi-objective mission cost and terrain confidence gating."""
    weight_rover_energy: float = 1.0     # Multiplier for rover surface energy (Wh)
    weight_drone_energy: float = 1.0     # Multiplier for drone flight overhead (Wh)
    weight_route_risk: float = 15.0      # Multiplier for cumulative route hazard exposure
    drone_reserve_margin: float = 0.20   # Required safety reserve buffer for drone battery (20%)
    critical_rover_battery_pct: float = 25.0  # Low rover battery threshold
    # Terrain confidence gating parameters
    confidence_threshold: float = 0.60   # Gate threshold: confidence < threshold triggers scout evaluation
    confidence_lookahead_m: float = 15.0 # Lookahead corridor length along upcoming path (meters)
    confidence_hysteresis: float = 0.05  # Hysteresis buffer to prevent decision flip-flopping [threshold +- margin]
    weight_observed_ground: float = 1.0  # Knowledge weight for direct rover ground contact
    weight_observed_aerial: float = 0.88 # Knowledge weight for aerial drone survey
    weight_inferred: float = 0.40        # Knowledge weight for AI spatial prior
    weight_unobserved: float = 0.00      # Knowledge weight for unobserved terrain
    confidence_hazard_weight: float = 0.25  # Attenuation factor reducing confidence for high terrain hazard


def compute_mission_cost(
    rover_energy_wh: float,
    drone_energy_wh: float,
    route_risk: float,
    config: Optional[MissionCostConfig] = None
) -> float:
    """
    Computes combined multi-objective mission cost.
    Cost = (w_rover * E_rover) + (w_drone * E_drone) + (w_risk * Risk)
    """
    cfg = config or MissionCostConfig()
    total_cost = (
        cfg.weight_rover_energy * float(rover_energy_wh) +
        cfg.weight_drone_energy * float(drone_energy_wh) +
        cfg.weight_route_risk * float(route_risk)
    )
    return float(round(total_cost, 2))


@dataclass
class DecisionAssessment:
    recommendation: DecisionRecommendation
    confidence_score: float              # 0.0 to 1.0 (computed terrain confidence score)
    terrain_complexity_score: float      # 0.0 to 1.0 (roughness + slope variance + obstacle density)
    path_uncertainty_score: float        # Shannon entropy / ignorance proxy in mission sector [0, 1]
    estimated_rover_saving_wh: float     # Estimated rover energy saved by avoiding detours
    estimated_scout_cost_wh: float       # Drone flight energy required
    expected_net_benefit_wh: float       # Net energy benefit (+ = savings, - = overhead)
    rational: str                        # Human-interpretable rationale for the crew
    terrain_confidence_score: float = 0.0 # Real computed terrain confidence score along upcoming path
    confidence_threshold: float = 0.60   # Configured gate threshold
    gating_bypassed: bool = False        # True if scouting was skipped due to high terrain confidence

    @property
    def rationale(self) -> str:
        return self.rational


class AdaptiveDecisionEngine:
    """Intelligent decision-theoretic arbiter for collaborative exploration."""

    def __init__(
        self,
        rover_model: RoverEnergyModel = None,
        drone_model: DroneEnergyModel = None,
        cost_config: MissionCostConfig = None
    ):
        self.rover_model = rover_model or RoverEnergyModel()
        self.drone_model = drone_model or DroneEnergyModel()
        self.cost_config = cost_config or MissionCostConfig()

    @staticmethod
    def _generate_provisional_corridor(
        start: Tuple[int, int],
        goal: Tuple[int, int],
        grid_size: int,
        steps: int = 25
    ) -> List[Tuple[int, int]]:
        """Generates direct linear waypoint corridor from start to goal."""
        sy, sx = start
        gy, gx = goal
        corridor = []
        for i in range(steps + 1):
            t = i / steps
            py = int(np.clip(round(sy + t * (gy - sy)), 0, grid_size - 1))
            px = int(np.clip(round(sx + t * (gx - sx)), 0, grid_size - 1))
            if not corridor or corridor[-1] != (py, px):
                corridor.append((py, px))
        return corridor

    def evaluate_scout_tradeoff(
        self,
        environment,
        terrain,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        rover_battery_wh: float = 120.0,
        drone_battery_wh: float = 40.0,
        cost_config: Optional[MissionCostConfig] = None,
        fusion_grid: Optional[Any] = None,
        planned_path: Optional[List[Tuple[int, int]]] = None,
        current_idx: int = 0,
        is_scouting_active: bool = False,
        override_confidence: Optional[float] = None
    ) -> DecisionAssessment:
        """
        Analyzes terrain confidence, obstacle cluster density, slope variance, and battery levels
        to recommend whether aerial scouting should be launched.

        GATE LOGIC:
        1. If terrain_confidence >= effective_threshold (with hysteresis):
           Skip scouting immediately! Rover drives direct to conserve drone flight energy.
        2. If terrain_confidence < effective_threshold:
           Evaluate expected-benefit trade-off and drone battery sufficiency:
           - If drone battery < required flight energy + reserve: mandate direct drive.
           - Else if expected net benefit or complexity is high: recommend aerial survey.
           - Else: drive direct.
        """
        cfg = cost_config or self.cost_config
        sy, sx = start
        gy, gx = goal
        dist_direct = np.hypot(gy - sy, gx - sx) * environment.resolution_m

        # 1. Analyze the direct mission corridor (bounding box + buffer)
        y_min = max(0, min(sy, gy) - 8)
        y_max = min(environment.size, max(sy, gy) + 9)
        x_min = max(0, min(sx, gx) - 8)
        x_max = min(environment.size, max(sx, gx) + 9)

        corridor_hazards = terrain.hazard_index[y_min:y_max, x_min:x_max]
        corridor_slopes = terrain.slope_deg[y_min:y_max, x_min:x_max]
        corridor_obstacles = environment.obstacles[y_min:y_max, x_min:x_max]

        obstacle_ratio = float(np.mean(corridor_obstacles))
        mean_hazard = float(np.mean(corridor_hazards))
        slope_variance = float(np.var(corridor_slopes))

        # Terrain complexity index [0, 1]
        complexity = float(np.clip(
            0.45 * (obstacle_ratio / 0.12) +
            0.35 * (mean_hazard / 0.40) +
            0.20 * min(1.0, slope_variance / 40.0),
            0.0, 1.0
        ))

        # Path uncertainty proxy strictly bounded in [0.0, 1.0]
        path_uncertainty = float(np.clip(0.50 + 0.50 * complexity, 0.0, 1.0))

        # 2. Compute Terrain Confidence along upcoming path
        if override_confidence is not None:
            terrain_conf = float(np.clip(override_confidence, 0.0, 1.0))
        elif fusion_grid is not None and planned_path is not None:
            terrain_conf = fusion_grid.compute_path_confidence(
                path=planned_path,
                current_idx=current_idx,
                lookahead_distance_m=cfg.confidence_lookahead_m,
                resolution_m=environment.resolution_m,
                weight_observed_ground=cfg.weight_observed_ground,
                weight_observed_aerial=cfg.weight_observed_aerial,
                weight_inferred=cfg.weight_inferred,
                weight_unobserved=cfg.weight_unobserved,
                hazard_penalty_weight=cfg.confidence_hazard_weight
            )
        elif fusion_grid is not None:
            corridor = self._generate_provisional_corridor(start, goal, environment.size)
            terrain_conf = fusion_grid.compute_path_confidence(
                path=corridor,
                current_idx=0,
                lookahead_distance_m=cfg.confidence_lookahead_m,
                resolution_m=environment.resolution_m,
                weight_observed_ground=cfg.weight_observed_ground,
                weight_observed_aerial=cfg.weight_observed_aerial,
                weight_inferred=cfg.weight_inferred,
                weight_unobserved=cfg.weight_unobserved,
                hazard_penalty_weight=cfg.confidence_hazard_weight
            )
        else:
            # When unobserved and no fusion grid provided, default confidence reflects unobserved corridor
            terrain_conf = 0.0

        # Apply Schmitt-trigger hysteresis to prevent decision flip-flopping
        # If scouting is active/requested, threshold moves up by hysteresis margin to clear
        # If driving normally, threshold moves down by hysteresis margin to trigger
        if is_scouting_active:
            effective_threshold = min(1.0, cfg.confidence_threshold + cfg.confidence_hysteresis)
        else:
            effective_threshold = max(0.0, cfg.confidence_threshold - cfg.confidence_hysteresis)

        # 3. Drone cost estimation for corridor flight
        scout_telemetry = self.drone_model.flight_energy(dist_direct * 1.15)
        scout_cost_wh = float(scout_telemetry["flight_energy_wh"])

        # 4. Expected rover blind traversal detour penalty
        expected_blind_detour_ratio = 1.0 + 1.6 * (complexity ** 1.5)
        nominal_rover_wh = dist_direct * 0.18
        rover_saving_wh = float(nominal_rover_wh * (expected_blind_detour_ratio - 1.0))
        net_benefit_wh = float(rover_saving_wh - scout_cost_wh)

        drone_energy_required = scout_cost_wh * (1.0 + cfg.drone_reserve_margin)

        # 5. GATED DECISION LOGIC
        # GATE 1: High Confidence Check (terrain is well-known -> skip scouting)
        if terrain_conf >= effective_threshold:
            rec = DecisionRecommendation.DRIVE_DIRECT
            gating_bypassed = True
            rationale = (
                f"High terrain confidence ({terrain_conf:.2f} >= threshold {cfg.confidence_threshold:.2f}, "
                f"hysteresis margin: {cfg.confidence_hysteresis:.2f}). Ground ahead is sufficiently known; "
                f"skipping aerial reconnaissance to conserve {scout_cost_wh:.1f} Wh drone flight energy "
                f"(expected net benefit: {net_benefit_wh:+.2f} Wh). Path uncertainty: {path_uncertainty:.2f}. "
                f"Direct rover driving is optimal."
            )
        # GATE 2: Low Confidence -> Evaluate existing scout-or-drive tradeoff & battery feasibility
        elif drone_battery_wh < drone_energy_required:
            rec = DecisionRecommendation.DRIVE_DIRECT
            gating_bypassed = False
            rationale = (
                f"Low terrain confidence ({terrain_conf:.2f} < threshold {cfg.confidence_threshold:.2f}) "
                f"indicates uncertain terrain ahead, but Drone battery is critically low ({drone_battery_wh:.1f} Wh "
                f"available vs {drone_energy_required:.1f} Wh required including {int(cfg.drone_reserve_margin*100)}% reserve). "
                f"Direct driving mandated to prevent drone stranding. Path uncertainty: {path_uncertainty:.2f}, "
                f"expected net benefit: {net_benefit_wh:+.2f} Wh, scout cost: {scout_cost_wh:.1f} Wh."
            )
        elif complexity > 0.45 or net_benefit_wh > 1.5:
            rec = DecisionRecommendation.SCOUT_RECOMMENDED
            gating_bypassed = False
            rationale = (
                f"Low terrain confidence ({terrain_conf:.2f} < threshold {cfg.confidence_threshold:.2f}) "
                f"with high terrain complexity ({complexity:.2f}), obstacle density ({obstacle_ratio*100:.1f}%), "
                f"and path uncertainty {path_uncertainty:.2f}. Aerial scouting requires {scout_cost_wh:.1f} Wh flight energy "
                f"and yields an expected net benefit of {net_benefit_wh:+.2f} Wh by avoiding severe rover backtracking."
            )
        elif complexity > 0.25:
            rec = DecisionRecommendation.TARGETED_SCOUT
            gating_bypassed = False
            rationale = (
                f"Low terrain confidence ({terrain_conf:.2f} < threshold {cfg.confidence_threshold:.2f}) "
                f"with moderate terrain complexity ({complexity:.2f}) and path uncertainty {path_uncertainty:.2f}. "
                f"Targeted chokepoint inspection requires {scout_cost_wh:.1f} Wh flight energy "
                f"and yields an expected net benefit of {net_benefit_wh:+.2f} Wh."
            )
        else:
            rec = DecisionRecommendation.DRIVE_DIRECT
            gating_bypassed = False
            rationale = (
                f"Low terrain confidence ({terrain_conf:.2f} < threshold {cfg.confidence_threshold:.2f}), "
                f"but terrain complexity ({complexity:.2f}) and obstacles are minimal with path uncertainty {path_uncertainty:.2f}. "
                f"Direct rover driving is optimal; deploying drone would incur {scout_cost_wh:.1f} Wh flight overhead "
                f"with expected net benefit of {net_benefit_wh:+.2f} Wh."
            )

        return DecisionAssessment(
            recommendation=rec,
            confidence_score=round(terrain_conf, 2),
            terrain_complexity_score=round(complexity, 2),
            path_uncertainty_score=round(path_uncertainty, 2),
            estimated_rover_saving_wh=round(rover_saving_wh, 2),
            estimated_scout_cost_wh=round(scout_cost_wh, 2),
            expected_net_benefit_wh=round(net_benefit_wh, 2),
            rational=rationale,
            terrain_confidence_score=round(terrain_conf, 2),
            confidence_threshold=round(cfg.confidence_threshold, 2),
            gating_bypassed=gating_bypassed
        )
