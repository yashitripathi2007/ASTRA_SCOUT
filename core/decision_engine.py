"""
Adaptive Decision Engine for ASTRA-SCOUT.
Evaluates the core question: "Scout or Drive?"
Computes Expected Information Gain (EIG), Energy Trade-off ROI, and Battery Feasibility
to determine whether deploying the aerial scout drone pays off.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple, Optional
import numpy as np
from .energy_model import DroneEnergyModel, RoverEnergyModel


class DecisionRecommendation(str, Enum):
    SCOUT_RECOMMENDED = "Deploy Scout (High Value)"
    DRIVE_DIRECT = "Drive Direct (Low Risk)"
    TARGETED_SCOUT = "Targeted Chokepoint Scout"


@dataclass
class MissionCostConfig:
    """Configurable weights and constraints for multi-objective mission cost."""
    weight_rover_energy: float = 1.0     # Multiplier for rover surface energy (Wh)
    weight_drone_energy: float = 1.0     # Multiplier for drone flight overhead (Wh)
    weight_route_risk: float = 15.0      # Multiplier for cumulative route hazard exposure
    drone_reserve_margin: float = 0.20   # Required safety reserve buffer for drone battery (20%)
    critical_rover_battery_pct: float = 25.0  # Low rover battery threshold


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
    confidence_score: float              # 0.0 to 1.0
    terrain_complexity_score: float      # 0.0 to 1.0 (roughness + slope variance + obstacle density)
    path_uncertainty_score: float        # Shannon entropy / ignorance proxy in mission sector [0, 1]
    estimated_rover_saving_wh: float     # Estimated rover energy saved by avoiding detours
    estimated_scout_cost_wh: float       # Drone flight energy required
    expected_net_benefit_wh: float       # Net energy benefit (+ = savings, - = overhead)
    rational: str                        # Human-interpretable rationale for the crew

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

    def evaluate_scout_tradeoff(
        self,
        environment,
        terrain,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        rover_battery_wh: float = 120.0,
        drone_battery_wh: float = 40.0,
        cost_config: Optional[MissionCostConfig] = None
    ) -> DecisionAssessment:
        """
        Analyzes terrain corridor, obstacle cluster density, slope variance, and battery levels
        to recommend whether aerial scouting should be launched.
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

        # 2. Drone cost estimation for corridor flight
        scout_telemetry = self.drone_model.flight_energy(dist_direct * 1.15)
        scout_cost_wh = float(scout_telemetry["flight_energy_wh"])

        # 3. Expected rover blind traversal detour penalty
        expected_blind_detour_ratio = 1.0 + 1.6 * (complexity ** 1.5)
        nominal_rover_wh = dist_direct * 0.18
        rover_saving_wh = float(nominal_rover_wh * (expected_blind_detour_ratio - 1.0))

        net_benefit_wh = float(rover_saving_wh - scout_cost_wh)

        # 4. Formulate Recommendation factoring in battery capacity and safety margins
        drone_energy_required = scout_cost_wh * (1.0 + cfg.drone_reserve_margin)

        if drone_battery_wh < drone_energy_required:
            # Insufficient drone battery to complete flight safely
            rec = DecisionRecommendation.DRIVE_DIRECT
            conf = 0.99
            rationale = (
                f"Drone battery critically low ({drone_battery_wh:.1f} Wh available vs {drone_energy_required:.1f} Wh required "
                f"including {int(cfg.drone_reserve_margin*100)}% reserve). Direct driving mandated to prevent drone stranding. "
                f"Corridor uncertainty: {path_uncertainty:.2f}, expected net benefit: {net_benefit_wh:+.2f} Wh, scout cost: {scout_cost_wh:.1f} Wh."
            )
        elif complexity > 0.45 or net_benefit_wh > 1.5:
            rec = DecisionRecommendation.SCOUT_RECOMMENDED
            conf = min(0.98, 0.65 + 0.35 * complexity)
            rationale = (
                f"High terrain complexity ({complexity:.2f}) and obstacle density ({obstacle_ratio*100:.1f}%) "
                f"with path uncertainty {path_uncertainty:.2f}. Aerial scouting requires {scout_cost_wh:.1f} Wh flight energy "
                f"and yields an expected net benefit of {net_benefit_wh:+.2f} Wh by avoiding severe rover backtracking."
            )
        elif complexity > 0.25:
            rec = DecisionRecommendation.TARGETED_SCOUT
            conf = 0.75
            rationale = (
                f"Moderate terrain complexity ({complexity:.2f}) with path uncertainty {path_uncertainty:.2f}. "
                f"Targeted chokepoint inspection requires {scout_cost_wh:.1f} Wh flight energy "
                f"and yields an expected net benefit of {net_benefit_wh:+.2f} Wh."
            )
        else:
            rec = DecisionRecommendation.DRIVE_DIRECT
            conf = min(0.95, 0.70 + 0.30 * (1.0 - complexity))
            rationale = (
                f"Low terrain complexity ({complexity:.2f}) and minimal obstacles with path uncertainty {path_uncertainty:.2f}. "
                f"Direct rover driving is optimal; deploying drone would incur {scout_cost_wh:.1f} Wh flight overhead "
                f"with expected net benefit of {net_benefit_wh:+.2f} Wh."
            )

        return DecisionAssessment(
            recommendation=rec,
            confidence_score=round(conf, 2),
            terrain_complexity_score=round(complexity, 2),
            path_uncertainty_score=round(path_uncertainty, 2),
            estimated_rover_saving_wh=round(rover_saving_wh, 2),
            estimated_scout_cost_wh=round(scout_cost_wh, 2),
            expected_net_benefit_wh=round(net_benefit_wh, 2),
            rational=rationale
        )
