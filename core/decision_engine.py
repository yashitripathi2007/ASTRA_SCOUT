"""
Adaptive Decision Engine for ASTRA-SCOUT.
Evaluates the core question: "Scout or Drive?"
Computes Expected Information Gain (EIG) and Energy Trade-off ROI to determine
whether deploying the aerial scout drone pays off.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple
import numpy as np
from .energy_model import DroneEnergyModel, RoverEnergyModel


class DecisionRecommendation(str, Enum):
    SCOUT_RECOMMENDED = "Deploy Scout (High Value)"
    DRIVE_DIRECT = "Drive Direct (Low Risk)"
    TARGETED_SCOUT = "Targeted Chokepoint Scout"


@dataclass
class DecisionAssessment:
    recommendation: DecisionRecommendation
    confidence_score: float              # 0.0 to 1.0
    terrain_complexity_score: float      # 0.0 to 1.0 (roughness + slope variance + obstacle density)
    path_uncertainty_score: float        # Shannon entropy / ignorance proxy in mission sector
    estimated_rover_saving_wh: float     # Estimated rover energy saved by avoiding detours
    estimated_scout_cost_wh: float       # Drone flight energy required
    expected_net_benefit_wh: float       # Net energy benefit (+ = savings, - = overhead)
    rational: str                        # Human-interpretable rationale for the crew


class AdaptiveDecisionEngine:
    """Intelligent decision-theoretic arbiter for collaborative exploration."""

    def __init__(
        self,
        rover_model: RoverEnergyModel = None,
        drone_model: DroneEnergyModel = None
    ):
        self.rover_model = rover_model or RoverEnergyModel()
        self.drone_model = drone_model or DroneEnergyModel()

    def evaluate_scout_tradeoff(
        self,
        environment,
        terrain,
        start: Tuple[int, int],
        goal: Tuple[int, int]
    ) -> DecisionAssessment:
        """
        Analyzes terrain corridor, obstacle cluster density, and slope variance
        to recommend whether aerial scouting should be launched.
        """
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
        complexity = np.clip(
            0.45 * (obstacle_ratio / 0.12) +
            0.35 * (mean_hazard / 0.40) +
            0.20 * min(1.0, slope_variance / 40.0),
            0.0, 1.0
        )

        # Path uncertainty proxy
        path_uncertainty = float(np.clip(0.5 + 0.5 * complexity, 0.0, 1.0))

        # 2. Drone cost estimation for corridor flight
        scout_telemetry = self.drone_model.flight_energy(dist_direct * 1.15)
        scout_cost_wh = scout_telemetry["flight_energy_wh"]

        # 3. Expected rover blind traversal detour penalty
        # Higher complexity causes more blind obstacle encounters and costly backtracking
        expected_blind_detour_ratio = 1.0 + 1.6 * (complexity ** 1.5)
        nominal_rover_wh = dist_direct * 0.18
        rover_saving_wh = nominal_rover_wh * (expected_blind_detour_ratio - 1.0)

        net_benefit_wh = rover_saving_wh - scout_cost_wh

        # 4. Formulate Recommendation
        if complexity > 0.45 or net_benefit_wh > 1.5:
            rec = DecisionRecommendation.SCOUT_RECOMMENDED
            conf = min(0.98, 0.65 + 0.35 * complexity)
            rationale = (
                f"High terrain complexity ({complexity:.2f}) and obstacle density ({obstacle_ratio*100:.1f}%) "
                f"indicate that an unguided rover risks multiple impassable dead-ends and severe detour penalties. "
                f"Aerial scouting saves an estimated {rover_saving_wh:.1f} Wh of rover backtracking."
            )
        elif complexity > 0.25:
            rec = DecisionRecommendation.TARGETED_SCOUT
            conf = 0.75
            rationale = (
                f"Moderate terrain complexity ({complexity:.2f}). A narrow corridor sweep or targeted chokepoint "
                f"inspection will secure a safe passage without paying the full energy cost of wide-area mapping."
            )
        else:
            rec = DecisionRecommendation.DRIVE_DIRECT
            conf = min(0.95, 0.70 + 0.30 * (1.0 - complexity))
            rationale = (
                f"Low terrain complexity ({complexity:.2f}) and minimal obstacles. "
                f"Direct rover driving is optimal; deploying the drone would incur an unnecessary "
                f"{scout_cost_wh:.1f} Wh flight overhead."
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
