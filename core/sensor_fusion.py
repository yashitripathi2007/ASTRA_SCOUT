"""
Sensor Fusion & Multi-Tier Knowledge Representation for ASTRA-SCOUT.

CRITICAL REQUIREMENT:
Distinguishes between:
1. What the drone scout observed from the air (wide-area, macro relief, medium confidence).
2. What the rover observed from the ground (local radius, micro obstacles, high confidence).
3. What the system has inferred / predicted via AI priors (spatial extrapolation, high uncertainty).
4. What remains completely unobserved.
"""

from enum import IntEnum
from typing import Dict, Tuple, List
import numpy as np


class ObservationSource(IntEnum):
    """Categorical source of terrain knowledge."""
    UNOBSERVED = 0     # Completely unknown terrain
    INFERRED = 1       # AI / statistical spatial prior with high uncertainty
    AERIAL_SCOUT = 2   # Drone aerial survey (wide footprint, macro hazards)
    ROVER_GROUND = 3   # Rover direct sensor observation (ground truth)


class SensorFusionGrid:
    """
    Maintains the multi-source belief map, uncertainty layers, and hazard assessments
    fusing aerial drone telemetry and rover surface perception.
    """

    def __init__(self, size: int):
        self.size = size
        # Knowledge source per cell (ObservationSource values)
        self.source_map = np.full((size, size), ObservationSource.UNOBSERVED, dtype=int)
        
        # Perceived terrain layers (what the navigation system knows)
        self.elevation_belief = np.zeros((size, size), dtype=float)
        self.hazard_belief = np.full((size, size), 0.30, dtype=float)  # Default caution prior
        self.obstacle_belief = np.zeros((size, size), dtype=bool)
        
        # Epistemic uncertainty grid: 1.0 = total ignorance, 0.0 = absolute ground truth
        self.uncertainty_map = np.ones((size, size), dtype=float)

    def apply_drone_observation(
        self,
        coverage_mask: np.ndarray,
        true_elevation: np.ndarray,
        true_hazard: np.ndarray,
        true_obstacles: np.ndarray,
        sensor_confidence: float = 0.85
    ) -> None:
        """
        Integrates aerial survey data from the drone scout.
        Overwrites UNOBSERVED and INFERRED cells with AERIAL_SCOUT data.
        Does NOT downgrade ROVER_GROUND cells because ground observations are ground truth.
        """
        update_mask = coverage_mask & (self.source_map != ObservationSource.ROVER_GROUND)

        self.source_map[update_mask] = ObservationSource.AERIAL_SCOUT
        self.elevation_belief[update_mask] = true_elevation[update_mask]
        
        # Aerial scout detects macro slopes and major boulder fields with slight sensor noise
        self.hazard_belief[update_mask] = true_hazard[update_mask]
        self.obstacle_belief[update_mask] = true_obstacles[update_mask]
        
        # Drone survey reduces uncertainty significantly
        self.uncertainty_map[update_mask] = 1.0 - sensor_confidence

    def apply_rover_observation(
        self,
        rover_footprint: np.ndarray,
        true_elevation: np.ndarray,
        true_hazard: np.ndarray,
        true_obstacles: np.ndarray
    ) -> None:
        """
        Integrates high-resolution surface sensor data (LiDAR, stereo cameras, tactile wheel slip)
        from the rover. Surface observations have highest authority (ground truth).
        """
        self.source_map[rover_footprint] = ObservationSource.ROVER_GROUND
        self.elevation_belief[rover_footprint] = true_elevation[rover_footprint]
        self.hazard_belief[rover_footprint] = true_hazard[rover_footprint]
        self.obstacle_belief[rover_footprint] = true_obstacles[rover_footprint]
        
        # Zero uncertainty for directly observed ground cells
        self.uncertainty_map[rover_footprint] = 0.0

    def apply_ai_inference(self, kernel_radius: int = 5) -> None:
        """
        Applies AI spatial interpolation / Gaussian process prior to unobserved cells
        surrounding observed terrain. Marks these cells as INFERRED.
        """
        observed_mask = (self.source_map == ObservationSource.AERIAL_SCOUT) | (
            self.source_map == ObservationSource.ROVER_GROUND
        )
        if not np.any(observed_mask):
            return

        from scipy.ndimage import distance_transform_edt, uniform_filter

        # Compute distance to nearest observed cell
        dist_to_observed = distance_transform_edt(~observed_mask)
        inferred_candidates = (~observed_mask) & (dist_to_observed <= kernel_radius)

        if np.any(inferred_candidates):
            # Smooth elevation and hazard extrapolation
            smoothed_z = uniform_filter(self.elevation_belief, size=kernel_radius * 2 + 1)
            smoothed_hazard = uniform_filter(self.hazard_belief, size=kernel_radius * 2 + 1)

            self.source_map[inferred_candidates] = ObservationSource.INFERRED
            self.elevation_belief[inferred_candidates] = smoothed_z[inferred_candidates]
            self.hazard_belief[inferred_candidates] = smoothed_hazard[inferred_candidates]
            
            # Uncertainty scales with distance from actual observation
            dist_scaled = np.clip(dist_to_observed[inferred_candidates] / float(kernel_radius), 0.0, 1.0)
            self.uncertainty_map[inferred_candidates] = 0.45 + 0.40 * dist_scaled

    @property
    def known_mask(self) -> np.ndarray:
        """Any cell that has been observed or inferred."""
        return self.source_map != ObservationSource.UNOBSERVED

    @property
    def aerial_mask(self) -> np.ndarray:
        """Cells observed exclusively by aerial drone scout."""
        return self.source_map == ObservationSource.AERIAL_SCOUT

    @property
    def ground_mask(self) -> np.ndarray:
        """Cells observed directly by rover ground sensors."""
        return self.source_map == ObservationSource.ROVER_GROUND

    @property
    def inferred_mask(self) -> np.ndarray:
        """Cells predicted by AI spatial extrapolation."""
        return self.source_map == ObservationSource.INFERRED

    @property
    def unobserved_mask(self) -> np.ndarray:
        """Cells with zero observations."""
        return self.source_map == ObservationSource.UNOBSERVED

    def compute_path_confidence(
        self,
        path: List[Tuple[int, int]],
        current_idx: int = 0,
        lookahead_distance_m: float = 15.0,
        resolution_m: float = 1.0,
        weight_observed_ground: float = 1.0,
        weight_observed_aerial: float = 0.88,
        weight_inferred: float = 0.40,
        weight_unobserved: float = 0.00,
        hazard_penalty_weight: float = 0.25
    ) -> float:
        """
        Computes the terrain confidence score in [0.0, 1.0] for upcoming cells along the path.

        CONCEPT:
        Quantifies "how well do I know the ground I am about to drive on?"
        Evaluated over a lookahead window W of upcoming path cells:
        W = path[current_idx + 1 : current_idx + 1 + N]
        where N = max(1, round(lookahead_distance_m / resolution_m)).

        FORMULA:
        For each upcoming cell i in W:
            1. Knowledge weight K_i based on provenance source:
               - ROVER_GROUND:  weight_observed_ground (1.00) -> direct ground-truth contact
               - AERIAL_SCOUT:  weight_observed_aerial (0.88) -> aerial survey observation
               - INFERRED:      weight_inferred        (0.40) -> AI spatial prior
               - UNOBSERVED:    weight_unobserved      (0.00) -> completely unknown
            2. Hazard modulation:
               h_i = hazard_belief[i]  in [0.0, 1.0]
               c_i = clip(K_i * (1.0 - hazard_penalty_weight * h_i), 0.0, 1.0)
            3. Window Aggregation:
               Confidence = (1 / |W|) * sum(c_i) if |W| > 0 else 1.0

        Returns:
            float in [0.0, 1.0]
        """
        if not path:
            return 1.0

        # Number of lookahead steps corresponding to lookahead distance
        num_steps = max(1, int(round(lookahead_distance_m / max(0.1, resolution_m))))

        # Slice upcoming path cells ahead of the current position
        if current_idx < len(path) - 1:
            upcoming_cells = path[current_idx + 1 : current_idx + 1 + num_steps]
        else:
            # At destination or single-point path
            return 1.0

        if not upcoming_cells:
            return 1.0

        weights_map = {
            ObservationSource.ROVER_GROUND: float(weight_observed_ground),
            ObservationSource.AERIAL_SCOUT: float(weight_observed_aerial),
            ObservationSource.INFERRED: float(weight_inferred),
            ObservationSource.UNOBSERVED: float(weight_unobserved),
        }

        cell_scores = []
        for y, x in upcoming_cells:
            if not (0 <= y < self.size and 0 <= x < self.size):
                continue
            src = self.source_map[y, x]
            k_weight = weights_map.get(src, 0.0)
            h_val = float(self.hazard_belief[y, x])
            # High hazard degrades traversability confidence even if cell was seen
            c_val = float(np.clip(k_weight * (1.0 - hazard_penalty_weight * h_val), 0.0, 1.0))
            cell_scores.append(c_val)

        if not cell_scores:
            return 1.0

        return float(np.clip(np.mean(cell_scores), 0.0, 1.0))

    def get_coverage_breakdown(self) -> Dict[str, float]:
        """Returns percentage breakdown of terrain knowledge categories."""
        total_cells = float(self.size * self.size)
        return {
            "unobserved_pct": 100.0 * np.sum(self.unobserved_mask) / total_cells,
            "inferred_pct": 100.0 * np.sum(self.inferred_mask) / total_cells,
            "aerial_scout_pct": 100.0 * np.sum(self.aerial_mask) / total_cells,
            "rover_ground_pct": 100.0 * np.sum(self.ground_mask) / total_cells,
            "total_known_pct": 100.0 * np.sum(self.known_mask) / total_cells,
            "mean_uncertainty": float(np.mean(self.uncertainty_map)),
        }
