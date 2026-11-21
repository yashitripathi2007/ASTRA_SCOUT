"""
ASTRA-SCOUT Core Package
AI-Powered Collaborative Rover-Drone Exploration System
"""

from .environment import PlanetaryEnvironment, generate_planetary_terrain
from .terrain_analysis import TerrainIntelligence, compute_terrain_layers
from .sensor_fusion import SensorFusionGrid, ObservationSource
from .energy_model import RoverEnergyModel, DroneEnergyModel
from .scout import DroneScout, FlightPattern
from .rover import RoverAgent
from .decision_engine import (
    AdaptiveDecisionEngine,
    DecisionRecommendation,
    MissionCostConfig,
    compute_mission_cost,
)
from .navigation import PathPlanner, plan_mission, StrategyMode, MissionResult

__all__ = [
    "PlanetaryEnvironment",
    "generate_planetary_terrain",
    "TerrainIntelligence",
    "compute_terrain_layers",
    "SensorFusionGrid",
    "ObservationSource",
    "RoverEnergyModel",
    "DroneEnergyModel",
    "DroneScout",
    "FlightPattern",
    "RoverAgent",
    "AdaptiveDecisionEngine",
    "DecisionRecommendation",
    "MissionCostConfig",
    "compute_mission_cost",
    "PathPlanner",
    "plan_mission",
    "StrategyMode",
    "MissionResult",
]
