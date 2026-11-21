"""
Tests for Path Planning and Dynamic Mission Simulation.
"""

import numpy as np
import pytest
from core.environment import generate_planetary_terrain
from core.terrain_analysis import compute_terrain_layers
from core.sensor_fusion import SensorFusionGrid, ObservationSource
from core.navigation import PathPlanner, plan_mission
from core.decision_engine import AdaptiveDecisionEngine
from core.scout import DroneScout


def test_planner_finds_path():
    size = 40
    env = generate_planetary_terrain(size=size, seed=7)
    terrain = compute_terrain_layers(env.elevation, env.obstacles)
    fusion = SensorFusionGrid(size)
    fusion.apply_rover_observation(
        np.ones((size, size), dtype=bool),
        env.elevation,
        terrain.hazard_index,
        env.obstacles
    )

    planner = PathPlanner()
    path, cost = planner.plan(fusion, env.elevation, env.start, env.goal)

    assert path is not None
    assert len(path) > 1
    assert path[0] == env.start
    assert path[-1] == env.goal
    assert cost < float("inf")

    # Path must not traverse confirmed obstacles
    for step in path:
        assert not env.obstacles[step]


def test_dynamic_mission_simulation():
    env = generate_planetary_terrain(size=50, seed=42)
    terrain = compute_terrain_layers(env.elevation, env.obstacles)
    decision = AdaptiveDecisionEngine()
    scout = DroneScout()

    result = plan_mission(
        environment=env,
        terrain=terrain,
        decision_engine=decision,
        scout=scout,
        rover_radius=5
    )

    assert result.rover_only_path is not None
    assert result.cooperative_path is not None
    assert len(result.rover_only_path) > 0
    assert len(result.cooperative_path) > 0
    assert result.cooperative_path[0] == env.start
    assert result.cooperative_path[-1] == env.goal
    assert "recommendation" in result.decision_assessment
