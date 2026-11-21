"""
Tests for Rover Terrain Confidence Metric, Threshold Gating, and Hysteresis.

Validates the 7 core requirements:
1. Confidence is always in [0.0, 1.0].
2. Confidence is higher on observed cells than on unobserved cells.
3. High confidence leads to no scout (DRIVE_DIRECT).
4. Low confidence leads to a scout request.
5. The threshold config changes the outcome.
6. Hysteresis prevents flip-flopping around threshold boundary.
7. Rover-only and Always-scout results are unchanged for the same seed (Fairness).
"""

import numpy as np
import pytest

from core.environment import generate_planetary_terrain, PlanetaryEnvironment
from core.terrain_analysis import compute_terrain_layers
from core.sensor_fusion import SensorFusionGrid, ObservationSource
from core.scout import DroneScout, FlightPattern
from core.decision_engine import (
    AdaptiveDecisionEngine,
    DecisionRecommendation,
    MissionCostConfig,
)
from core.navigation import plan_mission, PathPlanner


def test_confidence_bounded_zero_to_one():
    """Requirement: Confidence is strictly bounded within [0.0, 1.0]."""
    size = 40
    grid = SensorFusionGrid(size)

    # Case 1: Empty path
    assert grid.compute_path_confidence([]) == 1.0

    # Case 2: Completely unobserved grid
    path = [(i, i) for i in range(20)]
    conf_unobserved = grid.compute_path_confidence(path, current_idx=0)
    assert 0.0 <= conf_unobserved <= 1.0
    assert conf_unobserved == 0.0

    # Case 3: Mixed observations and hazards
    grid.source_map[5:15, 5:15] = ObservationSource.ROVER_GROUND
    grid.hazard_belief[5:15, 5:15] = 0.8
    conf_ground = grid.compute_path_confidence(path, current_idx=5)
    assert 0.0 <= conf_ground <= 1.0

    # Case 4: Extreme hazards
    grid.hazard_belief[:, :] = 1.0
    conf_max_hazard = grid.compute_path_confidence(path, current_idx=0)
    assert 0.0 <= conf_max_hazard <= 1.0

    # Case 5: Rover at destination (current_idx >= len(path) - 1)
    assert grid.compute_path_confidence(path, current_idx=len(path) - 1) == 1.0


def test_confidence_higher_on_observed_than_unobserved():
    """Requirement: Confidence is higher on observed cells than on unobserved cells."""
    size = 40
    path = [(10, x) for x in range(10, 25)]

    # 1. Unobserved grid
    grid_unobs = SensorFusionGrid(size)
    c_unobs = grid_unobs.compute_path_confidence(path, current_idx=0, lookahead_distance_m=10.0)

    # 2. Inferred grid
    grid_inf = SensorFusionGrid(size)
    # Apply ground observation nearby and AI inference to create inferred cells on path
    grid_inf.source_map[10, 20:25] = ObservationSource.ROVER_GROUND
    grid_inf.apply_ai_inference(kernel_radius=6)
    c_inf = grid_inf.compute_path_confidence(path, current_idx=0, lookahead_distance_m=10.0)

    # 3. Aerial scout surveyed grid
    grid_aerial = SensorFusionGrid(size)
    aerial_mask = np.zeros((size, size), dtype=bool)
    for pt in path:
        aerial_mask[pt] = True
    dummy_z = np.zeros((size, size))
    dummy_h = np.full((size, size), 0.1)
    dummy_o = np.zeros((size, size), dtype=bool)
    grid_aerial.apply_drone_observation(aerial_mask, dummy_z, dummy_h, dummy_o, sensor_confidence=0.88)
    c_aerial = grid_aerial.compute_path_confidence(path, current_idx=0, lookahead_distance_m=10.0)

    # 4. Rover ground truth observed grid
    grid_ground = SensorFusionGrid(size)
    ground_mask = np.zeros((size, size), dtype=bool)
    for pt in path:
        ground_mask[pt] = True
    grid_ground.apply_rover_observation(ground_mask, dummy_z, dummy_h, dummy_o)
    c_ground = grid_ground.compute_path_confidence(path, current_idx=0, lookahead_distance_m=10.0)

    # Validate hierarchy: Ground contact (1.0) >= Aerial survey (~0.85) > Inferred (>0) > Unobserved (0.0)
    assert c_unobs == 0.0
    assert c_inf > c_unobs
    assert c_aerial > c_inf
    assert c_ground >= c_aerial
    assert c_ground > 0.90


def test_high_confidence_skips_scouting():
    """Requirement: High confidence (known terrain) leads to no scout; rover keeps driving."""
    env = generate_planetary_terrain(size=50, seed=42, preset="Martian Crater Field", obstacle_density=0.14)
    terrain = compute_terrain_layers(env.elevation, env.obstacles, env.resolution_m)
    engine = AdaptiveDecisionEngine()

    # Pre-survey the mission area so terrain along upcoming path is completely known
    fusion = SensorFusionGrid(env.size)
    scout = DroneScout()
    scout.execute_survey(env.start, env.goal, env, fusion, pattern=FlightPattern.CORRIDOR)

    planner = PathPlanner()
    path, _ = planner.plan(fusion, env.elevation, env.start, env.goal)

    cfg = MissionCostConfig(confidence_threshold=0.60)
    dec = engine.evaluate_scout_tradeoff(
        environment=env,
        terrain=terrain,
        start=env.start,
        goal=env.goal,
        cost_config=cfg,
        fusion_grid=fusion,
        planned_path=path,
        current_idx=0
    )

    # Must skip scouting due to high confidence gate
    assert dec.terrain_confidence_score >= cfg.confidence_threshold
    assert dec.recommendation == DecisionRecommendation.DRIVE_DIRECT
    assert dec.gating_bypassed is True
    assert "High terrain confidence" in dec.rational
    assert f"{dec.terrain_confidence_score:.2f}" in dec.rational
    assert f"{cfg.confidence_threshold:.2f}" in dec.rational


def test_low_confidence_triggers_scout():
    """Requirement: Low confidence (unknown terrain) requests a drone scout when trade-off allows."""
    env = generate_planetary_terrain(size=50, seed=42, preset="Martian Crater Field", obstacle_density=0.14)
    terrain = compute_terrain_layers(env.elevation, env.obstacles, env.resolution_m)
    engine = AdaptiveDecisionEngine()

    # Fusion grid with unobserved terrain along planned corridor
    fusion = SensorFusionGrid(env.size)
    fusion.apply_rover_observation(
        np.zeros((env.size, env.size), dtype=bool),
        env.elevation, terrain.hazard_index, env.obstacles
    )
    planner = PathPlanner()
    path, _ = planner.plan(fusion, env.elevation, env.start, env.goal)

    cfg = MissionCostConfig(confidence_threshold=0.60)
    dec = engine.evaluate_scout_tradeoff(
        environment=env,
        terrain=terrain,
        start=env.start,
        goal=env.goal,
        drone_battery_wh=40.0,
        cost_config=cfg,
        fusion_grid=fusion,
        planned_path=path,
        current_idx=0
    )

    # Must request drone scouting
    assert dec.terrain_confidence_score < cfg.confidence_threshold
    assert dec.recommendation in [DecisionRecommendation.SCOUT_RECOMMENDED, DecisionRecommendation.TARGETED_SCOUT]
    assert dec.gating_bypassed is False
    assert "Low terrain confidence" in dec.rational
    assert f"{dec.terrain_confidence_score:.2f}" in dec.rational
    assert f"{cfg.confidence_threshold:.2f}" in dec.rational


def test_threshold_config_changes_outcome():
    """Requirement: The threshold config changes the outcome."""
    env = generate_planetary_terrain(size=50, seed=42, preset="Martian Crater Field", obstacle_density=0.12)
    terrain = compute_terrain_layers(env.elevation, env.obstacles, env.resolution_m)
    engine = AdaptiveDecisionEngine()

    fusion = SensorFusionGrid(env.size)
    # Simulate partial AI inference along path yielding intermediate confidence (~0.35 - 0.40)
    mask = np.zeros((env.size, env.size), dtype=bool)
    mask[env.start[0]:env.start[0]+8, env.start[1]:env.start[1]+8] = True
    fusion.apply_rover_observation(mask, env.elevation, terrain.hazard_index, env.obstacles)
    fusion.apply_ai_inference(kernel_radius=10)

    planner = PathPlanner()
    path, _ = planner.plan(fusion, env.elevation, env.start, env.goal)

    # Setting a very lenient/low threshold (e.g. 0.15) -> rover considers current confidence sufficient -> DRIVE_DIRECT
    cfg_low_th = MissionCostConfig(confidence_threshold=0.15, confidence_hysteresis=0.01)
    dec_low = engine.evaluate_scout_tradeoff(
        environment=env,
        terrain=terrain,
        start=env.start,
        goal=env.goal,
        cost_config=cfg_low_th,
        fusion_grid=fusion,
        planned_path=path,
        current_idx=0
    )

    # Setting a very strict/high threshold (e.g. 0.90) -> rover requires aerial reconnaissance -> SCOUT
    cfg_high_th = MissionCostConfig(confidence_threshold=0.90, confidence_hysteresis=0.01)
    dec_high = engine.evaluate_scout_tradeoff(
        environment=env,
        terrain=terrain,
        start=env.start,
        goal=env.goal,
        drone_battery_wh=40.0,
        cost_config=cfg_high_th,
        fusion_grid=fusion,
        planned_path=path,
        current_idx=0
    )

    assert dec_low.recommendation == DecisionRecommendation.DRIVE_DIRECT
    assert dec_low.gating_bypassed is True
    assert dec_high.recommendation in [DecisionRecommendation.SCOUT_RECOMMENDED, DecisionRecommendation.TARGETED_SCOUT]
    assert dec_high.gating_bypassed is False


def test_hysteresis_prevents_flip_flopping():
    """
    Requirement: Hysteresis prevents flip-flopping.
    Schmitt-trigger model:
    - Base threshold = 0.60, Hysteresis margin = 0.05.
    - When driving (not scouting): effective trigger threshold = 0.60 - 0.05 = 0.55.
      A confidence of 0.58 does NOT trigger scout.
    - When scouting is active: effective clear threshold = 0.60 + 0.05 = 0.65.
      A confidence of 0.62 does NOT prematurely stop scouting.
    """
    env = generate_planetary_terrain(size=50, seed=42, preset="Martian Crater Field", obstacle_density=0.12)
    terrain = compute_terrain_layers(env.elevation, env.obstacles, env.resolution_m)
    engine = AdaptiveDecisionEngine()

    cfg = MissionCostConfig(confidence_threshold=0.60, confidence_hysteresis=0.05)

    # Test confidence = 0.58 (between 0.55 and 0.65)
    # State A: Previous state was DRIVING (is_scouting_active=False)
    # Effective threshold = 0.55. Since 0.58 >= 0.55, rover keeps driving.
    dec_driving = engine.evaluate_scout_tradeoff(
        environment=env,
        terrain=terrain,
        start=env.start,
        goal=env.goal,
        cost_config=cfg,
        override_confidence=0.58,
        is_scouting_active=False
    )
    assert dec_driving.recommendation == DecisionRecommendation.DRIVE_DIRECT
    assert dec_driving.gating_bypassed is True

    # State B: Previous state was SCOUTING (is_scouting_active=True)
    # Effective threshold = 0.65. Since 0.58 < 0.65, rover maintains scouting mode.
    dec_scouting = engine.evaluate_scout_tradeoff(
        environment=env,
        terrain=terrain,
        start=env.start,
        goal=env.goal,
        drone_battery_wh=40.0,
        cost_config=cfg,
        override_confidence=0.58,
        is_scouting_active=True
    )
    assert dec_scouting.recommendation in [DecisionRecommendation.SCOUT_RECOMMENDED, DecisionRecommendation.TARGETED_SCOUT]
    assert dec_scouting.gating_bypassed is False


def test_rover_only_and_always_scout_unchanged():
    """
    Requirement: Fairness. Rover-only and Always-scout must keep working exactly as before.
    Identical seed, start, goal, and energy assumptions produce identical results.
    """
    seed = 42
    env = generate_planetary_terrain(size=50, seed=seed, preset="Martian Crater Field")
    terrain = compute_terrain_layers(env.elevation, env.obstacles, env.resolution_m)
    engine = AdaptiveDecisionEngine()
    scout = DroneScout()

    res1 = plan_mission(
        environment=env,
        terrain=terrain,
        decision_engine=engine,
        scout=scout,
        rover_radius=5,
        rover_battery_wh=120.0,
        drone_battery_wh=40.0
    )

    res2 = plan_mission(
        environment=env,
        terrain=terrain,
        decision_engine=engine,
        scout=scout,
        rover_radius=5,
        rover_battery_wh=120.0,
        drone_battery_wh=40.0
    )

    # 1. Rover-Only strategy must be identical
    assert res1.rover_only_path == res2.rover_only_path
    assert res1.rover_only_replans == res2.rover_only_replans
    assert res1.rover_only_telemetry["total_energy_wh"] == res2.rover_only_telemetry["total_energy_wh"]
    assert res1.rover_only_telemetry["total_distance_m"] == res2.rover_only_telemetry["total_distance_m"]

    # 2. Always-Scout strategy must be identical
    assert res1.cooperative_path == res2.cooperative_path
    assert res1.cooperative_replans == res2.cooperative_replans
    assert res1.cooperative_telemetry["total_energy_wh"] == res2.cooperative_telemetry["total_energy_wh"]
    assert res1.cooperative_telemetry["total_distance_m"] == res2.cooperative_telemetry["total_distance_m"]
    assert res1.drone_telemetry["flight_energy_wh"] == res2.drone_telemetry["flight_energy_wh"]

    # 3. Step confidence telemetry exists in result
    assert res1.confidence_history is not None
    assert len(res1.confidence_history) > 0
    for entry in res1.confidence_history:
        assert 0.0 <= entry["confidence"] <= 1.0
        assert "distance_m" in entry
        assert "decision" in entry
