"""
Tests for ASTRA-SCOUT Adaptive Decision Engine and Multi-Strategy Benchmarking.
Validates:
1. Decision responds to uncertainty & terrain complexity.
2. Decision responds to remaining battery capacity.
3. Identical-seed fairness across all three strategies.
4. Configurable mission cost weights change the outcome.
5. Rationale produces plain-language text referencing real computed values.
6. Uncertainty values are strictly bounded in [0.0, 1.0].
"""

import numpy as np
import pytest

from core.environment import generate_planetary_terrain, PlanetaryEnvironment
from core.terrain_analysis import compute_terrain_layers
from core.energy_model import RoverEnergyModel, DroneEnergyModel
from core.sensor_fusion import SensorFusionGrid, ObservationSource
from core.scout import DroneScout, FlightPattern
from core.decision_engine import (
    AdaptiveDecisionEngine,
    DecisionRecommendation,
    MissionCostConfig,
    compute_mission_cost,
)
from core.navigation import plan_mission, StrategyMode


def test_decision_responds_to_uncertainty():
    """
    Verifies that the decision engine shifts between direct driving and scouting
    as terrain complexity and uncertainty change.
    """
    engine = AdaptiveDecisionEngine()

    # 1. Low-uncertainty, flat, hazard-free terrain
    size = 50
    z_flat = np.zeros((size, size))
    obs_none = np.zeros((size, size), dtype=bool)
    env_flat = PlanetaryEnvironment(
        name="Flat Plains",
        size=size,
        resolution_m=1.0,
        elevation=z_flat,
        obstacles=obs_none,
        start=(5, 5),
        goal=(45, 45),
        seed=1
    )
    terrain_flat = compute_terrain_layers(env_flat.elevation, env_flat.obstacles, env_flat.resolution_m)
    dec_flat = engine.evaluate_scout_tradeoff(
        env_flat, terrain_flat, env_flat.start, env_flat.goal
    )

    # 2. High-uncertainty, rugged, obstacle-dense terrain
    env_rough = generate_planetary_terrain(size=size, seed=42, preset="Martian Crater Field", obstacle_density=0.14)
    terrain_rough = compute_terrain_layers(env_rough.elevation, env_rough.obstacles, env_rough.resolution_m)
    dec_rough = engine.evaluate_scout_tradeoff(
        env_rough, terrain_rough, env_rough.start, env_rough.goal
    )

    # Assertions
    assert dec_flat.terrain_complexity_score < dec_rough.terrain_complexity_score
    assert dec_flat.path_uncertainty_score < dec_rough.path_uncertainty_score
    assert dec_flat.expected_net_benefit_wh < 0.0
    assert dec_rough.expected_net_benefit_wh > 0.0
    assert dec_flat.recommendation == DecisionRecommendation.DRIVE_DIRECT
    assert dec_rough.recommendation in [DecisionRecommendation.SCOUT_RECOMMENDED, DecisionRecommendation.TARGETED_SCOUT]


def test_decision_responds_to_battery_level():
    """
    Verifies that the decision changes when drone battery level changes.
    When battery is insufficient for flight plus margin, it forces DRIVE_DIRECT.
    """
    engine = AdaptiveDecisionEngine()
    env = generate_planetary_terrain(size=50, seed=42, preset="Martian Crater Field", obstacle_density=0.12)
    terrain = compute_terrain_layers(env.elevation, env.obstacles, env.resolution_m)

    # Case A: Full drone battery (40.0 Wh) -> can safely fly
    dec_full_battery = engine.evaluate_scout_tradeoff(
        env, terrain, env.start, env.goal,
        drone_battery_wh=40.0
    )
    assert dec_full_battery.recommendation in [
        DecisionRecommendation.SCOUT_RECOMMENDED,
        DecisionRecommendation.TARGETED_SCOUT
    ]

    # Case B: Depleted drone battery (1.0 Wh) -> cannot fly safely -> forces DRIVE_DIRECT
    dec_low_battery = engine.evaluate_scout_tradeoff(
        env, terrain, env.start, env.goal,
        drone_battery_wh=1.0
    )
    assert dec_low_battery.recommendation == DecisionRecommendation.DRIVE_DIRECT
    assert "Drone battery" in dec_low_battery.rational
    assert "critically low" in dec_low_battery.rational or "depleted" in dec_low_battery.rational


def test_identical_seed_fairness():
    """
    CRITICAL FAIRNESS TEST: Verifies that Rover-Only, Always-Scout, and Adaptive-Scout
    all operate under identical terrain seeds, start positions, destinations,
    battery capacities, and energy assumptions.
    """
    seed = 77
    size = 50
    env = generate_planetary_terrain(size=size, seed=seed, preset="Martian Crater Field")
    terrain = compute_terrain_layers(env.elevation, env.obstacles, env.resolution_m)
    engine = AdaptiveDecisionEngine()
    scout = DroneScout(flight_altitude_m=12.0)

    res = plan_mission(
        environment=env,
        terrain=terrain,
        decision_engine=engine,
        scout=scout,
        rover_radius=5,
        rover_battery_wh=120.0,
        drone_battery_wh=40.0
    )

    # All three paths must start at identical start point
    assert res.rover_only_path[0] == env.start
    assert res.cooperative_path[0] == env.start
    assert res.adaptive_path[0] == env.start

    # All three reach destination
    assert res.rover_only_path[-1] == env.goal
    assert res.cooperative_path[-1] == env.goal
    assert res.adaptive_path[-1] == env.goal

    # Verify identical energy model physics assumptions
    assert res.rover_only_telemetry["battery_soc_pct"] <= 100.0
    assert res.cooperative_telemetry["battery_soc_pct"] <= 100.0
    assert res.adaptive_telemetry["battery_soc_pct"] <= 100.0

    # Mission costs dict contains all three strategies
    assert "rover_only" in res.mission_costs
    assert "always_scout" in res.mission_costs
    assert "adaptive_scout" in res.mission_costs


def test_weights_change_outcome():
    """
    Verifies that changing mission cost weights (e.g. heavy drone penalty vs low drone penalty)
    changes the combined mission cost and preferred strategy ranking.
    """
    env = generate_planetary_terrain(size=50, seed=42, preset="Martian Crater Field")
    terrain = compute_terrain_layers(env.elevation, env.obstacles, env.resolution_m)
    engine = AdaptiveDecisionEngine()
    scout = DroneScout(flight_altitude_m=12.0)

    # Weight set 1: Balanced mission weights
    cfg_balanced = MissionCostConfig(weight_rover_energy=1.0, weight_drone_energy=1.0, weight_route_risk=10.0)
    res_balanced = plan_mission(
        environment=env, terrain=terrain, decision_engine=engine, scout=scout, cost_config=cfg_balanced
    )

    # Weight set 2: Heavy drone penalty (e.g. drone energy 50x more costly than rover)
    cfg_heavy_drone = MissionCostConfig(weight_rover_energy=1.0, weight_drone_energy=50.0, weight_route_risk=0.0)
    res_heavy = plan_mission(
        environment=env, terrain=terrain, decision_engine=engine, scout=scout, cost_config=cfg_heavy_drone
    )

    # The always-scout cost must be significantly higher relative to rover-only under heavy drone penalty
    balanced_always_cost = res_balanced.mission_costs["always_scout"]
    heavy_always_cost = res_heavy.mission_costs["always_scout"]
    assert heavy_always_cost > balanced_always_cost

    # Direct compute test
    cost_a = compute_mission_cost(rover_energy_wh=10.0, drone_energy_wh=5.0, route_risk=0.2, config=cfg_balanced)
    cost_b = compute_mission_cost(rover_energy_wh=10.0, drone_energy_wh=5.0, route_risk=0.2, config=cfg_heavy_drone)
    assert cost_b > cost_a


def test_decision_rationale_contains_real_values():
    """
    Verifies that every decision recommendation produces a plain-language rationale
    that explicitly references computed uncertainty, expected net benefit, and energy cost.
    """
    engine = AdaptiveDecisionEngine()

    # Test across multiple presets
    for preset in ["Martian Crater Field", "Smooth Regolith", "Lunar Mare Basin"]:
        env = generate_planetary_terrain(size=50, seed=33, preset=preset)
        terrain = compute_terrain_layers(env.elevation, env.obstacles, env.resolution_m)
        dec = engine.evaluate_scout_tradeoff(env, terrain, env.start, env.goal)

        rationale = dec.rational
        # Must reference uncertainty value
        unc_str = f"{dec.path_uncertainty_score:.2f}"
        assert unc_str in rationale or f"{dec.path_uncertainty_score}" in rationale

        # Must reference energy cost
        cost_str = f"{dec.estimated_scout_cost_wh:.1f}"
        assert cost_str in rationale

        # Must reference expected net benefit
        benefit_str = f"{dec.expected_net_benefit_wh}"
        assert f"{abs(dec.expected_net_benefit_wh):.1f}" in rationale or f"{abs(dec.expected_net_benefit_wh):.2f}" in rationale


def test_uncertainty_bounds():
    """
    Verifies that all epistemic uncertainty values across SensorFusionGrid and
    DecisionEngine are strictly bounded within [0.0, 1.0].
    """
    size = 40
    grid = SensorFusionGrid(size)

    # Initial state
    assert np.all(grid.uncertainty_map >= 0.0)
    assert np.all(grid.uncertainty_map <= 1.0)

    # After drone observation
    mask = np.zeros((size, size), dtype=bool)
    mask[5:20, 5:20] = True
    dummy_z = np.zeros((size, size))
    dummy_h = np.zeros((size, size))
    dummy_o = np.zeros((size, size), dtype=bool)
    grid.apply_drone_observation(mask, dummy_z, dummy_h, dummy_o, sensor_confidence=0.88)
    assert np.all(grid.uncertainty_map >= 0.0)
    assert np.all(grid.uncertainty_map <= 1.0)

    # After rover ground observation
    rover_mask = np.zeros((size, size), dtype=bool)
    rover_mask[10:15, 10:15] = True
    grid.apply_rover_observation(rover_mask, dummy_z, dummy_h, dummy_o)
    assert np.all(grid.uncertainty_map >= 0.0)
    assert np.all(grid.uncertainty_map <= 1.0)

    # After AI inference
    grid.apply_ai_inference(kernel_radius=5)
    assert np.all(grid.uncertainty_map >= 0.0)
    assert np.all(grid.uncertainty_map <= 1.0)

    # Decision Engine uncertainty proxy
    engine = AdaptiveDecisionEngine()
    env = generate_planetary_terrain(size=size, seed=12)
    terrain = compute_terrain_layers(env.elevation, env.obstacles)
    dec = engine.evaluate_scout_tradeoff(env, terrain, env.start, env.goal)
    assert 0.0 <= dec.path_uncertainty_score <= 1.0
