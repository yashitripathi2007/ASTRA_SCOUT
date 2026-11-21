"""
Tests for Drone Scout Trajectory, Coverage, and Sensor Fusion Layering.
"""

import numpy as np
import pytest
from core.environment import generate_planetary_terrain
from core.sensor_fusion import SensorFusionGrid, ObservationSource
from core.scout import DroneScout, FlightPattern


def test_distinguishable_sensor_fusion_layers():
    """
    CRITICAL TEST: Verifies that drone aerial, rover ground, AI inferred,
    and unobserved cells maintain distinct IDs and confidence properties.
    """
    size = 50
    grid = SensorFusionGrid(size)

    # Initial state must be completely unobserved
    assert np.all(grid.source_map == ObservationSource.UNOBSERVED)
    assert np.all(grid.uncertainty_map == 1.0)

    # Apply Drone Aerial Observation to left half
    drone_mask = np.zeros((size, size), dtype=bool)
    drone_mask[:, :25] = True
    dummy_z = np.ones((size, size))
    dummy_haz = np.full((size, size), 0.2)
    dummy_obs = np.zeros((size, size), dtype=bool)

    grid.apply_drone_observation(drone_mask, dummy_z, dummy_haz, dummy_obs, sensor_confidence=0.85)

    # Left half must be AERIAL_SCOUT
    assert np.all(grid.source_map[:, :25] == ObservationSource.AERIAL_SCOUT)
    assert np.allclose(grid.uncertainty_map[:, :25], 0.15)
    # Right half still UNOBSERVED
    assert np.all(grid.source_map[:, 25:] == ObservationSource.UNOBSERVED)

    # Apply Rover Ground Observation to small center patch
    rover_mask = np.zeros((size, size), dtype=bool)
    rover_mask[20:30, 20:30] = True
    grid.apply_rover_observation(rover_mask, dummy_z, dummy_haz, dummy_obs)

    # Ground truth overrides aerial observations and unobserved
    assert np.all(grid.source_map[20:30, 20:30] == ObservationSource.ROVER_GROUND)
    assert np.all(grid.uncertainty_map[20:30, 20:30] == 0.0)

    # Apply AI inference
    grid.apply_ai_inference(kernel_radius=4)
    # Neighboring cells should be marked INFERRED
    inferred_count = np.sum(grid.source_map == ObservationSource.INFERRED)
    assert inferred_count > 0

    # Ensure all four categories exist and are distinguishable
    breakdown = grid.get_coverage_breakdown()
    assert breakdown["aerial_scout_pct"] > 0
    assert breakdown["rover_ground_pct"] > 0
    assert breakdown["inferred_pct"] > 0
    assert breakdown["unobserved_pct"] > 0


def test_drone_scout_flight_and_energy():
    env = generate_planetary_terrain(size=50, seed=15)
    scout = DroneScout(flight_altitude_m=12.0)
    waypoints = scout.generate_flight_path(env.start, env.goal, env.size, FlightPattern.CORRIDOR)

    assert len(waypoints) >= 2
    for wp in waypoints:
        assert 0 <= wp[0] < env.size
        assert 0 <= wp[1] < env.size

    telemetry = scout.energy_model.flight_energy(flight_distance_m=100.0, hover_duration_s=15.0)
    assert telemetry["flight_energy_wh"] > 0.0
    assert telemetry["battery_remaining_pct"] > 0.0
