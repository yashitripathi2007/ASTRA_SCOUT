"""
Tests for Physics-Based Energy Models (Rover tractive forces & Incline Asymmetry).
"""

import numpy as np
import pytest
from core.energy_model import RoverEnergyModel, DroneEnergyModel


def test_rover_incline_energy_asymmetry():
    """
    CRITICAL TEST: Driving uphill against gravity must consume significantly more energy
    than flat ground, which in turn consumes more than downhill.
    """
    model = RoverEnergyModel(mass_kg=120.0, nominal_speed_ms=0.15)
    p1 = (10, 10)
    p2 = (10, 11)  # 1 meter horizontal step

    # 1. Flat travel (delta_z = 0)
    flat_step = model.step_energy(p1, p2, z1=0.0, z2=0.0, roughness=0.1, hazard=0.1)

    # 2. Uphill travel (delta_z = +0.3m, ~17 deg slope)
    uphill_step = model.step_energy(p1, p2, z1=0.0, z2=0.3, roughness=0.1, hazard=0.1)

    # 3. Downhill travel (delta_z = -0.3m)
    downhill_step = model.step_energy(p1, p2, z1=0.3, z2=0.0, roughness=0.1, hazard=0.1)

    # Verification of physics
    assert uphill_step["energy_wh"] > flat_step["energy_wh"]
    assert flat_step["energy_wh"] > downhill_step["energy_wh"]
    assert uphill_step["power_w"] > flat_step["power_w"]


def test_drone_energy_monotonicity():
    model = DroneEnergyModel()
    short_flight = model.flight_energy(flight_distance_m=50.0)
    long_flight = model.flight_energy(flight_distance_m=150.0)

    assert long_flight["flight_energy_wh"] > short_flight["flight_energy_wh"]
    assert long_flight["flight_time_s"] > short_flight["flight_time_s"]
    assert long_flight["battery_remaining_pct"] < short_flight["battery_remaining_pct"]
