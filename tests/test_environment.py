"""
Tests for Planetary Environment Generation and Data Ingestion.
"""

import os
import numpy as np
import pytest
from core.environment import generate_planetary_terrain, PlanetaryEnvironment


def test_terrain_generation_dimensions():
    size = 60
    env = generate_planetary_terrain(size=size, seed=123, preset="Martian Crater Field")
    assert env.elevation.shape == (size, size)
    assert env.obstacles.shape == (size, size)
    assert not np.isnan(env.elevation).any()
    assert env.elevation.dtype == float


def test_start_and_goal_clearance():
    size = 70
    env = generate_planetary_terrain(size=size, seed=42)
    sy, sx = env.start
    gy, gx = env.goal

    # Check bounds
    assert 0 <= sy < size and 0 <= sx < size
    assert 0 <= gy < size and 0 <= gx < size
    assert env.start != env.goal

    # Start and goal must be clear of obstacles
    assert not env.obstacles[sy, sx]
    assert not env.obstacles[gy, gx]

    # Immediate 3x3 neighborhood around start must be clear
    assert not env.obstacles[max(0, sy-1):min(size, sy+2), max(0, sx-1):min(size, sx+2)].any()


def test_environment_save_and_load(tmp_path):
    env = generate_planetary_terrain(size=40, seed=99)
    save_file = str(tmp_path / "test_env.npz")
    env.save(save_file)
    assert os.path.exists(save_file)

    loaded = PlanetaryEnvironment.load(save_file)
    assert loaded.size == env.size
    assert loaded.name == env.name
    assert np.allclose(loaded.elevation, env.elevation)
    assert np.array_equal(loaded.obstacles, env.obstacles)
    assert loaded.start == env.start
    assert loaded.goal == env.goal
