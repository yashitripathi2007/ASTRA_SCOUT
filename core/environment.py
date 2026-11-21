"""
Planetary Environment Generator & Loader for ASTRA-SCOUT.
Models Martian and Lunar topography including impact craters, rolling hills,
ridge systems, and boulder fields with guaranteed traversal viability.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class PlanetaryEnvironment:
    """Represents a planetary terrain environment for rover-drone missions."""
    name: str
    size: int
    resolution_m: float           # Meters per grid cell (default 1.0m)
    elevation: np.ndarray         # 2D float array of elevations in meters
    obstacles: np.ndarray         # 2D boolean mask of impassable rocks/boulders
    start: Tuple[int, int]        # (y, x) start coordinates
    goal: Tuple[int, int]         # (y, x) goal coordinates
    seed: int

    @property
    def shape(self) -> Tuple[int, int]:
        return self.elevation.shape

    def save(self, filepath: str) -> None:
        """Saves environment data to a compressed numpy archive."""
        np.savez_compressed(
            filepath,
            name=self.name,
            size=self.size,
            resolution_m=self.resolution_m,
            elevation=self.elevation,
            obstacles=self.obstacles,
            start=np.array(self.start),
            goal=np.array(self.goal),
            seed=self.seed
        )

    @classmethod
    def load(cls, filepath: str) -> "PlanetaryEnvironment":
        """Loads environment from a saved numpy archive."""
        data = np.load(filepath)
        return cls(
            name=str(data["name"]),
            size=int(data["size"]),
            resolution_m=float(data["resolution_m"]),
            elevation=data["elevation"],
            obstacles=data["obstacles"],
            start=tuple(int(x) for x in data["start"]),
            goal=tuple(int(x) for x in data["goal"]),
            seed=int(data["seed"])
        )


def generate_planetary_terrain(
    size: int = 70,
    seed: int = 42,
    preset: str = "Martian Crater Field",
    resolution_m: float = 1.0,
    obstacle_density: float = 0.08,
    custom_start: Optional[Tuple[int, int]] = None,
    custom_goal: Optional[Tuple[int, int]] = None,
) -> PlanetaryEnvironment:
    """
    Generates realistic planetary digital elevation models (DEM) with geological features.

    Presets:
    - 'Martian Crater Field': Broad impact craters with raised rims and rocky ejecta.
    - 'Lunar Mare Basin': Rolling regolith undulations with basaltic boulder fields.
    - 'Rugged Canyon': Steeper tectonic rifts with challenging slopes.
    - 'Smooth Regolith': Low-hazard training terrain.
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size].astype(float)

    # Base macro-topography
    if preset == "Rugged Canyon":
        z = 4.5 * np.sin(x / 11.0) + 3.0 * np.cos(y / 8.0)
        # Deep canyon trench running diagonally
        canyon_axis = np.abs(y - 0.7 * x - 10)
        z -= 6.0 * np.exp(-(canyon_axis**2) / (2 * 4.5**2))
    elif preset == "Lunar Mare Basin":
        z = 1.2 * np.sin(x / 14.0) + 1.0 * np.cos(y / 16.0)
    elif preset == "Smooth Regolith":
        z = 0.8 * np.sin(x / 15.0) + 0.6 * np.cos(y / 15.0)
    else:  # Martian Crater Field (default)
        z = 2.2 * np.sin(x / 10.0) + 1.8 * np.cos(y / 12.0)

    # Add geological craters with raised rims
    num_craters = 4 if preset != "Smooth Regolith" else 2
    for _ in range(num_craters):
        cx, cy = rng.integers(8, size - 8, size=2)
        radius = rng.uniform(4.0, 10.0)
        depth = rng.uniform(2.5, 6.0)
        dist_sq = (x - cx)**2 + (y - cy)**2
        # Crater bowl (depression)
        bowl = depth * np.exp(-dist_sq / (2 * (radius * 0.75)**2))
        # Raised rim
        rim = (depth * 0.45) * np.exp(-((np.sqrt(dist_sq) - radius)**2) / (2 * 1.5**2))
        z += (rim - bowl)

    # Add mounds, dunes and hills
    num_hills = 6
    for _ in range(num_hills):
        cx, cy = rng.integers(6, size - 6, size=2)
        amp = rng.uniform(-3.5, 5.0)
        sigma = rng.uniform(4.0, 9.0)
        z += amp * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))

    # High-frequency geological regolith texture
    noise_sigma = 0.22 if preset != "Smooth Regolith" else 0.10
    z += rng.normal(0, noise_sigma, size=(size, size))

    # Obstacle generation (boulders, rock clusters)
    obstacles = np.zeros((size, size), dtype=bool)
    num_clusters = int(max(6, size * size * obstacle_density / 22.0))

    for _ in range(num_clusters):
        cx, cy = rng.integers(3, size - 3, size=2)
        cluster_radius = rng.integers(1, 3)
        # Cluster of 1 to 4 boulder cells
        cluster_mask = (x - cx)**2 + (y - cy)**2 <= cluster_radius**2
        # 75% probability of solid obstruction within cluster
        scatter = rng.uniform(0, 1, size=(size, size)) < 0.75
        obstacles |= (cluster_mask & scatter)

    # Impassable steep cliff obstacles where local grade is dangerous
    gy, gx = np.gradient(z)
    steep_slopes = np.degrees(np.arctan(np.sqrt(gx**2 + gy**2))) >= 38.0
    obstacles |= steep_slopes

    # Default Start and Goal locations
    start = custom_start if custom_start is not None else (size // 2, 4)
    goal = custom_goal if custom_goal is not None else (size // 2, size - 5)

    # Guarantee clearance in a 5x5 zone around start and goal
    def clear_zone(pt, r=2):
        y_min = max(0, pt[0] - r)
        y_max = min(size, pt[0] + r + 1)
        x_min = max(0, pt[1] - r)
        x_max = min(size, pt[1] + r + 1)
        obstacles[y_min:y_max, x_min:x_max] = False

    clear_zone(start)
    clear_zone(goal)

    # Carve a narrow safety passage if connectivity is completely blocked
    _ensure_reachability(obstacles, start, goal, z)

    return PlanetaryEnvironment(
        name=preset,
        size=size,
        resolution_m=resolution_m,
        elevation=z,
        obstacles=obstacles,
        start=start,
        goal=goal,
        seed=seed
    )


def _ensure_reachability(obstacles: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int], z: np.ndarray) -> None:
    """Verifies that at least one viable passage exists between start and goal."""
    from collections import deque
    h, w = obstacles.shape
    visited = np.zeros((h, w), dtype=bool)
    queue = deque([start])
    visited[start] = True

    reached = False
    while queue:
        cy, cx = queue.popleft()
        if (cy, cx) == goal:
            reached = True
            break
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w:
                if not obstacles[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((ny, nx))

    # If completely cut off, clear a winding traverse path
    if not reached:
        sy, sx = start
        gy, gx = goal
        steps = max(abs(gy - sy), abs(gx - sx))
        for step in range(steps + 1):
            t = step / max(1, steps)
            cy = int(np.clip(sy + t * (gy - sy), 0, h - 1))
            cx = int(np.clip(sx + t * (gx - sx), 0, w - 1))
            obstacles[max(0, cy-1):min(h, cy+2), max(0, cx-1):min(w, cx+2)] = False
