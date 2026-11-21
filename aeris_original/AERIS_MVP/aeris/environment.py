import numpy as np

def generate_terrain(size=70, seed=7):
    """Create a synthetic elevation grid and a separate obstacle mask."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size]
    z = 2.0 * np.sin(x / 9.0) + 1.5 * np.cos(y / 11.0)
    # Add smooth hills and depressions.
    for _ in range(7):
        cx, cy = rng.integers(5, size-5, size=2)
        amp = rng.uniform(-5, 7)
        sigma = rng.uniform(3, 9)
        z += amp * np.exp(-((x-cx)**2 + (y-cy)**2) / (2*sigma**2))
    z += rng.normal(0, 0.18, size=(size, size))

    obstacles = np.zeros((size, size), dtype=bool)
    for _ in range(max(8, size // 5)):
        cx, cy = rng.integers(4, size-4, size=2)
        radius = int(rng.integers(1, 3))
        obstacles[(x-cx)**2 + (y-cy)**2 <= radius**2] = True

    start = (size // 2, 3)
    goal = (size // 2, size - 4)
    obstacles[start] = False
    obstacles[goal] = False
    # Keep a small start/goal neighbourhood accessible.
    obstacles[start[0]-2:start[0]+3, start[1]-2:start[1]+3] = False
    obstacles[goal[0]-2:goal[0]+3, goal[1]-2:goal[1]+3] = False
    return z, obstacles, start, goal

def terrain_layers(z, obstacles):
    """Return slope, local roughness proxy and combined hazard score."""
    gy, gx = np.gradient(z)
    slope = np.degrees(np.arctan(np.sqrt(gx**2 + gy**2)))
    # Local roughness proxy from deviation from a 3x3 neighbourhood mean.
    padded = np.pad(z, 1, mode="edge")
    neighbourhoods = [padded[dy:dy+z.shape[0], dx:dx+z.shape[1]]
                      for dy in range(3) for dx in range(3)]
    stack = np.stack(neighbourhoods)
    roughness = np.std(stack, axis=0)
    # Normalised composite risk; weights are illustrative MVP parameters.
    s = np.clip(slope / 35.0, 0, 1)
    q = np.clip(roughness / 2.0, 0, 1)
    hazard = 0.55*s + 0.30*q + 0.15*obstacles.astype(float)
    hazard[obstacles] = 1.0
    return slope, roughness, hazard
