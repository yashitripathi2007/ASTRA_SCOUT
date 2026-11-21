"""
Terrain Intelligence & Hazard Analysis for ASTRA-SCOUT.
Extracts slope gradient, aspect, local surface roughness, geological classification,
and composite traversability indices.
"""

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np


@dataclass
class TerrainIntelligence:
    """Structured analytical layers for planetary terrain."""
    slope_deg: np.ndarray         # Slope angle in degrees [0, 90]
    aspect_deg: np.ndarray        # Compass direction of slope [0, 360)
    roughness: np.ndarray         # Micro-relief roughness standard deviation
    curvature: np.ndarray         # Surface Laplacian / profile curvature
    hazard_index: np.ndarray      # Normalized composite risk [0.0, 1.0]
    traversability: np.ndarray    # Traversability ease score [0.0, 1.0]
    classification: np.ndarray    # Categorical terrain type (int codes)


# Categorical geological terrain classifications
CLASS_BEDROCK = 0       # Firm bedrock / flat regolith (optimal travel)
CLASS_SAND = 1          # Loose sand / regolith dunes (moderate slip)
CLASS_ROCKY = 2         # Gravel & rock strewn field (increased vibrations)
CLASS_BOULDERS = 3      # Dense boulder field (high hazard)
CLASS_IMPASSABLE = 4    # Vertical cliff or impassable obstacle


def compute_terrain_layers(
    z: np.ndarray,
    obstacles: np.ndarray,
    resolution_m: float = 1.0
) -> TerrainIntelligence:
    """
    Computes slope, aspect, roughness, classification, and composite hazard index.
    """
    h, w = z.shape

    # 1. Slope & Aspect via directional gradients
    # Note: numpy gradient spacing accounts for physical resolution in meters
    gy, gx = np.gradient(z, resolution_m)
    grad_mag = np.sqrt(gx**2 + gy**2)
    slope_deg = np.degrees(np.arctan(grad_mag))

    # Aspect: 0 deg = East, 90 deg = North, 180 deg = West, 270 deg = South
    aspect_rad = np.arctan2(gy, gx)
    aspect_deg = (np.degrees(aspect_rad) + 360.0) % 360.0

    # 2. Local Roughness (3x3 neighborhood standard deviation)
    padded = np.pad(z, 1, mode="edge")
    neighbourhoods = [
        padded[dy : dy + h, dx : dx + w]
        for dy in range(3)
        for dx in range(3)
    ]
    stack = np.stack(neighbourhoods, axis=0)
    roughness = np.std(stack, axis=0)

    # 3. Curvature (Laplacian of elevation) - detects ridges (+) and gullies (-)
    gyy, _ = np.gradient(gy, resolution_m)
    _, gxx = np.gradient(gx, resolution_m)
    curvature = gxx + gyy

    # 4. Geological Terrain Classification
    classification = np.full((h, w), CLASS_BEDROCK, dtype=int)
    classification[(slope_deg < 8.0) & (roughness > 0.35)] = CLASS_SAND
    classification[(slope_deg >= 8.0) & (slope_deg < 18.0)] = CLASS_ROCKY
    classification[(slope_deg >= 18.0) | (roughness > 0.8)] = CLASS_BOULDERS
    classification[obstacles | (slope_deg >= 32.0)] = CLASS_IMPASSABLE

    # 5. Continuous Hazard Index H in [0.0, 1.0]
    # Physics parameters:
    # - Slopes > 25 deg pose serious rollover/slip risk for exploration rovers.
    # - Roughness > 1.2m exceeds typical rover wheel diameter (e.g. 0.5m).
    norm_slope = np.clip(slope_deg / 28.0, 0.0, 1.0)
    norm_rough = np.clip(roughness / 1.5, 0.0, 1.0)
    hazard_index = 0.50 * norm_slope + 0.35 * norm_rough + 0.15 * obstacles.astype(float)
    hazard_index[obstacles] = 1.0
    hazard_index = np.clip(hazard_index, 0.0, 1.0)

    # 6. Traversability Index T = 1.0 - Hazard
    traversability = np.clip(1.0 - hazard_index, 0.0, 1.0)
    traversability[obstacles] = 0.0

    return TerrainIntelligence(
        slope_deg=slope_deg,
        aspect_deg=aspect_deg,
        roughness=roughness,
        curvature=curvature,
        hazard_index=hazard_index,
        traversability=traversability,
        classification=classification,
    )
