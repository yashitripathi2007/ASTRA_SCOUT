# ASTRA-SCOUT Planetary Datasets

This directory contains pre-computed and custom digital elevation models (DEM) and obstacle masks for planetary exploration testing.

## Data Structure

Saved `.npz` files contain:
- `name`: String identifier of the geological terrain preset.
- `size`: Grid size (e.g., 70 for a $70 \times 70$ meter mesh).
- `resolution_m`: Spatial resolution in meters per cell (default $1.0\text{ m/cell}$).
- `elevation`: 2D float array containing surface topography (meters).
- `obstacles`: 2D boolean array where `True` indicates impassable boulders or vertical cliffs.
- `start`: `(y, x)` tuple representing the rover landing site / deployment base.
- `goal`: `(y, x)` tuple representing the scientific sample site / exploration waypoint.
- `seed`: Random seed used for reproducibility.

## Included Sample Presets

1. **`martian_crater.npz`**: Simulated Martian crater field featuring central impact depression, raised rim, ejecta blanket, and scattered basaltic boulders.
2. **`lunar_mare.npz`**: Rolling regolith undulations with subtle slopes and localized boulder fields.
3. **`rugged_canyon.npz`**: Tectonic rift valley with steep grade changes and challenging traversability corridors.

## Ingesting External NASA / USGS DEMs

To import real HiRISE (Mars) or LOLA (Moon) GeoTIFF or NumPy datasets:
1. Resample raster to a regular 2D grid.
2. Calculate slope and mark impassable cells.
3. Save as `.npz` matching the schema above or load directly via `pages/2_Terrain_Analysis.py`.
