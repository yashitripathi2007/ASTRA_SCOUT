"""
Visualization Engine for ASTRA-SCOUT.
Interactive 3D surfaces and 2D multi-source sensor fusion heatmaps powered by Plotly.
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.sensor_fusion import SensorFusionGrid, ObservationSource


def create_3d_terrain_surface(
    elevation: np.ndarray,
    rover_path: Optional[List[Tuple[int, int]]] = None,
    coop_path: Optional[List[Tuple[int, int]]] = None,
    drone_waypoints: Optional[List[Tuple[int, int]]] = None,
    start: Optional[Tuple[int, int]] = None,
    goal: Optional[Tuple[int, int]] = None,
    subsample: int = 1,
    adaptive_path: Optional[List[Tuple[int, int]]] = None
) -> go.Figure:
    """
    Renders an interactive 3D topography surface with overlaid rover paths and drone flight.
    """
    z = elevation[::subsample, ::subsample]
    y = np.arange(0, elevation.shape[0], subsample)
    x = np.arange(0, elevation.shape[1], subsample)

    fig = go.Figure()

    # 3D Surface
    fig.add_trace(go.Surface(
        z=z,
        x=x,
        y=y,
        colorscale="Earth",
        opacity=0.88,
        showscale=True,
        colorbar=dict(title="Elevation (m)", len=0.6, thickness=14),
        name="Topography"
    ))

    # Rover-Only Baseline Route (Red dashed)
    if rover_path and len(rover_path) > 1:
        rx = [p[1] for p in rover_path]
        ry = [p[0] for p in rover_path]
        rz = [elevation[p[0], p[1]] + 0.35 for p in rover_path]
        fig.add_trace(go.Scatter3d(
            x=rx, y=ry, z=rz,
            mode="lines+markers",
            line=dict(color="#FF4136", width=4, dash="dash"),
            marker=dict(size=2, color="#FF4136"),
            name="Rover-Only Baseline (Replanning)"
        ))

    # Always-Scout Cooperative Route (Cyan solid)
    if coop_path and len(coop_path) > 1:
        cx = [p[1] for p in coop_path]
        cy = [p[0] for p in coop_path]
        cz = [elevation[p[0], p[1]] + 0.5 for p in coop_path]
        fig.add_trace(go.Scatter3d(
            x=cx, y=cy, z=cz,
            mode="lines+markers",
            line=dict(color="#00D4FF", width=5),
            marker=dict(size=3, color="#00D4FF"),
            name="Always-Scout Route"
        ))

    # Adaptive-Scout Route (Gold solid)
    if adaptive_path and len(adaptive_path) > 1:
        ax = [p[1] for p in adaptive_path]
        ay = [p[0] for p in adaptive_path]
        az = [elevation[p[0], p[1]] + 0.65 for p in adaptive_path]
        fig.add_trace(go.Scatter3d(
            x=ax, y=ay, z=az,
            mode="lines+markers",
            line=dict(color="#FFD700", width=6),
            marker=dict(size=3, color="#FFD700"),
            name="Adaptive-Scout Route"
        ))

    # Drone Flight Trajectory (High altitude cyan flightline)
    if drone_waypoints and len(drone_waypoints) > 1:
        dx = [p[1] for p in drone_waypoints]
        dy = [p[0] for p in drone_waypoints]
        dz = [elevation[p[0], p[1]] + 12.0 for p in drone_waypoints]
        fig.add_trace(go.Scatter3d(
            x=dx, y=dy, z=dz,
            mode="lines+markers",
            line=dict(color="#00D4FF", width=4, dash="dot"),
            marker=dict(size=5, color="#00D4FF", symbol="diamond"),
            name="Drone Aerial Flight (12m AGL)"
        ))

    # Start & Goal markers
    if start:
        fig.add_trace(go.Scatter3d(
            x=[start[1]], y=[start[0]], z=[elevation[start] + 1.0],
            mode="markers+text",
            marker=dict(size=8, color="#00FF00", symbol="circle"),
            text=["START"],
            textposition="top center",
            name="Start Zone"
        ))
    if goal:
        fig.add_trace(go.Scatter3d(
            x=[goal[1]], y=[goal[0]], z=[elevation[goal] + 1.0],
            mode="markers+text",
            marker=dict(size=10, color="#FFCC00", symbol="cross"),
            text=["GOAL"],
            textposition="top center",
            name="Mission Goal"
        ))

    fig.update_layout(
        title=dict(
            text="Interactive 3D Planetary Topography & Cooperative Trajectories",
            y=0.98,
            x=0.02,
            xanchor="left"
        ),
        scene=dict(
            xaxis_title="X Grid (m)",
            yaxis_title="Y Grid (m)",
            zaxis_title="Elevation (m)",
            aspectmode="data"
        ),
        margin=dict(l=10, r=10, b=30, t=75),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.08,
            xanchor="center",
            x=0.5
        )
    )
    return fig


def create_multisource_knowledge_map(
    fusion_grid: SensorFusionGrid,
    rover_path: Optional[List[Tuple[int, int]]] = None,
    coop_path: Optional[List[Tuple[int, int]]] = None,
    start: Optional[Tuple[int, int]] = None,
    goal: Optional[Tuple[int, int]] = None
) -> go.Figure:
    """
    Renders the 2D Multi-Tier Knowledge Representation.
    
    DISTINGUISHES:
    - 0: Unobserved (Dark Charcoal)
    - 1: AI Inferred / Predicted (Amber)
    - 2: Aerial Drone Scout (Deep Sky Blue)
    - 3: Rover Ground Truth (Emerald Green)
    """
    source_map = fusion_grid.source_map
    h, w = source_map.shape

    # Custom discrete colorscale matching ObservationSource enum
    colorscale = [
        [0.00, "#1A1A24"],   # 0: Unobserved
        [0.25, "#1A1A24"],
        [0.25, "#FF9900"],   # 1: AI Inferred Prior
        [0.50, "#FF9900"],
        [0.50, "#00A2FF"],   # 2: Aerial Drone Observation
        [0.75, "#00A2FF"],
        [0.75, "#00E676"],   # 3: Rover Ground Truth
        [1.00, "#00E676"],
    ]

    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        z=source_map,
        colorscale=colorscale,
        zmin=0,
        zmax=3,
        colorbar=dict(
            title="Knowledge Source",
            tickvals=[0.375, 1.125, 1.875, 2.625],
            ticktext=["Unobserved", "AI Inferred", "Drone Aerial", "Rover Ground"],
            len=0.7,
            thickness=14
        ),
        name="Observation Source"
    ))

    # Overlays
    if rover_path:
        fig.add_trace(go.Scatter(
            x=[p[1] for p in rover_path],
            y=[p[0] for p in rover_path],
            mode="lines",
            line=dict(color="#FF4136", width=2.5, dash="dash"),
            name="Rover-Only Path"
        ))
    if coop_path:
        fig.add_trace(go.Scatter(
            x=[p[1] for p in coop_path],
            y=[p[0] for p in coop_path],
            mode="lines",
            line=dict(color="#FFFF00", width=3),
            name="ASTRA-SCOUT Route"
        ))
    if start:
        fig.add_trace(go.Scatter(
            x=[start[1]], y=[start[0]],
            mode="markers",
            marker=dict(size=12, color="#00FF00", symbol="circle"),
            name="Start"
        ))
    if goal:
        fig.add_trace(go.Scatter(
            x=[goal[1]], y=[goal[0]],
            mode="markers",
            marker=dict(size=14, color="#FF00FF", symbol="star"),
            name="Goal"
        ))

    fig.update_layout(
        title=dict(
            text="Multi-Source Sensor Fusion Grid (Provenanced Observation Layers)",
            y=0.98,
            x=0.02,
            xanchor="left"
        ),
        xaxis_title="X Grid (m)",
        yaxis_title="Y Grid (m)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        margin=dict(l=20, r=20, b=60, t=75),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5
        )
    )
    return fig


def create_hazard_traversability_map(
    hazard_index: np.ndarray,
    obstacles: np.ndarray,
    coop_path: Optional[List[Tuple[int, int]]] = None
) -> go.Figure:
    """Renders composite hazard heatmap with confirmed obstacles."""
    display_grid = np.copy(hazard_index)
    display_grid[obstacles] = 1.0

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=display_grid,
        colorscale="Inferno",
        zmin=0.0,
        zmax=1.0,
        colorbar=dict(title="Hazard Index", len=0.7, thickness=14),
        name="Terrain Hazard"
    ))

    if coop_path:
        fig.add_trace(go.Scatter(
            x=[p[1] for p in coop_path],
            y=[p[0] for p in coop_path],
            mode="lines",
            line=dict(color="#00FFAA", width=3),
            name="Safe Cooperative Route"
        ))

    fig.update_layout(
        title=dict(
            text="Composite Terrain Hazard & Obstacle Field",
            y=0.98,
            x=0.02,
            xanchor="left"
        ),
        xaxis_title="X Grid (m)",
        yaxis_title="Y Grid (m)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        margin=dict(l=20, r=20, b=60, t=75),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5
        )
    )
    return fig


def create_energy_comparison_figure(
    rover_only_wh: float,
    coop_rover_wh: float,
    drone_wh: float,
    adaptive_rover_wh: Optional[float] = None,
    adaptive_drone_wh: Optional[float] = None
) -> go.Figure:
    """Renders detailed energy breakdown comparing baseline vs always-scout vs adaptive-scout."""
    if adaptive_rover_wh is not None:
        categories = ["Rover-Only", "Always-Scout", "Adaptive-Scout"]
        rover_vals = [rover_only_wh, coop_rover_wh, adaptive_rover_wh]
        drone_vals = [0.0, drone_wh, adaptive_drone_wh or 0.0]
        rover_colors = ["#FF5722", "#00BCD4", "#4CAF50"]
        drone_colors = ["#000000", "#00838F", "#2E7D32"]
    else:
        categories = ["Rover-Only Baseline", "ASTRA-SCOUT Cooperative"]
        rover_vals = [rover_only_wh, coop_rover_wh]
        drone_vals = [0.0, drone_wh]
        rover_colors = ["#FF5722", "#4CAF50"]
        drone_colors = ["#000000", "#00BCD4"]

    fig = go.Figure(data=[
        go.Bar(
            name="Rover Surface Energy",
            x=categories,
            y=rover_vals,
            marker_color=rover_colors,
            text=[f"{v:.1f} Wh" for v in rover_vals],
            textposition="auto"
        ),
        go.Bar(
            name="Drone Aerial Energy Overhead",
            x=categories,
            y=drone_vals,
            marker_color=drone_colors,
            text=[f"{v:.1f} Wh" if v > 0 else "" for v in drone_vals],
            textposition="auto"
        )
    ])

    fig.update_layout(
        barmode="stack",
        title=dict(
            text="Mission Energy Consumption Breakdown",
            y=0.98,
            x=0.02,
            xanchor="left"
        ),
        yaxis_title="Energy (Watt-hours)",
        margin=dict(l=20, r=20, b=60, t=85),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5
        )
    )
    return fig


def create_telemetry_history_chart(
    rover_only_telemetry: Dict,
    coop_telemetry: Dict
) -> go.Figure:
    """Renders mission duration and distance metrics."""
    metrics = ["Distance (m)", "Mission Time (s)", "Replanning Events"]
    rover_vals = [
        rover_only_telemetry.get("total_distance_m", 0),
        rover_only_telemetry.get("total_time_s", 0),
        rover_only_telemetry.get("replanning_events_count", 0),
    ]
    coop_vals = [
        coop_telemetry.get("total_distance_m", 0),
        coop_telemetry.get("total_time_s", 0),
        coop_telemetry.get("replanning_events_count", 0),
    ]

    fig = make_subplots(rows=1, cols=3, subplot_titles=metrics)
    fig.add_trace(go.Bar(name="Rover-Only", x=["Rover-Only"], y=[rover_vals[0]], marker_color="#FF5722"), row=1, col=1)
    fig.add_trace(go.Bar(name="ASTRA-SCOUT", x=["ASTRA-SCOUT"], y=[coop_vals[0]], marker_color="#00FFAA"), row=1, col=1)

    fig.add_trace(go.Bar(name="Rover-Only", x=["Rover-Only"], y=[rover_vals[1]], marker_color="#FF5722", showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(name="ASTRA-SCOUT", x=["ASTRA-SCOUT"], y=[coop_vals[1]], marker_color="#00FFAA", showlegend=False), row=1, col=2)

    fig.add_trace(go.Bar(name="Rover-Only", x=["Rover-Only"], y=[rover_vals[2]], marker_color="#FF5722", showlegend=False), row=1, col=3)
    fig.add_trace(go.Bar(name="ASTRA-SCOUT", x=["ASTRA-SCOUT"], y=[coop_vals[2]], marker_color="#00FFAA", showlegend=False), row=1, col=3)

    fig.update_layout(
        title=dict(
            text="Mission Performance & Efficiency Comparison",
            y=0.98,
            x=0.02,
            xanchor="left"
        ),
        margin=dict(l=20, r=20, b=60, t=85),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5
        ),
        showlegend=True
    )
    return fig
