"""
Page 2: Planetary Terrain Analysis
In-depth geological and physical terrain assessment: 3D DEM, slope gradients,
surface roughness, traversability indices, and rock classification.
"""

import os
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.environment import generate_planetary_terrain, PlanetaryEnvironment
from core.terrain_analysis import compute_terrain_layers, CLASS_BEDROCK, CLASS_SAND, CLASS_ROCKY, CLASS_BOULDERS, CLASS_IMPASSABLE
from utils.styles import apply_shared_styles

st.set_page_config(page_title="Terrain Analysis | ASTRA-SCOUT", page_icon="🗺️", layout="wide")
apply_shared_styles()

st.title("🗺️ AI Planetary Terrain Intelligence")
st.caption("Morphological & Geotechnical Analysis for Planetary Surface Navigation")

if "env" not in st.session_state:
    st.warning("Please visit Mission Control or run the initial simulation first.")
    st.stop()

env = st.session_state["env"]
terrain = st.session_state["terrain"]

# Preset Selector / External Ingestion Toolbar
col_ctrl1, col_ctrl2 = st.columns([3, 2])
with col_ctrl1:
    st.markdown(f"**Active Topography:** `{env.name}` | Resolution: `{env.resolution_m} m/cell` | Grid: `{env.size} x {env.size}`")
with col_ctrl2:
    sample_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sample")
    if os.path.exists(sample_dir):
        available_files = [f for f in os.listdir(sample_dir) if f.endswith(".npz")]
        if available_files:
            selected_npz = st.selectbox("Load Sample Geological Dataset (.npz)", ["-- Active Session --"] + available_files)
            if selected_npz != "-- Active Session --":
                filepath = os.path.join(sample_dir, selected_npz)
                loaded_env = PlanetaryEnvironment.load(filepath)
                st.session_state["env"] = loaded_env
                st.session_state["terrain"] = compute_terrain_layers(loaded_env.elevation, loaded_env.obstacles, loaded_env.resolution_m)
                st.rerun()

# 1. Morphological Statistics Summary
s1, s2, s3, s4 = st.columns(4)
s1.metric("Elevation Range", f"{np.min(env.elevation):.1f} to {np.max(env.elevation):.1f} m", f"Δ {np.ptp(env.elevation):.1f} m")
s2.metric("Mean Surface Slope", f"{np.mean(terrain.slope_deg):.1f}°", f"Max: {np.max(terrain.slope_deg):.1f}°")
s3.metric("Roughness Proxy (σ)", f"{np.mean(terrain.roughness):.2f} m", f"Max: {np.max(terrain.roughness):.2f} m")
s4.metric("Obstacle Density", f"{np.mean(env.obstacles)*100:.1f}%", f"{int(np.sum(env.obstacles))} cells blocked")

st.markdown("---")

# 2. Main Visualization Tabs
tabs = st.tabs(["3D Elevation Surface", "Slope & Aspect Analysis", "Roughness & Hazards", "Geological Classification"])

with tabs[0]:
    # 3D Interactive DEM
    fig_dem = go.Figure(data=[go.Surface(
        z=env.elevation,
        colorscale="Earth",
        colorbar=dict(title="Elevation (m)", len=0.7)
    )])
    fig_dem.update_layout(
        title=dict(
            text=f"3D Topographical Surface: {env.name}",
            y=0.98,
            x=0.02,
            xanchor="left"
        ),
        scene=dict(xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Elevation (m)", aspectmode="data"),
        margin=dict(l=0, r=0, b=0, t=65)
    )
    st.plotly_chart(fig_dem, use_container_width=True)

with tabs[1]:
    col1, col2 = st.columns(2)
    with col1:
        # Slope Heatmap
        fig_slope = go.Figure(data=go.Heatmap(
            z=terrain.slope_deg,
            colorscale="Viridis",
            colorbar=dict(title="Slope (°)", len=0.8)
        ))
        fig_slope.update_layout(
            title=dict(text="Terrain Incline (Degrees)", y=0.96, x=0.02, xanchor="left"),
            yaxis=dict(scaleanchor="x", scaleratio=1),
            margin=dict(l=20, r=20, b=20, t=55)
        )
        st.plotly_chart(fig_slope, use_container_width=True)
    with col2:
        # Aspect Heatmap (Compass Direction of slope)
        fig_aspect = go.Figure(data=go.Heatmap(
            z=terrain.aspect_deg,
            colorscale="HSV",
            colorbar=dict(title="Aspect (°)", len=0.8)
        ))
        fig_aspect.update_layout(
            title=dict(text="Slope Aspect Heading (0°=East, 90°=North, 180°=West, 270°=South)", y=0.96, x=0.02, xanchor="left"),
            yaxis=dict(scaleanchor="x", scaleratio=1),
            margin=dict(l=20, r=20, b=20, t=55)
        )
        st.plotly_chart(fig_aspect, use_container_width=True)

with tabs[2]:
    col3, col4 = st.columns(2)
    with col3:
        # Roughness Heatmap
        fig_rough = go.Figure(data=go.Heatmap(
            z=terrain.roughness,
            colorscale="Plasma",
            colorbar=dict(title="Roughness σ (m)", len=0.8)
        ))
        fig_rough.update_layout(
            title=dict(text="Micro-Relief Surface Roughness", y=0.96, x=0.02, xanchor="left"),
            yaxis=dict(scaleanchor="x", scaleratio=1),
            margin=dict(l=20, r=20, b=20, t=55)
        )
        st.plotly_chart(fig_rough, use_container_width=True)
    with col4:
        # Hazard Heatmap
        fig_haz = go.Figure(data=go.Heatmap(
            z=terrain.hazard_index,
            colorscale="Inferno",
            colorbar=dict(title="Hazard Index", len=0.8)
        ))
        fig_haz.update_layout(
            title=dict(text="Composite Traversability Hazard Index", y=0.96, x=0.02, xanchor="left"),
            yaxis=dict(scaleanchor="x", scaleratio=1),
            margin=dict(l=20, r=20, b=20, t=55)
        )
        st.plotly_chart(fig_haz, use_container_width=True)

with tabs[3]:
    # Geological Terrain Classification Map
    class_names = ["Firm Bedrock", "Loose Sand", "Rocky Plains", "Boulder Field", "Impassable Cliff/Obstacle"]
    class_colors = [
        [0.0, "#2E7D32"],   # Firm Bedrock: Dark Green
        [0.2, "#2E7D32"],
        [0.2, "#FBC02D"],   # Loose Sand: Yellow
        [0.4, "#FBC02D"],
        [0.4, "#FB8C00"],   # Rocky Plains: Orange
        [0.6, "#FB8C00"],
        [0.6, "#D32F2F"],   # Boulder Field: Crimson
        [0.8, "#D32F2F"],
        [0.8, "#212121"],   # Impassable: Charcoal
        [1.0, "#212121"],
    ]

    fig_class = go.Figure(data=go.Heatmap(
        z=terrain.classification,
        colorscale=class_colors,
        zmin=0,
        zmax=4,
        colorbar=dict(
            title="Terrain Type",
            tickvals=[0.4, 1.2, 2.0, 2.8, 3.6],
            ticktext=class_names,
            len=0.7
        )
    ))
    fig_class.update_layout(
        title=dict(text="AI Geological Surface Classification", y=0.96, x=0.02, xanchor="left"),
        yaxis=dict(scaleanchor="x", scaleratio=1),
        margin=dict(l=20, r=20, b=20, t=55)
    )
    st.plotly_chart(fig_class, use_container_width=True)
    st.info("The classification engine segments terrain into distinct geotechnical traversability zones. Boulder fields and steep cliffs automatically receive high traversal cost penalties in the path planner.")
