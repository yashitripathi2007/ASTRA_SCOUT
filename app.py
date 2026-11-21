"""
ASTRA-SCOUT: AI-Powered Collaborative Rover-Drone Exploration System
Main Application Entrypoint & Executive Command Dashboard.
"""

import os
import streamlit as st
import numpy as np

from core.environment import generate_planetary_terrain, PlanetaryEnvironment
from core.terrain_analysis import compute_terrain_layers
from core.sensor_fusion import SensorFusionGrid
from core.scout import DroneScout, FlightPattern
from core.rover import RoverAgent
from core.decision_engine import AdaptiveDecisionEngine
from core.navigation import plan_mission
from utils.metrics import compute_comparative_metrics
from utils.styles import apply_shared_styles

st.set_page_config(
    page_title="ASTRA-SCOUT | Collaborative Planetary Exploration",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded"
)
apply_shared_styles()

# Initialize Global Session State
if "initialized" not in st.session_state:
    st.session_state["seed"] = 42
    st.session_state["grid_size"] = 70
    st.session_state["preset"] = "Martian Crater Field"
    st.session_state["flight_altitude"] = 12.0
    st.session_state["scout_pattern"] = FlightPattern.CORRIDOR
    st.session_state["rover_radius"] = 5
    st.session_state["caution_penalty"] = 2.5

    # Generate initial mission baseline
    env = generate_planetary_terrain(
        size=st.session_state["grid_size"],
        seed=st.session_state["seed"],
        preset=st.session_state["preset"]
    )
    terrain = compute_terrain_layers(env.elevation, env.obstacles, env.resolution_m)
    dec_engine = AdaptiveDecisionEngine()
    scout = DroneScout(flight_altitude_m=st.session_state["flight_altitude"])
    result = plan_mission(
        environment=env,
        terrain=terrain,
        decision_engine=dec_engine,
        scout=scout,
        rover_radius=st.session_state["rover_radius"],
        scout_pattern=st.session_state["scout_pattern"],
        caution_penalty=st.session_state["caution_penalty"]
    )

    st.session_state["env"] = env
    st.session_state["terrain"] = terrain
    st.session_state["scout"] = scout
    st.session_state["dec_engine"] = dec_engine
    st.session_state["result"] = result
    st.session_state["initialized"] = True

# Header & Branding
col_logo, col_title = st.columns([1, 5])
with col_logo:
    logo_path = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
    if os.path.exists(logo_path):
        st.image(logo_path, width=120)
with col_title:
    st.title("ASTRA-SCOUT")
    st.markdown('<p class="hero-subtitle">AI-Powered Collaborative Rover–Drone Planetary Exploration</p>', unsafe_allow_html=True)
    st.caption("Next-Generation Multi-Agent Aerial-Ground Co-Exploration Architecture")

st.markdown("---")

# Quick Navigation & High-Level Architecture Overview
st.markdown("""
Welcome to **ASTRA-SCOUT**, an autonomous multi-agent simulation framework that coordinates an aerial scout drone 
and a ground exploration rover to conquer hazardous extraterrestrial terrains (Mars & Moon).

### System Mission Modules
Select a module from the **left sidebar** or click through below:
""")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.info("### 🛸 1. Mission Control\nInteractive 3D planetary map, live mission execution, and telemetry feeds.")
with c2:
    st.success("### 🗺️ 2. Terrain Analysis\nDigital Elevation Models (DEM), slope gradients, micro-roughness & geological hazards.")
with c3:
    st.warning("### 👁️ 3. Scouting Intelligence\nMulti-source sensor fusion: provenanced aerial, ground, and AI-inferred observation layers.")
with c4:
    st.error("### 📊 4. Route Comparison\nComparative energy analysis, replanning reductions, and Monte Carlo multi-seed benchmarking.")

st.markdown("---")

# Executive Mission Summary Metrics
res = st.session_state["result"]
stats = compute_comparative_metrics(
    res.rover_only_telemetry,
    res.cooperative_telemetry,
    res.drone_telemetry,
    res.coverage_stats
)

st.markdown("### 🛰️ Active Mission Baseline Status")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Current Terrain", f"{st.session_state['preset']}")
m2.metric("Net Energy Saved", f"{stats['net_energy_savings_wh']} Wh", f"{stats['energy_savings_pct']}%")
m3.metric("Replans Avoided", f"{stats['replans_avoided']} events", f"-{res.rover_only_replans - res.cooperative_replans}")
m4.metric("Cooperative Distance", f"{stats['coop_distance_m']} m", f"-{stats['distance_savings_m']} m")
m5.metric("Terrain Surveyed", f"{stats['coverage_total_pct']}%", f"{stats['coverage_aerial_pct']}% Aerial")

# Decision Recommendation Callout
dec = res.decision_assessment
rec_color = "success" if "Deploy" in dec["recommendation"] else "info"
st.divider()
st.markdown(f"""
> **Adaptive Decision Engine Recommendation:** `{dec['recommendation']}`  
> **Confidence:** {dec['confidence_score'] * 100:.0f}% | **Terrain Complexity:** {dec['terrain_complexity']:.2f}  
> *"{dec['rationale']}"*
""")

st.caption("ASTRA-SCOUT Mission Framework | Built for High-Autonomy Planetary Exploration")
