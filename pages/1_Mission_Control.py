"""
Page 1: Mission Control Console
Interactive 3D planetary map, live simulation playback, multi-agent telemetry,
and three-way strategy execution (Rover-Only, Always-Scout, Adaptive-Scout).
"""

import streamlit as st
import numpy as np

from core.environment import generate_planetary_terrain
from core.terrain_analysis import compute_terrain_layers
from core.scout import DroneScout, FlightPattern
from core.decision_engine import AdaptiveDecisionEngine, MissionCostConfig
from core.navigation import plan_mission, StrategyMode
from utils.visualization import (
    create_3d_terrain_surface,
    create_multisource_knowledge_map,
    create_energy_comparison_figure,
)
from utils.metrics import compute_comparative_metrics

st.set_page_config(page_title="Mission Control | ASTRA-SCOUT", page_icon="🛸", layout="wide")

st.title("🛸 Mission Control Center")
st.caption("Real-Time Multi-Agent Surface & Aerial Telemetry Console")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Mission Parameters")
    preset = st.selectbox(
        "Geological Terrain Preset",
        ["Martian Crater Field", "Lunar Mare Basin", "Rugged Canyon", "Smooth Regolith"],
        index=0
    )
    seed = st.slider("Terrain Seed", min_value=1, max_value=200, value=st.session_state.get("seed", 42))
    grid_size = st.select_slider("Terrain Grid (m)", options=[50, 60, 70, 80, 90], value=st.session_state.get("grid_size", 70))

    st.subheader("🎯 Strategy View")
    strategy_view = st.selectbox(
        "Strategy Visualizer Mode",
        ["All 3 Strategies (Overlay)", "Adaptive-Scout", "Always-Scout", "Rover-Only"],
        index=0
    )

    st.subheader("🛸 Drone Scout Settings")
    flight_alt = st.slider("Flight Altitude (m AGL)", min_value=8.0, max_value=24.0, value=12.0, step=1.0)
    drone_battery_avail = st.slider("Drone Battery Available (Wh)", min_value=0.0, max_value=40.0, value=40.0, step=2.0)
    scout_pattern = st.selectbox(
        "Reconnaissance Flight Pattern",
        [FlightPattern.CORRIDOR, FlightPattern.BOUSTROPHEDON, FlightPattern.TARGETED_CHOKEPOINTS],
        index=0
    )

    st.subheader("🚜 Rover Sensor Settings")
    rover_radius = st.slider("Rover Local Sensor Radius (m)", min_value=3, max_value=8, value=5)
    caution_penalty = st.slider("Unknown Terrain Caution Penalty", min_value=1.0, max_value=5.0, value=2.5, step=0.1)

    st.subheader("⚖️ Combined Mission Cost Weights")
    w_rover = st.slider("Rover Energy Weight", min_value=0.2, max_value=3.0, value=1.0, step=0.1)
    w_drone = st.slider("Drone Energy Weight", min_value=0.2, max_value=3.0, value=1.0, step=0.1)
    w_risk = st.slider("Route Risk Weight", min_value=1.0, max_value=30.0, value=15.0, step=1.0)

    re_run = st.button("🚀 Run Mission Simulation", type="primary", use_container_width=True)

cost_cfg = MissionCostConfig(
    weight_rover_energy=w_rover,
    weight_drone_energy=w_drone,
    weight_route_risk=w_risk
)

# Run or Update Simulation
if re_run or "env" not in st.session_state:
    with st.spinner("Executing three-way mission simulation..."):
        env = generate_planetary_terrain(size=grid_size, seed=seed, preset=preset)
        terrain = compute_terrain_layers(env.elevation, env.obstacles, env.resolution_m)
        dec_engine = AdaptiveDecisionEngine(cost_config=cost_cfg)
        scout = DroneScout(flight_altitude_m=flight_alt)
        result = plan_mission(
            environment=env,
            terrain=terrain,
            decision_engine=dec_engine,
            scout=scout,
            rover_radius=rover_radius,
            scout_pattern=scout_pattern,
            caution_penalty=caution_penalty,
            drone_battery_wh=drone_battery_avail,
            cost_config=cost_cfg
        )

        st.session_state["env"] = env
        st.session_state["terrain"] = terrain
        st.session_state["scout"] = scout
        st.session_state["dec_engine"] = dec_engine
        st.session_state["result"] = result
        st.session_state["seed"] = seed
        st.session_state["grid_size"] = grid_size
        st.session_state["preset"] = preset
        st.session_state["cost_config"] = cost_cfg

env = st.session_state["env"]
terrain = st.session_state["terrain"]
res = st.session_state["result"]
stats = compute_comparative_metrics(
    rover_only_telemetry=res.rover_only_telemetry,
    coop_telemetry=res.cooperative_telemetry,
    drone_telemetry=res.drone_telemetry,
    coverage_stats=res.coverage_stats,
    baseline_coverage_stats=res.baseline_coverage_stats,
    adaptive_telemetry=res.adaptive_telemetry,
    adaptive_drone_telemetry=res.adaptive_drone_telemetry,
    adaptive_coverage_stats=res.adaptive_coverage_stats,
    mission_costs=res.mission_costs,
    route_risks=res.route_risks,
    cost_config=res.cost_config or cost_cfg
)

# 1. Decision Recommendation Alert
dec = res.decision_assessment
if "Deploy" in dec["recommendation"]:
    st.success(f"**Adaptive Arbiter: {dec['recommendation']}** — {dec['rationale']}")
elif "Chokepoint" in dec["recommendation"]:
    st.warning(f"**Adaptive Arbiter: {dec['recommendation']}** — {dec['rationale']}")
else:
    st.info(f"**Adaptive Arbiter: {dec['recommendation']}** — {dec['rationale']}")

# 2. Key Telemetry Metrics Row
c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Adaptive Net Energy Saved",
    f"{stats['adaptive_net_energy_savings_wh']} Wh",
    f"{stats['adaptive_energy_savings_pct']}% vs Baseline"
)
c2.metric(
    "Replanning Avoided (Adaptive)",
    f"{stats['adaptive_replans_avoided']}",
    f"Baseline: {res.rover_only_replans} | Adaptive: {res.adaptive_replans}"
)
c3.metric(
    "Adaptive Combined Mission Cost",
    f"{stats['adaptive_scout_cost']:.1f}",
    f"Baseline: {stats['rover_only_cost']:.1f} | Always: {stats['always_scout_cost']:.1f}"
)
c4.metric(
    "Terrain Mapped (Adaptive)",
    f"{stats['adaptive_coverage_total_pct']}%",
    f"Epistemic σ: {stats['adaptive_mean_uncertainty']:.3f}"
)

# 3. Main Visualization Tabs
tab1, tab2, tab3 = st.tabs(["🌐 Interactive 3D Terrain", "🗺️ Multi-Source Knowledge Map", "⚡ Energy & Strategy Telemetry"])

with tab1:
    drone_wps = res.drone_telemetry.get("waypoints") if res.drone_telemetry else None

    # Filter paths according to strategy_view
    display_rover = res.rover_only_path if strategy_view in ["All 3 Strategies (Overlay)", "Rover-Only"] else None
    display_coop = res.cooperative_path if strategy_view in ["All 3 Strategies (Overlay)", "Always-Scout"] else None
    display_adaptive = res.adaptive_path if strategy_view in ["All 3 Strategies (Overlay)", "Adaptive-Scout"] else None

    fig_3d = create_3d_terrain_surface(
        elevation=env.elevation,
        rover_path=display_rover,
        coop_path=display_coop,
        adaptive_path=display_adaptive,
        drone_waypoints=drone_wps,
        start=env.start,
        goal=env.goal
    )
    st.plotly_chart(fig_3d, use_container_width=True)
    st.caption("🔴 Red Dashed: Rover-Only Baseline (Replanning). 🔵 Cyan Solid: Always-Scout Route. 🟡 Gold Solid: Adaptive-Scout Route.")

with tab2:
    # Sensor Fusion Map
    from core.sensor_fusion import SensorFusionGrid
    fusion_grid = SensorFusionGrid(env.size)
    scout_wps = res.drone_telemetry.get("waypoints") if res.drone_telemetry else None
    if scout_wps:
        st.session_state["scout"].execute_survey(
            env.start, env.goal, env, fusion_grid, scout_pattern
        )
    for pt in res.cooperative_path:
        mask = (np.mgrid[0:env.size, 0:env.size][0] - pt[0])**2 + (np.mgrid[0:env.size, 0:env.size][1] - pt[1])**2 <= rover_radius**2
        fusion_grid.apply_rover_observation(mask, env.elevation, terrain.hazard_index, env.obstacles)

    fig_fusion = create_multisource_knowledge_map(
        fusion_grid=fusion_grid,
        rover_path=res.rover_only_path,
        coop_path=res.adaptive_path or res.cooperative_path,
        start=env.start,
        goal=env.goal
    )
    st.plotly_chart(fig_fusion, use_container_width=True)
    st.caption("Provenanced Layers: Dark Charcoal = Unobserved, Amber = AI Inferred Prior, Blue = Aerial Drone Observation, Green = Rover Direct Ground Sensors.")

with tab3:
    col_e1, col_e2 = st.columns([3, 2])
    with col_e1:
        drone_wh = res.drone_telemetry.get("flight_energy_wh", 0.0) if res.drone_telemetry else 0.0
        fig_energy = create_energy_comparison_figure(
            rover_only_wh=res.rover_only_telemetry.get("total_energy_wh", 0.0),
            coop_rover_wh=res.cooperative_telemetry.get("total_energy_wh", 0.0),
            drone_wh=drone_wh,
            adaptive_rover_wh=stats["adaptive_rover_energy_wh"],
            adaptive_drone_wh=stats["adaptive_drone_energy_wh"]
        )
        st.plotly_chart(fig_energy, use_container_width=True)
    with col_e2:
        st.subheader("Three-Strategy Telemetry Accounting")
        st.table({
            "Metric": [
                "Rover Surface Energy (Wh)",
                "Drone Flight Overhead (Wh)",
                "Total Mission Energy (Wh)",
                "Route Risk Index",
                "Combined Mission Cost",
                "Dynamic Replanning Count",
                "Mission Duration (s)",
                "Total Distance Traveled (m)"
            ],
            "Rover-Only": [
                f"{stats['rover_only_energy_wh']:.2f}",
                "0.00",
                f"{stats['rover_only_energy_wh']:.2f}",
                f"{stats['rover_only_risk']:.3f}",
                f"{stats['rover_only_cost']:.2f}",
                f"{stats['rover_only_replans']}",
                f"{stats['rover_only_time_s']:.1f}",
                f"{stats['rover_only_distance_m']:.2f}"
            ],
            "Always-Scout": [
                f"{stats['coop_rover_energy_wh']:.2f}",
                f"{stats['drone_energy_wh']:.2f}",
                f"{stats['coop_total_energy_wh']:.2f}",
                f"{stats['always_scout_risk']:.3f}",
                f"{stats['always_scout_cost']:.2f}",
                f"{stats['coop_replans']}",
                f"{stats['coop_time_s']:.1f}",
                f"{stats['coop_distance_m']:.2f}"
            ],
            "Adaptive-Scout": [
                f"{stats['adaptive_rover_energy_wh']:.2f}",
                f"{stats['adaptive_drone_energy_wh']:.2f}",
                f"{stats['adaptive_total_energy_wh']:.2f}",
                f"{stats['adaptive_scout_risk']:.3f}",
                f"{stats['adaptive_scout_cost']:.2f}",
                f"{stats['adaptive_replans']}",
                f"{stats['adaptive_time_s']:.1f}",
                f"{stats['adaptive_distance_m']:.2f}"
            ]
        })
