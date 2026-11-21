"""
Page 3: Scouting Intelligence & Sensor Fusion
Detailed breakdown of distinguishable knowledge layers:
1. Drone Aerial Observation (wide-swath, macro elevation)
2. Rover Ground Observation (local high-res, verified ground truth)
3. AI Inferred Prior (spatial interpolation with epistemic uncertainty)
4. Unobserved Unknown Terrain
Plus the Adaptive Decision Engine ("Scout or Drive?") evaluation.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go

from core.sensor_fusion import SensorFusionGrid, ObservationSource
from core.scout import DroneScout, FlightPattern
from core.decision_engine import AdaptiveDecisionEngine
from utils.visualization import create_multisource_knowledge_map

st.set_page_config(page_title="Scouting Intelligence | ASTRA-SCOUT", page_icon="👁️", layout="wide")

st.title("👁️ Scouting Intelligence & Sensor Fusion")
st.caption("Multi-Agent Perceptual Layering & Epistemic Uncertainty Breakdown")

if "env" not in st.session_state:
    st.warning("Please visit Mission Control or run the initial simulation first.")
    st.stop()

env = st.session_state["env"]
terrain = st.session_state["terrain"]
scout = st.session_state["scout"]
dec_engine = st.session_state["dec_engine"]
res = st.session_state["result"]

# Build live SensorFusionGrid for deep dive
fusion_grid = SensorFusionGrid(env.size)
scout_telemetry = scout.execute_survey(
    start=env.start,
    goal=env.goal,
    environment=env,
    fusion_grid=fusion_grid,
    pattern=st.session_state.get("scout_pattern", FlightPattern.CORRIDOR)
)
# Add rover ground truth
for pt in res.cooperative_path:
    mask = (np.mgrid[0:env.size, 0:env.size][0] - pt[0])**2 + (np.mgrid[0:env.size, 0:env.size][1] - pt[1])**2 <= 5**2
    fusion_grid.apply_rover_observation(mask, env.elevation, terrain.hazard_index, env.obstacles)

breakdown = fusion_grid.get_coverage_breakdown()

# 1. Multi-Tier Knowledge Layering Cards (CRITICAL REQUIREMENT)
st.markdown("### 🔍 Provenanced Multi-Tier Knowledge Accounting")
st.markdown("""
Unlike baseline systems that blend all observations into a single binary grid, **ASTRA-SCOUT rigorously distinguishes 
between observation sources, sensor confidences, and inference horizons.**
""")

k1, k2, k3, k4 = st.columns(4)
k1.metric("🛰️ Aerial Drone Observed", f"{breakdown['aerial_scout_pct']:.1f}%", "Macro relief / Medium confidence")
k2.metric("🚜 Rover Ground Observed", f"{breakdown['rover_ground_pct']:.1f}%", "Verified ground truth / 100% confidence")
k3.metric("🧠 AI Inferred / Predicted", f"{breakdown['inferred_pct']:.1f}%", "Spatial prior / High uncertainty")
k4.metric("⬛ Completely Unobserved", f"{breakdown['unobserved_pct']:.1f}%", "Total ignorance / Caution prior")

st.markdown("---")

# 2. Visualizations
tab_map, tab_unc, tab_decision = st.tabs([
    "🗺️ Multi-Source Knowledge Map",
    "📊 Epistemic Uncertainty Field",
    "⚖️ 'Scout or Drive?' Decision Engine"
])

with tab_map:
    fig_f = create_multisource_knowledge_map(
        fusion_grid=fusion_grid,
        rover_path=res.rover_only_path,
        coop_path=res.cooperative_path,
        start=env.start,
        goal=env.goal
    )
    st.plotly_chart(fig_f, use_container_width=True)
    st.info("""
    **Layer Provenance Legend:**
    - **Dark Charcoal:** Unobserved terrain. The navigation planner treats these cells cautiously with an unknown terrain prior.
    - **Amber:** AI Inferred. Interpolated spatial estimate based on proximity to observed cells.
    - **Deep Sky Blue:** Aerial Drone Scout observation footprint ($12\\text{m}$ AGL downward camera FOV).
    - **Emerald Green:** Rover Direct Ground Sensors ($5\\text{m}$ radius high-resolution LiDAR and wheel traction).
    """)

with tab_unc:
    fig_u = go.Figure(data=go.Heatmap(
        z=fusion_grid.uncertainty_map,
        colorscale="Cividis",
        zmin=0.0,
        zmax=1.0,
        colorbar=dict(title="Uncertainty σ", len=0.7)
    ))
    fig_u.update_layout(
        title="Epistemic Uncertainty Heatmap (0.0 = Absolute Ground Truth, 1.0 = Total Ignorance)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        margin=dict(l=20, r=20, b=20, t=35)
    )
    st.plotly_chart(fig_u, use_container_width=True)
    st.caption("Low uncertainty along the drone flight corridor enables the path planner to select an energy-optimal route without excessive caution penalties.")

with tab_decision:
    st.subheader("⚖️ Adaptive Decision Engine Analysis")
    dec = res.decision_assessment

    col_d1, col_d2 = st.columns([3, 2])
    with col_d1:
        st.markdown(f"""
        ### Active Recommendation: `{dec['recommendation']}`
        - **Confidence Score:** `{dec['confidence_score'] * 100:.0f}%`
        - **Corridor Terrain Complexity:** `{dec['terrain_complexity']:.2f}` (Scale 0 to 1)
        - **Estimated Rover Backtracking Savings:** `{dec['estimated_rover_saving_wh']:.2f} Wh`
        - **Drone Aerial Survey Energy Cost:** `{dec['estimated_scout_cost_wh']:.2f} Wh`
        - **Expected Net Energy Benefit:** `{dec['expected_net_benefit_wh']:.2f} Wh`
        
        #### Strategic Rationale:
        > *"{dec['rationale']}"*
        """)
    with col_d2:
        # Decision trade-off gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=dec["expected_net_benefit_wh"],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Expected Net Energy Benefit (Wh)"},
            delta={'reference': 0.0},
            gauge={
                'axis': {'range': [-10, 25]},
                'bar': {'color': "#00FFAA" if dec["expected_net_benefit_wh"] > 0 else "#FF4136"},
                'steps': [
                    {'range': [-10, 0], 'color': "#331111"},
                    {'range': [0, 25], 'color': "#113322"}
                ],
                'threshold': {
                    'line': {'color': "yellow", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.0
                }
            }
        ))
        fig_gauge.update_layout(margin=dict(l=20, r=20, b=20, t=40), height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
