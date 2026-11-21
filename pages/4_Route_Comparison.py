"""
Page 4: Route & Strategy Benchmarking
Quantitative performance comparison between the unassisted rover baseline
and the cooperative ASTRA-SCOUT system, including Monte Carlo multi-seed benchmarking.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.environment import generate_planetary_terrain
from core.terrain_analysis import compute_terrain_layers
from core.scout import DroneScout
from core.decision_engine import AdaptiveDecisionEngine
from core.navigation import plan_mission
from utils.visualization import create_energy_comparison_figure, create_telemetry_history_chart
from utils.metrics import compute_comparative_metrics

st.set_page_config(page_title="Route Comparison | ASTRA-SCOUT", page_icon="📊", layout="wide")

st.title("📊 Mission Route & Strategy Comparison")
st.caption("Quantitative Benchmarking: Blind Rover Baseline vs Cooperative ASTRA-SCOUT")

if "result" not in st.session_state:
    st.warning("Please run a mission simulation in Mission Control first.")
    st.stop()

res = st.session_state["result"]
stats = compute_comparative_metrics(
    res.rover_only_telemetry,
    res.cooperative_telemetry,
    res.drone_telemetry,
    res.coverage_stats
)

# 1. Headline Advantage Metric
adv_color = "success" if stats["cooperative_advantage"] else "warning"
if stats["cooperative_advantage"]:
    st.success(f"""
    **Cooperative Flight Advantage Confirmed:**  
    In this mission terrain, ASTRA-SCOUT achieved **{stats['energy_savings_pct']}% net energy savings** ({stats['net_energy_savings_wh']} Wh) 
    and eliminated **{stats['replans_avoided']} emergency replanning stops**, even after factoring in the drone's flight overhead!
    """)
else:
    st.warning(f"""
    **Drone Flight Overhead Not Recovered:**  
    In this terrain layout, the drone overhead ({stats['drone_energy_wh']} Wh) exceeded rover detour savings ({stats['net_energy_savings_wh']} Wh).  
    *Scientific Note: This is an authentic result. On flat or open terrain, the Adaptive Decision Engine correctly recommends direct driving.*
    """)

# 2. Main Comparison Charts
col1, col2 = st.columns([1, 1])

with col1:
    fig_energy = create_energy_comparison_figure(
        rover_only_wh=stats["rover_only_energy_wh"],
        coop_rover_wh=stats["coop_rover_energy_wh"],
        drone_wh=stats["drone_energy_wh"]
    )
    st.plotly_chart(fig_energy, use_container_width=True)

with col2:
    fig_hist = create_telemetry_history_chart(
        res.rover_only_telemetry,
        res.cooperative_telemetry
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# 3. Comprehensive Metric Table
st.markdown("### 📋 Detailed Mission Telemetry Accounting")
df_metrics = pd.DataFrame({
    "Performance Metric": [
        "Rover Surface Energy (Wh)",
        "Drone Flight Overhead (Wh)",
        "Total Mission Energy (Wh)",
        "Net Energy Saved (Wh)",
        "Net Energy Savings (%)",
        "Traversed Distance (m)",
        "Mission Time (s)",
        "Dynamic Replanning Halts",
        "Terrain Knowledge Coverage (%)",
        "Average Epistemic Uncertainty (σ)"
    ],
    "Rover-Only Baseline": [
        f"{stats['rover_only_energy_wh']:.2f}",
        "0.00",
        f"{stats['rover_only_energy_wh']:.2f}",
        "-",
        "-",
        f"{stats['rover_only_distance_m']:.2f}",
        f"{stats['rover_only_time_s']:.1f}",
        f"{stats['rover_only_replans']}",
        f"{stats['coverage_ground_pct']:.1f}%",
        "0.850"
    ],
    "ASTRA-SCOUT Cooperative": [
        f"{stats['coop_rover_energy_wh']:.2f}",
        f"{stats['drone_energy_wh']:.2f}",
        f"{stats['coop_total_energy_wh']:.2f}",
        f"{stats['net_energy_savings_wh']:+.2f}",
        f"{stats['energy_savings_pct']:+.1f}%",
        f"{stats['coop_distance_m']:.2f}",
        f"{stats['coop_time_s']:.1f}",
        f"{stats['coop_replans']}",
        f"{stats['coverage_total_pct']:.1f}%",
        f"{stats['mean_uncertainty']:.3f}"
    ],
    "Delta Improvement": [
        f"{(stats['rover_only_energy_wh'] - stats['coop_rover_energy_wh']):+.2f} Wh",
        f"-{stats['drone_energy_wh']:.2f} Wh",
        f"{stats['net_energy_savings_wh']:+.2f} Wh",
        f"{stats['net_energy_savings_wh']:+.2f} Wh",
        f"{stats['energy_savings_pct']:+.1f}%",
        f"{stats['distance_savings_m']:+.2f} m",
        f"{stats['time_savings_s']:+.1f} s",
        f"-{stats['replans_avoided']} replans",
        f"+{stats['coverage_total_pct'] - stats['coverage_ground_pct']:.1f}%",
        f"{(0.850 - stats['mean_uncertainty']):+.3f}"
    ]
})
st.dataframe(df_metrics, use_container_width=True, hide_index=True)

st.markdown("---")

# 4. Monte Carlo Multi-Seed Benchmark Runner
st.markdown("### 🎲 Monte Carlo Multi-Seed Statistical Benchmark")
st.markdown("""
To validate scientific claims rigorously, run repeated simulations across randomized terrain seeds.
This verifies the statistical distribution of energy savings and replanning reductions across diverse Martian topologies.
""")

col_mc1, col_mc2 = st.columns([1, 3])
with col_mc1:
    num_runs = st.select_slider("Number of Random Terrain Seeds", options=[5, 10, 15, 20], value=10)
    run_benchmark = st.button("▶️ Run Monte Carlo Benchmark", type="primary", use_container_width=True)

if run_benchmark:
    progress_bar = st.progress(0)
    status_text = st.empty()
    benchmark_records = []

    dec_eng = AdaptiveDecisionEngine()
    scout_agent = DroneScout(flight_altitude_m=12.0)

    for i in range(num_runs):
        seed_val = 100 + i * 17
        status_text.text(f"Simulating Seed {seed_val} ({i+1}/{num_runs})...")
        progress_bar.progress((i + 1) / num_runs)

        b_env = generate_planetary_terrain(size=60, seed=seed_val, preset="Martian Crater Field")
        b_terrain = compute_terrain_layers(b_env.elevation, b_env.obstacles, b_env.resolution_m)
        b_res = plan_mission(b_env, b_terrain, dec_eng, scout_agent, rover_radius=5)
        b_stats = compute_comparative_metrics(
            b_res.rover_only_telemetry,
            b_res.cooperative_telemetry,
            b_res.drone_telemetry,
            b_res.coverage_stats
        )

        benchmark_records.append({
            "Seed": seed_val,
            "Rover Baseline (Wh)": b_stats["rover_only_energy_wh"],
            "ASTRA Total (Wh)": b_stats["coop_total_energy_wh"],
            "Net Savings (Wh)": b_stats["net_energy_savings_wh"],
            "Savings (%)": b_stats["energy_savings_pct"],
            "Rover Replans": b_res.rover_only_replans,
            "ASTRA Replans": b_res.cooperative_replans,
            "Replans Avoided": b_stats["replans_avoided"],
            "Decision": b_res.decision_assessment["recommendation"]
        })

    status_text.text("Monte Carlo Benchmark Complete!")
    df_bm = pd.DataFrame(benchmark_records)

    # Box plot of energy savings
    fig_box = go.Figure()
    fig_box.add_trace(go.Box(
        y=df_bm["Savings (%)"],
        name="Net Energy Savings (%)",
        boxpoints='all',
        jitter=0.3,
        pointpos=-1.8,
        marker_color="#00FFAA"
    ))
    fig_box.update_layout(
        title=f"Distribution of Net Energy Savings Across {num_runs} Random Terrain Seeds",
        yaxis_title="Energy Savings (%)",
        margin=dict(l=20, r=20, b=20, t=35)
    )
    st.plotly_chart(fig_box, use_container_width=True)

    # Summary Statistics
    mean_sav = df_bm["Savings (%)"].mean()
    mean_wh = df_bm["Net Savings (Wh)"].mean()
    mean_replans = df_bm["Replans Avoided"].mean()
    win_rate = (df_bm["Net Savings (Wh)"] > 0).mean() * 100.0

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Mean Energy Savings", f"{mean_sav:.1f}%")
    b2.metric("Mean Energy Saved", f"{mean_wh:.1f} Wh")
    b3.metric("Mean Replans Avoided", f"{mean_replans:.1f} stops")
    b4.metric("Cooperative Win Rate", f"{win_rate:.0f}%")

    st.dataframe(df_bm, use_container_width=True, hide_index=True)
