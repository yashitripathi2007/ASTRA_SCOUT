import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from aeris.environment import generate_terrain, terrain_layers
from aeris.scout import scout_observation
from aeris.navigation import plan_route, simulate_rover_only, route_energy
from aeris.metrics import summarise

st.set_page_config(page_title="AERIS | Planetary Exploration", layout="wide")
st.title("AERIS")
st.caption("Aerial Exploration and Rover Intelligence System — cooperative planetary navigation simulator")

with st.sidebar:
    st.header("Mission settings")
    seed = st.slider("Terrain seed", 1, 50, 7)
    grid_size = st.select_slider("Terrain grid", options=[50, 60, 70, 80, 90], value=70)
    scout_radius = st.slider("Scout observation radius", 5, 22, 14)
    rover_radius = st.slider("Rover local sensor radius", 2, 10, 5)
    scout_cost = st.slider("Scout energy cost (Wh)", 0.0, 30.0, 8.0, 0.5)
    unknown_penalty = st.slider("Unknown-terrain caution penalty", 1.0, 5.0, 2.5, 0.1)
    run = st.button("Run mission simulation", type="primary", use_container_width=True)

if "result" not in st.session_state or run:
    z, obstacles, start, goal = generate_terrain(grid_size, seed)
    slope, roughness, hazard = terrain_layers(z, obstacles)
    scout_known = scout_observation(grid_size, start, goal, scout_radius)
    rover_known = simulate_rover_only(grid_size, start, rover_radius)

    # The rover-only baseline starts with its local observations. AERIS gets
    # the rover's local map plus the scout's communicated observations.
    aeris_known = np.logical_or(rover_known, scout_known)

    rover_route, rover_cost = plan_route(
        z, slope, roughness, hazard, rover_known, start, goal,
        unknown_penalty=unknown_penalty
    )
    aeris_route, aeris_cost = plan_route(
        z, slope, roughness, hazard, aeris_known, start, goal,
        unknown_penalty=unknown_penalty
    )

    st.session_state["result"] = {
        "z": z, "obstacles": obstacles, "start": start, "goal": goal,
        "slope": slope, "roughness": roughness, "hazard": hazard,
        "scout_known": scout_known, "rover_known": rover_known,
        "aeris_known": aeris_known, "rover_route": rover_route,
        "aeris_route": aeris_route, "rover_cost": rover_cost,
        "aeris_cost": aeris_cost, "scout_cost": scout_cost
    }

r = st.session_state["result"]
if r["rover_route"] is None or r["aeris_route"] is None:
    st.error("A route could not be found. Change the seed or grid settings and run again.")
    st.stop()

rover_energy = route_energy(r["rover_route"], r["slope"], r["roughness"], r["hazard"])
aeris_rover_energy = route_energy(r["aeris_route"], r["slope"], r["roughness"], r["hazard"])
total_aeris_energy = aeris_rover_energy + r["scout_cost"]
coverage = 100 * np.mean(r["aeris_known"])
baseline_coverage = 100 * np.mean(r["rover_known"])
stats = summarise(r["rover_route"], r["aeris_route"], rover_energy, total_aeris_energy)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rover-only route", f"{len(r['rover_route'])} cells")
c2.metric("AERIS rover route", f"{len(r['aeris_route'])} cells")
c3.metric("Known terrain with AERIS", f"{coverage:.1f}%")
c4.metric("AERIS total energy", f"{total_aeris_energy:.1f} Wh")

tabs = st.tabs(["Mission overview", "Terrain intelligence", "Performance comparison", "How it works"])
with tabs[0]:
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(r["z"], cmap="terrain", origin="lower")
    ax.imshow(np.ma.masked_where(~r["obstacles"], r["obstacles"]), cmap="Reds", alpha=.85, origin="lower")
    if r["rover_route"]:
        ax.plot([p[1] for p in r["rover_route"]], [p[0] for p in r["rover_route"]],
                color="deepskyblue", linewidth=2, label="Rover-only route")
    ax.plot([p[1] for p in r["aeris_route"]], [p[0] for p in r["aeris_route"]],
            color="yellow", linewidth=2.2, label="AERIS route")
    ax.scatter(r["start"][1], r["start"][0], c="lime", s=90, marker="o", label="Start")
    ax.scatter(r["goal"][1], r["goal"][0], c="magenta", s=110, marker="*", label="Goal")
    ax.set_title("Simulated planetary terrain and planned routes")
    ax.set_xlabel("X grid cell"); ax.set_ylabel("Y grid cell"); ax.legend(loc="upper right")
    st.pyplot(fig); plt.close(fig)
    st.info("Blue = rover-only route. Yellow = scout-assisted AERIS route. Red cells = simulated obstacles.")

with tabs[1]:
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        im = ax.imshow(r["slope"], cmap="magma", origin="lower")
        ax.set_title("Terrain slope (relative scale)"); fig.colorbar(im, ax=ax, fraction=.046)
        st.pyplot(fig); plt.close(fig)
    with col2:
        fig, ax = plt.subplots()
        im = ax.imshow(r["roughness"], cmap="viridis", origin="lower")
        ax.set_title("Local roughness proxy"); fig.colorbar(im, ax=ax, fraction=.046)
        st.pyplot(fig); plt.close(fig)
    fig, ax = plt.subplots()
    ax.imshow(r["aeris_known"], cmap="Blues", origin="lower", vmin=0, vmax=1)
    ax.set_title("AERIS known-terrain map (white = unknown, blue = observed)")
    st.pyplot(fig); plt.close(fig)
    st.caption("Roughness is approximated using local elevation variation. It is not a direct measurement of soil strength or wheel sinkage.")

with tabs[2]:
    st.subheader("Estimated energy comparison")
    fig, ax = plt.subplots()
    labels = ["Rover-only\n(rover energy)", "AERIS\n(rover + scout energy)"]
    values = [rover_energy, total_aeris_energy]
    bars = ax.bar(labels, values)
    ax.set_ylabel("Estimated energy (Wh; simplified model)")
    ax.set_title("Total mission energy — include scout overhead")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(), f"{v:.1f}", ha="center", va="bottom")
    st.pyplot(fig); plt.close(fig)
    st.write({
        "Rover-only estimated energy (Wh)": round(rover_energy, 2),
        "AERIS rover estimated energy (Wh)": round(aeris_rover_energy, 2),
        "Scout energy budget (Wh)": round(r["scout_cost"], 2),
        "AERIS total estimated energy (Wh)": round(total_aeris_energy, 2),
        "Rover-only observed map coverage (%)": round(baseline_coverage, 1),
        "AERIS observed map coverage (%)": round(coverage, 1),
        "Rover-only route length (cells)": len(r["rover_route"]),
        "AERIS route length (cells)": len(r["aeris_route"])
    })
    if total_aeris_energy < rover_energy:
        st.success("In this run, the estimated rover savings exceed the scout energy overhead.")
    else:
        st.warning("In this run, scout energy overhead is not recovered by rover energy savings. This is a valid result: scouting is not always beneficial.")

with tabs[3]:
    st.markdown("""
**AERIS workflow**

1. The environment creates a terrain height map and obstacle map.
2. The virtual scout observes a region ahead of the rover.
3. Terrain intelligence calculates slope, roughness and hazard indicators.
4. The scout's observed map is shared with the rover.
5. A* pathfinding searches for a low-cost route while treating unknown terrain cautiously.
6. The dashboard compares the result with a rover-only baseline.

**Important modelling assumptions**
- The terrain generator has full ground truth, but the rover and scout receive only partial maps.
- The scout observation is a simulated top-down camera footprint, not a real image-based 3D reconstruction.
- Energy values are simplified estimates for comparing strategies, not flight-qualified rover power predictions.
- This MVP uses a synthetic environment. Real HiRISE/LOLA DEM ingestion can be added after the simulation is stable.
""")

st.divider()
st.caption("AERIS MVP | Demonstration simulator. Use repeated terrain seeds before making performance claims.")
