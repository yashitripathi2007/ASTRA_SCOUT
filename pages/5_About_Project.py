"""
Page 5: About Project & Scientific Documentation
System architecture, mathematical models, physical formulas, and transition roadmap.
"""

import streamlit as st
from utils.styles import apply_shared_styles

st.set_page_config(page_title="About ASTRA-SCOUT", page_icon="📖", layout="wide")
apply_shared_styles()

st.title("📖 About ASTRA-SCOUT")
st.caption("AI-Powered Collaborative Rover–Drone Planetary Exploration Architecture")

st.markdown("""
## 🌌 Project Overview
**ASTRA-SCOUT** (*Autonomous Systems for Topographical Reconnaissance and Aerial-Surface Co-Exploration*) is a next-generation 
software simulator for cooperative planetary surface missions on Mars and the Moon. 

Extraterrestrial ground rovers (e.g. Curiosity, Perseverance) have historically traveled at low average speeds (< 0.15 m/s) 
due to limited onboard obstacle lookahead and the hazards of blind driving. ASTRA-SCOUT integrates an autonomous multirotor 
aerial scout drone (inspired by the Mars *Ingenuity* rotorcraft) that surveys terrain ahead of the rover, eliminating 
impassable dead-ends, reducing drive energy, and maximizing scientific exploration range.

---

## 🏛️ System Architecture

```
                               ┌─────────────────────────────────┐
                               │     Planetary Environment       │
                               │  (Elevation DEM, Hazards, Rocks)│
                               └────────────────┬────────────────┘
                                                │
                       ┌────────────────────────┴────────────────────────┐
                       ▼                                                 ▼
        ┌─────────────────────────────┐                   ┌─────────────────────────────┐
        │        Drone Scout          │                   │        Rover Agent          │
        │ Wide-area aerial survey     │                   │ Local tactile / LiDAR scan  │
        │ Altitude, FOV, Aerodynamics │                   │ Incline physics, Traction   │
        └──────────────┬──────────────┘                   └──────────────┬──────────────┘
                       │                                                 │
                       └────────────────────────┬────────────────────────┘
                                                ▼
                               ┌─────────────────────────────────┐
                               │       Sensor Fusion Grid        │
                               │ 1. Drone Aerial (Macro/Med Conf)│
                               │ 2. Rover Ground (Micro/Ground)  │
                               │ 3. AI Inferred (Spatial Prior)  │
                               │ 4. Unobserved (Caution Prior)   │
                               └────────────────┬────────────────┘
                                                │
                                                ▼
                               ┌─────────────────────────────────┐
                               │     Adaptive Decision Engine    │
                               │   "Scout or Drive?" Evaluation  │
                               │ Expected Info Gain vs Flight Wh │
                               └────────────────┬────────────────┘
                                                │
                                                ▼
                               ┌─────────────────────────────────┐
                               │   Energy-Aware Path Planner     │
                               │ Multi-Objective Directional A*  │
                               │ Dynamic Replanning Execution    │
                               └────────────────┬────────────────┘
                                                │
                                                ▼
                               ┌─────────────────────────────────┐
                               │    ASTRA-SCOUT Mission Control  │
                               │ 3D Maps, Telemetry & Analytics  │
                               └─────────────────────────────────┘
```

---

## 📐 Mathematical & Physical Modeling

### 1. Rover Tractive Physics & Directional Incline Energy
Unlike simplistic symmetric distance models, ASTRA-SCOUT calculates directional mechanical work:
$$F_{\\text{tractive}} = m g \\sin\\theta + C_{rr} m g \\cos\\theta$$

Where:
- $m$: Rover mass ($120\\text{ kg}$)
- $g$: Planetary gravitational acceleration (Mars: $3.72\\text{ m/s}^2$, Moon: $1.62\\text{ m/s}^2$)
- $\\theta$: Directional slope angle: $\\theta = \\arctan\\left(\\frac{z_2 - z_1}{d}\\right)$
  - Climbing uphill ($\\theta > 0$): Gravitational force opposes motion, increasing power demand.
  - Driving downhill ($\\theta < 0$): Gravity assists forward motion, reducing motor power to hotel loads.
- $C_{rr}$: Rolling resistance coefficient of planetary regolith ($C_{rr} \\approx 0.08$)
- Total Electrical Power:
  $$P_{\\text{elec}} = \\frac{F_{\\text{tractive}} \\cdot v}{\\eta} + P_{\\text{hotel}}$$

### 2. Drone Scout Aerodynamic Power Model
Representative of multirotor flight in thin planetary atmospheres (Ingenuity class):
$$P_{\\text{drone}} = P_{\\text{hover}} + P_{\\text{cruise}} + P_{\\text{climb}}$$
- Takeoff and landing spool-up overhead: $E_{\\text{takeoff\\_land}} = 1.2\\text{ Wh}$
- Flight energy integrated along survey waypoints:
  $$E_{\\text{flight}} = \\int P_{\\text{drone}}(t) \\, dt$$

### 3. Multi-Objective Directional A* Planner
Search minimizes a composite cost function:
$$J(u, v) = d(u, v) \\cdot \\Big(w_d + w_h H(v) + w_s S(u, v) + w_u U(v)\\Big)$$
- $H(v)$: Composite hazard index $\\in [0.0, 1.0]$
- $S(u, v)$: Directional slope between cells $u$ and $v$
- $U(v)$: Epistemic uncertainty of cell $v$ (penalizes exploring unverified territory)
- If cell $v$ is a confirmed obstacle: $J(u, v) = \\infty$ (strictly impassable)

### 4. "Scout or Drive?" Adaptive Decision Theory
The decision arbiter evaluates Expected Information Gain (EIG) and Energy Return on Investment (ROI):
$$\\Delta E_{\\text{expected}} = E_{\\text{rover, unguided}} - \\big(E_{\\text{rover, guided}} + E_{\\text{drone}}\\big)$$
- If Terrain Complexity Index $> 0.45$: Aerial scouting eliminates costly dead-ends and yields high net savings $\\implies$ `SCOUT_RECOMMENDED`.
- If Terrain Complexity Index $< 0.25$: Direct rover driving is optimal $\\implies$ `DRIVE_DIRECT`.

---

## 🚀 Transition from AERIS MVP to ASTRA-SCOUT

| Feature | Legacy AERIS MVP | ASTRA-SCOUT Production System |
| :--- | :--- | :--- |
| **Architecture** | Single monolithic script (`app.py`) | Modular 5-page Streamlit app + independent `core/` package |
| **Observation Modeling** | Flattened binary `known` mask | **4-tier distinguishable knowledge layers** (Drone, Rover, AI Inferred, Unobserved) |
| **Rover Execution** | Blind traversal through obstacles | **Dynamic step-by-step traversal with real-time replanning** |
| **Scout Model** | Static geometric rectangle & circles | **Waypoint flight planner, camera FOV projection & aerodynamics** |
| **Energy Physics** | Linear scalar formula | **Directional uphill/downhill incline work + rotorcraft power** |
| **Decision Intelligence** | None (Drone always deployed) | **Autonomous "Scout or Drive?" Decision Engine** |
| **Visualization** | Static 2D Matplotlib plots | **Interactive 3D Plotly surfaces & zoomable heatmaps** |
| **Validation** | No automated tests | **Full pytest test suite (`test_*.py`)** |

---

## 🛡️ Scientific Honesty & Demonstration Integrity
1. Extraterrestrial simulations must account for aerial flight energy overhead. ASTRA-SCOUT always reports **total mission energy** ($E_{\\text{rover}} + E_{\\text{drone}}$).
2. Aerial scouting is not a universal panacea: on completely flat or open terrain, drone deployment consumes flight energy without commensurate rover detour savings. ASTRA-SCOUT highlights these trade-offs transparently.
""")
