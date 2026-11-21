# AERIS MVP — beginner setup guide

AERIS is a small, software-only planetary exploration simulator. It demonstrates a virtual aerial scout sharing partial terrain observations with a ground rover, which then uses A* pathfinding to plan a route.

## What you need

- A Windows 10/11 laptop
- Internet for the initial Python/package installation
- About 1 GB of free space
- No Arduino, drone, rover or GPU required

## Part 1 — Install Python

1. Open https://www.python.org/downloads/
2. Download a current Python 3.11 or 3.12 Windows installer.
3. Run it.
4. **Important:** tick the checkbox labelled `Add Python to PATH`.
5. Click Install.
6. Open the Windows Start menu, type `cmd`, and open Command Prompt.
7. Type `python --version` and press Enter. You should see a Python version.

If Windows opens the Microsoft Store instead, install Python from python.org and restart Command Prompt.

## Part 2 — Open the project folder

1. Extract `AERIS_MVP.zip`.
2. Put the extracted `AERIS_MVP` folder somewhere easy to find, such as Desktop.
3. Open that folder in File Explorer.
4. Click the address bar, type `cmd`, and press Enter. This opens Command Prompt in the project folder.

## Part 3 — Create an isolated Python environment

In Command Prompt, run these commands one at a time:

```bash
python -m venv .venv
.venv\\Scripts\\activate
```

If activation is successful, `(.venv)` appears at the beginning of the command line.

## Part 4 — Install the required libraries

Run:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This installs:
- NumPy: numerical arrays and terrain calculations
- Matplotlib: map plots
- Streamlit: the browser-based dashboard

## Part 5 — Run AERIS

Run:

```bash
streamlit run app.py
```

A browser tab should open automatically. If it does not, copy the local URL shown in Command Prompt (usually `http://localhost:8501`) into Chrome.

To stop the app, return to Command Prompt and press `Ctrl + C`.

## What to click in the dashboard

1. Choose a terrain seed in the left sidebar.
2. Change the scout observation radius or rover sensor radius.
3. Click **Run mission simulation**.
4. Look at the Mission overview tab to compare routes.
5. Look at Terrain intelligence to see slope, roughness and observed areas.
6. Look at Performance comparison to inspect the energy estimates.
7. Change the seed and rerun to test different terrain layouts.

## What the code means — in plain English

- `app.py`: the dashboard; it calls the other files and displays results.
- `aeris/environment.py`: creates the simulated landscape and calculates terrain features.
- `aeris/scout.py`: decides which cells the aerial scout has observed.
- `aeris/navigation.py`: finds routes using A* and estimates movement energy.
- `aeris/metrics.py`: gathers a few comparison values.

A **grid cell** is one square in the simulated map. A **DEM** is a digital elevation model: a grid in which each cell stores terrain height. **A*** is a pathfinding algorithm that searches for a low-cost route from a start cell to a goal cell. **Known terrain** means a simulated agent has observed that part of the map; it does not mean the area is safe.

## Important scientific honesty

This MVP uses a synthetic height map. The virtual scout's observation is a simulated top-down camera footprint over that map. It does not reconstruct terrain from a real photograph. The energy calculation is a simplified comparative estimate, not a validated model of a specific rover.

Do not claim that AERIS has proven a real-world percentage energy saving from one simulation. Run several terrain seeds, report the mean result, and show cases where scouting helps and where its energy overhead does not pay off.

## Suggested 12–15 hour team plan

| Time | Work |
|---|---|
| 0–1.5 h | Install Python, install packages, run the app |
| 1.5–3 h | Understand the dashboard and explain each module |
| 3–5 h | Test multiple seeds and fix any setup/runtime issues |
| 5–7 h | Improve the scout observation pattern and add a scout marker/coverage visual |
| 7–9 h | Add a rover battery reserve rule and a return-to-base check |
| 9–11 h | Run repeated experiments; save results for at least 5–10 seeds |
| 11–13 h | Improve dashboard labels, presentation and demo sequence |
| 13–15 h | Rehearse pitch, prepare screenshots/video backup, test on presentation laptop |

If your team has fewer than 12 hours, do not add machine learning, ROS, photogrammetry or real DEM downloads. A stable, well-explained simulation is stronger than several unfinished features.

## Recommended demo sequence

1. Run a seed with a narrow scout radius.
2. Show the terrain map, hazards and both routes.
3. Increase scout radius and rerun.
4. Show how the rover's known map and route can change.
5. Open the energy comparison.
6. Explain that total energy includes scout overhead and that scouting is not guaranteed to help every mission.

## Troubleshooting

**`python is not recognized`**
- Reinstall Python from python.org and tick Add Python to PATH.
- Close and reopen Command Prompt.

**`No module named streamlit`**
- Make sure `(.venv)` is visible in Command Prompt.
- Run `pip install -r requirements.txt` again.

**Browser page does not open**
- Copy the `http://localhost:8501` address printed in Command Prompt into your browser.

**Port 8501 is already in use**
- Run `streamlit run app.py --server.port 8502`.
- Open `http://localhost:8502`.

**The app shows that no route could be found**
- Change the terrain seed and click Run mission simulation.
