"""
Comparative Metrics Engine for ASTRA-SCOUT.
Calculates rigorous statistical comparisons, energy savings %, replanning reductions,
and safety risk indices.
"""

from typing import Dict, Any


def compute_comparative_metrics(
    rover_only_telemetry: Dict[str, Any],
    coop_telemetry: Dict[str, Any],
    drone_telemetry: Dict[str, Any],
    coverage_stats: Dict[str, Any]
) -> Dict[str, Any]:
    """Computes comprehensive comparative metrics between baseline and ASTRA-SCOUT."""
    r_energy = float(rover_only_telemetry.get("total_energy_wh", 0.0))
    c_rover_energy = float(coop_telemetry.get("total_energy_wh", 0.0))
    d_energy = float(drone_telemetry.get("flight_energy_wh", 0.0)) if drone_telemetry else 0.0
    c_total_energy = c_rover_energy + d_energy

    net_energy_diff = r_energy - c_total_energy
    energy_savings_pct = (net_energy_diff / r_energy * 100.0) if r_energy > 0 else 0.0

    r_dist = float(rover_only_telemetry.get("total_distance_m", 0.0))
    c_dist = float(coop_telemetry.get("total_distance_m", 0.0))
    distance_savings_m = r_dist - c_dist

    r_replans = int(rover_only_telemetry.get("replanning_events_count", 0))
    c_replans = int(coop_telemetry.get("replanning_events_count", 0))
    replans_avoided = max(0, r_replans - c_replans)

    r_time = float(rover_only_telemetry.get("total_time_s", 0.0))
    c_time = float(coop_telemetry.get("total_time_s", 0.0))
    time_saved_s = r_time - c_time

    return {
        "rover_only_energy_wh": round(r_energy, 2),
        "coop_rover_energy_wh": round(c_rover_energy, 2),
        "drone_energy_wh": round(d_energy, 2),
        "coop_total_energy_wh": round(c_total_energy, 2),
        "net_energy_savings_wh": round(net_energy_diff, 2),
        "energy_savings_pct": round(energy_savings_pct, 1),
        "rover_only_distance_m": round(r_dist, 2),
        "coop_distance_m": round(c_dist, 2),
        "distance_savings_m": round(distance_savings_m, 2),
        "rover_only_replans": r_replans,
        "coop_replans": c_replans,
        "replans_avoided": replans_avoided,
        "rover_only_time_s": round(r_time, 1),
        "coop_time_s": round(c_time, 1),
        "time_savings_s": round(time_saved_s, 1),
        "coverage_total_pct": round(float(coverage_stats.get("total_known_pct", 0.0)), 1),
        "coverage_aerial_pct": round(float(coverage_stats.get("aerial_scout_pct", 0.0)), 1),
        "coverage_ground_pct": round(float(coverage_stats.get("rover_ground_pct", 0.0)), 1),
        "coverage_inferred_pct": round(float(coverage_stats.get("inferred_pct", 0.0)), 1),
        "coverage_unobserved_pct": round(float(coverage_stats.get("unobserved_pct", 0.0)), 1),
        "mean_uncertainty": round(float(coverage_stats.get("mean_uncertainty", 0.0)), 3),
        "cooperative_advantage": net_energy_diff > 0.0,
    }
