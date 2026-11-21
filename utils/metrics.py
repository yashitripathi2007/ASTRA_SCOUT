"""
Comparative Metrics Engine for ASTRA-SCOUT.
Calculates rigorous statistical comparisons, energy savings %, replanning reductions,
safety risk indices, and multi-objective combined mission costs.
"""

from typing import Dict, Any, Optional
from core.decision_engine import MissionCostConfig, compute_mission_cost


def compute_comparative_metrics(
    rover_only_telemetry: Dict[str, Any],
    coop_telemetry: Dict[str, Any],
    drone_telemetry: Optional[Dict[str, Any]],
    coverage_stats: Dict[str, Any],
    baseline_coverage_stats: Optional[Dict[str, Any]] = None,
    adaptive_telemetry: Optional[Dict[str, Any]] = None,
    adaptive_drone_telemetry: Optional[Dict[str, Any]] = None,
    adaptive_coverage_stats: Optional[Dict[str, Any]] = None,
    mission_costs: Optional[Dict[str, float]] = None,
    route_risks: Optional[Dict[str, float]] = None,
    cost_config: Optional[MissionCostConfig] = None
) -> Dict[str, Any]:
    """Computes comprehensive comparative metrics between baseline, always-scout, and adaptive-scout."""
    cfg = cost_config or MissionCostConfig()

    # 1. Rover Baseline
    r_energy = float(rover_only_telemetry.get("total_energy_wh", 0.0))
    r_dist = float(rover_only_telemetry.get("total_distance_m", 0.0))
    r_replans = int(rover_only_telemetry.get("replanning_events_count", 0))
    r_time = float(rover_only_telemetry.get("total_time_s", 0.0))
    base_cov = baseline_coverage_stats or {}
    baseline_uncertainty = float(base_cov.get("mean_uncertainty", 0.850))

    # 2. Always-Scout (Cooperative)
    c_rover_energy = float(coop_telemetry.get("total_energy_wh", 0.0))
    d_energy = float(drone_telemetry.get("flight_energy_wh", 0.0)) if drone_telemetry else 0.0
    c_total_energy = c_rover_energy + d_energy

    net_energy_diff = r_energy - c_total_energy
    energy_savings_pct = (net_energy_diff / r_energy * 100.0) if r_energy > 0 else 0.0

    c_dist = float(coop_telemetry.get("total_distance_m", 0.0))
    distance_savings_m = r_dist - c_dist

    c_replans = int(coop_telemetry.get("replanning_events_count", 0))
    replans_avoided = max(0, r_replans - c_replans)

    c_time = float(coop_telemetry.get("total_time_s", 0.0))
    time_saved_s = r_time - c_time

    # 3. Adaptive-Scout Strategy
    adap_rover = adaptive_telemetry or coop_telemetry
    adap_drone = adaptive_drone_telemetry
    adap_rover_energy = float(adap_rover.get("total_energy_wh", c_rover_energy))
    adap_drone_energy = float(adap_drone.get("flight_energy_wh", 0.0)) if adap_drone else 0.0
    adap_total_energy = adap_rover_energy + adap_drone_energy

    adap_net_energy_diff = r_energy - adap_total_energy
    adap_energy_savings_pct = (adap_net_energy_diff / r_energy * 100.0) if r_energy > 0 else 0.0

    adap_dist = float(adap_rover.get("total_distance_m", c_dist))
    adap_dist_savings_m = r_dist - adap_dist

    adap_replans = int(adap_rover.get("replanning_events_count", c_replans))
    adap_replans_avoided = max(0, r_replans - adap_replans)

    adap_time = float(adap_rover.get("total_time_s", c_time))
    adap_time_saved_s = r_time - adap_time

    adap_cov = adaptive_coverage_stats or coverage_stats
    adap_total_pct = float(adap_cov.get("total_known_pct", coverage_stats.get("total_known_pct", 0.0)))
    adap_mean_unc = float(adap_cov.get("mean_uncertainty", coverage_stats.get("mean_uncertainty", 0.0)))

    # 4. Route Risks and Combined Mission Costs
    risks = route_risks or {}
    costs = mission_costs or {}
    r_risk = float(risks.get("rover_only", 0.25))
    always_risk = float(risks.get("always_scout", 0.15))
    adap_risk = float(risks.get("adaptive_scout", 0.15))

    rover_cost = float(costs.get("rover_only", compute_mission_cost(r_energy, 0.0, r_risk, cfg)))
    always_cost = float(costs.get("always_scout", compute_mission_cost(c_rover_energy, d_energy, always_risk, cfg)))
    adap_cost = float(costs.get("adaptive_scout", compute_mission_cost(adap_rover_energy, adap_drone_energy, adap_risk, cfg)))

    return {
        # Rover-Only Baseline
        "rover_only_energy_wh": round(r_energy, 2),
        "rover_only_distance_m": round(r_dist, 2),
        "rover_only_replans": r_replans,
        "rover_only_time_s": round(r_time, 1),
        "baseline_mean_uncertainty": round(baseline_uncertainty, 3),

        # Always-Scout (Cooperative)
        "coop_rover_energy_wh": round(c_rover_energy, 2),
        "drone_energy_wh": round(d_energy, 2),
        "coop_total_energy_wh": round(c_total_energy, 2),
        "net_energy_savings_wh": round(net_energy_diff, 2),
        "energy_savings_pct": round(energy_savings_pct, 1),
        "coop_distance_m": round(c_dist, 2),
        "distance_savings_m": round(distance_savings_m, 2),
        "coop_replans": c_replans,
        "replans_avoided": replans_avoided,
        "coop_time_s": round(c_time, 1),
        "time_savings_s": round(time_saved_s, 1),
        "coverage_total_pct": round(float(coverage_stats.get("total_known_pct", 0.0)), 1),
        "coverage_aerial_pct": round(float(coverage_stats.get("aerial_scout_pct", 0.0)), 1),
        "coverage_ground_pct": round(float(coverage_stats.get("rover_ground_pct", 0.0)), 1),
        "coverage_inferred_pct": round(float(coverage_stats.get("inferred_pct", 0.0)), 1),
        "coverage_unobserved_pct": round(float(coverage_stats.get("unobserved_pct", 0.0)), 1),
        "mean_uncertainty": round(float(coverage_stats.get("mean_uncertainty", 0.0)), 3),
        "cooperative_advantage": net_energy_diff > 0.0,

        # Adaptive-Scout
        "adaptive_rover_energy_wh": round(adap_rover_energy, 2),
        "adaptive_drone_energy_wh": round(adap_drone_energy, 2),
        "adaptive_total_energy_wh": round(adap_total_energy, 2),
        "adaptive_net_energy_savings_wh": round(adap_net_energy_diff, 2),
        "adaptive_energy_savings_pct": round(adap_energy_savings_pct, 1),
        "adaptive_distance_m": round(adap_dist, 2),
        "adaptive_distance_savings_m": round(adap_dist_savings_m, 2),
        "adaptive_replans": adap_replans,
        "adaptive_replans_avoided": adap_replans_avoided,
        "adaptive_time_s": round(adap_time, 1),
        "adaptive_coverage_total_pct": round(adap_total_pct, 1),
        "adaptive_mean_uncertainty": round(adap_mean_unc, 3),

        # Route Risks & Combined Multi-Objective Costs
        "rover_only_risk": round(r_risk, 3),
        "always_scout_risk": round(always_risk, 3),
        "adaptive_scout_risk": round(adap_risk, 3),
        "rover_only_cost": round(rover_cost, 2),
        "always_scout_cost": round(always_cost, 2),
        "adaptive_scout_cost": round(adap_cost, 2),
    }
