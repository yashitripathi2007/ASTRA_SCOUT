"""
Physics-Based Energy Models for ASTRA-SCOUT.
Models directional rover wheel tractive dynamics (uphill vs downhill gravity components,
rolling resistance, turning overhead) and multirotor drone flight aerodynamics.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np


@dataclass
class RoverEnergyModel:
    """
    Physical energy model for a planetary exploration rover.
    
    Physics:
    - Force balance: F_tractive = m * g * sin(theta) + C_rr * m * g * cos(theta)
    - Directional slope: uphill requires climbing against gravity; downhill assists motion.
    - Power = (F_tractive * velocity) / drivetrain_efficiency + P_hotel
    """
    mass_kg: float = 120.0             # Rover total mass (kg)
    gravity_ms2: float = 3.72          # Gravitational acceleration (Mars = 3.72, Moon = 1.62)
    nominal_speed_ms: float = 0.15     # Surface speed (m/s)
    rolling_resistance: float = 0.08   # Regolith rolling resistance coefficient
    motor_efficiency: float = 0.74     # Electrical-to-mechanical power efficiency
    hotel_load_w: float = 18.0         # Avionics, computing, telemetry background power (W)
    turn_penalty_wh: float = 0.015     # Energy penalty per 45-degree steering maneuver

    def step_energy(
        self,
        p1: Tuple[int, int],
        p2: Tuple[int, int],
        z1: float,
        z2: float,
        roughness: float,
        hazard: float,
        resolution_m: float = 1.0,
        heading_change_deg: float = 0.0
    ) -> Dict[str, float]:
        """Calculates electrical energy (Wh) and power for moving from p1 to p2."""
        dy = p2[0] - p1[0]
        dx = p2[1] - p1[1]
        dist_m = np.hypot(dy, dx) * resolution_m
        delta_z = z2 - z1
        incline_rad = np.arctan2(delta_z, dist_m)
        incline_deg = np.degrees(incline_rad)

        mg = self.mass_kg * self.gravity_ms2
        # Gravity force along slope (+ uphill, - downhill)
        f_gravity = mg * np.sin(incline_rad)
        # Normal force & rolling resistance
        f_normal = mg * np.cos(incline_rad)
        # Surface roughness increases effective rolling friction
        effective_crr = self.rolling_resistance * (1.0 + 1.2 * min(1.0, roughness))
        f_rolling = effective_crr * f_normal

        # Net tractive mechanical force
        f_net = f_gravity + f_rolling

        if f_net > 0:
            # Overcoming gravity and friction
            p_mech = f_net * self.nominal_speed_ms
            p_elec = (p_mech / self.motor_efficiency) + self.hotel_load_w
        else:
            # Gravity assist on downhill: motors operate in holding/regenerative braking mode
            # Still consumes hotel load and minimum motor holding current
            p_elec = self.hotel_load_w + 3.0

        # Duration of step in seconds
        duration_s = dist_m / max(0.01, self.nominal_speed_ms)
        step_wh = (p_elec * duration_s) / 3600.0

        # Steering maneuver energy
        if heading_change_deg > 10.0:
            step_wh += (heading_change_deg / 45.0) * self.turn_penalty_wh

        return {
            "energy_wh": float(step_wh),
            "power_w": float(p_elec),
            "duration_s": float(duration_s),
            "distance_m": float(dist_m),
            "incline_deg": float(incline_deg),
        }

    def compute_path_energy(
        self,
        path: List[Tuple[int, int]],
        elevation: np.ndarray,
        roughness: np.ndarray,
        hazard: np.ndarray,
        resolution_m: float = 1.0
    ) -> Dict[str, float]:
        """Computes full mission energy and telemetry for a rover path."""
        if not path or len(path) < 2:
            return {"total_energy_wh": 0.0, "total_distance_m": 0.0, "total_time_s": 0.0}

        total_wh = 0.0
        total_dist = 0.0
        total_time = 0.0
        prev_heading = 0.0

        for i in range(len(path) - 1):
            p1, p2 = path[i], path[i + 1]
            z1, z2 = elevation[p1], elevation[p2]
            rough = roughness[p2]
            haz = hazard[p2]

            curr_heading = np.degrees(np.arctan2(p2[0] - p1[0], p2[1] - p1[1]))
            heading_change = 0.0 if i == 0 else abs(curr_heading - prev_heading)
            if heading_change > 180.0:
                heading_change = 360.0 - heading_change
            prev_heading = curr_heading

            step_data = self.step_energy(
                p1, p2, z1, z2, rough, haz, resolution_m, heading_change
            )
            total_wh += step_data["energy_wh"]
            total_dist += step_data["distance_m"]
            total_time += step_data["duration_s"]

        return {
            "total_energy_wh": float(round(total_wh, 2)),
            "total_distance_m": float(round(total_dist, 2)),
            "total_time_s": float(round(total_time, 1)),
            "average_power_w": float(round((total_wh * 3600.0) / max(1.0, total_time), 1)),
        }


@dataclass
class DroneEnergyModel:
    """
    Aerodynamic power model for a planetary exploration scout drone (multirotor).
    Representative of Mars rotorcraft (Ingenuity class).
    """
    drone_mass_kg: float = 1.8           # Flight vehicle mass (kg)
    gravity_ms2: float = 3.72           # Planetary gravity (m/s^2)
    hover_power_w: float = 240.0        # Power required to maintain steady hover (W)
    cruise_speed_ms: float = 4.0        # Horizontal flight velocity (m/s)
    cruise_power_w: float = 210.0       # Forward cruising aerodynamic power (W)
    climb_power_w: float = 280.0        # Ascending flight power (W)
    takeoff_landing_wh: float = 1.2     # Energy overhead for spool-up, takeoff and touchdown (Wh)
    battery_capacity_wh: float = 40.0   # Onboard battery capacity (Wh)

    def flight_energy(
        self,
        flight_distance_m: float,
        hover_duration_s: float = 10.0,
        altitude_m: float = 12.0
    ) -> Dict[str, float]:
        """Calculates electrical flight energy (Wh) for an aerial survey mission."""
        cruise_time_s = flight_distance_m / max(0.5, self.cruise_speed_ms)
        climb_time_s = (altitude_m * 2.0) / 1.5  # Climb and descent at 1.5 m/s

        e_cruise = (self.cruise_power_w * cruise_time_s) / 3600.0
        e_hover = (self.hover_power_w * hover_duration_s) / 3600.0
        e_climb = (self.climb_power_w * climb_time_s) / 3600.0
        total_energy_wh = e_cruise + e_hover + e_climb + self.takeoff_landing_wh

        total_time_s = cruise_time_s + hover_duration_s + climb_time_s
        battery_consumed_pct = 100.0 * (total_energy_wh / self.battery_capacity_wh)

        return {
            "flight_energy_wh": float(round(total_energy_wh, 2)),
            "flight_distance_m": float(round(flight_distance_m, 1)),
            "flight_time_s": float(round(total_time_s, 1)),
            "battery_consumed_pct": float(round(battery_consumed_pct, 1)),
            "battery_remaining_pct": float(round(max(0.0, 100.0 - battery_consumed_pct), 1)),
        }
