from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.interpolate import CubicSpline

Waypoint = Tuple[float, float]


@dataclass
class TrajectoryResult:
    mode: str
    time: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    speed: np.ndarray
    acceleration_magnitude: np.ndarray
    distance: float
    duration: float
    energy: float


def _arc_length_parameterization(waypoints: List[Waypoint]) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(waypoints, dtype=float)
    if len(points) < 2:
        raise ValueError("At least two waypoints are required.")

    deltas = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    arc_length = np.concatenate(([0.0], np.cumsum(segment_lengths)))

    # Remove duplicate points to keep the spline well-conditioned.
    unique_indices = np.concatenate(([0], np.where(np.diff(arc_length) > 1e-9)[0] + 1))
    arc_length = arc_length[unique_indices]
    points = points[unique_indices]
    if len(points) < 2:
        raise ValueError("Waypoints collapse to a single point after filtering duplicates.")
    return arc_length, points


def _smoothstep(u: np.ndarray) -> np.ndarray:
    return 6.0 * u**5 - 15.0 * u**4 + 10.0 * u**3


def _smoothstep_derivative(u: np.ndarray) -> np.ndarray:
    return 30.0 * u**4 - 60.0 * u**3 + 30.0 * u**2


def _smoothstep_second_derivative(u: np.ndarray) -> np.ndarray:
    return 120.0 * u**3 - 180.0 * u**2 + 60.0 * u


def _trajectory_profile(mode: str, total_distance: float) -> tuple[float, str]:
    if mode == "minimum-time":
        return max(14.0, total_distance / 8.0), "constant"
    if mode == "minimum-energy":
        return max(7.0, total_distance / 15.0), "smooth"
    raise ValueError(f"Unsupported trajectory mode: {mode}")


def generate_trajectory(waypoints: List[Waypoint], mode: str) -> TrajectoryResult:
    arc_length, points = _arc_length_parameterization(waypoints)
    total_distance = float(arc_length[-1])
    speed_setting, profile = _trajectory_profile(mode, total_distance)

    spline_x = CubicSpline(arc_length, points[:, 0], bc_type="natural")
    spline_y = CubicSpline(arc_length, points[:, 1], bc_type="natural")

    if profile == "constant":
        duration = total_distance / speed_setting
        sample_count = max(240, int(duration * 40))
        time = np.linspace(0.0, duration, sample_count)
        tau = time / duration if duration > 0 else np.zeros_like(time)
        arc_positions = total_distance * tau
        arc_velocity = np.full_like(time, total_distance / duration if duration > 0 else 0.0)
        arc_acceleration = np.zeros_like(time)
    else:
        duration = total_distance / speed_setting * 1.25
        sample_count = max(260, int(duration * 50))
        time = np.linspace(0.0, duration, sample_count)
        tau = time / duration if duration > 0 else np.zeros_like(time)
        arc_positions = total_distance * _smoothstep(tau)
        arc_velocity = total_distance * _smoothstep_derivative(tau) / duration if duration > 0 else np.zeros_like(time)
        arc_acceleration = total_distance * _smoothstep_second_derivative(tau) / (duration**2) if duration > 0 else np.zeros_like(time)

    x = spline_x(arc_positions)
    y = spline_y(arc_positions)
    positions = np.column_stack((x, y))

    dx_ds = spline_x.derivative(1)(arc_positions)
    dy_ds = spline_y.derivative(1)(arc_positions)
    d2x_ds2 = spline_x.derivative(2)(arc_positions)
    d2y_ds2 = spline_y.derivative(2)(arc_positions)

    velocities = np.column_stack((dx_ds * arc_velocity, dy_ds * arc_velocity))
    accelerations = np.column_stack(
        (
            d2x_ds2 * arc_velocity**2 + dx_ds * arc_acceleration,
            d2y_ds2 * arc_velocity**2 + dy_ds * arc_acceleration,
        )
    )

    speed = np.linalg.norm(velocities, axis=1)
    acceleration_magnitude = np.linalg.norm(accelerations, axis=1)
    energy = float(np.trapezoid(speed**2 + 0.25 * acceleration_magnitude**2, time))

    return TrajectoryResult(
        mode=mode,
        time=time,
        positions=positions,
        velocities=velocities,
        accelerations=accelerations,
        speed=speed,
        acceleration_magnitude=acceleration_magnitude,
        distance=total_distance,
        duration=duration,
        energy=energy,
    )
