from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
import numpy as np

from formation import compute_headings, create_letter_a_formation, formation_positions
from map_setup import default_map, results_dir
from path_planner import astar_path
from trajectory import TrajectoryResult, generate_trajectory


RESULTS_DIR = results_dir()


def _setup_axes(ax, config):
    ax.set_xlim(0, config.width)
    ax.set_ylim(0, config.height)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.25)
    obstacle = Circle(config.obstacle_center, config.obstacle_radius, color="#b23a48", alpha=0.35, label="Obstacle")
    inflated = Circle(config.obstacle_center, config.inflated_radius, color="#b23a48", alpha=0.12, linestyle="--", fill=True)
    ax.add_patch(inflated)
    ax.add_patch(obstacle)
    ax.scatter(*config.start, c="#2f9e44", s=80, marker="o", label="Start")
    ax.scatter(*config.goal, c="#e67700", s=80, marker="*", label="Goal")


def _plot_path(config, waypoints):
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    _setup_axes(ax, config)
    waypoints_array = np.asarray(waypoints, dtype=float)
    ax.plot(waypoints_array[:, 0], waypoints_array[:, 1], color="#1c7ed6", linewidth=2.5, marker="o", label="A* path")
    ax.set_title("Collision-Free Path Around the Obstacle")
    ax.legend(loc="upper left")
    fig.savefig(RESULTS_DIR / "path_plot.png", dpi=200)
    plt.close(fig)


def _plot_trajectory_comparison(min_time: TrajectoryResult, min_energy: TrajectoryResult):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    axes[0].plot(min_time.time, min_time.speed, label="Minimum-time", color="#1c7ed6")
    axes[0].plot(min_energy.time, min_energy.speed, label="Minimum-energy", color="#e8590c")
    axes[0].set_title("Speed vs Time")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Speed")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(min_time.time, min_time.acceleration_magnitude, label="Minimum-time", color="#1c7ed6")
    axes[1].plot(min_energy.time, min_energy.acceleration_magnitude, label="Minimum-energy", color="#e8590c")
    axes[1].set_title("Acceleration vs Time")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Acceleration")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle("Trajectory Comparison")
    fig.savefig(RESULTS_DIR / "trajectory_comparison.png", dpi=200)
    plt.close(fig)


def _sample_trajectory_state(trajectory: TrajectoryResult, query_time: float) -> tuple[np.ndarray, float]:
    time = trajectory.time
    heading_series = compute_headings(trajectory.time, trajectory.positions)
    if query_time <= time[0]:
        heading = float(heading_series[0])
        return trajectory.positions[0], heading
    if query_time >= time[-1]:
        heading = float(heading_series[-1])
        return trajectory.positions[-1], heading

    x = float(np.interp(query_time, time, trajectory.positions[:, 0]))
    y = float(np.interp(query_time, time, trajectory.positions[:, 1]))
    heading = float(np.interp(query_time, time, heading_series))
    return np.array([x, y]), heading


def _formation_at_time(trajectory: TrajectoryResult, formation, query_time: float) -> np.ndarray:
    centroid, heading = _sample_trajectory_state(trajectory, query_time)
    return formation_positions(centroid[None, :], np.array([heading]), formation.offsets)[0]


def _animate(config, waypoints, min_time: TrajectoryResult, min_energy: TrajectoryResult, formation):
    waypoints_array = np.asarray(waypoints, dtype=float)
    max_duration = max(min_time.duration, min_energy.duration)
    frame_count = 180
    frame_times = np.linspace(0.0, max_duration, frame_count)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    titles = ["Minimum-time formation flight", "Minimum-energy formation flight"]
    trajectories = [min_time, min_energy]
    drone_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(formation.offsets)))
    line_color = "#495057"
    scatters = []
    centroid_points = []
    connection_lines = []

    for ax, title in zip(axes, titles):
        _setup_axes(ax, config)
        ax.plot(waypoints_array[:, 0], waypoints_array[:, 1], color="#1c7ed6", linewidth=2.0, alpha=0.7, label="Planned path")
        ax.set_title(title)
        initial_positions = _formation_at_time(trajectories[len(scatters)], formation, 0.0)
        centroid_point = ax.scatter([], [], c="#111111", s=40, marker="x", label="Centroid")
        centroid_points.append(centroid_point)
        scatter = ax.scatter(initial_positions[:, 0], initial_positions[:, 1], s=70, c=drone_colors, edgecolors="#ffffff", linewidths=0.6, zorder=4)
        scatters.append(scatter)
        lines = []
        for _ in formation.connections:
            line, = ax.plot([], [], color=line_color, linewidth=1.4, alpha=0.8)
            lines.append(line)
        connection_lines.append(lines)
        ax.legend(loc="upper left")

    time_texts = [axes[0].text(0.02, 0.95, "", transform=axes[0].transAxes), axes[1].text(0.02, 0.95, "", transform=axes[1].transAxes)]

    def init():
        for scatter in scatters:
            scatter.set_offsets(np.empty((0, 2)))
        for centroid_point in centroid_points:
            centroid_point.set_offsets(np.empty((0, 2)))
        for lines in connection_lines:
            for line in lines:
                line.set_data([], [])
        for text in time_texts:
            text.set_text("")
        return scatters + centroid_points + [line for lines in connection_lines for line in lines] + time_texts

    def update(frame_index):
        current_time = frame_times[frame_index]
        artists = []
        for i, trajectory in enumerate(trajectories):
            centroid, heading = _sample_trajectory_state(trajectory, current_time)
            drone_positions = _formation_at_time(trajectory, formation, current_time)
            scatters[i].set_offsets(drone_positions)
            centroid_points[i].set_offsets(centroid[None, :])
            for line, (start_idx, end_idx) in zip(connection_lines[i], formation.connections):
                segment = drone_positions[[start_idx, end_idx]]
                line.set_data(segment[:, 0], segment[:, 1])
            time_texts[i].set_text(f"t = {current_time:.2f} s")
            artists.extend([scatters[i], centroid_points[i], *connection_lines[i], time_texts[i]])
        return artists

    animation = FuncAnimation(fig, update, frames=len(frame_times), init_func=init, blit=True, interval=50)
    animation.save(RESULTS_DIR / "formation_animation.gif", writer=PillowWriter(fps=20))
    plt.close(fig)


def _print_summary(min_time: TrajectoryResult, min_energy: TrajectoryResult):
    print("Simulation summary")
    print(f"  Minimum-time:   duration={min_time.duration:.2f}s, distance={min_time.distance:.2f}, energy={min_time.energy:.2f}")
    print(f"  Minimum-energy:  duration={min_energy.duration:.2f}s, distance={min_energy.distance:.2f}, energy={min_energy.energy:.2f}")
    time_diff = min_energy.duration - min_time.duration
    energy_diff = min_time.energy - min_energy.energy
    print(f"  Time advantage:  {time_diff:.2f}s longer for minimum-energy")
    print(f"  Energy savings:  {energy_diff:.2f} proxy units in minimum-energy")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config = default_map()
    formation = create_letter_a_formation(scale=1.0)
    path_result = astar_path(config)
    waypoints = path_result.waypoints

    min_time = generate_trajectory(waypoints, mode="minimum-time")
    min_energy = generate_trajectory(waypoints, mode="minimum-energy")

    _plot_path(config, waypoints)
    _plot_trajectory_comparison(min_time, min_energy)
    _animate(config, waypoints, min_time, min_energy, formation)
    _print_summary(min_time, min_energy)
    print(f"Visited nodes during A*: {path_result.visited_nodes}")
    print(f"Saved outputs to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
