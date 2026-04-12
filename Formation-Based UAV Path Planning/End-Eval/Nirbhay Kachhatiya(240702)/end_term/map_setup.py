import numpy as np
import matplotlib.pyplot as plt
import os

GRID_SIZE = 100
START = (5, 50)
GOAL  = (95, 50)
OBSTACLE_CENTER = (50, 50)
OBSTACLE_RADIUS = 10
SAFETY_MARGIN   = 3      

def is_in_obstacle(x, y, margin=0):
    cx, cy = OBSTACLE_CENTER
    return (x - cx) ** 2 + (y - cy) ** 2 <= (OBSTACLE_RADIUS + margin) ** 2


def build_grid(margin=SAFETY_MARGIN):
    grid = np.ones((GRID_SIZE, GRID_SIZE), dtype=bool)
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            if is_in_obstacle(x, y, margin):
                grid[y, x] = False
    return grid


def visualise_map(path=None, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")
    circle = plt.Circle(OBSTACLE_CENTER, OBSTACLE_RADIUS, color="#e05252", alpha=0.85, zorder=3)
    margin_circle = plt.Circle(OBSTACLE_CENTER, OBSTACLE_RADIUS + SAFETY_MARGIN,
                                color="#e05252", alpha=0.25, linestyle="--", fill=True, zorder=2)
    ax.add_patch(margin_circle)
    ax.add_patch(circle)    
    ax.scatter(*START, color="#52e08a", s=120, zorder=5, label="Start")
    ax.scatter(*GOAL,  color="#52b8e0", s=120, zorder=5, label="Goal")
    ax.annotate("START", START, textcoords="offset points", xytext=(6, 6),
                color="#52e08a", fontsize=9)
    ax.annotate("GOAL",  GOAL,  textcoords="offset points", xytext=(6, 6),
                color="#52b8e0", fontsize=9)
    if path is not None:
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], color="#f0c040", linewidth=2,
                zorder=4, label="Planned path")
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_aspect("equal")
    ax.set_title("Map Setup — Obstacle, Start & Goal", color="white", fontsize=13)
    ax.set_xlabel("X", color="white"); ax.set_ylabel("Y", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.legend(facecolor="#1a1f2e", labelcolor="white")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
    plt.show()


if __name__ == "__main__":
    grid = build_grid()
    print(f"Grid shape : {grid.shape}")
    print(f"Free cells : {grid.sum()} / {GRID_SIZE * GRID_SIZE}")
    visualise_map()
