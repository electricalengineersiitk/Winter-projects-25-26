import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

MAP_WIDTH  = 100
MAP_HEIGHT = 100

START = (5, 50)
GOAL  = (95, 50)

OBSTACLE_CENTER = (50, 50)
OBSTACLE_RADIUS = 10
SAFETY_MARGIN = 3          
GRID_RESOLUTION = 1        


def build_obstacle_grid(resolution=GRID_RESOLUTION,
                        safety_margin=SAFETY_MARGIN):
    cols = int(MAP_WIDTH  / resolution)
    rows = int(MAP_HEIGHT / resolution)
    grid = np.zeros((rows, cols), dtype=bool)

    cx, cy = OBSTACLE_CENTER
    effective_r = OBSTACLE_RADIUS + safety_margin

    for row in range(rows):
        for col in range(cols):
            # cell centre in world coordinates
            wx = col * resolution + resolution / 2
            wy = row * resolution + resolution / 2
            if (wx - cx) ** 2 + (wy - cy) ** 2 <= effective_r ** 2:
                grid[row, col] = True
    return grid


def world_to_grid(x, y, resolution=GRID_RESOLUTION):
    col = int(x / resolution)
    row = int(y / resolution)
    return col, row


def grid_to_world(col, row, resolution=GRID_RESOLUTION):
    x = col * resolution + resolution / 2
    y = row * resolution + resolution / 2
    return x, y

def visualise_map(path=None, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, MAP_WIDTH)
    ax.set_ylim(0, MAP_HEIGHT)
    ax.set_aspect('equal')
    ax.set_facecolor('#f0f4f8')
    fig.patch.set_facecolor('#e8edf2')

    safety_circle = plt.Circle(
        OBSTACLE_CENTER,
        OBSTACLE_RADIUS + SAFETY_MARGIN,
        color='#f4c430', alpha=0.3, label='Safety margin'
    )
    ax.add_patch(safety_circle)

    # Obstacle
    obs_circle = plt.Circle(
        OBSTACLE_CENTER,
        OBSTACLE_RADIUS,
        color='#c0392b', alpha=0.85, label='Obstacle'
    )
    ax.add_patch(obs_circle)

    # Start & goal
    ax.plot(*START, 's', color='#27ae60', markersize=12,
            zorder=5, label='Start')
    ax.plot(*GOAL,  '*', color='#2980b9', markersize=14,
            zorder=5, label='Goal')

    if path is not None:
        px, py = zip(*path)
        ax.plot(px, py, '-o', color='#8e44ad', linewidth=2,
                markersize=3, label='Planned path', zorder=4)

    ax.set_title('UAV Formation Path Planning — Map Setup', fontsize=13, fontweight='bold')
    ax.set_xlabel('X (units)')
    ax.set_ylabel('Y (units)')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"[map_setup] Map saved → {save_path}")
    plt.close(fig)
    return fig, ax


if __name__ == '__main__':
    grid = build_obstacle_grid()
    print(f"Grid shape : {grid.shape}")
    print(f"Blocked cells : {grid.sum()}")
    sc, sr = world_to_grid(*START)
    gc, gr = world_to_grid(*GOAL)
    print(f"Start grid cell : col={sc}, row={sr}  blocked={grid[sr, sc]}")
    print(f"Goal  grid cell : col={gc}, row={gr}  blocked={grid[gr, gc]}")
    visualise_map(save_path=os.path.join('results', 'map_only.png'))
    print("Map visualisation complete.")