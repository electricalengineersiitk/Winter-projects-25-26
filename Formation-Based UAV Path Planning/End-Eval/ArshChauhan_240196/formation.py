import numpy as np
import matplotlib.pyplot as plt
from map_setup import plot_map, START

N_DRONES = 5

RAW_OFFSETS = [
    ( 0.0,  0.0),
    (-2.0, -2.0),
    (-4.0, -4.0),
    (-2.0,  2.0),
    (-4.0,  4.0),
]

OFFSETS = np.array(RAW_OFFSETS, dtype=float)

COLORS = ["red", "blue", "green", "purple", "orange"]

def get_drone_positions(cx, cy, heading=0.0):
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)

    positions = np.zeros((N_DRONES, 2))
    for i in range(N_DRONES):
        dx, dy = OFFSETS[i]
        rx = dx * cos_h - dy * sin_h
        ry = dx * sin_h + dy * cos_h
        positions[i] = [cx + rx, cy + ry]

    return positions

def compute_all_positions(traj):
    T = len(traj["x"])
    all_positions = np.zeros((T, N_DRONES, 2))

    for step in range(T):
        cx = traj["x"][step]
        cy = traj["y"][step]
        vx = traj["vx"][step]
        vy = traj["vy"][step]
        heading = np.arctan2(vy, vx) if (abs(vx) + abs(vy)) > 0.01 else 0.0
        all_positions[step] = get_drone_positions(cx, cy, heading)

    return all_positions

if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.set_title("V-Formation Shape (5 Drones)")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    for i, (dx, dy) in enumerate(OFFSETS):
        ax.scatter(dx, dy, s=200, color=COLORS[i], zorder=5)
        ax.text(dx + 0.1, dy + 0.1, f"D{i}", fontsize=9)
    ax.set_xlabel("dx offset")
    ax.set_ylabel("dy offset")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    
    plot_map(axes[1], title="Formation Preview at Start")
    preview = get_drone_positions(*START, heading=0.0)
    for i in range(N_DRONES):
        axes[1].scatter(*preview[i], s=80, color=COLORS[i],
                        label=f"D{i}", zorder=6)
    axes[1].legend(loc="upper right", fontsize=7)

    plt.tight_layout()
    plt.savefig("results/formation_preview.png", dpi=100)
    plt.show()
    print("Saved: results/formation_preview.png")
