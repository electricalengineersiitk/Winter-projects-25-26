import numpy as np
import matplotlib.pyplot as plt

FORMATION_OFFSETS_RAW = np.array([
    (-2.0, -2.0),
    (-1.5, -0.5),
    (-1.0,  1.0),
    ( 2.0, -2.0),
    ( 1.5, -0.5),
    ( 1.0,  1.0),
    ( 0.0,  2.5),
    (-0.8,  0.0),
    ( 0.8,  0.0),
], dtype=float)

N_DRONES = len(FORMATION_OFFSETS_RAW)
FORMATION_OFFSETS = FORMATION_OFFSETS_RAW - FORMATION_OFFSETS_RAW.mean(axis=0)


def get_drone_positions(centroid_x, centroid_y):
    centroid = np.array([centroid_x, centroid_y])
    return centroid + FORMATION_OFFSETS


def get_all_positions_along_traj(trajectory):
    cx = trajectory[:, 0]
    cy = trajectory[:, 1]
    centroids = np.stack([cx, cy], axis=1)[:, np.newaxis, :]     
    offsets   = FORMATION_OFFSETS[np.newaxis, :, :]                
    return centroids + offsets                                    


def visualise_formation():
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, N_DRONES))
    for i, (dx, dy) in enumerate(FORMATION_OFFSETS):
        ax.scatter(dx, dy, color=colors[i], s=180, zorder=3)
        ax.annotate(f"D{i}", (dx, dy), textcoords="offset points",
                    xytext=(5, 5), color="white", fontsize=8)
    ax.scatter(0, 0, marker="+", color="white", s=200, zorder=4, label="Centroid")
    ax.set_title("Formation Shape — 'A' (9 UAVs)", color="white", fontsize=12)
    ax.set_xlabel("dx", color="white"); ax.set_ylabel("dy", color="white")
    ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#333")
    ax.set_aspect("equal")
    ax.legend(facecolor="#1a1f2e", labelcolor="white")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print(f"Number of drones : {N_DRONES}")
    print("Offsets (dx, dy):")
    for i, off in enumerate(FORMATION_OFFSETS):
        print(f"  Drone {i:2d} : ({off[0]:+.2f}, {off[1]:+.2f})")
    visualise_formation()
