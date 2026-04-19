import numpy as np
import matplotlib.pyplot as plt
import os

N = 5
FORMATION = "V-shape"


def get_formation_offsets(n=N):
    spacing = 5.0
    offsets = []
    half = n // 2
    for i in range(n):
        slot = i - half
        dx = -abs(slot) * spacing
        dy = slot * spacing
        offsets.append((dx, dy))
    return offsets


def assign_drones(centroid_x, centroid_y, heading_deg=0.0):
    offsets = get_formation_offsets(N)
    angle = np.radians(heading_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    positions = []
    for dx, dy in offsets:
        wx = centroid_x + cos_a * dx - sin_a * dy
        wy = centroid_y + sin_a * dx + cos_a * dy
        positions.append((wx, wy))
    return positions


def formation_info():
    print(f"Formation : {FORMATION}")
    print(f"UAVs      : {N}")
    for i, (dx, dy) in enumerate(get_formation_offsets()):
        print(f"  Drone {i+1}: dx={dx:+.1f}, dy={dy:+.1f}")


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    formation_info()

    cx, cy = 50.0, 50.0
    positions = assign_drones(cx, cy, heading_deg=0)

    fig, ax = plt.subplots(figsize=(6, 5))
    for i, (x, y) in enumerate(positions):
        ax.plot(x, y, "o", markersize=14, label=f"D{i+1}")
        ax.annotate(f"D{i+1}", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9)

    ax.plot(cx, cy, "k+", markersize=14, markeredgewidth=2, label="Centroid")
    ax.set_title(f"{FORMATION} Formation (N={N})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig("results/formation_preview.png", dpi=120)
    plt.show()
    print("Saved -> results/formation_preview.png")
