import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

MAP_W = 100
MAP_H = 100

START = (5, 50)
GOAL  = (95, 50)

OBS_X = 45   
OBS_Y = 45   
OBS_W = 10
OBS_H = 10

SAFETY_MARGIN = 7

def point_in_obstacle(x, y, margin=0):
    return (OBS_X - margin <= x <= OBS_X + OBS_W + margin and
            OBS_Y - margin <= y <= OBS_Y + OBS_H + margin)

def segment_hits_obstacle(p1, p2, margin=0, steps=50):
    for i in range(steps + 1):
        t = i / steps
        x = p1[0] + t * (p2[0] - p1[0])
        y = p1[1] + t * (p2[1] - p1[1])
        if point_in_obstacle(x, y, margin):
            return True
    return False

def plot_map(ax, title="Map"):
    ax.set_xlim(0, MAP_W)
    ax.set_ylim(0, MAP_H)
    ax.set_aspect("equal")

    margin_rect = patches.Rectangle(
        (OBS_X - SAFETY_MARGIN, OBS_Y - SAFETY_MARGIN),
        OBS_W + 2 * SAFETY_MARGIN, OBS_H + 2 * SAFETY_MARGIN,
        linewidth=1, edgecolor="orange", facecolor="orange", alpha=0.15,
        linestyle="--", label="Safety margin"
    )
    ax.add_patch(margin_rect)

    rect = patches.Rectangle((OBS_X, OBS_Y), OBS_W, OBS_H,
                               linewidth=1, edgecolor="black", facecolor="red")
    ax.add_patch(rect)
    ax.text(50, 50, "OBS", ha="center", va="center", fontsize=8, color="white")

    ax.plot(*START, "g^", markersize=10, label="Start")
    ax.plot(*GOAL,  "b*", markersize=12, label="Goal")

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    ax.grid(True, alpha=0.3)

if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_map(ax, title="Map Preview")
    plt.tight_layout()
    plt.savefig("results/map_preview.png", dpi=100)
    plt.show()
    print("Map saved to results/map_preview.png")
