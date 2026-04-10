import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os


#  Letter 'G' — 9 drones
#  Coordinate system: x = right, y = up (centred at 0,0)
#  Scale: roughly fits in a ±4 × ±5 box

#   Visual sketch (y top → bottom):
#
#       D1  D2  D3
#     D4
#     D5
#     D6        D9
#       D7  D8  D9  ← D9 is the inner-right crossbar drone
#
#  More precisely:
#        D1(−3, 4)  D2(0, 4)  D3(3, 4)   ← top bar
#        D4(−4, 1)                         ← left upper stem
#        D5(−4,−1)                         ← left lower stem
#        D6(−4,−4)  D7(−1,−4)  D8(2,−4)  ← bottom bar
#        D9(3,−1)                          ← right stem bottom-half
#       D10( 3, 1)                         ← right inner crossbar  (making 10?)
#
#  We keep it at exactly 9 drones by merging bottom-right into the crossbar:

FORMATION_NAME = "G"

# (dx, dy) offsets from centroid — 9 drones
FORMATION_OFFSETS = np.array([
    # Top bar
    (-3.0,  4.0),   # D0
    ( 0.0,  4.0),   # D1
    ( 3.0,  4.0),   # D2
    # Left vertical bar
    (-4.0,  1.5),   # D3
    (-4.0, -1.5),   # D4
    # Bottom bar
    (-3.0, -4.0),   # D5
    ( 0.0, -4.0),   # D6
    ( 3.0, -4.0),   # D7
    # Right crossbar (inner horizontal stroke of G)
    ( 3.0,  0.0),   # D8
], dtype=float)

N_DRONES = len(FORMATION_OFFSETS)


def get_drone_positions(centroid_x, centroid_y, offsets=None):
    if offsets is None:
        offsets = FORMATION_OFFSETS
    centroid = np.array([centroid_x, centroid_y], dtype=float)
    return centroid + offsets          


def expand_to_drones(centroid_traj):
    drone_trajs = []
    for i, (dx, dy) in enumerate(FORMATION_OFFSETS):
        d = {}
        for key, val in centroid_traj.items():
            if key == 'x':
                d['x'] = val + dx
            elif key == 'y':
                d['y'] = val + dy
            else:
                d[key] = val         
        d['drone_id'] = i
        drone_trajs.append(d)
    return drone_trajs


def visualise_formation(save_path=None):
    fig, ax = plt.subplots(figsize=(5, 6))
    ax.set_aspect('equal')
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#16213e')

    colours = plt.cm.plasma(np.linspace(0.2, 0.9, N_DRONES))

    for i, (dx, dy) in enumerate(FORMATION_OFFSETS):
        ax.scatter(dx, dy, s=200, color=colours[i], zorder=5,
                   edgecolors='white', linewidths=0.8)
        ax.text(dx + 0.3, dy + 0.3, f'D{i}', color='white',
                fontsize=8, fontweight='bold', zorder=6)

    skeleton = [
        (2, 0),   # top bar: D2–D1–D0
        (1, 0),
        (0, 3),   # D0 → D3 (top-left to left-upper)
        (3, 4),   # D3 → D4 (left-upper to left-lower)
        (4, 5),   # D4 → D5 (left-lower to bottom-left)
        (5, 6),   # bottom bar: D5–D6–D7
        (6, 7),
        (7, 8),   # D7 → D8 (bottom-right to crossbar)
    ]
    pts = FORMATION_OFFSETS
    for i, j in skeleton:
        ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]],
                '-', color='#a0a0c0', linewidth=1.5, alpha=0.6, zorder=3)

    ax.set_xlim(-7, 7)
    ax.set_ylim(-7, 7)
    ax.set_title("Formation Shape — Letter 'G'  (9 drones)",
                 color='white', fontsize=12, fontweight='bold')
    ax.set_xlabel('dx (units)', color='white')
    ax.set_ylabel('dy (units)', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#4a4a6a')
    ax.grid(True, alpha=0.2, color='#4a4a6a')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"[formation] Formation shape saved → {save_path}")
    plt.close(fig)
    return fig

if __name__ == '__main__':
    print(f"Formation : {FORMATION_NAME}")
    print(f"N_DRONES  : {N_DRONES}")
    print("Offsets (dx, dy):")
    for i, (dx, dy) in enumerate(FORMATION_OFFSETS):
        print(f"  D{i:02d}  ({dx:+.1f}, {dy:+.1f})")

    positions = get_drone_positions(50, 50)
    print(f"\nDrone positions at centroid (50,50):\n{positions}")

    visualise_formation(save_path=os.path.join('results', 'formation_shape.png'))
    print("Formation visualisation complete.")