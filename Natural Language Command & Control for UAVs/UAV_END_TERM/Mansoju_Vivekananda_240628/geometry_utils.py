import numpy as np
import matplotlib.pyplot as plt

def generate_circle(center_n, center_e, radius, num_points, alt):
    waypoints = []
    for i in range(num_points + 1):
        angle = 2 * np.pi * i / num_points
        n = center_n + radius * np.cos(angle)
        e = center_e + radius * np.sin(angle)
        waypoints.append({"north": round(n, 2), "east": round(e, 2), "alt": alt})
    return waypoints

def generate_helix(radius, start_alt, end_alt, laps, num_points):
    waypoints = []
    total_points = num_points * laps
    for i in range(total_points + 1):
        angle = 2 * np.pi * i / num_points
        alt = start_alt + (end_alt - start_alt) * i / total_points
        n = radius * np.cos(angle)
        e = radius * np.sin(angle)
        waypoints.append({"north": round(n, 2), "east": round(e, 2), "alt": round(alt, 2)})
    return waypoints

def generate_orbit(pole_n, pole_e, radius, alt, num_points):
    waypoints = []
    for i in range(num_points + 1):
        angle = 2 * np.pi * i / num_points
        n = pole_n + radius * np.cos(angle)
        e = pole_e + radius * np.sin(angle)
        waypoints.append({"north": round(n, 2), "east": round(e, 2), "alt": alt})
    return waypoints

def generate_scan(area_width, area_height, lane_spacing, alt):
    waypoints = []
    e = 0
    going_north = True
    while e <= area_width:
        if going_north:
            waypoints.append({"north": 0, "east": round(e, 2), "alt": alt})
            waypoints.append({"north": area_height, "east": round(e, 2), "alt": alt})
        else:
            waypoints.append({"north": area_height, "east": round(e, 2), "alt": alt})
            waypoints.append({"north": 0, "east": round(e, 2), "alt": alt})
        going_north = not going_north
        e += lane_spacing
    waypoints.append({"north": 0, "east": 0, "alt": alt})
    return waypoints

def plot_trajectory(waypoints, title):
    ns = [wp["north"] for wp in waypoints]
    es = [wp["east"]  for wp in waypoints]
    alts = [wp["alt"] for wp in waypoints]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=14)

    axes[0].plot(es, ns, 'b-o', markersize=3)
    axes[0].plot(es[0], ns[0], 'go', markersize=10, label='Start')
    axes[0].plot(es[-1], ns[-1], 'rs', markersize=10, label='End')
    axes[0].set_xlabel("East (m)")
    axes[0].set_ylabel("North (m)")
    axes[0].set_title("Top View (N vs E)")
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_aspect('equal')

    axes[1].plot(range(len(alts)), alts, 'r-o', markersize=3)
    axes[1].set_xlabel("Waypoint index")
    axes[1].set_ylabel("Altitude (m)")
    axes[1].set_title("Altitude Profile")
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    print(f"Plot saved: {title.replace(' ', '_')}.png")
    plt.show()

if __name__ == "__main__":
    print("=" * 50)
    print("PHASE 4 - Trajectory Generation Test")
    print("=" * 50)

    print("\n1. Generating circle...")
    circle = generate_circle(0, 0, 10, 36, 10)
    print(f"   Circle: {len(circle)} waypoints")
    plot_trajectory(circle, "Circle Trajectory")

    print("\n2. Generating helix...")
    helix = generate_helix(8, 5, 20, 3, 36)
    print(f"   Helix: {len(helix)} waypoints")
    plot_trajectory(helix, "Helix Trajectory")

    print("\n3. Generating orbit (around pole at N=10, E=10)...")
    orbit = generate_orbit(10, 10, 5, 10, 36)
    print(f"   Orbit: {len(orbit)} waypoints")
    plot_trajectory(orbit, "Orbit Trajectory")

    print("\n4. Generating scan pattern...")
    scan = generate_scan(20, 20, 5, 10)
    print(f"   Scan: {len(scan)} waypoints")
    plot_trajectory(scan, "Scan Trajectory")

    print("\nPhase 4 Complete!")
    print("Check the PNG files saved in your project folder.")  