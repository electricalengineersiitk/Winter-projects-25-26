import math
import matplotlib.pyplot as plt

def generate_circle(center_n, center_e, radius, num_points=20, alt=10):
    """Generates a flat circular path."""
    waypoints = []
    for i in range(num_points + 1):
        angle = (2 * math.pi * i) / num_points
        n = center_n + (radius * math.cos(angle))
        e = center_e + (radius * math.sin(angle))
        waypoints.append({"north": round(n, 2), "east": round(e, 2), "alt": alt})
    return waypoints

def generate_helix(radius, start_alt, end_alt, laps=2, num_points=40):
    """Generates a rising spiral (Helix)."""
    waypoints = []
    alt_step = (end_alt - start_alt) / num_points
    for i in range(num_points + 1):
        angle = (2 * math.pi * laps * i) / num_points
        n = radius * math.cos(angle)
        e = radius * math.sin(angle)
        current_alt = start_alt + (alt_step * i)
        waypoints.append({"north": round(n, 2), "east": round(e, 2), "alt": round(current_alt, 2)})
    return waypoints

def generate_orbit(pole_n, pole_e, radius, alt=10, num_points=20):
    """Orbits a specific fixed point (the pole)."""
    # Effectively a circle centered at the pole
    return generate_circle(pole_n, pole_e, radius, num_points, alt)

def visualize_path(waypoints, title="UAV Trajectory"):
    """Visualizes the 2D path using Matplotlib."""
    norths = [wp['north'] for wp in waypoints]
    easts = [wp['east'] for wp in waypoints]
    
    plt.figure(figsize=(6,6))
    plt.plot(easts, norths, marker='o', linestyle='-')
    plt.xlabel('East (meters)')
    plt.ylabel('North (meters)')
    plt.title(title)
    plt.grid(True)
    plt.axis('equal')
    plt.show()

# --- TEST AND VISUALIZE ---
if __name__ == "__main__":
    print("Generating Circle...")
    circle = generate_circle(0, 0, 10)
    visualize_path(circle, "10m Radius Circle")

    print("Generating Helix...")
    helix = generate_helix(5, 10, 30)
    visualize_path(helix, "Rising Helix (10m to 30m)")