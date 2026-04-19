import math
import matplotlib.pyplot as plt


# -------------------------------
# Circular Path Generator
# -------------------------------
def build_circle(cn, ce, r, pts, height):

    wp_list = []

    step = (2 * math.pi) / pts

    for k in range(pts):
        ang = k * step

        n_val = cn + r * math.cos(ang)
        e_val = ce + r * math.sin(ang)

        wp_list.append({
            "north": n_val,
            "east": e_val,
            "alt": height
        })

    # close the circle
    wp_list.append(dict(wp_list[0]))

    return wp_list


# orbit just reuses circle logic
def build_orbit(pn, pe, r, h, pts):
    return build_circle(pn, pe, r, pts, h)


# -------------------------------
# Helical Path Generator
# -------------------------------
def build_helix(r, h_start, h_end, turns, pts):

    path = []

    for i in range(pts + 1):

        frac = i / pts
        theta = turns * 2 * math.pi * frac

        n = r * math.cos(theta)
        e = r * math.sin(theta)

        altitude = h_start + (h_end - h_start) * frac

        path.append({
            "north": n,
            "east": e,
            "alt": altitude
        })

    return path


# -------------------------------
# Lawn-mower Scan Pattern
# -------------------------------
def build_scan(width, height, spacing, altitude):

    track = []

    current_n = 0
    forward = True

    while current_n <= height:

        if forward:
            track.append({"north": current_n, "east": 0, "alt": altitude})
            track.append({"north": current_n, "east": width, "alt": altitude})
        else:
            track.append({"north": current_n, "east": width, "alt": altitude})
            track.append({"north": current_n, "east": 0, "alt": altitude})

        current_n += spacing
        forward = not forward

    return track


# -------------------------------
# Utility for plotting
# -------------------------------
def plot_path(data, title):

    n_vals = [p["north"] for p in data]
    e_vals = [p["east"] for p in data]

    plt.figure()
    plt.plot(e_vals, n_vals, marker="o")
    plt.xlabel("East")
    plt.ylabel("North")
    plt.title(title)
    plt.grid(True)
    plt.show()


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":

    helix_path = build_helix(5, 5, 20, 3, 50)
    plot_path(helix_path, "Helix Path (Top View)")

    scan_path = build_scan(20, 20, 5, 10)
    plot_path(scan_path, "Area Scan Pattern")