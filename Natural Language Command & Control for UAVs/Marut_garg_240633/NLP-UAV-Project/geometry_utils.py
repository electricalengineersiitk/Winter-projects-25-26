import math

def generate_circle(center_n, center_e, radius, num_points, alt):
    waypoints = []

    for i in range(num_points):
        angle = 2 * math.pi * i / num_points

        n = center_n + radius * math.cos(angle)
        e = center_e + radius * math.sin(angle)

        waypoints.append({
            "north": n,
            "east": e,
            "alt": alt
        })

    waypoints.append(waypoints[0])

    return waypoints


def generate_orbit(pole_n, pole_e, radius, alt, num_points):
    return generate_circle(pole_n, pole_e, radius, num_points, alt)


import math

def generate_helix(radius, start_alt, end_alt, laps, num_points):
    waypoints = []

    for i in range(num_points + 1):
        angle = 2 * math.pi * laps * (i / num_points)

        north = radius * math.cos(angle)
        east = radius * math.sin(angle)

        alt = start_alt + (end_alt - start_alt) * (i / num_points)

        waypoints.append({
            "north": north,
            "east": east,
            "alt": alt
        })

    return waypoints


def generate_scan(area_width, area_height, lane_spacing, alt):
    waypoints = []

    direction = 1
    north = 0

    while north <= area_height:
        if direction == 1:
            waypoints.append({"north": north, "east": 0, "alt": alt})
            waypoints.append({"north": north, "east": area_width, "alt": alt})
        else:
            waypoints.append({"north": north, "east": area_width, "alt": alt})
            waypoints.append({"north": north, "east": 0, "alt": alt})

        north += lane_spacing
        direction *= -1

    return waypoints


import matplotlib.pyplot as plt

if __name__ == "__main__":

    helix = generate_helix(5, 5, 20, 3, 50)

    north = [wp["north"] for wp in helix]
    east = [wp["east"] for wp in helix]

    plt.plot(east, north, marker='o')
    plt.title("Helix (Top View)")
    plt.xlabel("East")
    plt.ylabel("North")
    plt.grid()
    plt.show()


    scan = generate_scan(20, 20, 5, 10)

    north = [wp["north"] for wp in scan]
    east = [wp["east"] for wp in scan]

    plt.plot(east, north, marker='o')
    plt.title("Scan Pattern")
    plt.xlabel("East")
    plt.ylabel("North")
    plt.grid()
    plt.show()
