from dronekit import connect, VehicleMode
import time
from geometry_utils import generate_circle
from safety_checker import validate_waypoints

vehicle = connect('udp:127.0.0.1:14550', wait_ready=True)


def arm_and_takeoff(target_alt):
    print("Setting GUIDED mode...")
    vehicle.mode = VehicleMode("GUIDED")

    while vehicle.mode.name != "GUIDED":
        print("Waiting for GUIDED mode...")
        time.sleep(1)

    print("Arming...")
    vehicle.armed = True

    while not vehicle.armed:
        print("Waiting for arming...")
        time.sleep(1)

    print("Taking off...")
    vehicle.simple_takeoff(target_alt)

    while True:
        alt = vehicle.location.global_relative_frame.alt
        print("Altitude:", alt)
        if alt >= target_alt * 0.95:
            print("Reached altitude")
            break
        time.sleep(1)

def move_waypoints(waypoints):
    home = vehicle.location.global_relative_frame

    for wp in waypoints:
        north = wp["north"]
        east = wp["east"]
        alt = wp["alt"]

        target = vehicle.location.global_relative_frame

        target.lat = home.lat + north * 0.00001
        target.lon = home.lon + east * 0.00001
        target.alt = alt

        vehicle.simple_goto(target)
        time.sleep(3)

def main():
    cmd = input("Enter command: ").lower()

    if "circle" in cmd or "around" in cmd:
        waypoints = generate_circle(0, 0, 5, 20, 10)

    elif "square" in cmd:
        waypoints = [
            {"north": 0, "east": 0, "alt": 10},
            {"north": 5, "east": 0, "alt": 10},
            {"north": 5, "east": 5, "alt": 10},
            {"north": 0, "east": 5, "alt": 10},
            {"north": 0, "east": 0, "alt": 10}
        ]

    elif "north" in cmd:
        waypoints = [
            {"north": 0, "east": 0, "alt": 10},
            {"north": 10, "east": 0, "alt": 10},
            {"north": 0, "east": 0, "alt": 10}
        ]

    elif "ground" in cmd:
        waypoints = [
            {"north": 0, "east": 0, "alt": 0}
        ]

    else:
        print("Command not supported")
        return

    home = {"north": 0, "east": 0}

    result = validate_waypoints(waypoints, home)

    if not result["valid"]:
        print("Mission rejected:", result["reason"])
        return

    print("Mission approved")

    arm_and_takeoff(10)
    move_waypoints(waypoints)

    print("Returning home")
    vehicle.mode = VehicleMode("RTL")

main()
