from dronekit import connect, VehicleMode, LocationGlobalRelative
import time


def connect_drone():
    link = "udp:127.0.0.1:14550"
    print("Establishing connection to vehicle...")
    drone = connect(link, wait_ready=True)
    print("Connection successful\n")
    return drone


def arm_and_takeoff(drone, target_alt):

    print("Switching to GUIDED mode...")
    drone.mode = VehicleMode("GUIDED")

    print("Arming motors...")
    drone.armed = True

    while not drone.armed:
        print("Waiting for drone to arm...")
        time.sleep(1)

    print("Initiating takeoff...")
    drone.simple_takeoff(target_alt)

    while True:
        current_alt = drone.location.global_relative_frame.alt
        print("Current altitude:", current_alt)

        if current_alt >= target_alt * 0.9:
            print("Target altitude reached\n")
            break

        time.sleep(1)


def fly_square(drone):

    home_lat = drone.location.global_frame.lat
    home_lon = drone.location.global_frame.lon

    mission_points = [
        LocationGlobalRelative(home_lat + 0.0001, home_lon, 10),
        LocationGlobalRelative(home_lat + 0.0001, home_lon + 0.0001, 10),
        LocationGlobalRelative(home_lat, home_lon + 0.0001, 10),
        LocationGlobalRelative(home_lat, home_lon, 10)
    ]

    for idx, wp in enumerate(mission_points):
        print(f"Navigating to waypoint {idx+1}")
        drone.simple_goto(wp)
        time.sleep(10)


def return_home(drone):
    print("\nSwitching to RTL mode...")
    drone.mode = VehicleMode("RTL")


def main():

    vehicle = connect_drone()

    arm_and_takeoff(vehicle, 10)

    fly_square(vehicle)

    return_home(vehicle)

    vehicle.close()
    print("Vehicle connection closed")


if __name__ == "__main__":
    main()