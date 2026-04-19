from dronekit import connect, VehicleMode, LocationGlobalRelative
import time
import math

def get_distance_metres(loc1, loc2):
    dlat = loc2.lat - loc1.lat
    dlon = loc2.lon - loc1.lon
    return math.sqrt((dlat*dlat) + (dlon*dlon)) * 1.113195e5

def wait_until_reached(target, tolerance=1.5):
    print(f"  Flying to waypoint...")
    while True:
        current = vehicle.location.global_relative_frame
        dist = get_distance_metres(current, target)
        print(f"  Distance to waypoint: {dist:.1f} m")
        if dist <= tolerance:
            print("  Waypoint reached!")
            break
        time.sleep(1)

def arm_and_takeoff(target_altitude):
    print("Setting GUIDED mode...")
    vehicle.mode = VehicleMode("GUIDED")
    
    print("Waiting for armable...")
    while not vehicle.is_armable:
        print("  Not armable yet...")
        time.sleep(1)
    
    print("Arming motors...")
    vehicle.armed = True
    while not vehicle.armed:
        print("  Waiting for arm...")
        time.sleep(1)
    
    print(f"Taking off to {target_altitude}m...")
    vehicle.simple_takeoff(target_altitude)
    
    while True:
        alt = vehicle.location.global_relative_frame.alt
        print(f"  Altitude: {alt:.1f} m")
        if alt >= target_altitude * 0.95:
            print("Target altitude reached!")
            break
        time.sleep(1)

print("Connecting to SITL...")
vehicle = connect('tcp:127.0.0.1:5762', wait_ready=True)
print("Connected!")

arm_and_takeoff(10)
time.sleep(2)

home = vehicle.location.global_relative_frame
print(f"Home location: lat={home.lat}, lon={home.lon}")

# 1 metre North  = +0.0000090 degrees latitude
# 1 metre East   = +0.0000112 degrees longitude
# 10m square so multiply by 10

print("\n--- Flying 10x10m Square Pattern ---")

print("\nLeg 1: Flying to WP1 (10m North)...")
wp1 = LocationGlobalRelative(
    home.lat + 0.0000898,
    home.lon,
    10
)
vehicle.simple_goto(wp1)
wait_until_reached(wp1)
time.sleep(1)

print("\nLeg 2: Flying to WP2 (10m North, 10m East)...")
wp2 = LocationGlobalRelative(
    home.lat + 0.0000898,
    home.lon + 0.0001112,
    10
)
vehicle.simple_goto(wp2)
wait_until_reached(wp2)
time.sleep(1)

print("\nLeg 3: Flying to WP3 (0m North, 10m East)...")
wp3 = LocationGlobalRelative(
    home.lat,
    home.lon + 0.0001112,
    10
)
vehicle.simple_goto(wp3)
wait_until_reached(wp3)
time.sleep(1)

print("\nLeg 4: Returning to Launch (RTL)...")
vehicle.mode = VehicleMode("RTL")

print("Waiting for drone to land...")
while True:
    alt = vehicle.location.global_relative_frame.alt
    print(f"  Altitude: {alt:.1f} m")
    if alt <= 0.5:
        print("Landed!")
        break
    time.sleep(1)

vehicle.close()
print("\nPhase 2 Complete! Square pattern done.")