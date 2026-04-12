from dronekit import connect, VehicleMode, LocationGlobalRelative
import time

# 1. Connect to SITL (using your 14551 port)
vehicle = connect('127.0.0.1:14551', wait_ready=True)

def arm_and_takeoff(target_altitude):
    print("Pre-arm checks...")
    while not vehicle.is_armable:
        time.sleep(1)
    
    print("Arming and taking off...")
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True
    
    while not vehicle.armed:
        time.sleep(1)
        
    vehicle.simple_takeoff(target_altitude)
    
    # Wait until altitude is reached
    while True:
        alt = vehicle.location.global_relative_frame.alt
        if alt >= target_altitude * 0.95:
            print("Reached target altitude")
            break
        time.sleep(1)

def fly_to_offset(dNorth, dEast, target_alt):
    """
    Sends the drone to a point (meters) relative to its HOME position.
    """
    # 1 degree Lat is approx 111,319 meters
    # 0.00009 degrees is approx 10 meters
    new_lat = home_location.lat + (dNorth * 0.000009)
    new_lon = home_location.lon + (dEast * 0.000011)
    
    target_point = LocationGlobalRelative(new_lat, new_lon, target_alt)
    print(f"Moving to Point: North {dNorth}m, East {dEast}m")
    vehicle.simple_goto(target_point)
    
    # Give it time to travel 10 meters
    time.sleep(10)

# --- START MISSION ---
try:
    # Get the starting position (Home)
    home_location = vehicle.location.global_relative_frame

    # Phase 2 Goal: Takeoff, Square, and Land
    arm_and_takeoff(10)

    print("Starting 10m x 10m Square Pattern")
    
    # Point 1: 10m North, 0m East
    fly_to_offset(10, 0, 10)
    
    # Point 2: 10m North, 10m East
    fly_to_offset(10, 10, 10)
    
    # Point 3: 0m North, 10m East
    fly_to_offset(0, 10, 10)
    
    # Point 4: 0m North, 0m East (Back to Start)
    fly_to_offset(0, 0, 10)

    print("Mission Complete. Returning to Launch...")
    vehicle.mode = VehicleMode("RTL")

except Exception as e:
    print(f"Error: {e}")

finally:
    # Keep connection open briefly to see RTL start in QGC
    time.sleep(5)
    vehicle.close()
    print("Vehicle connection closed.")