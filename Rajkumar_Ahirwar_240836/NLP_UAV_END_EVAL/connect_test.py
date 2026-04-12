from dronekit import connect, VehicleMode, LocationGlobalRelative
import time
print(f"--- Connecting to SITL on: 127.0.0.1:14551 ---")
# Change 14550 to 14551
vehicle = connect('127.0.0.1:14551', wait_ready=True)

def arm_and_takeoff(target_altitude):
    print("Performing Pre-arm checks...")
    # Don't try to arm until autopilot is ready
    while not vehicle.is_armable:
        print(" Waiting for vehicle to initialize (GPS/Gyros)...")
        time.sleep(1)

    print("Switching to GUIDED mode...")
    vehicle.mode = VehicleMode("GUIDED")
    
    print("Arming motors...")
    vehicle.armed = True

    # Confirm vehicle armed before attempting takeoff
    while not vehicle.armed:
        print(" Waiting for arming confirmation...")
        time.sleep(1)

    print(f"Taking off to {target_altitude}m!")
    vehicle.simple_takeoff(target_altitude)

    # Monitor altitude
    while True:
        curr_alt = vehicle.location.global_relative_frame.alt
        print(f" Current Altitude: {curr_alt:.2f}m")
        if curr_alt >= target_altitude * 0.95:
            print("Reached target altitude.")
            break
        time.sleep(1)

# --- EXECUTION ---
try:
    arm_and_takeoff(10)
    
    print("Hovering for 5 seconds...")
    time.sleep(5)

    print("Setting mode to LAND...")
    vehicle.mode = VehicleMode("LAND")

    # Wait for the drone to touch down
    while vehicle.armed:
        print(f" Landing... Altitude: {vehicle.location.global_relative_frame.alt:.2f}m")
        time.sleep(1)

    print("Drone safely on the ground. Mission complete.")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    print("Closing vehicle connection.")
    vehicle.close()