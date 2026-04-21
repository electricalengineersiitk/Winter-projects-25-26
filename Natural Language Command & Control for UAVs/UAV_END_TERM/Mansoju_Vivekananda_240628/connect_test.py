from dronekit import connect, VehicleMode
import time

print("Connecting to SITL...")
vehicle = connect('tcp:127.0.0.1:5762', wait_ready=True)

print("Arming and taking off...")

def arm_and_takeoff(target_altitude):
    vehicle.mode = VehicleMode("GUIDED")
    while not vehicle.is_armable:
        print(f"Waiting for armable... GPS: {vehicle.gps_0}, EKF OK: {vehicle.ekf_ok}, System status: {vehicle.system_status.state}")
        time.sleep(1)

    vehicle.armed = True
    while not vehicle.armed:
        print("Waiting for arm...")
        time.sleep(1)

    vehicle.simple_takeoff(target_altitude)

    while True:
        alt = vehicle.location.global_relative_frame.alt
        print(f"Altitude: {alt:.1f} m")
        if alt >= target_altitude * 0.95:
            print("Target altitude reached.")
            break
        time.sleep(1)

arm_and_takeoff(10)

print("Hovering for 5 seconds...")
time.sleep(5)

print("Landing...")
vehicle.mode = VehicleMode("LAND")
time.sleep(10)

vehicle.close()
print("Done.")
