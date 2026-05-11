from dronekit import connect, VehicleMode, LocationGlobalRelative
import time

print("Connecting...")
vehicle = connect('udp:127.0.0.1:14550', wait_ready=True)

vehicle.mode = VehicleMode("GUIDED")
vehicle.armed = True

while not vehicle.armed:
    print("Waiting for arming...")
    time.sleep(1)

print("Taking off...")
vehicle.simple_takeoff(10)

while True:
    alt = vehicle.location.global_relative_frame.alt
    print("Altitude:", alt)
    if alt >= 9:
        break
    time.sleep(1)

lat = vehicle.location.global_frame.lat
lon = vehicle.location.global_frame.lon

points = [
    LocationGlobalRelative(lat + 0.0001, lon, 10),
    LocationGlobalRelative(lat + 0.0001, lon + 0.0001, 10),
    LocationGlobalRelative(lat, lon + 0.0001, 10),
    LocationGlobalRelative(lat, lon, 10)
]

for i, point in enumerate(points):
    print("Going to point", i+1)
    vehicle.simple_goto(point)
    time.sleep(10)

print("Returning home...")
vehicle.mode = VehicleMode("RTL")

vehicle.close()
