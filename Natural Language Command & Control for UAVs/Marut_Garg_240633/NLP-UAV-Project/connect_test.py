from dronekit import connect

print("Connecting to vehicle...")

vehicle = connect('udp:127.0.0.1:14550', wait_ready=True)

print("Connected!")

print("Vehicle is armable:", vehicle.is_armable)
print("Mode:", vehicle.mode.name)
print("Location:", vehicle.location.global_frame)

vehicle.close()
