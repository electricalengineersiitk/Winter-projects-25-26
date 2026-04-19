from dronekit import connect


def connect_vehicle():

    connection_string = "udp:127.0.0.1:14550"

    print("Attempting vehicle connection...")

    drone = connect(connection_string, wait_ready=True)

    print("Vehicle connection established\n")

    print("Armable status  :", drone.is_armable)
    print("Current mode    :", drone.mode.name)
    print("Global position :", drone.location.global_frame)

    return drone


if __name__ == "__main__":

    vehicle = connect_vehicle()

    # close the link after reading info
    vehicle.close()

    print("\nConnection closed.")