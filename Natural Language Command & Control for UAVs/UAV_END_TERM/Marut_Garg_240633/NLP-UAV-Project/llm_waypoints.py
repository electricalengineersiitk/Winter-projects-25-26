import json

def parse_command(cmd):
    cmd = cmd.lower()

    if "north" in cmd:
        return {"waypoints": [{"north": 5, "east": 0, "alt": 10}]}

    elif "square" in cmd:
        return {
            "waypoints": [
                {"north": 0, "east": 5, "alt": 10},
                {"north": 5, "east": 5, "alt": 10},
                {"north": 5, "east": 0, "alt": 10},
                {"north": 0, "east": 0, "alt": 10}
            ]
        }

    elif "forward" in cmd:
        return {"waypoints": [{"north": 8, "east": 0, "alt": 10}]}

    elif "right" in cmd:
        return {"waypoints": [{"north": 0, "east": 4, "alt": 10}]}

    else:
        return {"waypoints": [{"north": 0, "east": 0, "alt": 10}]}


while True:
    cmd = input("Enter command: ")

    if cmd == "exit":
        break

    result = parse_command(cmd)

    print("Parsed waypoints:", json.dumps(result, indent=2))
