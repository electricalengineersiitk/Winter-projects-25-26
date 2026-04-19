import json


def interpret_input(user_text):

    text = user_text.strip().lower()

    command_map = {
        "north": [{"north": 5, "east": 0, "alt": 10}],
        "forward": [{"north": 8, "east": 0, "alt": 10}],
        "right": [{"north": 0, "east": 4, "alt": 10}]
    }

    if "square" in text:
        path = [
            {"north": 0, "east": 5, "alt": 10},
            {"north": 5, "east": 5, "alt": 10},
            {"north": 5, "east": 0, "alt": 10},
            {"north": 0, "east": 0, "alt": 10}
        ]
        return {"waypoints": path}

    for key in command_map:
        if key in text:
            return {"waypoints": command_map[key]}

    return {"waypoints": [{"north": 0, "east": 0, "alt": 10}]}


def main():

    while True:

        user_command = input("Command > ")

        if user_command.lower() == "exit":
            print("Stopping command parser...")
            break

        output = interpret_input(user_command)

        print("\nGenerated Waypoints:")
        print(json.dumps(output, indent=2))
        print()


if __name__ == "__main__":
    main()