import math

def distance(p1, p2):
    return math.sqrt((p1["north"] - p2["north"])**2 + (p1["east"] - p2["east"])**2)

def validate_waypoints(waypoints, home_pos):
    obstacles = [{"north": 50, "east": 50}]

    for wp in waypoints:
        if wp["alt"] < 2:
            return {"valid": False, "reason": "Altitude too low", "safe_waypoints": []}

        if wp["alt"] > 50:
            return {"valid": False, "reason": "Altitude too high", "safe_waypoints": []}

        if distance(wp, home_pos) > 100:
            return {"valid": False, "reason": "Too far from home", "safe_waypoints": []}

        for obs in obstacles:
            if distance(wp, obs) < 2:
                return {"valid": False, "reason": "Too close to obstacle", "safe_waypoints": []}

    if distance(waypoints[-1], home_pos) > 5:
        return {"valid": False, "reason": "Does not return home", "safe_waypoints": []}

    return {"valid": True, "reason": "All checks passed", "safe_waypoints": waypoints}

if __name__ == "__main__":
    home = {"north": 0, "east": 0}

    test_cases = [
        [{"north": 1, "east": 1, "alt": 10}, {"north": 0, "east": 0, "alt": 10}],
        [{"north": 2, "east": 2, "alt": 20}, {"north": 0, "east": 0, "alt": 20}],
        [{"north": 3, "east": 3, "alt": 30}, {"north": 0, "east": 0, "alt": 30}],
        [{"north": 200, "east": 0, "alt": 10}],
        [{"north": 1, "east": 1, "alt": 1}]
    ]

    for i, case in enumerate(test_cases):
        result = validate_waypoints(case, home)
        print(f"Test {i+1}: {'PASS' if result['valid'] else 'FAIL'} - {result['reason']}")
