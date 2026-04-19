import math

def get_distance(n1, e1, n2, e2):
    return math.sqrt((n2 - n1)**2 + (e2 - e1)**2)

def validate_waypoints(waypoints, home_pos=(0, 0), obstacles=[]):
    MIN_ALT = 2
    MAX_ALT = 50
    MAX_RANGE = 100
    OBSTACLE_CLEARANCE = 2
    HOME_RETURN_TOLERANCE = 5

    for i, wp in enumerate(waypoints):
        n = wp["north"]
        e = wp["east"]
        alt = wp["alt"]

        # Check minimum altitude
        if alt < MIN_ALT:
            return {
                "valid": False,
                "reason": f"WP{i+1} altitude {alt}m is below minimum {MIN_ALT}m",
                "safe_waypoints": []
            }

        # Check maximum altitude
        if alt > MAX_ALT:
            return {
                "valid": False,
                "reason": f"WP{i+1} altitude {alt}m exceeds maximum {MAX_ALT}m",
                "safe_waypoints": []
            }

        # Check maximum range from home
        dist_from_home = get_distance(home_pos[0], home_pos[1], n, e)
        if dist_from_home > MAX_RANGE:
            return {
                "valid": False,
                "reason": f"WP{i+1} is {dist_from_home:.1f}m from home, exceeds max {MAX_RANGE}m",
                "safe_waypoints": []
            }

        # Check obstacle proximity
        for j, obs in enumerate(obstacles):
            dist_to_obs = get_distance(n, e, obs[0], obs[1])
            if dist_to_obs < OBSTACLE_CLEARANCE:
                return {
                    "valid": False,
                    "reason": f"WP{i+1} is {dist_to_obs:.1f}m from obstacle {j+1}, minimum clearance is {OBSTACLE_CLEARANCE}m",
                    "safe_waypoints": []
                }

    # Check final waypoint is near home
    last = waypoints[-1]
    dist_last_to_home = get_distance(
        home_pos[0], home_pos[1],
        last["north"], last["east"]
    )
    if dist_last_to_home > HOME_RETURN_TOLERANCE:
        return {
            "valid": False,
            "reason": f"Final waypoint is {dist_last_to_home:.1f}m from home, must be within {HOME_RETURN_TOLERANCE}m",
            "safe_waypoints": []
        }

    return {
        "valid": True,
        "reason": "All checks passed",
        "safe_waypoints": waypoints
    }


# ── 5 Test Cases ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("PHASE 5 - Safety Checker Tests")
    print("=" * 55)

    # PASS 1 - Normal square flight
    test1 = [
        {"north": 10, "east": 0,  "alt": 10},
        {"north": 10, "east": 10, "alt": 10},
        {"north": 0,  "east": 10, "alt": 10},
        {"north": 0,  "east": 0,  "alt": 10},
    ]

    # PASS 2 - Orbit around a pole
    test2 = [
        {"north": 15, "east": 0,  "alt": 15},
        {"north": 10, "east": 10, "alt": 15},
        {"north": 0,  "east": 15, "alt": 15},
        {"north": 0,  "east": 0,  "alt": 15},
    ]

    # PASS 3 - Simple north flight and return
    test3 = [
        {"north": 20, "east": 0, "alt": 10},
        {"north": 0,  "east": 0, "alt": 10},
    ]

    # FAIL 1 - Altitude too low (below 2m)
    test4 = [
        {"north": 5, "east": 0, "alt": 1},
        {"north": 0, "east": 0, "alt": 1},
    ]

    # FAIL 2 - Too far from home (over 100m)
    test5 = [
        {"north": 200, "east": 0, "alt": 10},
        {"north": 0,   "east": 0, "alt": 10},
    ]

    obstacles = [(5, 5)]  # one obstacle at N=5, E=5

    tests = [
        (test1, "PASS 1 - Normal square flight"),
        (test2, "PASS 2 - Orbit pattern"),
        (test3, "PASS 3 - Simple north and return"),
        (test4, "FAIL 1 - Altitude too low"),
        (test5, "FAIL 2 - Too far from home"),
    ]

    for waypoints, description in tests:
        print(f"\nTest: {description}")
        result = validate_waypoints(waypoints, home_pos=(0, 0), obstacles=obstacles)

        if result["valid"]:
            print(f"  PASS - {result['reason']}")
            print(f"  Safe waypoints: {len(result['safe_waypoints'])} waypoints approved")
        else:
            print(f"  FAIL - {result['reason']}")

    print("\n" + "=" * 55)
    print("Phase 5 Complete!")
    print("=" * 55)