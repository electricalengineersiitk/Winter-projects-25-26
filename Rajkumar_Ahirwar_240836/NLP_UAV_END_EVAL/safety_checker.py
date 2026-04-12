import math

def get_distance(p1, p2):
    """Calculates 2D distance between two NED points (north, east)."""
    return math.sqrt((p1['north'] - p2['north'])**2 + (p1['east'] - p2['east'])**2)

def validate_waypoints(waypoints, home_pos={'north': 0, 'east': 0}):
    """
    Middleware validator to block unsafe LLM-generated flight plans.
    """
    # Define Constraints
    MIN_ALT = 2.0
    MAX_ALT = 50.0
    MAX_RANGE = 100.0
    OBSTACLES = [{'north': 15, 'east': 15}] # Example defined obstacle
    OBSTACLE_THRESHOLD = 2.0
    HOME_THRESHOLD = 5.0

    if not waypoints:
        return {'valid': False, 'reason': 'Empty mission', 'safe_waypoints': []}

    for i, wp in enumerate(waypoints):
        n, e, alt = wp['north'], wp['east'], wp['alt']
        
        # 1. Altitude Check
        if alt < MIN_ALT:
            return {'valid': False, 'reason': f"WP {i}: Altitude {alt}m is below min {MIN_ALT}m", 'safe_waypoints': []}
        if alt > MAX_ALT:
            return {'valid': False, 'reason': f"WP {i}: Altitude {alt}m is above max {MAX_ALT}m", 'safe_waypoints': []}

        # 2. Range Check (Distance from Home)
        dist_from_home = get_distance(wp, home_pos)
        if dist_from_home > MAX_RANGE:
            return {'valid': False, 'reason': f"WP {i}: Distance {dist_from_home:.1f}m exceeds max range {MAX_RANGE}m", 'safe_waypoints': []}

        # 3. Obstacle Check
        for obs in OBSTACLES:
            if get_distance(wp, obs) < OBSTACLE_THRESHOLD:
                return {'valid': False, 'reason': f"WP {i}: Too close to obstacle at {obs}", 'safe_waypoints': []}

    # 4. Return Path Check (Last WP must be near Home)
    last_wp = waypoints[-1]
    if get_distance(last_wp, home_pos) > HOME_THRESHOLD:
        return {'valid': False, 'reason': "Mission must end within 5m of home position", 'safe_waypoints': []}

    return {'valid': True, 'reason': 'All safety checks passed', 'safe_waypoints': waypoints}

# --- TEST CASES ---
test_cases = [
    # 1. Pass: Simple 10m North & back
    {'name': 'Simple Valid', 'wps': [{'north': 10, 'east': 0, 'alt': 10}, {'north': 0, 'east': 0, 'alt': 10}]},
    
    # 2. Pass: 10m Square
    {'name': 'Valid Square', 'wps': [{'north': 10, 'east': 0, 'alt': 10}, {'north': 10, 'east': 10, 'alt': 10}, {'north': 0, 'east': 10, 'alt': 10}, {'north': 0, 'east': 0, 'alt': 10}]},
    
    # 3. Pass: High Altitude (45m)
    {'name': 'High Alt Valid', 'wps': [{'north': 5, 'east': 5, 'alt': 45}, {'north': 0, 'east': 0, 'alt': 10}]},

    # 4. Fail: Too far (150m)
    {'name': 'Range Violation', 'wps': [{'north': 150, 'east': 0, 'alt': 10}, {'north': 0, 'east': 0, 'alt': 10}]},
    
    # 5. Fail: No Return to Home
    {'name': 'RTL Violation', 'wps': [{'north': 10, 'east': 10, 'alt': 10}]},
    
    # 6. Fail: Altitude too low
    {'name': 'Altitude Violation', 'wps': [{'north': 5, 'east': 0, 'alt': 1.5}, {'north': 0, 'east': 0, 'alt': 10}]}
]

if __name__ == "__main__":
    print(f"{'Test Name':<20} | {'Status':<10} | {'Reason'}")
    print("-" * 60)
    for case in test_cases:
        result = validate_waypoints(case['wps'])
        status = "PASS" if result['valid'] else "FAIL"
        print(f"{case['name']:<20} | {status:<10} | {result['reason']}")