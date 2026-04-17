import os
import json
import time
import math
from openai import OpenAI
from dotenv import load_dotenv
from dronekit import connect, VehicleMode, LocationGlobalRelative
import safety_checker  # Ensure this file exists in the same folder
import geometry_utils   # Ensure this file exists in the same folder

# 1. Setup & API
load_dotenv()
client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.getenv("GROQ_API_KEY"))
MODEL_ID = "llama-3.3-70b-versatile"

# 2. Connect to SITL
# Use 14550 for standard SITL connection
print("Connecting to SITL...")
vehicle = connect('127.0.0.1:14551', wait_ready=True)

SYSTEM_PROMPT = """
You are a UAV Flight Planner. Translate commands into a PERFECT SQUARE.
A perfect square MUST follow this 5-point sequence (Side = S):
1. (S, 0) - Move North
2. (S, S) - Move East
3. (0, S) - Move South
4. (0, 0) - Move West (Back to Home)
5. (0, 0) - Confirm Home

Return ONLY JSON: {"waypoints": [{"north": 10, "east": 0, "alt": 10}, ...]}
"""

def get_location_metres(original_location, dNorth, dEast):
    """
    Trigonometric conversion of meter offsets to Lat/Lon coordinates.
    """
    earth_radius = 6378137.0 
    dLat = dNorth / earth_radius
    dLon = dEast / (earth_radius * math.cos(math.pi * original_location.lat / 180))

    newlat = original_location.lat + (dLat * 180 / math.pi)
    newlon = original_location.lon + (dLon * 180 / math.pi)
    return LocationGlobalRelative(newlat, newlon, original_location.alt)

def arm_and_takeoff(target_alt):
    print("Pre-arm checks...")
    while not vehicle.is_armable:
        print(" Waiting for vehicle to initialise...")
        time.sleep(1)
    
    print("Arming motors")
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    while not vehicle.armed:
        print(" Waiting for arming...")
        time.sleep(1)

    print(f"Taking off to {target_alt}m...")
    vehicle.simple_takeoff(target_alt)

    while True:
        curr_alt = vehicle.location.global_relative_frame.alt
        print(f" Altitude: {curr_alt:.1f}m")
        if curr_alt >= target_alt * 0.95:
            print("Reached target altitude")
            break
        time.sleep(1)

def execute_mission(waypoints):
    # Capture the starting position to calculate offsets correctly
    home_loc = vehicle.location.global_relative_frame
    
    # 1. Takeoff first
    arm_and_takeoff(10)
    
    for i, wp in enumerate(waypoints):
        target_pos = get_location_metres(home_loc, wp['north'], wp['east'])
        target_pos.alt = wp['alt']
        
        print(f"\n--- Moving to Waypoint {i+1}: N:{wp['north']} E:{wp['east']} @ {wp['alt']}m ---")
        vehicle.simple_goto(target_pos)
        
        # 2. Distance Monitoring Loop
        start_time = time.time()
        while True:
            curr = vehicle.location.global_relative_frame
            # Recalculate distance to target
            dist = math.sqrt((wp['north'] - (curr.lat - home_loc.lat)*111319)**2 + 
                             (wp['east'] - (curr.lon - home_loc.lon)*111319)**2)
            
            print(f"Distance to WP: {dist:.2f}m", end='\r')
            
            # SUCCESS: Within 2 meters
            if dist < 2.5:
                print(f"\n✅ Waypoint {i+1} Reached!")
                time.sleep(2)
                break
            
            # FAILSAFE: Timeout after 45 seconds if drone is stuck
            if time.time() - start_time > 45:
                print(f"\n⚠️ Timeout reached for WP {i+1}. Proceeding to next...")
                break
                
            time.sleep(1)

    print("\nMission Complete. Returning to Launch...")
    vehicle.mode = VehicleMode("RTL")

def process_command(user_cmd):
    try:
        print(f"Asking AI for plan: '{user_cmd}'")
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_cmd}]
        )
        
        content = response.choices[0].message.content.strip()
        # Clean up JSON formatting
        raw_json = content.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw_json)
        
        # Phase 5: Run Safety Checker
        check = safety_checker.validate_waypoints(data['waypoints'])
        
        if check['valid']:
            execute_mission(check['safe_waypoints'])
        else:
            print(f"❌ Safety Denied: {check['reason']}")
            
    except Exception as e:
        print(f"❌ Error Processing Command: {e}")

if __name__ == "__main__":
    try:
        while True:
            cmd = input("\nEnter Command (e.g., 'Fly a 10m square' or 'exit'): ")
            if cmd.lower() == 'exit': 
                break
            process_command(cmd)
    except KeyboardInterrupt:
        print("\nManual Stop Triggered.")
    finally:
        print("Closing vehicle connection...")
        vehicle.close()