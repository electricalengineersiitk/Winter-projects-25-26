from dronekit import connect, VehicleMode, LocationGlobalRelative
from openai import OpenAI
from dotenv import load_dotenv
import time
import math
import json
import os


try:
    from voice_input import get_voice_command
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False

load_dotenv()

# LLM clients - Groq as primary, GPT as fallback
groq_client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)
openai_client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))


def get_distance(n1, e1, n2, e2):
    return math.sqrt((n2-n1)**2 + (e2-e1)**2)

def validate_waypoints(waypoints, home_pos=(0,0), obstacles=[]):
    MIN_ALT = 2
    MAX_ALT = 50
    MAX_RANGE = 100
    OBSTACLE_CLEARANCE = 2
    HOME_TOLERANCE = 5

    for i, wp in enumerate(waypoints):
        n, e, alt = wp["north"], wp["east"], wp["alt"]

        if alt < MIN_ALT:
            return {"valid": False,
                    "reason": f"WP{i+1} altitude {alt}m below minimum {MIN_ALT}m",
                    "safe_waypoints": []}
        if alt > MAX_ALT:
            return {"valid": False,
                    "reason": f"WP{i+1} altitude {alt}m exceeds maximum {MAX_ALT}m",
                    "safe_waypoints": []}

        dist_home = get_distance(home_pos[0], home_pos[1], n, e)
        if dist_home > MAX_RANGE:
            return {"valid": False,
                    "reason": f"WP{i+1} is {dist_home:.1f}m from home, max is {MAX_RANGE}m",
                    "safe_waypoints": []}

        for j, obs in enumerate(obstacles):
            dist_obs = get_distance(n, e, obs[0], obs[1])
            if dist_obs < OBSTACLE_CLEARANCE:
                return {"valid": False,
                        "reason": f"WP{i+1} is {dist_obs:.1f}m from obstacle {j+1}, need {OBSTACLE_CLEARANCE}m clearance",
                        "safe_waypoints": []}

    last = waypoints[-1]
    dist_last = get_distance(home_pos[0], home_pos[1], last["north"], last["east"])
    if dist_last > HOME_TOLERANCE:
        return {"valid": False,
                "reason": f"Final waypoint is {dist_last:.1f}m from home, must be within {HOME_TOLERANCE}m",
                "safe_waypoints": []}

    return {"valid": True, "reason": "All checks passed", "safe_waypoints": waypoints}



SYSTEM_PROMPT = """
You are a drone flight planner.
Convert the user's natural language command into NED waypoints.
Rules:
- NED = North/East offsets in metres from home
- Interpret the user's command as literally as possible
- If the user says "ground level", use alt=0. If they say "fly high", use a high altitude.
- Do NOT apply your own safety rules - a separate safety system will validate the plan
- always end with north=0, east=0 to return home
- return ONLY raw JSON, no markdown, no backticks

Format: {"waypoints": [{"north": 0, "east": 0, "alt": 10}, ...]}
"""

def get_waypoints_from_llm(command, feedback=None):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if feedback:
        messages.append({"role": "user", "content": command})
        messages.append({"role": "assistant", "content": feedback})
        messages.append({"role": "user", "content": f"That plan was rejected: {feedback}. Please generate a safe alternative."})
    else:
        messages.append({"role": "user", "content": command})


    apis = [
        ("Groq", groq_client, "llama-3.3-70b-versatile"),
        ("GPT", openai_client, "gpt-4o-mini"),
    ]

    for api_name, client, model in apis:
        try:
            print(f"  Trying {api_name}...")
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2
            )
            raw = response.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(raw)
            print(f"  {api_name} succeeded")
            return parsed["waypoints"]
        except Exception as e:
            print(f"  {api_name} failed: {e}")
            continue

    raise Exception("All LLM APIs failed! Check your API keys.")



def get_location_metres(original, dn, de):
    earth_radius = 6378137.0
    dlat = dn / earth_radius
    dlon = de / (earth_radius * math.cos(math.pi * original.lat / 180))
    return LocationGlobalRelative(
        original.lat + math.degrees(dlat),
        original.lon + math.degrees(dlon),
        original.alt
    )

def get_ned_distance(current, target_ned, home):
    target_gps = get_location_metres(home, target_ned["north"], target_ned["east"])
    dlat = target_gps.lat - current.lat
    dlon = target_gps.lon - current.lon
    return math.sqrt((dlat*dlat + dlon*dlon)) * 1.113195e5

def arm_and_takeoff(vehicle, altitude):
    print("  Setting GUIDED mode...")
    vehicle.mode = VehicleMode("GUIDED")
    while not vehicle.is_armable:
        print("  Waiting for armable...")
        time.sleep(1)
    print("  Arming...")
    vehicle.armed = True
    while not vehicle.armed:
        time.sleep(1)
    print(f"  Taking off to {altitude}m...")
    vehicle.simple_takeoff(altitude)
    while True:
        alt = vehicle.location.global_relative_frame.alt
        print(f"  Altitude: {alt:.1f}m")
        if alt >= altitude * 0.95:
            print("  Target altitude reached!")
            break
        time.sleep(1)

def execute_mission(vehicle, waypoints):
    home = vehicle.location.global_relative_frame
    print(f"\nExecuting {len(waypoints)} waypoints...")

    for i, wp in enumerate(waypoints):
        target = get_location_metres(home, wp["north"], wp["east"])
        target = LocationGlobalRelative(target.lat, target.lon, wp["alt"])

        print(f"\n  Flying to WP{i+1}: N={wp['north']}m E={wp['east']}m Alt={wp['alt']}m")
        vehicle.simple_goto(target)

        while True:
            current = vehicle.location.global_relative_frame
            dist = get_ned_distance(current, wp, home)
            print(f"  Distance to WP{i+1}: {dist:.1f}m")
            if dist <= 1.5:
                print(f"  WP{i+1} reached!")
                break
            time.sleep(1)

    print("\nAll waypoints done. Returning to launch...")
    vehicle.mode = VehicleMode("RTL")

    while True:
        alt = vehicle.location.global_relative_frame.alt
        print(f"  Descending: {alt:.1f}m")
        if alt <= 0.5:
            print("  Landed!")
            break
        time.sleep(1)


def main():
    print("=" * 55)
    print("UAV NLP MAIN CONTROLLER")
    print("=" * 55)

    if VOICE_AVAILABLE:
        print("\nInput modes available:")
        print("  1. Text input (type commands)")
        print("  2. Voice input (speak commands via microphone)")
        mode = input("Select mode (1 or 2, default=1): ").strip()
        use_voice = (mode == "2")
        if use_voice:
            print("Voice mode enabled! You will speak your commands.")
    else:
        use_voice = False
        print("\n(Voice input not available - install sounddevice and scipy to enable)")

    print("\nConnecting to SITL...")
    vehicle = connect('tcp:127.0.0.1:5762', wait_ready=True)
    print("Connected!")

    obstacles = []

    while True:
        print("\n" + "="*55)

        if use_voice:
            input("Press Enter when ready to speak...")
            command = get_voice_command(duration=5)
            if not command:
                print("Could not understand audio. Try again.")
                continue
            confirm = input(f"You said: \"{command}\" - Use this? (yes/no): ").strip().lower()
            if confirm != 'yes':
                continue
        else:
            command = input("Enter flight command (or 'quit' to exit): ").strip()

        if command.lower() == 'quit':
            print("Closing connection...")
            vehicle.close()
            print("Done. Goodbye!")
            break

        if not command:
            continue


        waypoints = None
        feedback = None
        MAX_RETRIES = 3

        for attempt in range(1, MAX_RETRIES + 1):
            print(f"\nAttempt {attempt}/{MAX_RETRIES} - Generating waypoints...")
            try:
                waypoints = get_waypoints_from_llm(command, feedback)
                print(f"GPT returned {len(waypoints)} waypoints")

                result = validate_waypoints(waypoints, obstacles=obstacles)

                if result["valid"]:
                    print("Safety check PASSED!")
                    waypoints = result["safe_waypoints"]
                    break
                else:
                    print(f"Safety check FAILED: {result['reason']}")
                    feedback = result["reason"]
                    waypoints = None

            except Exception as e:
                print(f"Error: {e}")
                feedback = str(e)
                waypoints = None

        if waypoints is None:
            print("\nCould not generate safe waypoints after 3 attempts.")
            print("Please try a different command.")
            continue

        print(f"\nFinal approved waypoints ({len(waypoints)}):")
        for i, wp in enumerate(waypoints):
            print(f"  WP{i+1}: N={wp['north']}m E={wp['east']}m Alt={wp['alt']}m")

        confirm = input("\nExecute this mission? (yes/no): ").strip().lower()
        if confirm != 'yes':
            print("Mission cancelled.")
            continue

        print("\nArming and taking off...")
        arm_and_takeoff(vehicle, 10)

        execute_mission(vehicle, waypoints)

        print("\nMission complete! Ready for next command.")

if __name__ == "__main__":
    main()