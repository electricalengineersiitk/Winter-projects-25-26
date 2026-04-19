from openai import OpenAI
import google.generativeai as genai
import json
import os
import time
from dotenv import load_dotenv

load_dotenv()


openai_client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))

groq_client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-2.0-flash-lite")

SYSTEM_PROMPT = """
You are a drone flight planner. 
Convert the user's natural language command into a JSON list of NED waypoints.
Rules:
- NED means North/East/Down offsets in metres from home position
- altitude (alt) must always be between 5 and 30 metres
- distances must be reasonable (under 50 metres)
- always return to home at the end (N=0, E=0)
- return ONLY raw JSON, no explanation, no markdown, no backticks

Output format must be exactly:
{"waypoints": [{"north": 0, "east": 0, "alt": 10}, ...]}
"""

def call_groq(prompt):
    """Call Groq API."""
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

def call_gpt(prompt):
    """Call OpenAI GPT API."""
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

def call_gemini(prompt):
    """Call Google Gemini API."""
    full_prompt = SYSTEM_PROMPT + f"\nUser command: {prompt}"
    response = gemini_model.generate_content(full_prompt)
    return response.text.strip()

def get_waypoints_from_llm(command):
    """Try Groq first, then GPT, then Gemini as fallback."""
    print(f"\nCommand: '{command}'")
    
    apis = [
        ("Groq (llama-3.3-70b)", call_groq),
        ("GPT (gpt-4o-mini)", call_gpt),
        ("Gemini (2.0-flash-lite)", call_gemini),
    ]
    
    for api_name, api_fn in apis:
        try:
            print(f"  Trying {api_name}...")
            raw = api_fn(command)
            
            print(f"  Raw response: {raw}")
            raw = raw.replace("```json", "").replace("```", "").strip()
            
            parsed = json.loads(raw)
            waypoints = parsed["waypoints"]
            
            print(f"  {api_name} succeeded - {len(waypoints)} waypoints:")
            for i, wp in enumerate(waypoints):
                print(f"    WP{i+1}: North={wp['north']}m, East={wp['east']}m, Alt={wp['alt']}m")
            
            return waypoints
        
        except Exception as e:
            print(f"  {api_name} failed: {e}")
            continue
    
    raise Exception("All APIs failed! Check your API keys and quotas.")


test_commands = [
    "Fly 5 meters north",
    "Fly a 10m square",
    "Go forward 8m then turn right 4m",
    "Go around the pole and come back"
]

print("=" * 50)
print("PHASE 3 - LLM Waypoint Generation Test")
print("=" * 50)

for command in test_commands:
    print(f"\n{'='*50}")
    waypoints = get_waypoints_from_llm(command)
    print(f"SUCCESS - {len(waypoints)} waypoints generated")
    print("="*50)
    time.sleep(2)

print("\nPhase 3 Complete!")