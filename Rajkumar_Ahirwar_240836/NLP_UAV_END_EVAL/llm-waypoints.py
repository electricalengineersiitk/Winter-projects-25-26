import os
import time
import json
from openai import OpenAI # Groq uses the OpenAI library
from dotenv import load_dotenv

load_dotenv()

# 1. Setup Groq Client
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

# Use Llama 3.3 70B - it is very smart at following JSON instructions
MODEL_ID = "llama-3.3-70b-versatile" 

SYSTEM_PROMPT = """
You are a UAV Flight Planner. Translate English commands into NED waypoints.
Return ONLY raw JSON. No prose.

Rules:
1. ALWAYS provide an "alt" (altitude). 
2. DEFAULT altitude is 10 if the user doesn't specify one. 
3. NEVER return "alt": 0 unless the user explicitly says "land".
4. Coordinates are relative to takeoff point (0,0).

Format: {"waypoints": [{"north": 0, "east": 5, "alt": 10}]}
"""

def get_flight_plan(user_command):
    print(f"\n--- Processing: '{user_command}' ---")
    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_command}
            ],
            temperature=0
        )
        
        raw_text = response.choices[0].message.content.strip()
        # Clean markdown
        raw_text = raw_text.replace("```json", "").replace("```", "").strip()
        
        data = json.loads(raw_text)
        print("✅ Successfully Parsed:")
        print(json.dumps(data, indent=4))
        return data

    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# Test the script
if __name__ == "__main__":
    test_commands = ["Fly 5 meters north", "Fly a 10 m square"]
    for cmd in test_commands:
        get_flight_plan(cmd)
        time.sleep(2) # Groq is so fast you only need a 2s delay!