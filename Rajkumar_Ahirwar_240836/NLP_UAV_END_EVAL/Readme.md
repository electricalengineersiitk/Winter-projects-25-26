# Natural Language Command and Control for UAVs

- Name: Rajkumar Ahirwar

## Overview

This project enables controlling a UAV using natural language commands. It leverages a Large Language Model (LLM) to interpret user commands, generates a flight path, validates it for safety, and executes the mission in a simulated environment.

## Prerequisites

- **OS**: Linux (preferred)
- **Python**: 3.1+
- **Tools**:
  - A SITL (Software-In-The-Loop) drone simulator, such as [ArduPilot](https://ardupilot.org/dev/docs/sitl-overview.html) with a ground control station like [QGroundControl](http://qgroundcontrol.com/) or [Mission Planner](https://ardupilot.org/planner/).

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/electricalengineersiitk/Winter-projects-25-26/
   cd Winter-projects-25-26/Rajkumar_Ahirwar_240836/NLP-UAV-Project
   ```
2. **Create a virtual environment (recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```
4. **Set up your API key:**
   Create a `.env` file in the project root and add your Groq API key:

   ```
   GROQ_API_KEY="your-groq-api-key"
   ```

## How to Run

1. **Launch the SITL Simulator:**
   Start your SITL instance. For example, with ArduPilot:
   (in ubuntu terminal)
   wait till the map is launched.

   ```bash
   cd "/mnt/d/ASUS/ArduPilot/ardupilot/ArduCopter" && \
   sim_vehicle.py -v ArduCopter --map --console --out=udp:127.0.0.1:14550 --out=udp:127.0.0.1:14551
   ```

   Wait for the simulator to be ready and for the drone's position to be established (you'll see GPS lock messages).
2. **Run the Main Controller:**
   In a new terminal, navigate to the project directory and run:
   (in ubuntu terminal)

   ```bash
   cd "/mnt/d/ASUS/NLC&C_UAVs/Winter-projects-25-26/Rajkumar_Ahirwar_240836/NLP-UAV-Project"
   source venv-ardupilot/bin/activate
   python3 main_controller.py
   ```
3. **Enter Commands:**
   When prompted, type a flight command like `"Fly a 10m square"` or `"orbit the takeoff point at a 15 meter radius"` and press Enter.

## File Descriptions

- `main_controller.py`: The central orchestrator that handles user input, LLM calls, safety checks, and drone control.
- `llm-waypoints.py`: Interfaces with the Groq API to translate natural language into a JSON flight plan.
- `safety_checker.py`: A middleware that validates the LLM's flight plan against safety rules (altitude, range, etc.).
- `geometry_utils.py`: A utility for generating waypoints for geometric shapes like circles and helices.
- `waypoint-nav.py`: A simple script for basic waypoint navigation without LLM integration.
- `connect_test.py`: A script to test basic takeoff and landing.
- `connect_test1.py`: A script to test flying a hardcoded square pattern.
- `requirements.txt`: A list of all Python dependencies for the project.

## Demo

[DEMO VIDEO LINK](https://drive.google.com/file/d/1zE-_5gqARWPsoXbCfG8A_tXrXmsft7G2/view?usp=sharing)

## What I Learned

This project was a deep dive into the practical application of Large Language Models for robotics control. I learned that effective prompt engineering is crucial; instructing the LLM to return only raw JSON and defining strict formatting rules was key to reliably parsing flight plans. On the drone control side, I gained hands-on experience with the DroneKit library, managing vehicle states, and translating between relative (North-East-Down) and global coordinate systems. A major takeaway was the importance of not blindly trusting AI-generated outputs. This led to the development of a critical safety checker middleware, as LLMs can sometimes generate unsafe or nonsensical commands, reinforcing the need for validation before execution. Integrating the Groq API taught me about handling asynchronous requests and the importance of robust error handling. The biggest challenge was creating a seamless pipeline from a high-level natural language command to a low-level, validated drone action in a simulated environment.

## Known Issues

It is not able to make a perfect circle. But it can make shapes like square rectangle, pentagon , hexagon etc. It is prompted only for square, triangle, rectangle.
