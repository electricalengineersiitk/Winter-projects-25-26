# Natural Language Command & Control for UAVs

Name: Marut Garg  
Roll No: 240633  
Mentors: Pratyush Srivastava, Akul Agarwal  

---

## 1. Project Description

This project builds a complete system that allows a user to control a UAV using simple natural language commands. Instead of manually specifying coordinates or using a joystick, the user can give commands like "Fly a 5 meter square and come back" or "Go around the pole". The system processes this input, converts it into structured waypoint trajectories, validates them for safety, and executes them on a simulated drone using ArduPilot SITL.

The main goal of this project is to demonstrate how high-level human instructions can be translated into precise, deterministic drone movements. It combines concepts from natural language processing, geometry-based trajectory planning, safety validation, and real-time drone control.

---

## 2. Overall System Flow

The entire system works as a pipeline:

User Command → Command Parsing → Waypoint Generation → Safety Validation → Drone Execution (SITL)

Each module plays a specific role in converting human intent into safe and executable drone actions.

---

## 3. Architecture Overview

The system is divided into the following main components:

1. Command Input  
2. Command Interpretation (LLM / rule-based parsing)  
3. Geometry-based Trajectory Generator  
4. Safety Checker (constraint validation layer)  
5. Drone Controller (DroneKit interface)  
6. ArduPilot SITL (simulation environment)

All modules are integrated together in a single controller script to form a complete working system.

---

## 4. Trajectory Generation (geometry_utils.py)

This module generates mathematical waypoint paths based on the command.

Implemented trajectories:

- Circle → evenly spaced points forming a closed loop  
- Helix → circular motion with increasing altitude  
- Orbit → circle around a fixed point (pole)  
- Scan (Lawnmower pattern) → area coverage path  

Key idea:
All trajectories are deterministic and based on simple trigonometry (sin, cos) and geometry.

Example:
Circle points are generated using:
north = center_n + r * cos(theta)  
east = center_e + r * sin(theta)

This ensures smooth and predictable motion.

---

## 5. Command Handling (main_controller.py + llm_waypoints.py)

The system interprets user input and maps it to trajectory functions.

Examples:
- "Fly a 5 m square and come back" → square waypoints  
- "Go around the pole" → orbit trajectory  
- "Fly north 10 meters" → linear path  

The commands are converted into structured waypoint dictionaries:

{
  "north": value,
  "east": value,
  "alt": value
}

These waypoints are then passed to the drone controller.

---

## 6. Safety Layer (safety_checker.py)

A critical part of the system is the safety validation layer.

Rules enforced:

- Minimum altitude: ≥ 2 meters  
- Maximum altitude: ≤ 50 meters  
- Maximum range: ≤ 100 meters from home  
- Obstacle avoidance: ≥ 2 meters away  
- Return condition: final waypoint must be near home  

If any rule is violated:
- The mission is rejected  
- A reason is returned  

This ensures that unsafe commands never reach the drone.

---

## 7. Drone Control (DroneKit)

The drone is controlled using DroneKit API.

Key operations:

- Connect to SITL  
- Set mode to GUIDED  
- Arm the drone  
- Takeoff to desired altitude  
- Move through waypoints using simple_goto()  
- Return to launch (RTL)  

The drone follows the generated path step-by-step.

---

## 8. Simulation Environment (SITL)

ArduPilot SITL is used to simulate the drone.

Features:
- Realistic flight behavior  
- GPS simulation  
- MAVLink communication  
- Visual map + console  

This allows testing without real hardware.

---

## 9. File Description

connect_test.py  
→ Tests connection between Python and SITL  

waypoint_nav.py  
→ Implements basic square waypoint navigation  

llm_waypoints.py  
→ Converts natural language commands into waypoint structures  

geometry_utils.py  
→ Generates trajectories (circle, helix, orbit, scan)  

safety_checker.py  
→ Validates waypoint safety rules  

main_controller.py  
→ Integrates all components into a full pipeline  

---

## 10. Prerequisites

- Ubuntu / WSL (recommended)  
- Python 3.8+  
- ArduPilot SITL  
- MAVProxy  
- DroneKit  
- pymavlink  
- numpy  
- matplotlib  

---

## 11. Installation

Clone the repository:

git clone <your-repo-link>  
cd NLP-UAV-Project  

Install dependencies:

pip install -r requirements.txt  

---

## 12. How to Run

Step 1: Start SITL

sim_vehicle.py -v ArduCopter --console --map --out=127.0.0.1:14550  

Step 2: Run the controller

python main_controller.py  

---

## 13. Example Commands

Fly a 5 m square and come back  
Go around the pole and come back  
Fly north 10 meters then hover  
Fly at ground level  

---

## 14. Demo

(Add your video link here)

The demo should show:
- Square path execution  
- Orbit maneuver  
- Safety rejection case  

---

## 15. Results

The system successfully:

- Interprets natural language commands  
- Generates accurate trajectories  
- Validates safety constraints  
- Executes missions in SITL  

All core functionalities worked as expected.

---

## 16. What I Learned

Through this project, I learned how high-level natural language commands can be converted into structured control instructions for a UAV. I understood how trajectory generation works using mathematical models and how important safety validation is in real-world systems. Working with DroneKit and ArduPilot helped me understand MAVLink communication and drone control mechanisms. I also gained experience in debugging simulation issues such as GPS delays and connection errors. Overall, this project helped me bridge the gap between AI-based command interpretation and real-time control systems.

---

## 17. Challenges Faced

- Setting up SITL and DroneKit connection  
- Handling GPS initialization delays  
- Debugging waypoint movement inaccuracies  
- Managing repeated simulation runs  
- Ensuring safety rules were correctly enforced  

---

## 18. Limitations

- Works only in simulation (SITL)  
- Uses simple command parsing (limited NLP capability)  
- Path accuracy depends on simple_goto()  
- No real-world obstacle detection  

---

## 19. Future Improvements

- Integrate advanced LLM for better command understanding  
- Add real-time obstacle detection  
- Improve path tracking accuracy  
- Extend to real drone hardware  
- Support more complex commands  

---

## 20. Conclusion

This project demonstrates a complete pipeline for controlling UAVs using natural language. It combines AI, geometry, and control systems into a single working solution. The system successfully converts human instructions into safe and executable drone actions, providing a strong foundation for future real-world autonomous UAV applications.
