



# Natural Language Command Interface for UAV Control

Name: Lalit Meena  
Roll No: 240591  
Mentor: Pratyush Srivastava & Akul Agarwal

---

# Overview

This project builds a system that allows a UAV to be controlled using simple natural language instructions. Instead of manually specifying coordinates, the user can give commands such as “Fly a 5 meter square and come back” or “Orbit around the pole.” The system interprets the command, generates waypoint trajectories, verifies safety constraints, and executes the mission using a simulated drone environment.

---

# Architecture Diagram

The system operates as a pipeline where a natural language command is gradually converted into executable drone actions.


User Command
|
v
Command Interpreter
|
v
Trajectory Generator
|
v
Safety Checker
|
v
Drone Controller
|
v
ArduPilot SITL Simulation


Each stage processes the output of the previous stage until the drone receives validated waypoint instructions.

---

# Prerequisites

Before running the project, make sure the following tools are installed:

Operating System  
Ubuntu / WSL recommended

Software Requirements

Python 3.8 or higher  
ArduPilot SITL  
MAVProxy  
DroneKit  
pymavlink  

Python Libraries

numpy  
matplotlib  

These tools are required to simulate the drone environment and communicate with it using Python.

---

# Installation

Clone the repository and install the required dependencies.

Clone the project repository:


git clone < git clone https://github.com/electricalengineersiitk/Winter-projects-25-26.git>
cd <Natural Language Command & Control for UAVs>


Install Python dependencies:


pip install -r requirements.txt


Make sure SITL and MAVProxy are properly installed before running the system.

---

# How to Run

First start the ArduPilot SITL simulation.


sim_vehicle.py -v ArduCopter --console --map --out=127.0.0.1:14550


Once the simulator is running, open another terminal and run the main controller.


python main_controller.py


You can now enter commands such as:


Fly a 5 meter square and come back
Go around the pole
Fly north 10 meters then hover


The system will generate waypoints and execute the mission in simulation.

---

# File Descriptions

connect_test.py  
Tests the communication between Python and the SITL simulator.

waypoint_nav.py  
Implements basic waypoint navigation such as square trajectories.

llm_waypoints.py  
Converts natural language commands into structured waypoint data.

geometry_utils.py  
Generates geometric flight paths including circle, helix, orbit, and scan patterns.

safety_checker.py  
Checks that generated waypoints satisfy safety rules.

main_controller.py  
Acts as the main entry point and coordinates the entire command pipeline.

---

# Demo

Example demonstrations of the system are shown below.


Safety Rejection Example  
Command: Fly at ground level  
Result: Mission rejected due to safety constraints

---

# What I Learned

Through this project I learned how different software components can be integrated to control a drone using high level instructions. I gained experience working with the ArduPilot SITL simulation environment and learned how drones can be controlled programmatically through DroneKit and MAVLink communication. I also understood how geometric trajectory generation can be used to create structured flight paths such as circles and scan patterns. Implementing the safety validation layer helped me understand the importance of enforcing constraints before executing drone commands. Overall, the project gave practical exposure to drone software architecture and natural language based control systems.

---

# Known Issues

The command interpreter currently supports only a limited set of predefined commands. More complex natural language inputs may not always be interpreted correctly. The system is currently tested only in the SITL simulation environment and has not been validated on real drone hardware. GPS initialization delays in SITL can sometimes cause the drone to take longer to begin execution. Future improvements could include a more robust language parser and additional safety checks.

