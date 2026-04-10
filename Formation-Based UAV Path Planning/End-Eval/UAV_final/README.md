# UAV_path_planning
# UAV Formation Flight Simulation

### Part 1 — What Is built?
Built a simulation of a UAV swarm navigating a 2D environment while avoiding a central circular obstacle. The swarm consists of 7 UAVs flying in a 'W' formation. For the pathfinding RRT* algo is implemented then the route is smoothed into both minimum-time and minimum-energy trajectory.

### Part 2 — Setup
To set up the project locally, run the following commands in your terminal:

```bash
git clone https://github.com/ayushy1012/UAV_path_planning.git
cd UAV_final/end_term
pip install -r requirements.txt
```

### Part 3 — How to run
```bash
python simulate.py
```
It may take 30-45 seconds for this script to execute , after execution is complete an animation will pop up which shows both minimum energy and minimum time trajectories 

### Part 4 — What each script does 
•	map_setup.py — defines the 2D grid, places the obstacle, sets start and goal coordinates

•	path_planner.py — implements RRT* algorithm to find a collision-free path

•	trajectory.py — converts the path into smooth min-time and min-energy trajectories

•	simulate.py — runs everything together and produces the animation and plots

### Part 5 — Results
<img src="end_term\results\formation_animation.gif" width="600" alt="UAV Formation Animation">

Min-Time Trajectory (Red):
  Total Time: 7.30 seconds
  Energy Cost (proxy): 5190.29

Min-Energy Trajectory (Blue):
  Total Time: 29.18 seconds
  Energy Cost (proxy): 79.78

### Part 6 — Formation details
The formation shape choosen is W , the number of UAVs are 7 , the formation shape/number of drones can be changed by editing the formation class in map_setup.py