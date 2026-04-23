This project is a python based simulation which uses dijkstra's algorithm to navigate around a circular obstacle. simulation shows 5 UAVs  flying from starting point to goal while maintaining a 'V' formation.

setup:
To set up the environment, follow these commands in your terminal:
create a fork of the public repository.    
 1. git clone https://github.com/[your-username]/[your-repo].git
 2. cd [your-repo]/end_term
 3. pip install -r requirements.txt

how to run:
 To run the full simulation and to save result use:
  1. python simulate.py
 This will open an animation window showing the drones in motion. It also automatically saves  plots and a GIF animation of the flight to the results folder'


function of each script:
  map_setup.py:
  Defines the 100x100 grid, places the central obstacle, and sets the start (5, 50) and goal (95, 50) coordinates. a function is_collision is also added to prevent collision from obstacle using a safety buffer .

  path_planner.py:
  It Implements Dijkstra's algorithm to find a collision-free waypoint path around the obstacle.

  trajectory.py:
   Uses Cubic Splines to convert jagged waypoints to make  smooth min-time and min-energy trajectories.
   
   formation.py: Defines the v shape  formation ,  A shape  formation and triangle shape formation offsets and assigns drones to their relative positions.
   
   simulate.py: It is the  main script that integrates all modules to run the simulation, animation, and data logging.

results
 Observation : The Minimum-Time trajectory reached the goal in 10 seconds, while the Minimum-Energy trajectory took 25 seconds the energy-efficient version used a lower velocity profile with more gradual speed changes to conserve power.

formation:
 I chose a 'V' shaped formation consisting of 5 UAVs."V" shaped can be change to "A" shape or traingle shaped formation by using script simulate.py.  A path was computed for the formation centroid. each drone was assigned a fixed offset relative to this centroid .these offsets remained constant throughout the flight to maintain the shape perfec
