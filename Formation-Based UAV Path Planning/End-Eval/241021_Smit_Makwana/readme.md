# End-Term UAV Path Planning Simulation

## Part 1: What did you build?
[cite_start]This project simulates 5 UAVs flying from a start point to a goal point while maintaining a 'V' formation shape[cite: 5]. [cite_start]I implemented the A* algorithm to find a collision-free path around a single obstacle[cite: 6, 7].

## Part 2: Setup
To run this project locally, clone the repository and install the dependencies:

\`\`\`bash
git clone https://github.com/Cosmos2911/Winter-projects-25-26.git
cd "Formation-Based UAV Path Planning/End-Eval/241021_Smit_Makwana"
pip install -r requirements.txt
\`\`\`

## Part 3: How to run
Execute the main simulation script:

\`\`\`bash
python simulate.py
\`\`\`

When this runs, it performs the A* path planning, generates both min-time and min-energy trajectories, applies the formation offsets, and then outputs three files to the \`results/\` directory: a static path plot, a trajectory comparison plot, and an animation GIF of the flight. [cite_start]It also prints a summary of the flight durations to the console[cite: 36, 75].

## Part 4: What each script does
* [cite_start]**map_setup.py** — defines the 2D grid, places the obstacle, and sets the start and goal coordinates[cite: 39, 50, 51, 52].
* **path_planner.py** — implements the A* algorithm to find a collision-free waypoint path avoiding the obstacle[cite: 40, 54, 55].
* [cite_start]**trajectory.py** — converts the waypoint path into smooth minimum-time and minimum-energy trajectories using cubic splines[cite: 41, 58, 59].
* [cite_start]**formation.py** — defines the 5-drone 'V' formation shape and computes individual drone paths using fixed offsets[cite: 42, 65, 66, 67].
* **simulate.py** — the main driver script that runs all components, produces the animation, and saves all plots to the results folder[cite: 43, 69, 70].

## Part 5: Results
### Static Path
![Path Plot](results/path_plot.png)

### Trajectory Comparison
![Trajectory Comparison](results/trajectory_comparison.png)

### Animation
![Formation Animation](results/formation_animation.gif)

**Observation:** *(Add your short observation here based on the printed output in your console. For example: "The minimum-time trajectory is faster, completing the flight in X.XX seconds compared to the minimum-energy trajectory's Y.YY seconds. However, the minimum-energy trajectory shows significantly smoother acceleration profiles." [cite: 46])*

## Part 6: Formation details
The simulation uses a 'V' formation shape consisting of N=5 UAVs[cite: 48]. Drones are assigned to positions using fixed coordinate offsets from the computed centroid trajectory (the A* path). [cite_start]These offsets remain constant throughout the flight to maintain the shape[cite: 68].