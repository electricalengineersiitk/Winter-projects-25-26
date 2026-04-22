
# Smart Throttle Control for EVs

## Project Summary

This project models and controls a throttle-driven electric vehicle motor using MATLAB and Simulink. A first-order motor model is developed, followed by the design and tuning of a PID controller. The controller is further extended into a gain-scheduled controller that adapts across different throttle zones.

The results demonstrate improved speed, stability, and adaptability compared to open-loop and fixed PID control.

---

## How to Run

Run the following MATLAB scripts in order:

1. plant_model.m  
2. pid_design.m  
3. gain_scheduled.m  
4. compare_results.m  

All plots are automatically saved in the `results/` folder.

The Simulink model (`throttle_model.slx`) can be opened directly to visualize the closed-loop and gain-scheduled control system.

---

## Plant Model

The motor is modeled as a first-order system:

G(s) = 1 / (0.5s + 1)

### Parameters:
- Gain K = 1  
- Time constant τ = 0.5 seconds  

### Open Loop Performance:
- Rise Time ≈ 1.1 s  
- Settling Time ≈ 2.0 s  
- Overshoot = 0%  

The system is stable but slow and requires control for faster response.

---

## PID Controller Design

Final tuned gains:

- Kp = 3  
- Ki = 2  
- Kd = 0.1  

### Performance:
- Rise Time ≈ 0.4 s  
- Settling Time ≈ 0.8–1.0 s  
- Overshoot ≈ 3–5%  
- Steady-state error ≈ 0  

### Tuning Strategy:
- Increased Kp to reduce rise time  
- Added Ki to eliminate steady-state error  
- Added Kd to reduce overshoot  

---

## Gain Scheduling

To improve performance across varying throttle levels, gain scheduling is implemented.

### Throttle Zones:

- Zone 1 (0–30%) → smooth response  
- Zone 2 (30–70%) → balanced response  
- Zone 3 (70–100%) → aggressive response  

### Gains Used:

| Zone | Kp | Ki | Kd |
|------|----|----|----|
| 1    | 2  | 3  | 0.05 |
| 2    | 3  | 2  | 0.1  |
| 3    | 4  | 1  | 0.15 |

### Throttle Profile:
- 0–5 s → 20%  
- 5–10 s → 50%  
- 10–15 s → 85%  

The controller dynamically switches gains based on throttle input while maintaining continuous system behavior.

---

## Results and Observations

### Open Loop Response
![Open Loop](results/open_loop_response.png)

The system is slow and lacks responsiveness.

---

### PID Response
![PID](results/pid_response.png)

The PID controller significantly improves response speed and eliminates steady-state error with minimal overshoot.

---

### Gain Scheduled Response
![Gain Scheduled](results/gainscheduled_response.png)

The system adapts to throttle changes, providing smooth control at low speeds and faster response at higher speeds.

---

### Comparison Plot
![Comparison](results/comparison_plot.png)

- Open-loop is slow  
- PID is fast and stable  
- Gain scheduling provides the most adaptable performance  

---

## Simulink Model

The Simulink model implements the closed-loop control system with gain scheduling.

### Key Components:
- Step input (throttle)
- Sum block (error calculation)
- PID Controller with external gains
- MATLAB Function block (gain scheduler)
- Transfer Function (motor model)
- Scope (output visualization)

The model demonstrates real-time adaptation of PID gains based on throttle input.

---

## Bonus Insight

Effect of PID parameter variation:

- High Kp → faster response but increased overshoot  
- High Ki → eliminates steady-state error but may cause instability  
- High Kd → reduces overshoot but increases noise sensitivity  

Example:
Excessively high Kp caused oscillations and overshoot greater than 20%.

---

## Conclusion

The project demonstrates that PID control significantly improves system performance over open-loop operation. Gain scheduling further enhances adaptability across different operating conditions, making the system suitable for real-world EV throttle control applications.

---

## Folder Structure

end_term/
├── README.md  
├── plant_model.m  
├── pid_design.m  
├── gain_scheduled.m  
├── compare_results.m  
├── throttle_model.slx  
└── results/  
    ├── open_loop_response.png  
    ├── pid_response.png  
    ├── gainscheduled_response.png  
    └── comparison_plot.png
