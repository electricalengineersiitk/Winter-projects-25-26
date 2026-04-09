# Smart Throttle Control for EVs

## Project Summary
This project models an electric vehicle (EV) throttle-motor system using a transfer function. A PID controller is designed to improve system performance, and gain scheduling is implemented to adapt controller behavior across different throttle zones.

---

## How to Run
1. Run `plant_model.m` → generates open-loop response  
2. Run `pid_design.m` → generates PID response  
3. Run `gain_scheduled.m` → generates gain-scheduled response  
4. Run `compare_results.m` → generates comparison plot  

---

## Plant Model

The motor is modeled as a first-order system:

G(s) = 1 / (0.5s + 1)

The values K = 1 and τ = 0.5 were assumed as standard values for a first-order motor model in the absence of real motor data.

---
## PID Tuning

The PID controller gains used are:

Kp = 3  
Ki = 5  
Kd = 0.1  

The PID gains were tuned manually. First, the proportional gain (Kp) was increased to improve the speed of response. Then, the integral gain (Ki) was increased to eliminate steady-state error. Finally, a small derivative gain (Kd) was added to reduce overshoot and improve stability.

The final controller provides a fast rise time, minimal overshoot, and negligible steady-state error compared to the open-loop system.

---
## Gain Scheduling
The throttle range is divided into three zones:

- Zone 1 (0–30%): Kp=2, Ki=6, Kd=0.1 (smooth response)  
- Zone 2 (30–70%): Kp=3, Ki=5, Kd=0.1 (balanced response)  
- Zone 3 (70–100%): Kp=5, Ki=2, Kd=0.1 (fast response)  

Different PID gains are used in each zone to adapt system performance.
The throttle range was divided into three zones based on expected operating conditions: low, medium, and high throttle. The transitions between zones were handled smoothly by segmenting the simulation time.

---

## Results and Observations

### Open-loop Response
The system responds slowly and takes time to reach steady state.
![Open Loop](results/open_loop_response.png)
### PID Response
The PID controller improves response speed and reduces error.
![PID](results/pid_response.png)
### Gain-Scheduled Response
The controller adapts to different throttle levels, providing smoother control at low speeds and faster response at high speeds.

![Gain Scheduled](results/gainscheduled_response.png)
### Comparison Plot
The comparison shows that the PID controller improves performance over open-loop, and gain scheduling adapts performance across operating conditions.
![Comparison](results/comparison_plot.png)
---

## Simulink Model
A closed-loop Simulink model was built using a PID controller and feedback loop to validate the system behavior.

---

## Conclusion
The project demonstrates how PID control and gain scheduling can improve EV throttle response and adapt performance across different operating conditions.

## Additional Observation
Improper PID tuning (e.g., very high Kp) resulted in oscillations, highlighting the importance of proper tuning.
