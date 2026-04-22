# Smart Throttle Control for EVs — End-Term Project

## 1. Project Summary

This project focuses on modeling and controlling a throttle–motor system for an electric vehicle using classical control techniques. A first-order plant model is developed and analyzed, followed by the design of a PID controller to meet specified performance criteria. The controller is further extended into a gain-scheduled framework to improve performance across varying operating regions. The results demonstrate clear improvements in response speed, accuracy, and adaptability compared to the open-loop system.

---

## 2. How to Run

Execute the scripts in the following sequence:

1. `plant_model.m`
   → Defines the system transfer function and generates the open-loop response.

2. `pid_design.m`
   → Designs and tunes the PID controller, and generates the closed-loop response with performance metrics.

3. `gain_scheduled.m`
   → Implements gain scheduling across multiple throttle zones and simulates system behavior.

4. `compare_results.m`
   → Plots open-loop, PID-controlled, and gain-scheduled responses together for comparison.

All plots are automatically saved in the specified results directory.

---

## 3. Plant Model

The throttle–motor system is modeled as a first-order transfer function:

G(s) = K / (τs + 1)

Where:

* K = 1 represents the steady-state gain
* τ = 0.5 seconds represents the system time constant

This simplified model captures the dominant dynamics of a throttle-driven motor system. The open-loop response exhibits a slow rise and delayed convergence, indicating limited responsiveness and motivating the need for feedback control.

---

## 4. PID Controller Design and Tuning

A PID controller was designed to meet the following performance targets:

* Rise time < 1 second
* Overshoot < 10%
* Steady-state error ≈ 0

### Final Tuned Gains:

* Kp = 6
* Ki = 3
* Kd = 0.1

### Tuning Strategy:

* **Proportional Gain (Kp):** Increased to improve response speed and reduce rise time
* **Integral Gain (Ki):** Introduced to eliminate steady-state error and ensure accurate tracking
* **Derivative Gain (Kd):** Used to dampen the response and control overshoot

The tuning process followed a sequential approach (P → PI → PID), ensuring stability at each stage. The final response satisfies all performance constraints and significantly improves system dynamics compared to the open-loop case.

---

## 5. Gain Scheduling Strategy

A single PID controller may not provide optimal performance across all operating conditions. To address this, a gain-scheduled controller was implemented by dividing the throttle range into three regions:

* **Zone 1 (0–30%) — Low Throttle:**
  Emphasis on smooth and stable response. Higher integral action ensures elimination of steady-state error.

* **Zone 2 (30–70%) — Mid Throttle:**
  Balanced response with moderate gains for both speed and stability.

* **Zone 3 (70–100%) — High Throttle:**
  Emphasis on fast response. Higher proportional gain and reduced integral action prevent excessive overshoot.

Each zone uses a distinct set of PID gains. The simulation demonstrates how system dynamics adapt across these regions, highlighting improved performance over a fixed-gain controller.

---

## 6. Results and Observations

### Open-Loop Response

The open-loop system exhibits slow dynamics with a gradual rise to steady state. While stable, it lacks responsiveness and is unsuitable for real-time control applications.

### PID-Controlled Response

The PID controller significantly enhances system performance:

* Faster rise time
* Minimal steady-state error
* Controlled overshoot

The system quickly reaches the desired speed and remains stable.

### Gain-Scheduled Response

The gain-scheduled controller adapts performance based on operating conditions:

* Smooth behavior at low throttle
* Balanced response in mid-range
* Aggressive and fast response at high throttle

This demonstrates improved adaptability and robustness compared to a fixed PID controller.

### Comparison of Responses

The comparison plot clearly shows:

* Open-loop response is slow and inefficient
* PID control improves speed and accuracy
* Gain scheduling provides performance optimization across different operating regions

---

## 7. Key Insights

* Even a simple first-order model can effectively capture essential system dynamics for controller design.
* PID tuning requires balancing speed, accuracy, and stability — increasing performance in one aspect often affects another.
* Gain scheduling is a practical method to handle system nonlinearity and varying operating conditions.
* Simulation-based validation is essential before real-world implementation.

---

## 8. Conclusion

This project demonstrates the complete control design workflow: modeling, analysis, controller design, and validation. The PID controller successfully meets performance requirements, while gain scheduling enhances adaptability across operating regions. The approach highlights how classical control techniques can be effectively applied to real-world systems such as EV throttle control.

---

