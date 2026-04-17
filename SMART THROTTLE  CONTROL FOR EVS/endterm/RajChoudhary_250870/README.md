\## Smart Throttle Control for EVs



\---



\### Part 1 — Project Summary



In this project, a simplified throttle–motor system was modeled using a first-order transfer function. A PID controller was designed and tuned to meet performance requirements such as rise time, overshoot, and steady-state error. The controller was then extended to a gain-scheduled version to handle different throttle ranges. The results show that the PID improves system response significantly, and gain scheduling allows better behavior across varying operating conditions.



\---



\### Part 2 — How to Run



1\. Run `plant\_model.m` → plots open-loop step response

2\. Run `pid\_design.m` → plots closed-loop PID response

3\. Run `gain\_scheduled.m` → plots gain-scheduled response

4\. Run `compare\_results.m` → plots all responses together



\---



\### Part 3 — Plant Model



The motor was modeled as a first-order transfer function:



G(s) = K / (τs + 1)



The chosen values were:



\* K = 1

\* τ = 0.1



The value of K = 1 ensures zero steady-state error for a unit input. The value of τ was selected after testing different values and observing that very small τ values lead to unrealistically fast responses. τ = 0.1 gives a fast but still reasonable system response for simulation.



\---



\### Part 4 — PID Tuning



A PID controller was designed using MATLAB’s pidtune function.



The controller gains obtained were:



\* Kp = (from pidtune result)

\* Ki = (from pidtune result)

\* Kd = (from pidtune result)



The system performance after tuning:



\* Rise time was less than 1 second

\* Overshoot was below 10%

\* Steady-state error was approximately zero



During tuning, the controller behavior was adjusted to reduce steady-state error and control overshoot. The PID controller significantly improved response speed and accuracy compared to the open-loop system.



\---



\### Part 5 — Gain Scheduling



To improve performance across different throttle levels, gain scheduling was implemented by dividing the throttle into three zones:



\* Zone 1 (0–30%): Higher Ki for smoother response and reduced steady-state error

\* Zone 2 (30–70%): Balanced PID gains

\* Zone 3 (70–100%): Higher Kp and lower Ki for faster response



Different PID gains were assigned to each zone, and switching between them was done based on the current throttle value. The system was simulated with a throttle sequence of 20% → 50% → 85%. The transitions between zones did not introduce significant spikes, indicating stable switching.



\---



\### Part 6 — Results and Observations



Open-loop response:

The system responds slowly based on the time constant and has no control over performance metrics like rise time or steady-state accuracy.



PID response:

The PID controller improves the response significantly by reducing rise time and eliminating steady-state error while keeping overshoot within limits.



Gain-scheduled response:

The gain-scheduled controller adapts to different throttle ranges, providing smoother control at low throttle and faster response at high throttle.



Comparison plot:

The comparison shows that the PID controller performs much better than the open-loop system, and gain scheduling provides additional flexibility across operating conditions.



