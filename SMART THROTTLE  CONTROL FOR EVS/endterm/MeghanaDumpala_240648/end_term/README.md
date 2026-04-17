Smart Throttle Control for EVs


Project Summary

A simplified electric vehicle throttle–motor system was modeled using a first-order transfer function. A PID controller was designed to improve response speed and reduce steady-state error. Gain scheduling was implemented to adapt controller gains across different throttle regions.


How to Run

1. Run `plant_model` → generates open-loop response
2. Run `pid_design` → generates PID response
3. Run `gain_scheduled` → generates gain-scheduled response
4. Run `compare_results` → generates comparison plot


Plant Model

Transfer Function:

G(s) = 1 / (0.5s + 1)

K = 1 and τ = 0.5 s were selected as reasonable assumptions for an EV motor model.


PID Tuning

Gain	Value
Kp	2.5
Ki	3
Kd	0.15



The proportional gain improves response speed, integral action removes steady-state error, and derivative action reduces overshoot.


Gain Scheduling

Three throttle zones were defined:

Zone 1 (0–30%): smooth response with higher Ki
Zone 2 (30–70%): balanced PID behaviour
Zone 3 (70–100%): faster acceleration using higher Kp

PID gains were updated dynamically as throttle changed.


Results and Observations

Open-loop response shows slow behaviour without control.
PID control improves rise time and accuracy.
Gain scheduling provides adaptive performance across operating conditions.
The comparison plot demonstrates overall improvement of closed-loop control.
