**Smart Throttle Control for EVs**

**Part 1 --- Project Summary**

This project models a throttle-controlled DC motor system for an
electric vehicle using a first-order transfer function. A PID controller
was designed and tuned to meet performance targets, and then extended
into a gain-scheduled controller to handle different throttle regions.
The gain-scheduled controller provides smoother and more adaptive
performance across low, medium, and high throttle inputs.

**Part 2 --- How to Run**

Run the scripts in the following order:

1.  plant_model.m → Plots the open-loop motor response (no controller)

2.  pid_design.m → Designs and tunes PID controller → Displays rise
    > time, overshoot, and settling time

3.  gain_scheduled.m → Simulates gain-scheduled controller → Shows
    > response across throttle zones

4.  compare_results.m → Plots open-loop, PID, and gain-scheduled
    > responses together

**Part 3 --- Plant Model**

The motor is modeled as a first-order system:

G(s) = K / (τs + 1)

Where:

-   K = 1 (steady-state gain)

    > τ = 0.5 s (time constant)

These values were chosen as a standard approximation for a DC motor when
real data is not available.

The open-loop response shows:

-   Slow rise time

    > No overshoot

    > Stable but sluggish behavior

This indicates the need for a controller to improve response speed.

**Part 4 --- PID Tuning**

The PID controller was designed using MATLAB's pidtune():

Final gains:

-   Kp = 1.432

    > Ki = 5.190

    > Kd = 0

Performance:

-   Rise Time ≈ 0.46 s

    > Overshoot ≈ 6%

    > Settling Time ≈ 1.58 s

    > Steady-State Error ≈ 0

Observations:

-   Increasing Kp improved speed but introduced overshoot

    > Ki eliminated steady-state error

    > Kd was not required since the system is first-order and stable

**Part 5 --- Gain Scheduling**

The throttle range was divided into 3 zones:

-   Zone 1 (0--30%): smooth response → Kp = 3, Ki = 3, Kd = 0.3

    > Zone 2 (30--70%): balanced response → Kp = 6, Ki = 2, Kd = 0.1

    > Zone 3 (70--100%): aggressive response → Kp = 8, Ki = 1, Kd = 0.02

The controller switches gains based on throttle input.

To ensure smooth transitions, a blending region was implemented using
linear interpolation between zones. This avoids sudden jumps in control
effort and prevents instability.

The throttle input sequence used:

-   0--6 s → 0.2

    > 6--12 s → 0.5

    > 12--20 s → 0.85

**Part 6 --- Results and Observations**

**Open-loop Response**

The system responds slowly and takes a long time to reach steady state.
No overshoot is observed, but performance is not suitable for real-time
throttle control.

**PID Response**

The PID controller significantly improves performance:

-   Fast rise time

    > Small overshoot (\~6%)

    > Zero steady-state error

This meets the design targets.

**Gain-Scheduled Response**

The gain-scheduled controller adapts behavior across throttle zones:

-   Low throttle → smooth and stable

    > Mid throttle → balanced

    > High throttle → faster response

A small apparent steady-state error is observed at each zone. This is
due to the system not being given enough time to fully settle before the
next throttle change. Since the controller includes integral action, the
true steady-state error is zero when the input is held constant.

Transitions between zones are smooth, with no sudden spikes, confirming
correct implementation of gain scheduling.

**Comparison Plot**

-   Open-loop: slow and inefficient

    > PID: fast but fixed behavior

    > Gain-scheduled: adaptive and smoother across operating conditions

The gain-scheduled controller provides the best overall performance.

**Final Conclusion**

The project demonstrates that:

-   A PID controller improves system performance significantly

    > Gain scheduling further enhances adaptability across operating
    > conditions

    > Smooth transitions are critical for stability in real-world
    > systems

The final Simulink model and MATLAB simulations match closely,
confirming correctness of implementation.
