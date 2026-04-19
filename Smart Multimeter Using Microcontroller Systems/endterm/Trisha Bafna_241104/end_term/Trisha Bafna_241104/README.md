# Smart Multimeter Simulation

## Part 1 — What did you build?
I built a software simulation of an industry-grade digital multimeter capable of measuring Resistance, Capacitance, and Inductance. The simulation includes an auto-ranging engine that evaluates readings and switches scales dynamically without user input. The overall simulation consistently operates with an average measurement error of less than 2%.

## Part 2 — How to set it up
//bash
git clone [https://github.com/Gallivanter01/smart-multimeter-simulation.git](https://github.com/Gallivanter01/smart-multimeter-simulation.git)
cd smart-multimeter-simulation/end_term/smart_multimeter
pip install -r requirements.txt

## Part 3 —  How to run simulation
python simulate.py

## Part 4 — results
| Method | R Error | C Error | L Error |
| :--- | :--- | :--- | :--- |
| Fixed-range (no auto) | 29.71% | 0.43% | 0.63% |
| Your auto-ranging sim | 0.52% | 0.43% | 0.63% |

## Part 5 - Known limitations
While the simulation accurately models ideal physics and baseline ADC quantization noise, real-world hardware implementation would introduce additional variances. Physical components would be subject to temperature drift over time, op-amp offset voltages would skew low-level readings, and the inherent probe resistance would necessitate a nulling feature for accurate low-range resistance measurements.
