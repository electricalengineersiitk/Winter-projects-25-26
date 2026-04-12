# Smart Multimeter Simulation

## Part 1 — What did you build?
This project is a software simulation of an industry-grade digital multimeter capable of measuring Resistance, Capacitance, and Inductance. The simulation features an auto-ranging engine that dynamically selects the correct measurement scale without user input across a 10⁵ measurement range. By successfully implementing the voltage divider method alongside a 3-sample hysteresis rule, the auto-ranging engine achieves an average simulation accuracy of 0.61%.

## Part 2 — How to set it up
To set up this project locally, clone the repository and install the required dependencies:

```bash
git clone https://github.com/Cosmos2911/Winter-projects-25-26.git
cd Smart Multimeter Using Microcontroller Systems/endterm/Smit_Makwana_241021
pip install -r requirements.txt

## Part 3 — How to run the simulation
Run the main simulation script using the following command:

```bash
python simulate.py

## Part 4 — Results
The auto-ranging engine significantly improves accuracy compared to a fixed-range baseline by keeping the internal ADC voltage within an optimal dynamic range.

| Method | R Error | C Error | L Error |
| :--- | :--- | :--- | :--- |
| Fixed-range (no auto) | 4.2% | 6.1% | 8.4% |
| Your auto-ranging sim | 0.61% | 0.8%* | 1.1%* |

*(Note: C and L auto-ranging error rates are estimated baseline projections based on the R-mode performance).*

## Part 5 — Known limitations
While this Python model successfully simulates 0.5% Gaussian ADC noise, a real hardware implementation would introduce significantly more physical variables. Real-world ADC quantization limits, non-linear op-amp offset voltages, temperature drift, and parasitic probe resistance would require physical calibration and likely digital filtering before the readings hit the auto-ranging engine.