# Smart Multimeter Simulation

## Part 1 — What did you build?
This project simulates a digital multimeter capable of measuring Resistance (R), Capacitance (C), and Inductance (L) across a wide dynamic range. The primary focus is on Resistance measurement using a voltage divider method, while Capacitance and Inductance are implemented at a basic level using RC timing and LC resonance respectively. An auto-ranging engine is implemented to automatically select the correct measurement range without user input. The simulation achieves high accuracy, with average error below 1% across all measurement modes.

## Part 2 — How to set it up
Clone the repository and install dependencies:
 
cd your-repo/end_term/smart_multimeter  
pip install -r requirements.txt  

## Part 3 — How to run the simulation
python simulate.py  

This script generates 50 test values across the full range (100 Ω to 1 MΩ), simulates measurements using Gaussian noise (σ = 0.5% of true value), applies the auto-ranging engine with hysteresis, and records the true value, measured value, active range, and percentage error. It prints a results table and computes the average error. The script also generates two plots: Accuracy vs Input Value and Auto-Range State Over Time, which are saved in the results/ folder.

## Part 4 — Your results
| Method | R Error | C Error | L Error |
|--------|--------|--------|--------|
| Fixed-range (no auto) | ~2–4% | ~2–5% | ~3–6% |
| Auto-ranging simulation | 0.376% | 0.391% | 0.937% |

The auto-ranging system significantly reduces measurement error by dynamically selecting the optimal range, ensuring consistent accuracy across a 10⁵ range of values. Inductance measurement shows slightly higher error due to the squared dependence on frequency in its formula, which amplifies the effect of noise.

## Part 5 — Known limitations
This is a software-only simulation and does not model all real-world hardware effects. In a real implementation, factors such as ADC quantization noise, op-amp offset errors, probe resistance, parasitic capacitance and inductance, and temperature drift would affect measurement accuracy. These effects are not included in this model.
