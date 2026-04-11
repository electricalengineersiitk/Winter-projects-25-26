# Smart Multimeter Simulation


This project simulates a digital multimeter for measuring Resistance (R), Capacitance (C), and Inductance (L) with automatic range selection. The auto-ranging engine selects the correct scale automatically without user input. All three modes achieve accuracy well under 2%.

# How to set it up

git clone https://github.com/your-username/your-repo.git
cd your-repo/end_term/smart_multimeter
pip install -r requirements.txt

## How to run the simulation

python simulate.py

Runs 50 test values across all 5 ranges for R, C, and L. Prints error per sample and a final results table. Saves both plots to results/.


##  results

| Method                | R Error | C Error | L Error |
|-----------------------|---------|---------|---------|
| Fixed-range (no auto) | ~4.5%   | ~3.0%   | ~3.5%   |
| Auto-ranging sim      | ~0.4%  | ~0.3%  | ~0.5%  |

All three modes pass the 2% target.


# Known limitations

This is a software simulation, so real hardware problems are not included.
In a real device, the measurement would be affected by:
- Small errors in the ADC chip
- Tiny voltage offsets in the op-amp
- Resistance in the probe wires
- Temperature changes affecting components
- Unwanted capacitance on the PCB board
