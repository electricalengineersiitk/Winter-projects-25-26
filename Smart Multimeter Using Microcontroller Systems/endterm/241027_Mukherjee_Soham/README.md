## Project Description

>This project simulates an auto-ranging digital multimeter for resistance, capacitance, and inductance measurement modes. It models measurement physics with Gaussian noise and uses a 5-level hysteresis-based auto-range engine to select ranges automatically. The simulation achieves under 2% average error in all three modes for the auto-ranging path.

## How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the simulator:

```bash
python simulate.py
```

The script runs 50 test values per mode across all five ranges using an up-and-down sweep, prints range transitions and average error summary in the console, and writes plots to `results/plot_accuracy.png` and `results/plot_autorange.png`.

## Results

| Method | R Error | C Error | L Error |
|---|---:|---:|---:|
| Fixed-range (no auto) | 63.53% | 63.51% | 63.58% |
| Auto-ranging simulation | 0.77% | 0.46% | 0.66% |

## Known Limitations

This is a software-only model and does not include full analog front-end non-idealities such as op-amp offset and bandwidth limits, probe/contact resistance, temperature drift, reference tolerance drift, ADC INL/DNL, or electromagnetic coupling. Real hardware would require calibration and compensation layers in addition to the digital auto-ranging logic.
