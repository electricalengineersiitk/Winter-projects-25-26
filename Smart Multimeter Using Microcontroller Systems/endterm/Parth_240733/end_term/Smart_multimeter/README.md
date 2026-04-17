# Smart Multimeter — Simulation & Design

> End-Term Project | Smart Multimeter | Simulation & Design | GitHub Submission

---

## Part 1 — What did you build?

This project implements a software simulation of an industry-grade digital multimeter capable of measuring **Resistance (R)**, **Capacitance (C)**, and **Inductance (L)** across a 10⁵ dynamic range. The **auto-ranging engine** automatically selects the correct measurement scale (one of five ranges) without any user input, using a hysteresis mechanism that requires 3 consecutive out-of-range readings before switching — preventing oscillation at range boundaries. The simulation achieves **≈98% accuracy** (≤ 2% average error) across all three measurement modes using real physics formulas with a Gaussian noise model (σ = 0.5% of true value).

---

## Part 2 — How to set it up

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo/end_term/smart_multimeter
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, numpy, matplotlib (see `requirements.txt`).

---

## Part 3 — How to run the simulation

```bash
python simulate.py
```

The script runs **50 log-spaced test values** across the full 10⁵ measurement range for each of the three modes (R, C, L). For each sample it calls the appropriate physics-based measurement function (with added Gaussian noise), passes the result through the auto-ranging engine, and records the active range and percentage error. It then prints a comparative results table (auto-ranging vs fixed-range baseline) and saves two plots to `results/`.

---

## Part 4 — Your results

| **Method**             | **R Error** | **C Error** | **L Error** |
|------------------------|-------------|-------------|-------------|
| Fixed-range (no auto)  | 0.76%       | 0.42%       | 0.89%       |
| Your auto-ranging sim  | 0.71%       | 0.43%       | 0.69%       |

All three modes achieve well under the 2% error target. The auto-ranging engine selects the correct range for every test value. Lower % error = better.

---

## Part 5 — Known limitations

In real hardware, several effects would increase measurement error beyond the simulated 0.5% Gaussian noise floor. ADC quantisation noise and non-linearity would introduce systematic offsets especially at the top and bottom of each range. Op-amp input offset voltages (typically 0.1–5 mV) would corrupt the voltage divider output for low-resistance measurements, and temperature drift (≈ 10 ppm/°C for resistors, more for capacitors) would cause slow baseline shifts over time. Probe contact resistance (0.1–1 Ω) would add a fixed offset that dominates measurements in Range 1. An LC resonance method for inductance would additionally be affected by parasitic capacitance of PCB traces and the series resistance (ESR) of the inductor itself. A real implementation would require calibration coefficients stored in EEPROM and periodic zero-offset corrections to maintain ≈98% accuracy across temperature and aging.

---

## Project Structure

```
end_term/
└── smart_multimeter/
    ├── README.md            ← this file
    ├── requirements.txt     ← numpy, matplotlib
    ├── simulate.py          ← main simulation entry point
    ├── autorange.py         ← auto-ranging engine with hysteresis
    ├── measurement.py       ← R, C, L physics formulas + noise model
    ├── protocol.py          ← OTG serial packet format (CRC-16)
    ├── docs/
    │   └── app_wireframe.png
    └── results/
        ├── plot_accuracy.png
        └── plot_autorange.png
```

---

## Physics Formulas Used

| Mode | Formula | Notes |
|------|----------|-------|
| Resistance | `R = R_ref × V_adc / (V_ref − V_adc)` | Voltage divider; R_ref auto-selected per range |
| Capacitance | `C = τ / R_ref` where `τ = RC` | Time for V to reach 63.2% of V_supply |
| Inductance | `L = 1 / ((2πf)² × C_ref)` | LC resonance; f = measured frequency |

---

## Auto-Ranging Engine

Five ranges per mode. Switching rules:

- **Step UP** if reading > 90% of current range max (hysteresis: 3 consecutive triggers)
- **Step DOWN** if reading < 10% of current range max (hysteresis: 3 consecutive triggers)
- **OL (Overload)** if reading exceeds Range 5 maximum

---

## OTG Serial Protocol

Each measurement is transmitted as a 12-byte packet over USB-OTG serial (115200 baud):

```
[0xAA][MODE][VALUE f32][ERROR u16][RANGE][CRC-16]
  1B    1B      4B        2B        1B      2B
```

See `protocol.py` for full encoding/decoding with CRC-16/CCITT-FALSE verification.
