# Smart Digital Multimeter — Python Simulation

## Part 1: What Was Built

This project is a Python simulation of an industry-grade digital multimeter capable of measuring **Resistance** (Ω), **Capacitance** (F), and **Inductance** (H) across a 10⁵ dynamic range. At its core is an **auto-ranging engine** with 5 switchable ranges per mode, a 3-reading hysteresis guard to prevent oscillation, and overload detection. All measurements are simulated with Gaussian ADC noise (σ = 0.5 % of true value), yielding a consistently **average error ≤ 2 %** across all modes. A companion **desktop GUI app** (`app.py`) replays the simulation as if readings were arriving over a USB-OTG serial link, displaying the live value, active range, and formatted serial packet on a DMM-style interface.

---

## Part 2: Setup Commands

```bash
# Clone the repository
git clone https://github.com/<your-username>/smart-multimeter.git

# Navigate into the project
cd smart-multimeter/end_term/smart_multimeter

# Install dependencies
pip install -r requirements.txt
```

---

## Part 3: Running the Simulation

```bash
python simulate.py
```

The script generates **50 test values** distributed log-uniformly across all 5 measurement ranges for each of the three modes. It prints a console results table showing average & maximum errors and settled-sample counts, then saves two plots (`plot_accuracy.png` and `plot_autorange.png`) to the `results/` folder. Sample OTG serial packets are also printed to the terminal.

---

## Part 4: Results Comparison

| Mode        | Avg Error — Auto-Range | Avg Error — Fixed Range (R3) | Improvement |
|-------------|------------------------|------------------------------|-------------|
| Resistance  | **0.38 %**             | 30.80 %                      | ~80×        |
| Capacitance | **0.34 %**             | 30.85 %                      | ~90×        |
| Inductance  | **0.80 %**             | 31.13 %                      | ~39×        |

*All values from a 50-sample sweep (seed=42). Fixed-range baseline uses Range 3 (mid-range); values outside that range saturate, causing large errors. Auto-range engine settles 48/50 samples per mode.*

---

## Part 5: Known Hardware Limitations

Real-world multimeter implementations face several physical limitations not captured in this simulation. **ADC noise** (quantisation error + thermal noise on the reference voltage) limits precision below a few LSBs and can vary with supply-rail ripple. **Op-amp input offset voltage** introduces a systematic DC error in the front-end conditioning stage that is especially significant at low signal levels. **Probe and contact resistance** (typically 0.1–2 Ω for cheap test leads) directly adds to low-resistance measurements, making sub-ohm accuracy extremely challenging without a 4-wire Kelvin connection. Finally, **temperature drift** in the reference resistor, timing oscillator, and ADC voltage reference causes gain errors that vary with ambient temperature at typically 50–100 ppm/°C for low-cost components, necessitating periodic calibration or on-chip temperature compensation for precision instruments.

---

## Part 6: Desktop OTG App (`app.py`)

A Tkinter-based desktop GUI simulates the **mobile/host-side of the OTG serial link**, displaying live readings as they arrive from the auto-ranging engine.

### How to run

```bash
python app.py
```

> tkinter is part of the Python standard library — no extra install required.

### Features

| Feature | Description |
|---|---|
| **DMM-style display** | Large readout with mode-coloured value (green=Resistance, cyan=Capacitance, magenta=Inductance) |
| **Mode selector** | Switch between Ω / F / H modes; replays 50-sample sweep for the selected mode |
| **Range indicator bar** | Ranges 1–5 highlighted in real time as the auto-ranger steps |
| **Status indicator** | `SETTLED` (bright) / `RANGING …` (orange) mirrors the hysteresis engine state |
| **OTG packet viewer** | Displays the raw decoded serial packet string for each reading |
| **Start / Stop controls** | 9-second animated replay at ~180 ms per sample |

### What it demonstrates

The app shows how a host device (phone or PC) connected over USB-OTG would consume the formatted packets from `protocol.py`: decoding the mode, active range, measured value, and error percentage and presenting them on a readable display — without any knowledge of the underlying physics or ranging logic.

