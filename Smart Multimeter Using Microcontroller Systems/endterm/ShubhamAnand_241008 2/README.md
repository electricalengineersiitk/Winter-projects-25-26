# Smart Multimeter — Simulation & Design

> End-Term Project | Simulation & Design | GitHub Submission

---

## Part 1 — What did we build?

This project simulates an industry-grade digital multimeter capable of measuring **Resistance (R)**, **Capacitance (C)**, and **Inductance (L)** across a 10⁵ dynamic range. The auto-ranging engine continuously monitors each measurement and selects the optimal scale automatically — no user input required. It uses a hysteresis rule (3 consecutive out-of-range readings before switching) to prevent oscillation between adjacent ranges. The simulation achieves an average measurement accuracy of **≤ 0.8%** error across all three modes, well under the 2% target, using Gaussian noise (σ = 0.5% of true value) to model ADC imprecision.

---

## Part 2 — How to set it up

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo/end_term/smart_multimeter
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, NumPy, Matplotlib

---

## Part 3 — How to run the simulation

```bash
python simulate.py
```

The script tests all three measurement modes (R, C, L) using 50 input values each, spread logarithmically across five ranges. For each sample it calls the physics-based measurement formula (with Gaussian noise), passes the result through the auto-ranging engine, and records the true value, measured value, active range, and percentage error. It then prints a full results table to the terminal and saves both required plots to the `results/` folder.

---

## Part 4 — Results

Average percentage error comparison — Auto-ranging simulation vs fixed-range baseline (Range 3 held constant):

| Method                    | R Error | C Error | L Error |
|---------------------------|---------|---------|---------|
| Fixed-range (no auto)     | 2.29%   | 2.19%   | 2.56%   |
| Auto-ranging simulation   | 0.54%   | 0.40%   | 0.80%   |

All three modes pass the ≤ 2% average error requirement. Auto-ranging reduces error by **4–6×** vs the fixed-range baseline by ensuring each measurement is taken in the most appropriate scale.

### Plot 1 — Accuracy vs Input Value
![Accuracy Plot](results/plot_accuracy.png)

### Plot 2 — Auto-Range State Over Time
![Auto-range Plot](results/plot_autorange.png)

---

## Part 5 — Known Limitations

The simulation assumes a noise-free voltage reference and ideal component values, but real hardware would introduce several additional error sources. ADC quantisation noise and op-amp input offset voltage would add a fixed error floor, particularly at the bottom of each range where signal levels are small. Probe lead resistance (typically 0.1–0.5 Ω) would add a systematic offset to resistance measurements. The LC resonance method for inductance assumes the reference capacitor C_ref is perfectly stable, but real capacitors drift with temperature by ±50–200 ppm/°C. Similarly, the RC timing method for capacitance is sensitive to supply voltage accuracy and the threshold detection circuit's propagation delay. Addressing these would require calibration routines, temperature compensation look-up tables, and differential measurement techniques in firmware.

---

