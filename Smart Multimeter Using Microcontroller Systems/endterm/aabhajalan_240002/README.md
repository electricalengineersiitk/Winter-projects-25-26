# Smart Multimeter — Simulation & Design

## Part 1 — What did you build?

This project is a software simulation of an industry-grade digital multimeter that measures **Resistance (R)**, **Capacitance (C)**, and **Inductance (L)** across a 10⁵ dynamic range using real physics formulas. The auto-ranging engine automatically selects one of five measurement scales based on the incoming reading, using a 3-sample hysteresis rule to prevent range oscillation. The simulation achieves **≤ 0.8% average error** across all three measurement modes with a Gaussian noise model (σ = 0.5 % of true value).

---

## Part 2 — How to set it up

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo/end_term/smart_multimeter
pip install -r requirements.txt
```

> Requires Python 3.10+ with `numpy` and `matplotlib`.

---

## Part 3 — How to run the simulation

```bash
python simulate.py
```

The script generates 50 log-spaced test values across the full 10⁵ range for each of the three measurement modes (R: 100 Ω → 1 MΩ, C: 10 nF → 100 µF, L: 10 µH → 100 mH). Each value is passed through the noise model and the auto-ranging engine. The script prints a full per-sample results table and an average error summary, then saves two plots to `results/`.

---

## Part 4 — Results

| **Method**               | **R Error** | **C Error** | **L Error** |
|--------------------------|:-----------:|:-----------:|:-----------:|
| Fixed-range (no auto)    | 39.65 %     | 39.71 %     | 39.99 %     |
| Your auto-ranging sim    |  0.38 %     |  0.34 %     |  0.80 %     |

Auto-ranging reduces average error by **~100×** compared to a fixed mid-range baseline. All three modes pass the ≤ 2 % target.

---

## Part 5 — Known limitations

In real hardware, ADC quantisation noise would replace the Gaussian model, introducing non-linearity at range boundaries; op-amp offset voltages (typically 1–5 mV) would add a fixed-floor error to low-value resistance measurements; probe contact resistance (~0.1–1 Ω) would corrupt sub-10 Ω readings; the RC method for capacitance assumes an ideal comparator threshold of exactly 63.2 %, which drifts with temperature; and the LC resonance method for inductance is sensitive to the tolerance of the reference capacitor (C_ref), so a 1 % capacitor error propagates directly as a 1 % inductance error across all five ranges.

---

## Project Structure

```
end_term/
└── smart_multimeter/
    ├── README.md
    ├── requirements.txt
    ├── simulate.py        ← main simulation entry point
    ├── autorange.py       ← auto-ranging logic (5 ranges, 3-sample hysteresis)
    ├── measurement.py     ← R, C, L formulas + noise model
    ├── protocol.py        ← OTG serial packet format (bonus)
    └── results/
        ├── plot_accuracy.png
        └── plot_autorange.png
```

---

## Measurement Formulas

| Mode | Formula | Method |
|------|---------|--------|
| Resistance | `R = R_ref × V_adc / (V_ref − V_adc)` | Voltage divider |
| Capacitance | `C = τ / R_ref` where τ is the RC time constant | Charge-discharge timing |
| Inductance | `L = 1 / ((2πf)² × C_ref)` | LC resonance frequency |

---

## Auto-Ranging Rules

| Condition | Action |
|-----------|--------|
| Reading > 90 % of range max (×3 in a row) | Step **UP** |
| Reading < 10 % of range max (×3 in a row) | Step **DOWN** |
| Reading in 10–90 % window | **SETTLED** — report value |
| Reading exceeds all ranges | **OL** (overload) |

The 3-sample hysteresis rule prevents range oscillation at boundary values.
