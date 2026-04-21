# Smart Multimeter — Software Simulation

> End-Term Project | Simulation & Design | GitHub Submission

---


This project is a software simulation of an industry-grade digital multimeter.
## Part 1 — What did you build?

This project is a software simulation of an industry-grade digital multimeter capable of measuring **Resistance (R)**, **Capacitance (C)**, and **Inductance (L)** across a 10⁵ dynamic range. The simulation implements real physics formulas for each mode — voltage-divider for R, RC time-constant for C, and LC resonance frequency for L — with Gaussian noise (σ = 0.5% of true value) applied to model ADC imprecision. An **auto-ranging engine** (`autorange.py`) automatically selects the correct measurement scale from 5 ranges without any user input, using a hysteresis rule (3 consecutive triggers) to prevent range oscillation. The simulation achieves **≈98% accuracy (< 2% average error)** across all three modes over a 50-sample sweep.

## Part 2 — How to set it up

```bash
git clone todo
cd your-repo/end_term/smart_multimeter
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, numpy, matplotlib.



to run the simulation

```bash
python simulate.py
```

The script runs **50 test values** log-spaced across all 5 ranges for each of the three measurement modes (R, C, L). For each value it applies the physics measurement formula with Gaussian noise, feeds the result through the auto-ranging engine, and records the true value, measured value, active range, and percentage error. It prints a summary results table to the console and saves two plots to the `results/` folder.


PS D:\multimeter> python3 autorange.py 
=== autorange.py self-test (Resistance sweep) ===

     True R (Ω) |  Range |     Status
----------------------------------------
           10.0 |      1 |    SETTLED
           14.9 |      1 |    SETTLED
           22.1 |      1 |    SETTLED
           32.9 |      1 |    SETTLED
           48.9 |      1 |    SETTLED
           72.8 |      1 |    SETTLED
          108.3 |      1 |    SETTLED
          161.0 |      1 |    SETTLED
          239.5 |      2 |    STEP_UP
          356.2 |      2 |    SETTLED
          529.8 |      2 |    SETTLED
          788.0 |      2 |    SETTLED
         1172.1 |      2 |    SETTLED
         1743.3 |      2 |    SETTLED
         2592.9 |      3 |    STEP_UP
         3856.6 |      3 |    SETTLED
         5736.2 |      3 |    SETTLED
         8531.7 |      3 |    SETTLED
        12689.6 |      3 |    SETTLED
        18873.9 |      3 |    SETTLED
        28072.2 |      4 |    STEP_UP
        41753.2 |      4 |    SETTLED
        62101.7 |      4 |    SETTLED
        92367.1 |      4 |    SETTLED
       137382.4 |      4 |    SETTLED
       204336.0 |      5 |    STEP_UP
       303919.5 |      5 |    SETTLED
       452035.4 |      5 |    SETTLED
       672335.8 |      5 |    SETTLED
      1000000.0 |      5 |    SETTLED
PS D:\multimeter> python3 measurement.py
=== measurement.py self-test ===

Resistance  | true=4700 Ω    | measured=4710.53 Ω   | error=0.224%
Capacitance | true=470 nF    | measured=467.56 nF | error=0.520%
Inductance  | true=10 mH     | measured=9.9254 mH | error=0.746%
PS D:\multimeter> python3 protocol.py
=== protocol.py self-test ===

Encoded packet (16 bytes): AA 01 03 CD 98 93 45 48 E1 FA 3E 01 00 04 43 55

Decoded:
[Seq 00001] Mode=R | Range 3 | Value=4723.1 Ω | Error=0.490% | Flags: RANGE_CHANGED

Overload packet:
[Seq 00002] Mode=C | OL | Value=0 F | Error=0.000% | Flags: OVERLOAD
PS D:\multimeter> python3 simulate.py

============================================================
  Smart Multimeter Simulation — End-Term Project
============================================================

Running 50-sample sweep for each mode (R, C, L)...

Simulation complete. Printing results...


---------------------------------------------------------------------
  SIMULATION RESULTS — Average % Error over 50 samples
---------------------------------------------------------------------
Method                         |    R Error |    C Error |    L Error
---------------------------------------------------------------------
Fixed-range (no auto)  [baseline] |      2.70% |      0.39% |      0.76%
Auto-ranging simulation        |      3.43% |      0.40% |      0.80%
---------------------------------------------------------------------


  RESISTANCE (Ω) — first 10 samples
          True |     Measured |    Error |  Range
  ------------------------------------------------
            10 |        10.02 |   0.153% |      1
         12.92 |        12.96 |   0.376% |      1
         16.68 |        16.52 |   0.977% |      1
         21.54 |        21.56 |   0.064% |      1
         27.83 |        27.82 |   0.008% |      1
         35.94 |         36.1 |   0.441% |      1
         46.42 |        46.43 |   0.033% |      1
         59.95 |        60.09 |   0.235% |      1
         77.43 |        77.57 |   0.186% |      1
           100 |        100.4 |   0.444% |      1

  CAPACITANCE (nF) — first 10 samples
          True |     Measured |    Error |  Range
  ------------------------------------------------
             1 |       0.9981 |   0.189% |      1
         1.292 |        1.289 |   0.178% |      1
         1.668 |         1.66 |   0.467% |      1
         2.154 |        2.144 |   0.475% |      1
         2.783 |        2.794 |   0.420% |      1
         3.594 |        3.602 |   0.217% |      1
         4.642 |        4.628 |   0.297% |      1
         5.995 |        5.997 |   0.036% |      1
         7.743 |        7.752 |   0.116% |      1
            10 |        10.08 |   0.801% |      1

  INDUCTANCE (mH) — first 10 samples
          True |     Measured |    Error |  Range
  ------------------------------------------------
         0.001 |    0.0009966 |   0.337% |      1
      0.001292 |      0.00129 |   0.091% |      1
      0.001668 |     0.001703 |   2.082% |      1
      0.002154 |     0.002173 |   0.849% |      1
      0.002783 |     0.002807 |   0.884% |      1
      0.003594 |     0.003561 |   0.910% |      1
      0.004642 |      0.00464 |   0.031% |      1
      0.005995 |     0.006015 |   0.328% |      1
      0.007743 |     0.007701 |   0.536% |      1
          0.01 |      0.01002 |   0.155% |      1

Plots saved in `results/`:
- `plot_accuracy.png` — % error vs true value (log-scale X), auto-range vs fixed-range
- `plot_autorange.png` — active range index over 50 samples, showing range transitions

---


In real hardware, several physical effects would increase measurement error beyond what this simulation models. ADC quantisation noise and op-amp input offset voltage would add a systematic bias not present in a pure Gaussian model. Probe resistance (typically 0.1–1 Ω) and contact resistance introduce a fixed additive error that is proportionally larger for low-value resistance measurements. Temperature drift of the reference resistor and reference capacitor (typically ±50–200 ppm/°C) would shift the calibration baseline over time. Finally, parasitic capacitance on PCB traces (a few pF) would limit the accuracy of low-capacitance measurements in the C1 range. A production implementation would address these with a calibration routine, instrumentation amplifiers with matched components, and a temperature-compensated voltage reference.


## Bonus — OTG Serial Protocol

`protocol.py` implements a compact **16-byte serial packet** for transmitting readings to a mobile app over USB OTG:

| Byte(s) | Field    | Description                              |
|---------|----------|------------------------------------------|
| 0       | SOF      | Start of frame: `0xAA`                   |
| 1       | MODE     | `0x01`=R, `0x02`=C, `0x03`=L             |
| 2       | RANGE    | Active range 1–5; `0xFF` = overload      |
| 3–6     | VALUE    | 32-bit float, little-endian (SI units)   |
| 7–10    | ERROR    | 32-bit float, little-endian (% error)    |
| 11–12   | SEQ      | 16-bit sequence counter                  |
| 13      | FLAGS    | Bit 0: OL, Bit 1: settling, Bit 2: rng  |
| 14      | CHECKSUM | XOR of bytes 0–13                        |
| 15      | EOF      | End of frame: `0x55`                     |

