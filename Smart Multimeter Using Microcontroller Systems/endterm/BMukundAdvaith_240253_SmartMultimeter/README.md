# Smart Multimeter using Microcontroller Systems

Electrical Engineering Association Winter Project
Simulation and Design
---

## Brief Overview of the Project

This project simulates an industry-grade digital multimeter capable of measuring Resistance (primary focus), Capacitance, and Inductance across a 10^5 dynamic range.  
The auto-ranging engine automatically selects the correct measurement scale by monitoring each reading against 10% and 90% thresholds, with a 3-sample hysteresis rule to prevent oscillation near range boundaries.  
The simulation achieves an average measurement accuracy of approximately 98% or better (under 2% error) across all
three modes using a Gaussian noise model with σ = 0.5% of the true value.

---

## How to set it up

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo/end_term/smart_multimeter
pip install -r requirements.txt
```

Requirements: Python 3.8 or higher, numpy, matplotlib.
No hardware is needed — this is a pure software simulation.

---

## How to run the simulation

```bash
python simulate.py
```

The script runs 50 test values spread log-evenly across all 5 ranges for each measurement mode (R, C, L). 
(Resistances are measured using the Voltage Divider Method, Capacitances are measured using the Time Constant Method and Inductnaces are measured using Resonant Frequency Method.)  
For every test value it adds Gaussian noise, passes the reading through the auto-ranging engine, and records the true value, measured value, active range, and percentage error. It then prints a summary table comparing auto-ranging accuracy against a fixed-range baseline, and saves two labelled plots to the results/ folder.

---

## Results

Results from a 50-sample sweep across all five ranges (seed = 42):

| Method                          | R Error | C Error | L Error |
|---------------------------------|---------|---------|---------|
| Fixed-range (no auto)           | 40.51%  | 39.87%  | 40.44%  |
| Auto-ranging simulation         |  1.32%  |  0.48%  |  0.83%  |

Auto-ranging consistently keeps error below the 2% target across the full 10^5 value range. Fixed-range performance degrades sharply at the low and high ends because a single range cannot cover the full scale accurately.

Accuracy plot:  results/plot_accuracy.png  
Range state plot: results/plot_autorange.png

---

## Known limitations

In a real hardware implementation several additional error sources would need to be accounted for.   
--> ADC quantisation noise is modelled here as Gaussian but in practice it includes non-linearity and integral non-linearity errors that vary across the conversion range.  
--> Op-amp offset voltage in the signal conditioning stage would introduce a systematic DC bias not captured by the symmetric noise model used here.  
--> Probe and PCB trace resistance (typically 0.1–1 Ω) would add a fixed offset to low-range resistance measurements, making Range 1 less accurate than simulated.  
--> Temperature drift in the reference resistor and capacitor would cause the calibration constants R_REF and C_REF to shift over time, requiring periodic recalibration.  
--> Finally, the LC resonance method for inductance is sensitive to parasitic capacitance in the PCB layout, which would shift themeasured resonant frequency and introduce a systematic error not present in the simulation.  

---

## File Structure

```
endterm/
└── BMukundAdvaith_240253_SmartMultimeter/
    ├── README.md
    ├── requirements.txt
    ├── simulate.py             — main simulation entry point
    ├── autorange.py            — auto-ranging logic with hysteresis
    ├── measurement.py          — R, C, L formulas + noise model
    ├── protocol.py             — OTG serial packet format
    ├── app_wireframe.png       — app wireframe 
    └── results/
        ├── plot_accuracy.png
        └── plot_autorange.png
```

---

## Measurement Formulas

| Mode        | Formula                                      | Method                  |
|-------------|----------------------------------------------|-------------------------|
| Resistance  | R = R_ref × V_adc / (V_ref − V_adc)         | Voltage divider         |
| Capacitance | C = τ / R_ref  (τ = RC time constant)        | Charge-discharge timing |
| Inductance  | L = 1 / ((2πf)² × C_ref)                    | LC resonance frequency  |

Noise model: Gaussian, σ = 0.5% of true value applied at the ADC reading stage.

---

## Auto-Ranging Engine

| Step | Condition                          | Action                        |
|------|------------------------------------|-------------------------------|
| 1    | reading > 90% of range max         | Increment up_count            |
| 2    | up_count reaches 3 (consecutive)   | Switch to next higher range   |
| 3    | reading < 10% of range max         | Increment down_count          |
| 4    | down_count reaches 3 (consecutive) | Switch to next lower range    |
| 5    | reading in 10–90% window           | Report SETTLED, reset counts  |
| 6    | reading > max of Range 5           | Report OL (overload)          |

---

## OTG Serial Packet Format

12-byte packet structure:

```
Byte  0     : Start byte   (0xAA)
Byte  1     : Mode         (0x01=R  0x02=C  0x03=L)
Byte  2     : Range index  (0x01 to 0x05)
Bytes 3–6   : Value        (32-bit float, big-endian)
Bytes 7–8   : Error × 100  (16-bit unsigned int)
Byte  9     : Status       (0x00=SETTLED  0x01=STEP_UP  0x02=STEP_DOWN  0x03=OL)
Byte  10    : Checksum     (XOR of bytes 1–9)
Byte  11    : Stop byte    (0xFF)
```

Baud rate: 115200. Checksum allows the receiver to detect corrupted packets.
