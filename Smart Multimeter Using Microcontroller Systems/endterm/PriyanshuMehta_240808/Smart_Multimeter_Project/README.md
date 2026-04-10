# Smart Multimeter Simulation Project

## Part 1 — What did I build?
I developed a **software simulation of a digital multimeter** that can measure:
* Resistance (R)
* Capacitance (C)
* Inductance (L)

The system includes an **auto-ranging feature** that automatically selects the best measurement range, improving accuracy and ease of use.

---

## Part 2 — How to set it up

```bash
git clone https://github.com/mehtapriyanshu172006-droid/Multimeter_Project_240808.git
cd Multimeter_Project_240808/end_term/smart_multimeter
pip install -r requirements.txt
```

---

## Part 3 — How to run the simulation

```bash
python simulate.py
```

This runs a **50-sample test** across R, C, and L values and stores plots in the `results/` folder.

---

## Part 4 — Results

This table compares the performance of a **fixed-range multimeter** vs the **auto-ranging system**:

| Method                  | R Error | C Error | L Error |
| ----------------------- | ------- | ------- | ------- |
| Fixed-range (no auto)   | 2.36%   | 8.19%   | 9.07%   |
| Auto-ranging (proposed) | 0.40%   | 0.34%   | 0.49%   |

> These values are average errors obtained from simulation runs.

### Resistance (R)
![R Accuracy Plot](results/plot_accuracy_R.png)
![R Auto-Range Plot](results/plot_autorange_R.png)

### Capacitance (C)
![C Accuracy Plot](results/plot_accuracy_C.png)
![C Auto-Range Plot](results/plot_autorange_C.png)

### Inductance (L)
![L Accuracy Plot](results/plot_accuracy_L.png)
![L Auto-Range Plot](results/plot_autorange_L.png)

---

## Part 5 — Known Limitations

In real hardware systems, additional factors would affect accuracy:
* ADC quantization noise
* Probe resistance
* Temperature variations

This simulation uses a **Gaussian noise model** to approximate real-world behavior, but actual circuits may introduce more complex effects.

---

## Project Structure
smart_multimeter/
│
├── simulate.py        # runs full simulation
├── autorange.py       # auto range logic
├── measurement.py     # R, C, L calculations
├── protocol.py        # JSON output formatting
├── draw_ui.py         # mobile UI simulation
├── results/           # output plots
└── docs/              # UI images

---

## Summary

The results clearly show that:
* **Auto-ranging significantly reduces measurement error**
* Fixed-range systems perform poorly when the measured value is not close to the selected range

This project demonstrates how intelligent range selection improves both **accuracy and usability** in measurement systems.
