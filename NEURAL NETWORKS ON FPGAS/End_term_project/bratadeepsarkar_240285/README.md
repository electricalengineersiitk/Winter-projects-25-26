# Neural Networks on FPGAs — Final Project

**Name:** Bratadeep Sarkar  
**Roll Number:** 240285  
**FPGA Board:** Basys3 (Artix-7 XC7A35T)

---

## Project Description

A 2-layer feedforward neural network implemented in Verilog that classifies
Iris flower inputs into 3 classes (Setosa, Versicolour, Virginica).
Weights are trained in Python (scikit-learn + TensorFlow), quantized to
Q8 fixed-point (16-bit signed), and exported as .mem files loaded by Verilog
using $readmemh. The design is synthesized for the Basys3 FPGA board.

Network architecture: 4 inputs → 8 hidden neurons (ReLU) → 3 output neurons (Softmax/argmax)

---

## How to Run the Python Script

```bash
cd python
pip install scikit-learn tensorflow numpy
python train_and_export.py
```

This generates all `.mem` files in the `weights/` folder.

---

## How to Simulate (Icarus Verilog)

```bash
# Neuron testbench
iverilog -o sim/tb_neuron.vvp sim/tb_neuron.v src/neuron.v
vvp sim/tb_neuron.vvp

# Layer testbench
iverilog -o sim/tb_layer.vvp sim/tb_layer.v src/layer.v src/neuron.v
vvp sim/tb_layer.vvp
```

---

## How to Synthesize in Vivado

1. Open Vivado → Create New RTL Project → select board: Basys3 (xc7a35tcpg236-1)
2. Add sources: all `.v` files from `src/`
3. Add constraints: `vivado/nn_top.xdc`
4. Click Run Synthesis → Run Implementation → Generate Bitstream
5. Open Hardware Manager → Auto Connect → Program Device

---

## Results

| Metric | Value |
|--------|-------|
| LUT Utilisation | <1% (14 LUTs) |
| Worst Negative Slack (WNS) | 6.866 ns |

---

## Verification Results (Simulation vs Implementation)

| Test # | Expected Class | FPGA Output (Sim) | Status |
|--------|----------------|-------------------|--------|
| 1      | 0              | 0                 | **PASS** |
| 2      | 1              | 1                 | **PASS** |
| 3      | 2              | 2                 | **PASS** |
| 4      | 0              | 0                 | **PASS** |
| 5      | 2              | 2                 | **PASS** |

