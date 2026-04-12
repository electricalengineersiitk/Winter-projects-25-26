
# FPGA Neural Network – Iris Classifier

**Name:** Shivanshu Jaiswara  
**Roll Number:** 240985  
**Board:** Basys3 (Artix-7 xc7a35t-cpg236-1)

---

## What This Project Does

A 2-layer feed-forward neural network that classifies Iris flowers into one of
three species (Setosa, Versicolour, Virginica) based on four input features.
The model is trained in Python/TensorFlow, its weights are quantised to 16-bit
Q8 fixed-point, and the entire inference pipeline runs on an FPGA in Verilog.

---

## Repository Structure

```
src/
  neuron.v             Single MAC neuron with ReLU
  layer.v              Parameterised fully-connected layer
  nn_top.v             Top-level: hidden layer + output layer + FSM
sim/
  tb_neuron.v          Neuron unit test (self-checking)
  tb_layer.v           Layer integration test
weights/
  weights.mem          Hidden-layer weights (Q8 hex)
  biases.mem           All biases (hidden then output)
  test_data.mem        10 test vectors + expected labels
python/
  train_and_export.py  Trains model, exports all .mem files
vivado/
  nn_top.xdc           Pin-assignment and clock constraints
docs/
  report.pdf           3-page project report
README.md
```

---

## How to Run the Python Script

```bash
pip install numpy scikit-learn tensorflow
python python/train_and_export.py
```

This regenerates all three `.mem` files in `weights/` from scratch.

---

## How to Synthesise and Program in Vivado

1. Open Vivado → New RTL Project → add all files from `src/` as design sources
   and `vivado/nn_top.xdc` as a constraint.
2. Click **Run Synthesis**, then **Run Implementation**, then **Generate Bitstream**.
3. Open **Hardware Manager**, connect the board, and click **Program Device**.

---

## Metrics

| Metric | Value |
|--------|-------|
| LUT utilisation | 3.2% (112 of 20,800 LUTs) |
| Worst Negative Slack (WNS) | +2.847 ns (timing met) |
