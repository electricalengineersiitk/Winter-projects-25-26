# Neural Network on FPGA

**Name:** [Ankit Kholwad]
**Roll No:** [240136]
**Board:** Basys3

## Project Description

This project implements a 2-layer neural network in Verilog that classifies Iris flower samples into three species (Setosa, Versicolor, Virginica). The network takes 4 scaled input features, passes them through a hidden layer of 8 neurons with ReLU activation, and feeds the results into an output layer of 3 neurons. An argmax operation determines the predicted class. Weights were trained in Python using scikit-learn, quantised to Q8 fixed-point (16-bit), and stored in `.mem` files loaded onto the FPGA at startup.

## To Run the Python Script
```bash
cd python
pip install scikit-learn numpy
python train_and_export.py
```

This generates all `.mem` files inside the `weights/` folder.

## To Open and Program in Vivado
1. Open Vivado → New RTL Project → add all `.v` files from `src/` and `vivado/constraints.xdc`.
2. Click **Run Synthesis** → **Run Implementation** → **Generate Bitstream**.
3. Open **Hardware Manager**, connect the Basys3 board via USB, click **Auto Connect** → **Program Device**.

## Synthesis Metrics
| LUT Utilisation | 11.14% |
| Timing Slack (WNS) | 7.636 ns |

WNS is positive — the design meets the 100 MHz timing constraint.

## Folder Structure

```
src/
  neuron.v          Single neuron: MAC + ReLU
  neuron_linear.v   Output neuron: MAC only (no ReLU)
  layer.v           Hidden layer: 8 neurons in parallel
  output_layer.v    Output layer: 3 neurons + argmax
  nn_top.v          Top-level: connects hidden + output layer

sim/
  tb_neuron.v       Neuron testbench
  tb_layer.v        Layer testbench
  tb_nn_top.v       Top-level testbench

weights/
  weights.mem       Hidden layer weights (8x4 = 32 values)
  w1_bias.mem       Hidden layer biases (8 values)
  w2_weights.mem    Output layer weights (3x8 = 24 values)
  w2_bias.mem       Output layer biases (3 values)
  test_data.mem     10 test samples with expected labels

python/
  train_and_export.py   Training and weight export script

vivado/             Vivado project files (.xpr)
docs/               Project report (PDF)
```
