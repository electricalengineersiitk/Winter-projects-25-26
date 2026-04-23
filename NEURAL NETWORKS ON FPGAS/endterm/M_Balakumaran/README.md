# FPGA Neural Network Final Project

**Name:** M Balakuamran
**Roll Number:** 240602
**FPGA Board:** Artix™ 7 FPGA AC701

## Project Description
This project implements a 2-layer neural network on an FPGA to classify data from the Iris dataset into three flower classes based on four input features. The hardware design is written in Verilog and utilizes a custom sequential Multiply-Accumulate (MAC) neuron module with a ReLU activation function. The model is first trained in Python using TensorFlow, and its floating-point weights and biases are quantized into 16-bit Q8 fixed-point integers. These integers are loaded into the FPGA's memory via `.mem` files, enabling efficient and accurate hardware-level inference.

## How to Run the Python Script
To train the model and generate the required `.mem` files (weights, biases, and test data), run the following command from the root directory:
```bash
python python/train.py
```

## How to Open and Program in Vivado
1. Open Vivado and create a new RTL project targeting your specific FPGA board. Add all Verilog files (`.v`) and your board's clock constraints file (`.xdc`).
2. Click **Run Synthesis**, then **Run Implementation**, and finally **Generate Bitstream**.
3. Once the bitstream is ready, open the **Hardware Manager**, auto-connect to your board via USB, select your `.bit` file, and click **Program Device** to flash the FPGA.

## Hardware Metrics
* **LUT Utilisation:** 0.99 %
* **Timing Slack (Worst Negative Slack):** 0.463 ns
