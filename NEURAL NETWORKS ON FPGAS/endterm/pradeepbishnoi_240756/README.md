# NAME
Pradeep Bishnoi

# ROLL NO
240756

# FPGA BOARD
Zynq-7000 (xc7z020clg400-1)

# PROJECT OVERVIEW 
This project focuses on designing and implementing a synthesizable 2-layer neural network in Verilog for FPGA deployment. The project implements a fully functional feed-forward neural network on an FPGA to classify the Iris flower dataset in hardware. The network consists of two layers, first one is a hidden layer with 8 neurons and then an output layer with 3 neurons (one per class), built entirely in Verilog using fixed-point Q8 arithmetic. The model is first trained in Python using TensorFlow, after which the weights and biases are quantised to 16-bit integers and exported as `.mem` files. These are then loaded directly into the FPGA design, where inference is performed cycle-accurately at 100 MHz with a latency of just a few clock cycles per prediction.

# RUNNING PYTHON SCRIPT
Making sure that Python 3, TensorFlow, and scikit-learn are all installed, then running:

```bash
python python/train_and_export.py
```

This trains the model on the Iris dataset, quantises the weights to Q8 fixed-point format, and saves `weights.mem`, `biases.mem`, and `test_data.mem` into the `weights/` folder automatically.

# PROGRAMMING IN VIVADO
Open Vivado 2025.2 and load the project by going to *File → Open Project* and selecting `vivado/project_vivado.xpr`. Once the project is open, click *Run Synthesis*, then *Run Implementation*, and finally *Generate Bitstream* from the Flow Navigator while all `.mem` files are already included as project sources. To program the board, connect it via USB, open the *Hardware Manager*, click *Auto Connect*, and select *Program Device* with the generated `nn_top.bit` bitstream file.

# SYNTHESIS RESULTS
| Metric               | Value                 |
|----------------------|-----------------------|
| LUT Utilisation      | 896 / 53200 (1.68%)   |
| Worst Negative Slack | +2.768 ns             |
