# Neural Networks on FPGAs

**Name:** Marut Garg

**Roll No:** 240633

**FPGA Board (Target):** Artix-7 AC701 Evaluation Platform

---

## 1. Project Description

This project implements a simple neural network layer on an FPGA using Verilog. Each neuron performs a multiply-accumulate (MAC) operation on the input data and then applies a ReLU activation function. Multiple neurons are grouped together to form a layer, and all of them run in parallel. The weights and biases are generated using a Python script and stored in memory files, which are later used in the hardware design. The aim of this project is to understand how neural network operations can be mapped to hardware and how FPGA parallelism can be used to speed up computation.

---

## 2. Overall Design Structure

The design is divided into three main parts:

* Neuron (basic computation unit)
* Layer (collection of neurons)
* Top module (connects everything together)

The system is clock-driven and uses control signals to manage the flow of data.

---

## 3. Neuron Architecture (`neuron.v`)

The neuron is the core unit of the design.

### Operation:

[
output = ReLU\left(\sum (data_in \times weight) + bias \right)
]

### How it works:

* Takes one input at a time
* Multiplies it with corresponding weight
* Adds the result to an accumulator
* When `last` signal comes:

  * Adds bias
  * Applies ReLU
  * Produces output

### Signals:

* Inputs:

  * `data_in`, `weight_in`, `bias`
  * `clk`, `rst_n`
  * `start`, `last`
* Outputs:

  * `out`
  * `valid`

### Important points:

* Uses sequential accumulation (not fully parallel MAC)
* ReLU is implemented using sign check
* Output becomes valid only after full computation

---

## 4. Layer Architecture (`layer.v`)

The layer module contains multiple neurons.

### Structure:

* 8 neurons are instantiated
* All neurons receive the same input stream
* Each neuron has its own weights and bias

### Outputs:

```id="0orenn"
out0, out1, ..., out7
```

### Key idea:

* All neurons run in parallel
* This shows how FPGA can do multiple computations at the same time

---

## 5. Top-Level Module (`nn_top.v`)

This module connects the layer to external inputs.

### Inputs:

* `clk` → clock
* `rst_n` → reset
* `start` → start signal
* `data_in` → input data
* `last` → end of input

### Outputs:

* `out0` to `out7`
* `valid`

### Role:

* Sends inputs to the layer
* Collects outputs from all neurons
* Acts as the main interface

---

## 6. Data Representation

* All values are stored in **16-bit signed format**
* Fixed-point arithmetic is used
* Floating-point is avoided because:

  * It uses more hardware
  * It is slower on FPGA

This makes the design efficient and simple.

---

## 7. Design Choices

Some design decisions were made to keep the hardware simple and efficient:

* Fixed-point arithmetic was used instead of floating point to reduce hardware cost
* Sequential accumulation was used inside each neuron to save resources
* Parallel neurons were used in the layer to improve throughput
* ReLU was chosen as activation because it is easy to implement in hardware

These choices helped balance performance and resource usage.

---

## 8. Memory and Data Flow

The design uses memory files for weights and inputs.

### Files:

* `weights.mem` → stores weights
* `biases.mem` → stores biases
* `test_data.mem` → input values

### Flow:

1. Data comes one by one
2. Each neuron processes it
3. Final output is generated after accumulation

---

## 9. File Description

### src/

* `neuron.v` → neuron logic
* `layer.v` → 8 parallel neurons
* `nn_top.v` → top-level integration

---

### sim/

* `tb_neuron.v` → tests neuron
* `tb_layer.v` → tests layer

---

### weights/

* All memory files used in simulation

---

### python/

* `train_and_export.py`

  * Generates weights and biases
  * Converts them to `.mem` format

---

### vivado/

* Complete Vivado project
* Includes synthesis and implementation

---

### docs/

* `report.pdf` → final report

---

## 10. How to Run Python Script

Run:

```bash id="cmd2"
python python/train_and_export.py
```

This generates all memory files required for simulation.

---

## 11. Simulation

Simulation was done using Vivado.

### Checked:

* Correct MAC operation
* ReLU working correctly
* Valid signal timing

Both neuron and layer were tested using testbenches.
Waveforms were also observed in Vivado to verify correct timing and signal behavior.

---

## 12. Vivado Flow (Synthesis and Programming)

Steps followed:

1. Open project:

   ```
   vivado/nn_project/nn_project.xpr
   ```

2. Run:

   * Synthesis
   * Implementation

3. Generate bitstream:

   ```
   write_bitstream -force nn_top.bit
   ```

The design completes all steps successfully.
DRC issues related to I/O constraints were handled appropriately before bitstream generation.

---

## 13. Results

### Functional Results

All outputs matched expected values during simulation.

### Resource Usage

* Slice LUTs: 194
* Flip-Flops: 385
* DSPs: 0

### Timing

* Worst Negative Slack (WNS): INF

This means timing constraints are fully satisfied.

---

## 14. Hardware Implementation

Bitstream (`nn_top.bit`) was generated successfully.

Actual FPGA testing was not done because a board was not available.
However, the design is ready to be programmed on the target FPGA.

---

## 15. Challenges Faced

* Handling file paths in Vivado
* Loading memory files correctly
* Understanding fixed-point behavior
* Debugging simulation errors

---

## 16. What I Learned

* How neural networks can be implemented in hardware
* Importance of parallel computation
* How FPGA design flow works (simulation → synthesis → implementation)
* Debugging using testbenches

---

## 17. Limitations

* Only a single layer is implemented
* No support for floating-point precision
* Sequential accumulation makes each neuron slightly slower
* No hardware testing was performed

These can be improved in future versions.

---

## 18. Future Improvements

* Add multiple layers to build a deeper network
* Use pipelining to improve speed
* Explore DSP blocks for faster multiplication
* Support more advanced activation functions

---

## 19. Conclusion

This project shows how a basic neural network layer can be implemented on FPGA using Verilog. It gives a clear idea of how hardware can be used to speed up neural network computations. The design can be extended further by adding more layers or improving performance using pipelining.

---

## 20. Project Structure

```id="finstr"
marutgarg_240633/
 ├── src/
 ├── sim/
 ├── weights/
 ├── python/
 ├── vivado/
 ├── docs/
 └── README.md
```

---
