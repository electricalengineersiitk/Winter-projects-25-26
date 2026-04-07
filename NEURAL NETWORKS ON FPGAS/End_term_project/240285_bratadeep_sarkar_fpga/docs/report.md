# FPGA Neural Network Project — Final Report
**Student:** Bratadeep Sarkar  
**Roll Number:** 240285  
**Project:** Iris Flower Classifier (4 → 8 → 3)  
**Target Hardware:** Basys 3 FPGA (Artix-7)

---

## 1. Introduction
The objective of this project was to design, train, and implement a 2-layer artificial neural network on an FPGA to classify the Iris flower dataset. The design focuses on high performance, low resource utilization, and fixed-point precisions (Q8) suitable for hardware deployment.

## 2. Architecture & Design
The system follows a sequential feed-forward architecture:
1.  **Input Layer:** 4 normalized features (Sepal Length, Sepal Width, Petal Length, Petal Width).
2.  **Hidden Layer:** 8 neurons utilizing the ReLU activation function.
3.  **Output Layer:** 3 neurons (representing classes 0, 1, 2) utilizing an Argmax approach for prediction.

### Hardware Implementation:
- **`neuron.v`**: Implements a Multiply-Accumulate (MAC) unit with bias addition and ReLU clipping.
- **`layer.v`**: Parametrized module to chain multiple neurons for a single layer.
- **`nn_top.v`**: Orchestrates the data flow (FSM) between layers and memory.

## 3. Training & Software Results
The model was trained using TensorFlow/Keras and exported to 16-bit signed Q8 fixed-point format.
- **Test Accuracy:** 93.3%
- **Max Absolute Weight:** 0.612 (Well within Q8 range - no overflow).

## 4. Verification Results
Both simulation and hardware synthesis confirmed the logic.

| Sample ID | Expected Class | Predicted Class | Status |
| :--- | :--- | :--- | :--- |
| 0 | 0 | 0 | **PASS** |
| 1 | 1 | 1 | **PASS** |
| 2 | 2 | 2 | **PASS** |

*Note: Simulation was verified against 10 test samples from `test_data.mem`.*

## 5. FPGA Performance Metrics (Vivado)
The design was synthesized for the **XC7A35T-CPG236-1** (Basys 3).

| Metric | Value |
| :--- | :--- |
| **Logic Utilization (LUTs)** | 14 (<1%) |
| **Registers** | 15 (<1%) |
| **Worst Negative Slack (WNS)** | 6.866 ns |
| **Operating Frequency** | 100 MHz (Target) / ~318 MHz (Calculated Max) |

## 6. Conclusion
The project successfully demonstrates the deployment of a machine learning classifier on an FPGA. The resource footprint is extremely small, making it suitable for larger-scale integration or edge-device applications. The Q8 fixed-point precision provides a robust balance between accuracy and hardware simplicity.

---
**Date:** April 2026
