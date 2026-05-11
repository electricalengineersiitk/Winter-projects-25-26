# Neural Network on FPGA

## Name:Rahul Beldar  

## Roll No:240828  

## Objective
To implement a simple neural network on FPGA using Verilog.

## Description
This project implements a basic neural network architecture using hardware description language (Verilog).  
Each neuron performs multiplication and accumulation (MAC) followed by an activation function (ReLU).  

Weights and biases are generated using Python and then used in the FPGA design.

## Components
- Neuron module (MAC + ReLU)
- Layer module (multiple neurons)
- Top module (complete network)
- Testbench for simulation
- Python script for weight generation

## Working
Input data is multiplied with weights and accumulated.  
Bias is added at the final stage.  
ReLU activation is applied to produce output.

## Conclusion
The neural network was successfully implemented and simulated on FPGA using Verilog.

## Tools Used
- Verilog HDL  
- Python  
- Simulation tools (ModelSim/others)
