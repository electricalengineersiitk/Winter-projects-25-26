Name:Harikrishna S
Roll number:240424


FPGA Board name : Artix7 AC701 Evaluation Platform (xc7a200tfbg676-2)

  This project aims to implement a neural network on an FPGA which uses trained weights and bias for the Iris dataset. It uses 3 layers- input, hidden layers (8) and output layers (3) with and argmax layer. Each neuron computes summation(W*D)+B. This summation is achieved by using accumulator which adds until last is activated. Next there is the hidden layer which calls 8 neurons and above it there is nn_top which encloses the layer and contains the output layer too. It calls layers once and calls 3 separate neurons for output layer and does argmax to give final output. 

The python file in python folder is used to create mem files containing all the weights, biases which will be automatically created when the python file executed.

Open vivado and create a new project and use the files from my vivado folder. Enter the board name and other details. To start programming the layers, just open it and under sources tab create a .v file and choose design. Also create a tb.v file by choosing sim.

LUT utilization - 1% (479).
Timing slack (WNS) - 0.579ns.

Thank you.
