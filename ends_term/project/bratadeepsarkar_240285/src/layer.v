`timescale 1ns / 1ps
//
// Hidden Layer — 8 Neurons in Parallel
// This module instantiates 8 neurons in parallel for the hidden layer.
// It passes weight/bias filenames so each neuron can load its own memory.
//
// Interface:
//   Pulse start=1 for 1 cycle. Then feed data_in for 4 consecutive cycles
//   (input_idx 0, 1, 2, 3). Assert last=1 on the 4th cycle (input_idx=3).
//   When valid_layer goes high, all 8 neuron outputs are ready in out_vector.
//
module layer #(
    parameter NUM_NEURONS  = 8,
    parameter NUM_INPUTS   = 4,
    parameter WEIGHT_FILE  = "weights_hidden.mem",
    parameter BIAS_FILE    = "biases_hidden.mem"
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,       // pulse to trigger (same as neuron start)
    input  wire [15:0] data_in,     // one input value per cycle
    input  wire [1:0]  input_idx,   // which input (0..3) is being fed
    input  wire        last,        // pulse on final input (input_idx == 3)
    output wire [127:0] out_vector, // 8 neurons × 16 bits = 128 bits
    output wire        valid_layer  // high for 1 cycle when all outputs ready
);

    // Memory for weights and biases
    reg [15:0] weights_mem [0:NUM_NEURONS*NUM_INPUTS-1];  // 32 values
    reg [15:0] biases_mem  [0:NUM_NEURONS-1];             // 8 values

    initial begin
        $readmemh(WEIGHT_FILE, weights_mem);
        $readmemh(BIAS_FILE, biases_mem);
    end

    // Generate the 8 neurons.
    // Each neuron gets a slice of the shared data_in bus.
    wire [7:0] neuron_valid;  // valid signal from each neuron

    genvar i;
    generate
        for (i = 0; i < NUM_NEURONS; i = i + 1) begin : neuron_inst
            neuron n (
                .clk       (clk),
                .rst_n     (rst_n),
                .start     (start),
                .data_in   (data_in),
                .weight_in (weights_mem[i * NUM_INPUTS + input_idx]),
                .bias      (biases_mem[i]),
                .last      (last),
                .out       (out_vector[i*16 +: 16]),
                .valid     (neuron_valid[i])
            );
        end
    endgenerate

    // All neurons receive the same start/last signals and run in lockstep,
    // so they all finish on the same cycle. Neuron 0's valid is sufficient.
    assign valid_layer = neuron_valid[0];

endmodule
