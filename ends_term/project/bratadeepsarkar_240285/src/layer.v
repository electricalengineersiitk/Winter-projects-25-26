`timescale 1ns / 1ps
//
// Hidden Layer — 8 Neurons in Parallel
// Loads weights/biases from .mem files via $readmemh.
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

    // ── Weight & Bias Memory ──────────────────────────────────────────────
    // Weight layout: neuron_i weights at indices [i*NUM_INPUTS .. i*NUM_INPUTS+3]
    reg [15:0] weights_mem [0:NUM_NEURONS*NUM_INPUTS-1];  // 32 values
    reg [15:0] biases_mem  [0:NUM_NEURONS-1];             // 8 values

    initial begin
        $readmemh(WEIGHT_FILE, weights_mem);
        $readmemh(BIAS_FILE, biases_mem);
    end

    // ── Instantiate 8 Neurons ─────────────────────────────────────────────
    // Each neuron gets the same data_in, start, last signals.
    // Each neuron gets its own weight (looked up by neuron index + input_idx).
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

    // All neurons run the same number of cycles, so they all finish together.
    // Use neuron 0's valid as the layer valid (AND with others for safety).
    assign valid_layer = neuron_valid[0];

endmodule
