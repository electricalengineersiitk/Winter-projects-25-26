`timescale 1ns / 1ps
//
// Neuron Module (MAC + ReLU)
// Q8 Fixed-point (16-bit signed)
//
// Protocol:

/*
 * Basic Neuron: MAC + ReLU
 *
 * This performs a Multiply-Accumulate (MAC) over a stream of features.
 * Once the 'last' signal is pulsed, it applies ReLU and sets 'valid' high.
 *
 * Protocol:
 *   Pulse start=1 on the FIRST input cycle.
 *   Pulse last=1 on the LAST input cycle.
 *   For a single-input neuron, both start=1 and last=1 on the same cycle.
 *   The product on the last cycle IS accumulated.
 */
module neuron (
    input  wire        clk,
    input  wire        rst_n,      // active-low reset
    input  wire        start,      // pulse high for 1 cycle to begin
    input  wire [15:0] data_in,    // one input value at a time (Q8 format)
    input  wire [15:0] weight_in,  // matching weight for that input
    input  wire [15:0] bias,       // bias value (Q8 format)
    input  wire        last,       // pulse high on the final input
    output reg  [15:0] out,        // result after ReLU (Q8 format)
    output reg         valid       // high for 1 cycle when output is ready
);

    // Internal 32-bit accumulator (Q16 format after multiply)
    reg signed [31:0] acc;

    // Signed multiplication: Q8 * Q8 = Q16 (32-bit)
    wire signed [31:0] product;
    assign product = $signed(data_in) * $signed(weight_in);

    // Bias shifted to Q16 alignment
    wire signed [31:0] bias_shifted;
    assign bias_shifted = $signed(bias) <<< 8;

    // Final sum for ReLU (computed combinationally for use in the last cycle)
    wire signed [31:0] final_sum_start_last;  // for start+last simultaneous
    wire signed [31:0] final_sum_last;        // for last only (multi-input)
    assign final_sum_start_last = product + bias_shifted;
    assign final_sum_last       = acc + product + bias_shifted;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset state
            acc   <= 32'sd0;
            out   <= 16'd0;
            valid <= 1'b0;
        end else if (start && last) begin
            // Single-input neuron: compute everything in one cycle
            if (final_sum_start_last[31]) begin
                out <= 16'd0;                          // ReLU: negative → 0
            end else begin
                out <= final_sum_start_last[23:8];     // Q16 → Q8 truncation
            end
            valid <= 1'b1;
            acc   <= 32'sd0;
        end else if (start) begin
            // First input: clear accumulator, load first product
            acc   <= product;
            valid <= 1'b0;
        end else if (last) begin
            // Last input: accumulate, add bias, apply ReLU
            if (final_sum_last[31]) begin
                out <= 16'd0;                          // ReLU: negative → 0
            end else begin
                out <= final_sum_last[23:8];           // Q16 → Q8 truncation
            end
            valid <= 1'b1;
            acc   <= 32'sd0;
        end else begin
            // Middle inputs: accumulate product
            acc   <= acc + product;
            valid <= 1'b0;
        end
    end

endmodule
