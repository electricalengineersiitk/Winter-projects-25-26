`timescale 1ns / 1ps

module neuron (
    input  wire        clk,
    input  wire        rst_n,      // active-low reset
    input  wire        start,      // pulse high for 1 cycle to begin
    input  wire [15:0] data_in,    // one input value at a time (Q8 format)
    input  wire [15:0] weight_in,  // matching weight for that input
    input  wire [15:0] bias,       // bias value
    input  wire        last,       // pulse high on the final input
    output reg  [15:0] out,        // result after ReLU
    output reg         valid       // high for 1 cycle when output is ready
);

    // Internal 32-bit accumulator (Stores values with 16 fractional bits)
    reg signed [31:0] acc;
    reg processing; // State flag to track if we are mid-sequence

    // Cast inputs to signed wires to ensure Verilog uses 2's complement math
    wire signed [15:0] s_data   = data_in;
    wire signed [15:0] s_weight = weight_in;
    
    // ALIGN THE BIAS: 
    // Bias is Q8 (8 fractional bits). The accumulator has 16 fractional bits.
    // We shift the bias left by 8 bits (adding 8'd0) to align the decimal points,
    // and sign-extend the top 8 bits so it fits the 32-bit wire.
    wire signed [31:0] s_bias   = { {8{bias[15]}}, bias, 8'd0 }; 

    // Multiply two Q8 values. This yields a product with 16 fractional bits.
    wire signed [31:0] product = s_data * s_weight;

    // Accumulate the full 32-bit precision
    wire signed [31:0] next_acc = (start ? 32'sd0 : acc) + product;

    // Final mathematical value including the bias
    wire signed [31:0] final_val = next_acc + s_bias;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc        <= 32'sd0;
            out        <= 16'd0;
            valid      <= 1'b0;
            processing <= 1'b0;
        end else begin
            // Default valid to 0 so it strictly pulses for 1 cycle
            valid <= 1'b0;

            if (start) begin
                processing <= 1'b1;
                
                if (last) begin
                    // Edge case: sequence is only 1 element long
                    if (final_val < 0) begin
                        out <= 16'd0; // Apply ReLU
                    end else begin
                        // EXTRACT Q8: Shift right by 8 to drop the extra 8 fractional bits
                        out <= final_val[23:8]; 
                    end
                    valid      <= 1'b1;
                    processing <= 1'b0;
                end else begin
                    acc <= next_acc; 
                end
                
            end else if (processing) begin
                if (last) begin
                    // Final element in the sequence
                    if (final_val < 0) begin
                        out <= 16'd0; // Apply ReLU
                    end else begin
                        // EXTRACT Q8: Shift right by 8 to drop the extra 8 fractional bits
                        out <= final_val[23:8]; 
                    end
                    valid      <= 1'b1;
                    processing <= 1'b0;
                end else begin
                    acc <= next_acc;
                end
            end
        end
    end

endmodule