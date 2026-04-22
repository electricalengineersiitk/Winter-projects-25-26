// neuron.v
// --------
// A single MAC (Multiply-Accumulate) neuron for Q8 fixed-point arithmetic.
//
// Operation
//   Each clock cycle while running, data_in is multiplied by weight_in and
//   the result is accumulated.  When 'last' is asserted on the same cycle as
//   the final input pair, the bias is added and ReLU is applied.  The
//   output is held on 'out' and 'valid' pulses high for one cycle.
//
// Numeric format
//   All inputs, weights and biases are 16-bit signed (Q8: 1 sign bit,
//   7 integer bits, 8 fractional bits).  The multiply produces a 32-bit
//   product; we accumulate in 32 bits and shift right by 8 before adding
//   the bias so the accumulator stays in Q8 throughout.

`timescale 1ns / 1ps

module neuron (
    input  wire        clk,
    input  wire        rst_n,      // active-low synchronous reset
    input  wire        start,      // pulse for 1 cycle to begin a new computation
    input  wire [15:0] data_in,    // Q8 signed input value
    input  wire [15:0] weight_in,  // Q8 signed weight for this input
    input  wire [15:0] bias,       // Q8 signed bias
    input  wire        last,       // assert on the cycle of the final input pair
    output reg  [15:0] out,        // Q8 result after ReLU
    output reg         valid       // high for 1 cycle when out is ready
);

    // 32-bit accumulator gives enough headroom for 4 × (Q8 × Q8) products
    reg signed [31:0] accumulator;
    reg               running;     // set on start, cleared when result is ready

    wire signed [15:0] data_s   = $signed(data_in);
    wire signed [15:0] weight_s = $signed(weight_in);
    wire signed [15:0] bias_s   = $signed(bias);

    // Full 32-bit product of two Q8 values; sum of N such products is Q16
    wire signed [31:0] product  = data_s * weight_s;

    always @(posedge clk) begin
        if (!rst_n) begin
            accumulator <= 32'sd0;
            out         <= 16'd0;
            valid       <= 1'b0;
            running     <= 1'b0;
        end else begin
            valid <= 1'b0;    // default: deassert every cycle

            if (start) begin
                // First input arrives on the same cycle as start
                accumulator <= product;
                running     <= 1'b1;
            end else if (running) begin
                if (last) begin
                    // Final input: accumulate, re-scale back to Q8, add bias
                    // and apply ReLU
                    reg signed [31:0] raw_sum;
                    raw_sum = accumulator + product;

                    // Re-scale from Q16 → Q8 by arithmetic right-shift 8
                    reg signed [31:0] scaled;
                    scaled = raw_sum >>> 8;

                    // Add Q8 bias (sign-extend to 32 bits)
                    reg signed [31:0] biased;
                    biased = scaled + {{16{bias_s[15]}}, bias_s};

                    // ReLU: clamp negative values to 0, then keep lower 16 bits
                    if (biased[31]) begin
                        out <= 16'd0;
                    end else if (|biased[31:16]) begin
                        // Positive overflow – saturate to max 16-bit value
                        out <= 16'h7FFF;
                    end else begin
                        out <= biased[15:0];
                    end

                    valid   <= 1'b1;
                    running <= 1'b0;
                    accumulator <= 32'sd0;
                end else begin
                    // Middle inputs: keep accumulating
                    accumulator <= accumulator + product;
                end
            end
        end
    end

endmodule
