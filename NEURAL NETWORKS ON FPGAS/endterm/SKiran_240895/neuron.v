`timescale 1ns / 1ps
module neuron(
    input  wire        clk,
    input  wire        rst_n,      // active-low reset
    input  wire        start,      // pulse high for 1 cycle to begin
    input  wire signed [15:0] data_in,    // one input value at a time (Q8 format)
    input  wire signed [15:0] weight_in,  // matching weight for that input
    input  wire [15:0] bias,       // bias value
    input  wire        last,       // pulse high on the final input
    output reg  [15:0] out,        // result after ReLU
    output reg         valid       // high for 1 cycle when output is ready
);

reg signed [31:0] acc;
wire signed [15:0] datains = data_in;
wire signed [15:0] weightin = weight_in;
wire signed [31:0] mult = datains * weightin;
wire signed [31:0] temp = acc + mult + ({{8{bias[15]}}, bias, 8'b0});
always@ (posedge clk) begin
    if (!rst_n) begin
            acc   <= 32'd0;
            valid <= 1'b0;
            out   <= 16'd0;
        end else begin
            valid <= 1'b0; 
            if (start) begin
                acc <= 32'd0;
            end else if (last) begin
                //temp sum to avoid overwrite
                //temp <= acc + mult + (bias << 8);
                if (temp > 0)
                    out <= temp[23:8];
                else
                    out <= 16'd0;
                valid <= 1'b1;
                acc   <= 32'd0; // Reset
            end else begin
                acc <= acc + mult;
            end
        end
    end
endmodule
