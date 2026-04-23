`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 08.04.2026 01:30:39
// Design Name: 
// Module Name: Nueron
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module neuron (
    input wire clk,
    input wire rst_n, // active-low reset
    input wire start, // pulse high for 1 cycle to begin
    input wire [15:0] data_in, // one input value at a time (Q8 format)
    input wire [15:0] weight_in, // matching weight for that input
    input wire [15:0] bias, // bias value
    input wire last, // pulse high on the final input
    output reg [15:0] out, // result after ReLU
    output reg valid // high for 1 cycle when output is ready
);
reg [31:0] acc;
wire [31:0] mult;
reg [31:0] sum;
assign mult= $signed(data_in)*$signed(weight_in);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        acc <=32'd0;
        out   <= 16'd0;
        valid <= 1'b0;
    end 
    else begin
        valid<=0;
        if (start) begin
                acc<=mult; // reset accumulator on start I am using mult instead of 0 here.
        end else begin
                
                acc <= acc + mult;
        end
        if(last) begin
                sum = acc+mult + $signed(({{16{bias[15]}}, bias})<<8);
                out <= (sum[31]) ? 16'd0 : sum[23:8]; // use bits [23:8] for Q8 truncation
                valid <= 1'b1;
                
        end
    end
    
end

endmodule