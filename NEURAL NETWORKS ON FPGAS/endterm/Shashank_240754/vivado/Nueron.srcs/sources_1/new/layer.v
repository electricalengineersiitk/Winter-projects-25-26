`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 10.04.2026 18:23:04
// Design Name: 
// Module Name: layer
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


module layer(
    input wire clk,
    input wire rst_n,
    input wire start,
    input wire [15:0] data_in,
    

    output wire [15:0] out0,
    output wire [15:0] out1,
    output wire [15:0] out2,
    output wire [15:0] out3,
    output wire [15:0] out4,
    output wire [15:0] out5,
    output wire [15:0] out6,
    output wire [15:0] out7,
    
    output wire valid
);
        
        // 8 neurons × 4 inputs = 32 weights
        reg [15:0] weight_mem [0:31];
        reg [15:0] bias_mem   [0:7];
        
        // load weights + bias
        initial begin
            $readmemh("D:/HARI NOTES/projects/FPGA prof/weights/weights.mem", weight_mem);
            $readmemh("D:/HARI NOTES/projects/FPGA prof/weights/biases.mem", bias_mem);
            $display("weight[0]=%h", weight_mem[0]);
            $display("bias[0]=%h", bias_mem[0]);
        end
        
        // input index (0 to 3) such that there it has 4 features
        reg [1:0] idx;
        
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n)
                idx<=0;
            else if (start)
                idx<=0;
            else if (idx<3)
                idx<=idx+1;
        end
        
        // neuron valid signals
        wire [7:0] valid_bus;
        wire lastint;
        assign lastint = (idx == 2'd3);

        wire [15:0] out_bus [0:7];
        assign out0 = out_bus[0];
        assign out1 = out_bus[1];
        assign out2 = out_bus[2];
        assign out3 = out_bus[3];
        assign out4 = out_bus[4];
        assign out5 = out_bus[5];
        assign out6 = out_bus[6];
        assign out7 = out_bus[7];

        genvar i;
        generate
            for (i = 0; i < 8; i = i + 1) begin : NEURONS
        
                neuron n_inst (
                    .clk(clk),
                    .rst_n(rst_n),
                    .start(start),
                    .data_in(data_in),
        
                    
                    
                    .weight_in(weight_mem[i*4 + idx]),
        
                    .bias(bias_mem[i]),
                    .last(lastint),
        
                    .out(out_bus[i]),
                    .valid(valid_bus[i])
                );
        
            end
        endgenerate
        
        assign valid = valid_bus[0];
        
endmodule
