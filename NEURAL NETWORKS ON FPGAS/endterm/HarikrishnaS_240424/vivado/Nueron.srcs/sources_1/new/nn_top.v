`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 11.04.2026 01:33:28
// Design Name: 
// Module Name: nn_top
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


module nn_top(
    input clk,
    input wire rst_n,
    input wire start,
    input wire [15:0] data_in,
    
    
    output reg[1:0]fout,
    output reg done
    );
    
    reg [15:0] weight_mem [0:55];
    reg [15:0] bias_mem   [0:10];
    
    // load weights + bias
    initial begin
        $readmemh("D:/HARI NOTES/projects/FPGA prof/weights/weights.mem", weight_mem);
        $readmemh("D:/HARI NOTES/projects/FPGA prof/weights/biases.mem", bias_mem);
    end

    wire [15:0] out_bus [0:7];
    
    wire valid;

    layer dut (
    .clk(clk),
    .rst_n(rst_n),
    .start(start),
    .data_in(data_in),
    

    .out0(out_bus[0]),
    .out1(out_bus[1]),
    .out2(out_bus[2]),
    .out3(out_bus[3]),
    .out4(out_bus[4]),
    .out5(out_bus[5]),
    .out6(out_bus[6]),
    .out7(out_bus[7]),

    .valid(valid)
);
    reg [3:0] idx;
    always @(posedge clk or negedge rst_n) begin
            if (!rst_n)
                idx<=0;
            else if (start)
                idx<=0;
            else if (valid&&idx<7)
                idx<=idx+1;
    end

    wire [7:0] valid_bus;
    wire lastint;
    assign lastint = (idx == 3'd7);
    reg start_int;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            start_int <= 0;
        else
            start_int <= valid;  // 1-cycle pulse
    end
    

    wire [15:0] pout [0:2];
    
    genvar i;
    generate
        for (i = 0; i < 3; i = i + 1) begin : NEURONS
    
            neuron n_inst (
                .clk(clk),
                .rst_n(rst_n),
                .start(start_int),
                .data_in(out_bus[idx]),
    
                
                
                .weight_in(weight_mem[(i+8)*4 + idx]),
    
                .bias(bias_mem[i+8]),
                .last(lastint),
    
                .out(pout[i]),
                .valid(valid_bus[i])
            );
    
        end
    endgenerate
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fout <= 0;
            done <= 0;
        end else begin
            if (valid_bus[0] && valid_bus[1] && valid_bus[2]) begin
                fout <= (pout[0] > pout[1]) ? 
                        ((pout[0] > pout[2]) ? 0 : 2) : 
                        ((pout[1] > pout[2]) ? 1 : 2);
                done <= 1;
            end else begin
                done <= 0;
            end
        end
    end
    
endmodule
