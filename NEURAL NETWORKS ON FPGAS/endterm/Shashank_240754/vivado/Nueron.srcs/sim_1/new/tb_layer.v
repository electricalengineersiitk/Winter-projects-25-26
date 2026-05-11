`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 11.04.2026 00:08:57
// Design Name: 
// Module Name: tb_layer
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


module tb_layer;
    reg clk;
    reg rst_n;
    reg start;
    
    reg [15:0] data_in;
    
    wire [15:0] out0, out1, out2, out3, out4, out5, out6, out7;
    wire valid;
    
    
    // DUT
    layer uut (
    .clk(clk),
    .rst_n(rst_n),
    .start(start),
    .data_in(data_in),
    

    .out0(out0),
    .out1(out1),
    .out2(out2),
    .out3(out3),
    .out4(out4),
    .out5(out5),
    .out6(out6),
    .out7(out7),

    .valid(valid)
);

    // clock generation
    always #5 clk = ~clk;
    
    initial begin
        clk = 0;
        rst_n = 0;
        start = 0;
        
        data_in = 0;
    
        // reset
        #10;
        rst_n = 1;
    
        // Cycle 0 first input
        @(posedge clk);
        start   = 1;
        
        data_in = 16'd256;
    
        // Cycle 1
        @(posedge clk);
        start   = 0;
        
        data_in = 16'd128;
    
        // Cycle 2
        @(posedge clk);
        
        data_in = 16'd64;
    
        // Cycle 3 
        @(posedge clk);
        
        data_in = 16'd32;
    
        // clear last
        @(posedge clk);
        
    
        // wait for output
        wait(valid);
    
        $display("%d %d %d %d %d %d %d %d",
            out0, out1, out2, out3,
            out4, out5, out6, out7
        );
    
        #10;
        $finish;
    end

endmodule
