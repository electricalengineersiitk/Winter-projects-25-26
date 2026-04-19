`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 09.04.2026 15:22:28
// Design Name: 
// Module Name: tb_neuron
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: `timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 08.04.2026 11:57:23
// Design Name: 
// Module Name: tb_nueron
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


module tb_nueron;
    
        reg clk;
        reg rst_n;
        reg start;
        reg [15:0] data_in;
        reg [15:0] weight_in;
        reg [15:0] bias;
        reg last;
    
        wire [15:0] out;
        wire valid;
    
        
        neuron uut (
            .clk(clk),
            .rst_n(rst_n),
            .start(start),
            .data_in(data_in),
            .weight_in(weight_in),
            .bias(bias),
            .last(last),
            .out(out),
            .valid(valid)
        );
    
        always #5 clk = ~clk;
    
        initial begin
            
            clk = 0;
            rst_n = 0;
            start = 0;
            data_in = 0;
            weight_in = 0;
            bias = 16'd256; // 1.0 in Q8
            last = 0;
    
            
            #10;
            rst_n = 1;
    
            
            #10;
            start = 1;
            #10;
            start = 0;
    
            
            data_in   = 16'd256;  // 1.0
            weight_in = 16'd256;  // 1.0
            last = 0;
            #10;
    
            
            data_in   = 16'd512;  // 2.0
            weight_in = 16'd128;  // 0.5
            #10;
    
            
            data_in   = -16'd256; // -1.0
            weight_in = 16'd256;  // 1.0
            #10;
    
            
            data_in   = 16'd128;  // 0.5
            weight_in = -16'd256; // -1.0
            last = 1;
            #10;
            last = 0;
    
            
            wait(valid);
    
            
            if (out == 16'd384) begin
                $display("PASS: Output = %d (Expected 384)", out);
            end else begin
                $display("FAIL: Output = %d (Expected 384)", out);
            end
    
            #10;
            $finish;
        end
endmodule
