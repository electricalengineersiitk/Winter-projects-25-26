`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 11.04.2026 03:58:02
// Design Name: 
// Module Name: tb_nn_top
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


module tb_nn_top;
    
        reg clk;
        reg rst_n;
        reg start;
        reg [15:0] data_in;
    
        wire [1:0] fout;
        wire done;
    
        // Instantiate DUT
        nn_top uut (
            .clk(clk),
            .rst_n(rst_n),
            .start(start),
            .data_in(data_in),
            .fout(fout),
            .done(done)
        );
    
        // Clock generation (10ns period)
        always #5 clk = ~clk;
    
        // Task to send 4 inputs
        task send_input;
            input [15:0] d0, d1, d2, d3;
            begin
                @(posedge clk);
                start = 1;
                data_in = d0;
    
                @(posedge clk);
                start = 0;
                data_in = d1;
    
                @(posedge clk);
                data_in = d2;
    
                @(posedge clk);
                data_in = d3;
            end
        endtask
    
        initial begin
            // Init
            clk = 0;
            rst_n = 0;
            start = 0;
            data_in = 0;
    
            // Reset
            #20;
            rst_n = 1;
    
            // Wait a bit
            #20;
    
            $display("---- TEST 1 ----");
    
            // Example input (Q8 format)
            send_input(16'h0100, 16'h0200, 16'hFF00, 16'h0080);
    
            // Wait for result
            wait(done == 1);
    
            $display("Prediction{d}: %d", fout);
    
            #20;
    
            $display("TEST 2");
    
            send_input(16'h0080, 16'h0100, 16'h0100, 16'h0100);
    
            wait(done == 1);
    
            $display("Prediction: %d", fout);
    
            #50;
    
            $finish;
        end


endmodule

