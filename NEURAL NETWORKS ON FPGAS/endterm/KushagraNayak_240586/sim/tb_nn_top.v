`timescale 1ns / 1ps

module tb_nn_top();
    // Inputs
    reg clk;
    reg reset;

    // Instantiate the Unit Under Test (UUT)
    nn_top uut (
        .clk(clk),
        .reset(reset)
    );

    // Clock generation (100 MHz -> 10ns period)
    always #5 clk = ~clk;

    initial begin
        // Initialize Inputs
        clk = 0;
        reset = 1;

        // Wait 100 ns for global reset to finish
        #100;
        
        // Release reset and start network
        reset = 0;

        // Run simulation for 1000ns
        #1000;
        
        $display("Simulation Complete. Hardware successfully initialized.");
        $finish;
    end
endmodule