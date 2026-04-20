`timescale 1ns / 1ps
`include "neuron.v"

module neuron_tb;

    // 1. Parameters & Signals
    reg clk;
    reg rst_n;
    reg start;
    reg [15:0] data_in;
    reg [15:0] weight_in;
    reg [15:0] bias;
    reg last;

    wire [15:0] out;
    wire valid;

    // 2. Instantiate the Unit Under Test (UUT)
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

    // 3. Clock Generation (100MHz)
    always #5 clk = ~clk;

    initial begin
        // This creates the file named "neuron_sim.vcd"
        $dumpfile("neuron_sim.vcd");
        
        // $dumpvars(level, module) 
        // 0 means "dump all signals in this module and everything inside it"
        $dumpvars(0, neuron_tb); 
    end

    // 4. Stimulus Procedure
    initial begin
        // Initialize Inputs
        clk = 0;
        rst_n = 0;
        start = 0;
        data_in = 0;
        weight_in = 0;
        bias = 16'd1280; // Bias = +5
        last = 0;

        // Reset the system
        #20 rst_n = 1;
        #10;

        // --- TEST CASE 1: Simple MAC ---
        // Calculation: (2 * 3) + (4 * 2) + Bias(5) = 6 + 8 + 5 = 19
        
        @(posedge clk);
        start     <= 1;
        data_in   <= 16'd512; // 2 in Q8 format
        weight_in <= 16'd768; // 3 in Q8 format
        
        @(posedge clk);
        start     <= 0;
        data_in   <= 16'd1024; // 4 in Q8 format
        weight_in <= 16'd512; // 2 in Q8 format
        last      <= 1; // This is the final input
        
        @(posedge clk);
        last      <= 0;
        data_in   <= 0;
        weight_in <= 0;

        // Wait for valid signal
        wait(valid);
        $display("Test 1 Result: %d (Expected 19)", out);

        #50;

        // --- TEST CASE 2: Negative Result (ReLU Check) ---
        // Calculation: (10 * -2) + Bias(5) = -20 + 5 = -15 -> ReLU should make this 0
        
        @(posedge clk);
        start     <= 1;
        data_in   <= 16'd2560;  // 10 in Q8 format (10 * 256)
        weight_in <= 16'hFE00;  // -2 in Q8 format (-512, or 0xFE00)
        last      <= 1;         // Only one input this time
        
        @(posedge clk);
        start     <= 0;
        last      <= 0;

        wait(valid);
        $display("Test 2 Result: %d (Expected 0 due to ReLU)", out);

        #100;
        $finish;
    end

endmodule