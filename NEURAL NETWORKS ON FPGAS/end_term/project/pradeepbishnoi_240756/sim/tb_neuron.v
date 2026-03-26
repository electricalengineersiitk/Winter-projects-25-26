`timescale 1ns / 1ps

module tb_neuron();

    // Signals
    reg         clk;
    reg         rst_n;
    reg         start;
    reg  [15:0] data_in;
    reg  [15:0] weight_in;
    reg  [15:0] bias;
    reg         last;
    wire [15:0] out;
    wire        valid;

    // Instantiating the UUT
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

    // Clock Generation (100MHz)
    always #5 clk = ~clk;

    // Test Stimulus
    initial begin
        // Input intialisation
        clk = 0;
        rst_n = 0;
        start = 0;
        last = 0;
        data_in = 0;
        weight_in = 0;
        bias = 16'h0100; // Bias = 1.0 (Q8: 1 * 256 = 256 = 0x0100)

        // Reset Sequence
        #20 rst_n = 1; 
        #20;

        // Neuron Calculation (4 Inputs) 
        
        // Cycle 1: Input 0.5 * Weight 2.0 = 1.0
        @(posedge clk);
        start = 1;
        data_in   = 16'h0080; // 0.5
        weight_in = 16'h0200; // 2.0
        
        // Cycle 2: Input 1.0 * Weight 1.0 = 1.0
        @(posedge clk);
        start = 0;
        data_in   = 16'h0100; // 1.0
        weight_in = 16'h0100; // 1.0

        // Cycle 3: Input -0.5 * Weight 2.0 = -1.0
        @(posedge clk);
        data_in   = 16'hff80; // -0.5
        weight_in = 16'h0200; // 2.0

        // Cycle 4: Input 1.0 * Weight -0.5 = -0.5 (LAST)
        @(posedge clk);
        last = 1;
        data_in   = 16'h0100; // 1.0
        weight_in = 16'hff80; // -0.5

        // End of Stream
        @(posedge clk);
        last = 0;
        data_in = 0;
        weight_in = 0;

        // Result
        wait(valid);
        #100;

        $display("--------------------------------------");
        $display("Calculation Finished!");
        $display("Neuron Output (Hex): %h", out);
        $display("Neuron Output (Dec): %f", $signed(out) / 256.0);
        $display("--------------------------------------");
        $dumpfile("neuron_results.vcd"); // storing the results for gtkwave generation in iverilog
        $dumpvars(0, tb_neuron);
        $finish;
    end
endmodule
