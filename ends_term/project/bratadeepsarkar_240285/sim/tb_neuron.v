`timescale 1ns / 1ps

module tb_neuron;

    reg        clk;
    reg        rst_n;
    reg        start;
    reg [15:0] data_in;
    reg [15:0] weight_in;
    reg [15:0] bias;
    reg        last;
    wire [15:0] out;
    wire        valid;

    // Instantiate unit under test
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

    // 100 MHz clock: period = 10 ns
    always #5 clk = ~clk;

    // Task to reset all inputs
    task reset_inputs;
        begin
            start     = 0;
            data_in   = 0;
            weight_in = 0;
            bias      = 0;
            last      = 0;
        end
    endtask

    integer pass_count;
    integer fail_count;

    initial begin
        // Initialise
        $dumpfile("tb_neuron.vcd");
        $dumpvars(0, tb_neuron);
        clk  = 0;
        rst_n = 0;
        pass_count = 0;
        fail_count = 0;
        reset_inputs;

        // Hold reset for 100 ns
        #100;
        rst_n = 1;
        #20;

        // ================================================================
        // TEST 1: Two-input MAC, bias = 0
        //   1.0 * 1.0 + 0.5 * 2.0 + 0.0 = 2.0
        //   Q8 values: 1.0=0x0100, 0.5=0x0080, 2.0=0x0200
        //   Expected output: 0x0200 (512 decimal = 2.0 in Q8)
        // ================================================================
        $display("--- TEST 1: Two-input MAC (bias=0) ---");

        // Cycle 1: first input (start=1)
        @(posedge clk); #1;
        data_in   = 16'h0100;   // 1.0
        weight_in = 16'h0100;   // 1.0
        bias      = 16'h0000;   // 0.0
        start     = 1;
        last      = 0;

        // Cycle 2: second input (last=1)
        @(posedge clk); #1;
        start     = 0;
        data_in   = 16'h0080;   // 0.5
        weight_in = 16'h0200;   // 2.0
        last      = 1;

        // Cycle 3: deassert last, wait for valid
        @(posedge clk); #1;
        last = 0;
        wait(valid);
        #1;
        if (valid !== 1'b1) begin
            $display("  FAIL: valid not asserted");
            fail_count = fail_count + 1;
        end else if (out !== 16'h0200) begin
            $display("  FAIL: out = %0d (0x%04h), expected 512 (0x0200)", out, out);
            fail_count = fail_count + 1;
        end else begin
            $display("  PASS: out = %0d (0x%04h)", out, out);
            pass_count = pass_count + 1;
        end
        reset_inputs;

        #20;

        // ================================================================
        // TEST 2: Single-input, negative result → ReLU clips to 0
        //   1.0 * (-1.0) + 0.0 = -1.0 → ReLU → 0
        //   Q8: -1.0 = 0xFF00 (two's complement)
        //   Expected output: 0x0000
        // ================================================================
        $display("--- TEST 2: Single-input ReLU (negative → 0) ---");

        @(posedge clk); #1;
        data_in   = 16'h0100;   // 1.0
        weight_in = 16'hFF00;   // -1.0
        bias      = 16'h0000;
        start     = 1;
        last      = 1;          // single input: start+last same cycle

        @(posedge clk); #1;
        start = 0;
        last = 0;
        wait(valid);
        #1;
        if (valid !== 1'b1) begin
            $display("  FAIL: valid not asserted");
            fail_count = fail_count + 1;
        end else if (out !== 16'h0000) begin
            $display("  FAIL: out = %0d, expected 0", out);
            fail_count = fail_count + 1;
        end else begin
            $display("  PASS: out = %0d", out);
            pass_count = pass_count + 1;
        end
        reset_inputs;

        #20;

        // ================================================================
        // TEST 3: Four-input MAC with bias (matches hidden layer shape)
        //   inputs:  [1.0, 0.5, -0.5, 0.25]
        //   weights: [2.0, 1.0,  1.0, 0.0 ]
        //   bias:    0.5
        //   Expected: 1.0*2.0 + 0.5*1.0 + (-0.5)*1.0 + 0.25*0.0 + 0.5
        //           = 2.0    + 0.5     - 0.5         + 0.0       + 0.5
        //           = 2.5
        //   Q8: 2.5 = 640 = 0x0280
        // ================================================================
        $display("--- TEST 3: Four-input MAC with bias ---");

        // Input 0 (start)
        @(posedge clk); #1;
        data_in   = 16'h0100;   // 1.0
        weight_in = 16'h0200;   // 2.0
        bias      = 16'h0080;   // 0.5
        start     = 1;
        last      = 0;

        // Input 1 (middle)
        @(posedge clk); #1;
        start     = 0;
        data_in   = 16'h0080;   // 0.5
        weight_in = 16'h0100;   // 1.0

        // Input 2 (middle)
        @(posedge clk); #1;
        data_in   = 16'hFF80;   // -0.5 (two's complement)
        weight_in = 16'h0100;   // 1.0

        // Input 3 (last)
        @(posedge clk); #1;
        data_in   = 16'h0040;   // 0.25
        weight_in = 16'h0000;   // 0.0
        last      = 1;

        @(posedge clk); #1;
        last = 0;
        wait(valid);
        #1;
        if (valid !== 1'b1) begin
            $display("  FAIL: valid not asserted");
            fail_count = fail_count + 1;
        end else if (out !== 16'h0280) begin
            $display("  FAIL: out = %0d (0x%04h), expected 640 (0x0280)", out, out);
            fail_count = fail_count + 1;
        end else begin
            $display("  PASS: out = %0d (0x%04h)", out, out);
            pass_count = pass_count + 1;
        end

        // ================================================================
        // SUMMARY
        // ================================================================
        #20;
        $display("");
        $display("========================================");
        $display("  RESULTS: %0d PASSED, %0d FAILED", pass_count, fail_count);
        $display("========================================");

        if (fail_count > 0)
            $display("  *** SOME TESTS FAILED — FIX neuron.v BEFORE PROCEEDING ***");
        else
            $display("  All tests passed. Proceed to TASK4_LAYER.md");

        $finish;
    end

endmodule
