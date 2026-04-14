// tb_neuron.v
// Testbench for the neuron module.
//
// Uses Neuron 1 weights from weights.mem with 4 known inputs.
// Expected output is pre-calculated:
//   inputs  : 256, 128, -64, 32
//   weights : 186, -28, 135, 263   (neuron 1, from weights.mem)
//   bias    : -73                  (from w1_bias.mem)
//   acc     : (256*186 + 128*-28 + -64*135 + 32*263) >> 8 each = 170
//   result  : 170 + (-73) = 97
//   ReLU    : 97  (positive, no clamp)
//   Expected out = 16'd97   (0x0061)
//
// Two checks:
//   1. valid goes high exactly one cycle after last
//   2. out == 16'd97

module tb_neuron;

    reg        clk   = 0;
    always #5  clk   = ~clk;  

    reg        rst_n  = 0;
    reg        start  = 0;
    reg signed [15:0] data_in   = 0;
    reg signed [15:0] weight_in = 0;
    reg signed [15:0] bias      = 0;
    reg        last   = 0;

    wire signed [15:0] out;
    wire               valid;

    neuron dut (
        .clk       (clk),
        .rst_n     (rst_n),
        .start     (start),
        .data_in   (data_in),
        .weight_in (weight_in),
        .bias      (bias),
        .last      (last),
        .out       (out),
        .valid     (valid)
    );

    // Bias is constant throughout
    initial bias = -16'sd73;

    integer pass_count = 0;
    integer fail_count = 0;

    task check;
        input [63:0] expected_out;
        input        expected_valid;
        begin
            if (valid !== expected_valid) begin
                $display("FAIL: valid = %b, expected %b  at time %0t", valid, expected_valid, $time);
                fail_count = fail_count + 1;
            end else if (expected_valid && out !== expected_out[15:0]) begin
                $display("FAIL: out = %0d (0x%04X), expected %0d (0x%04X)  at time %0t",
                         out, out, expected_out[15:0], expected_out[15:0], $time);
                fail_count = fail_count + 1;
            end else begin
                if (expected_valid)
                    $display("PASS: out = %0d  valid = %b", out, valid);
                pass_count = pass_count + 1;
            end
        end
    endtask

    initial begin
        $display("========== Neuron Testbench ==========");
        $display("Inputs: 256 128 -64 32");
        $display("Weights: 186 -28 135 263  Bias: -73");
        $display("Expected output: 97  (ReLU(170 + (-73)) = 97)");
        $display("");

        // Reset the module
        rst_n = 0;
        #12 rst_n = 1;

        // Cycle 1: Input 0 with start 
        @(negedge clk);
        data_in   = 16'sd256;
        weight_in = 16'sd186;
        start     = 1;
        last      = 0;

        // Cycle 2: Input 1 
        @(negedge clk);
        start     = 0;
        data_in   = 16'sd128;
        weight_in = -16'sd28;

        // Cycle 3: Input 2 
        @(negedge clk);
        data_in   = -16'sd64;
        weight_in = 16'sd135;

        // Cycle 4: Input 3 (last) 
        @(negedge clk);
        data_in   = 16'sd32;
        weight_in = 16'sd263;
        last      = 1;

        // Cycle 5: de-assert last, sample output 
        @(negedge clk);
        last = 0;

        // valid and out are registered sample one posedge later
        @(posedge clk); #1;
        $display("Sampling output...");
        check(16'd97, 1'b1);

        // Cycle 6: valid must self-clear 
        @(posedge clk); #1;
        if (valid !== 0)
            $display("FAIL: valid did not clear after one cycle");
        else begin
            $display("PASS: valid self-cleared");
            pass_count = pass_count + 1;
        end

        // Zero-input ReLU check: all zeros → bias only 
        $display("");
        $display("--- Second run: all-zero inputs, bias=-73 => ReLU(-73)=0 ---");
        @(negedge clk);
        data_in = 0; weight_in = 0; start = 1;
        @(negedge clk); start = 0;
        @(negedge clk);
        @(negedge clk); last = 1;
        @(negedge clk); last = 0;
        @(posedge clk); #1;
        check(16'd0, 1'b1);

        //Summary
        $display("Results: %0d PASS  %0d FAIL", pass_count, fail_count);
        if (fail_count == 0)
            $display("ALL TESTS PASSED");
        else
            $display("SOME TESTS FAILED — review output above");
        $display("======================================");

        #20 $finish;
    end

endmodule
