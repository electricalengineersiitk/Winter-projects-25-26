// tb_neuron.v
// -----------
// Testbench for the neuron module.
//
// Hand-computed expected result
//   inputs  : 0x0100 (1.0), 0x0200 (2.0), 0x0180 (1.5), 0x0080 (0.5)
//   weights : 0x0100 (1.0), 0x0080 (0.5), 0x0100 (1.0), 0x0100 (1.0)
//   bias    : 0x0040 (0.25)
//
//   Dot product = 1.0×1.0 + 2.0×0.5 + 1.5×1.0 + 0.5×1.0 = 4.0
//   After bias  = 4.0 + 0.25 = 4.25
//   ReLU        = 4.25  (positive → unchanged)
//   Q8 hex      = 4.25 × 256 = 1088 = 0x0440
//
// The simulation prints PASS or FAIL and then calls $finish.

`timescale 1ns / 1ps

module tb_neuron;

    // ── DUT signals ──────────────────────────────────────────────────────────
    reg        clk, rst_n, start, last;
    reg [15:0] data_in, weight_in, bias;
    wire [15:0] out;
    wire        valid;

    // ── DUT instantiation ────────────────────────────────────────────────────
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

    // ── clock: 10 ns period ──────────────────────────────────────────────────
    initial clk = 0;
    always #5 clk = ~clk;

    // ── test vectors (Q8: value × 256) ───────────────────────────────────────
    reg [15:0] inputs  [0:3];
    reg [15:0] weights [0:3];

    integer i;
    localparam [15:0] BIAS_VAL    = 16'h0040;   //  0.25
    localparam [15:0] EXPECTED    = 16'h0440;   //  4.25

    initial begin
        inputs[0]  = 16'h0100;  weights[0] = 16'h0100;  // 1.0 × 1.0
        inputs[1]  = 16'h0200;  weights[1] = 16'h0080;  // 2.0 × 0.5
        inputs[2]  = 16'h0180;  weights[2] = 16'h0100;  // 1.5 × 1.0
        inputs[3]  = 16'h0080;  weights[3] = 16'h0100;  // 0.5 × 1.0
    end

    // ── stimulus ─────────────────────────────────────────────────────────────
    initial begin
        $dumpfile("tb_neuron.vcd");
        $dumpvars(0, tb_neuron);

        // Initialise
        rst_n     = 0;
        start     = 0;
        last      = 0;
        data_in   = 0;
        weight_in = 0;
        bias      = BIAS_VAL;

        // Hold reset for 3 cycles
        @(negedge clk); @(negedge clk); @(negedge clk);
        rst_n = 1;
        @(negedge clk);

        // Feed inputs one by one; start is high on first, last on final
        for (i = 0; i < 4; i = i + 1) begin
            data_in   = inputs[i];
            weight_in = weights[i];
            start     = (i == 0) ? 1'b1 : 1'b0;
            last      = (i == 3) ? 1'b1 : 1'b0;
            @(negedge clk);
        end
        start = 0;
        last  = 0;

        // Wait for valid to go high (at most 10 cycles)
        repeat (10) begin
            @(posedge clk);
            if (valid) disable wait_valid;
        end
        wait_valid: begin end   // exit point

        @(posedge clk);  // sample output one cycle after valid

        // ── result check ─────────────────────────────────────────────────────
        $display("-----------------------------------------------------");
        $display("Neuron testbench result");
        $display("  Expected : 0x%04X (%0d)", EXPECTED, EXPECTED);
        $display("  Got      : 0x%04X (%0d)", out,      out);
        if (out === EXPECTED)
            $display("  STATUS   : PASS");
        else
            $display("  STATUS   : FAIL  <--- mismatch!");
        $display("-----------------------------------------------------");

        #20;
        $finish;
    end

endmodule
