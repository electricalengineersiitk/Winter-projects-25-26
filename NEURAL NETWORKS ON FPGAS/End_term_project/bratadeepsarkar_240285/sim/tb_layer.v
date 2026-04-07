`timescale 1ns / 1ps

module tb_layer;

    reg         clk;
    reg         rst_n;
    reg         start;
    reg  [15:0] data_in;
    reg  [1:0]  input_idx;
    reg         last;
    wire [127:0] out_vector;
    wire        valid_layer;

    // Instantiate UUT
    layer uut (
        .clk         (clk),
        .rst_n       (rst_n),
        .start       (start),
        .data_in     (data_in),
        .input_idx   (input_idx),
        .last        (last),
        .out_vector  (out_vector),
        .valid_layer (valid_layer)
    );

    // 100 MHz clock
    always #5 clk = ~clk;

    integer i;

    initial begin
        // Init
        clk       = 0;
        rst_n     = 0;
        start     = 0;
        data_in   = 0;
        input_idx = 0;
        last      = 0;

        // Reset
        #100;
        rst_n = 1;
        #20;

        $display("============================================");
        $display(" Layer Testbench (8 neurons, 4 inputs)");
        $display(" Weights from: weights/weights_hidden.mem");
        $display(" Biases from:  weights/biases_hidden.mem");
        $display("============================================");
        $display("");

        // Feed 4 inputs: [1.0, 0.5, 0.625, 0.125] in Q8
        // Q8 values: 1.0=0x0100, 0.5=0x0080, 0.625=0x00A0, 0.125=0x0020
        $display("Feeding inputs: [1.0, 0.5, 0.625, 0.125]");
        $display("");

        // Input 0 (start)
        @(posedge clk); #1;
        data_in   = 16'h0100;   // 1.0
        input_idx = 2'd0;
        start     = 1;
        last      = 0;

        // Input 1 (middle)
        @(posedge clk); #1;
        start     = 0;
        data_in   = 16'h0080;   // 0.5
        input_idx = 2'd1;

        // Input 2 (middle)
        @(posedge clk); #1;
        data_in   = 16'h00A0;   // 0.625
        input_idx = 2'd2;

        // Input 3 (last)
        @(posedge clk); #1;
        data_in   = 16'h0020;   // 0.125
        input_idx = 2'd3;
        last      = 1;

        @(posedge clk); #1;
        last = 0;
        data_in = 0;

        // Wait for valid
        wait(valid_layer);
        @(posedge clk); #1; // let output settle

        $display("--- Layer Outputs (Q8 hex) ---");
        for (i = 0; i < 8; i = i + 1) begin
            $display("  Neuron %0d: 0x%04h  (decimal: %0d)",
                     i,
                     out_vector[i*16 +: 16],
                     out_vector[i*16 +: 16]);
        end
        $display("");
        $display("Verify: each output should be >= 0 (ReLU enforces this).");
        $display("Values depend on trained weights in .mem files.");
        $display("");
        $display("Layer test DONE.");

        #100;
        $finish;
    end

endmodule
