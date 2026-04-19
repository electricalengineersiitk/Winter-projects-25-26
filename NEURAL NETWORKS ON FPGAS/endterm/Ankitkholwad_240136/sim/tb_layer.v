module tb_layer;

reg clk = 0;
always #5 clk = ~clk;

reg rst_n = 0;
reg start = 0;
reg signed [15:0] data_in;
reg last = 0;

wire [15:0] out0, out1, out2, out3, out4, out5, out6, out7;
wire [7:0] valid;

layer dut (
    .clk(clk),
    .rst_n(rst_n),
    .start(start),
    .data_in(data_in),
    .last(last),
    .out0(out0), .out1(out1), .out2(out2), .out3(out3),
    .out4(out4), .out5(out5), .out6(out6), .out7(out7),
    .valid(valid)
);

initial begin
    $display("Layer Testbench Start");

    // Reset
    rst_n = 0;
    #10 rst_n = 1;

    // Provide 4 inputs
    // Input 1
    data_in = 16'sd256; start = 1;
    #10 start = 0;

    // Input 2
    data_in = 16'sd128;
    #10;

    // Input 3
    data_in = -16'sd64;
    #10;

    // Input 4 (last)
    data_in = 16'sd32; last = 1;
    #10 last = 0;

    // valid is registered — it appears on the NEXT posedge after last.
    // We already advanced one clock above (#10); wait one more to be safe.
    #10;
    $display("Valid signals after last: %b", valid);
    if (valid == 8'b11111111) begin
        $display("PASS: All valid signals are high after last");
    end else begin
        $display("FAIL: Not all valid signals are high after last");
    end

    // Wait for outputs to stabilize
    #50;

    // Check results
    $display("Valid signals: %b", valid);
    $display("Outputs: %d %d %d %d %d %d %d %d",
             out0, out1, out2, out3, out4, out5, out6, out7);

    // ReLU can produce zeros — require activity and diversity, not all-positive outputs
    if (out0 != 0 || out1 != 0 || out2 != 0 || out3 != 0 ||
        out4 != 0 || out5 != 0 || out6 != 0 || out7 != 0) begin
        $display("PASS: At least one neuron output is non-zero");
    end else begin
        $display("FAIL: All outputs are zero");
    end

    if (!(out0 == out1 && out1 == out2 && out2 == out3 && out3 == out4 &&
          out4 == out5 && out5 == out6 && out6 == out7)) begin
        $display("PASS: Outputs differ across neurons");
    end else begin
        $display("FAIL: All outputs are identical");
    end

    $display("Layer Testbench End");
    #20 $finish;
end

endmodule