// tb_nn_top — stimulus protocol for nn_top
//
// Clock: 10 ns period (#5 half-period) => 100 MHz style timing.
// Hidden layer expects 4 sequential samples on data_in (one per clock after start).
//   Cycle 0: assert start=1 with first sample; then start=0 for remaining beats.
//   Last of the 4: assert h_last=1 for one cycle with that sample, then h_last=0.
// After hidden MAC completes, nn_top runs the output layer and pulses done; class_out
// holds 0..2. done_seen latches any cycle where done==1 for the final check.

module tb_nn_top;

reg clk = 0;
always #5 clk = ~clk;

reg rst_n = 0;
reg start = 0;
reg signed [15:0] data_in;
reg h_last = 0;

wire [1:0] class_out;
wire done;

reg done_seen;

nn_top dut (
    .clk(clk),
    .rst_n(rst_n),
    .start(start),
    .data_in(data_in),
    .h_last(h_last),
    .class_out(class_out),
    .done(done)
);

// Latch done signal
always @(posedge clk) begin
    if (done)
        done_seen <= 1;
end

initial begin
    $display("====== Neural Network Top-Level Test ======\n");

    rst_n = 0;
    done_seen = 0;
    #10 rst_n = 1;

    $display("Starting classification...");

    // Feed 4 inputs to hidden layer
    data_in = 16'sd256; start = 1;
    #10 start = 0;

    data_in = 16'sd128;
    #10;

    data_in = -16'sd64;
    #10;

    data_in = 16'sd32; h_last = 1;
    #10 h_last = 0;

    // Wait for computation to complete (max 10050 time units)
    #10050;

    if (done_seen) begin
        $display("PASS: Done signal asserted");
        $display("Predicted class: %d", class_out);
        
        if (class_out >= 0 && class_out <= 2)
            $display("PASS: Class is valid (0-2)");
        else
            $display("FAIL: Class out of range");
    end else begin
        $display("FAIL: Done signal never asserted");
    end

    $display("\n====== Test Complete ======\n");
    #20 $finish;
end

endmodule