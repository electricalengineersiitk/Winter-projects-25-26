`timescale 1ns / 1ps

module tb_nn_top;

    reg        clk;
    reg        rst_n;
    reg        start;
    wire [1:0] predicted_class;
    wire       done;

    nn_top uut (
        .clk             (clk),
        .rst_n           (rst_n),
        .start           (start),
        .predicted_class (predicted_class),
        .done            (done)
    );

    // 100 MHz clock
    always #5 clk = ~clk;

    initial begin
        clk   = 0;
        rst_n = 0;
        start = 0;

        #100;
        rst_n = 1;
        #20;

        $display("========================================");
        $display(" nn_top Integration Test");
        $display(" Test input: first sample from test_data.mem");
        $display("========================================");
        $display("");

        // Pulse start
        @(posedge clk); #1;
        start = 1;
        @(posedge clk); #1;
        start = 0;

        // Wait for done (timeout after 500 cycles)
        begin : wait_loop
            repeat (500) begin
                @(posedge clk); #1;
                if (done) disable wait_loop;
            end
        end

        if (done) begin
            $display("Inference COMPLETE.");
            $display("  Predicted class: %0d", predicted_class);
            $display("  (Compare with expected label from test_data.mem line 5)");
        end else begin
            $display("TIMEOUT: done never asserted after 500 cycles.");
            $display("  Check FSM logic in nn_top.v");
        end

        $display("");
        $display("========================================");
        #100;
        $finish;
    end

endmodule
