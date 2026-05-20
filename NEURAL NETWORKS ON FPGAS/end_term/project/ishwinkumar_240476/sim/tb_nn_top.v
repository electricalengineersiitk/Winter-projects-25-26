`timescale 1ns / 1ps

module tb_nn_top;

    reg clk;
    reg rst_n;
    reg start;
    reg [15:0] data_in;
    reg last;

    wire [1:0] pred_class;
    wire done;

    // 🔹 DUT
    nn_top uut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .data_in(data_in),
        .last(last),
        .pred_class(pred_class),
        .done(done)
    );

    // 🔹 Test data
    reg [15:0] test_data [0:100];
    integer i;
    integer base;

    // 🔹 Clock (10ns period)
    always #5 clk = ~clk;

    initial begin
        clk = 0;
        rst_n = 0;
        start = 0;
        last = 0;
        data_in = 0;

        // load inputs
        $readmemh("test_data.mem", test_data);

        // =====================
        // RESET
        // =====================
        @(posedge clk);
        rst_n = 1;

        // =====================
        // START SIGNAL
        // =====================
        @(posedge clk);
        start = 1;

        @(posedge clk);
        start = 0;

        // =====================
        // FEED INPUTS (4 values)
        // =====================
        base = 0;
        for (i = 0; i < 4; i = i + 1) begin
            data_in = test_data[base + i];
            last    = (i == 3);
            @(posedge clk);
        end

        // clear last
        @(posedge clk);
        last = 0;

        // =====================
        // WAIT FOR OUTPUT
        // =====================
        wait(done);

        $display("=================================");
        $display("Prediction = %d", pred_class);
        $display("=================================");
        $display("acc0=%d acc1=%d acc2=%d", uut.acc0, uut.acc1, uut.acc2);

        #20;
        $finish;
    end

endmodule