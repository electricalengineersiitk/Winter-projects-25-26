`timescale 1ns / 1ps

module tb_layer;

    reg clk;
    reg rst_n;
    reg start;
    reg [15:0] data_in;
    reg last;

    wire [15:0] out0, out1, out2, out3;
    wire [15:0] out4, out5, out6, out7;
    wire valid;

    // DUT
    layer uut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .data_in(data_in),
        .last(last),
        .out0(out0), .out1(out1), .out2(out2), .out3(out3),
        .out4(out4), .out5(out5), .out6(out6), .out7(out7),
        .valid(valid)
    );

    // Test data
    reg [15:0] test_data [0:100];
    integer i;
    integer base;

    // Clock
    always #5 clk = ~clk;

    initial begin
        clk = 0;
        rst_n = 0;
        start = 0;
        last = 0;
        data_in = 0;

        $readmemh("test_data.mem", test_data);

        // Reset
        @(posedge clk);
        rst_n = 1;

        // Start pulse
        @(posedge clk);
        start = 1;

        @(posedge clk);
        start = 0;

        // Feed 4 inputs (IMPORTANT: 1 per cycle)
       
        base =4;
        for (i = 0; i < 4; i = i + 1) begin
            data_in = test_data[base + i];
            last    = (i == 3);
            @(posedge clk);
        end

        // Clear last
        @(posedge clk);
        last = 0;

        // Wait for all neurons
        wait(valid);

        // Print outputs
        $display("---- LAYER OUTPUT ----");
        $display("%d(%h) %d(%h) %d(%h) %d(%h)", out0, out0, out1, out1, out2, out2, out3, out3);
        $display("%d(%h) %d(%h) %d(%h) %d(%h)" , out4, out4, out5, out5, out6, out6, out7, out7);

        #20;
        $finish;
    end

endmodule