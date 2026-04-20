`timescale 1ns/1ps
`include "top_nn.v"

module top_nn_tb;

    reg        clk, rst_n, start;
    reg [15:0] data_in    [0:3];

    wire [15:0] out_f   [0:2];
    wire        valid_f [0:2];

    two_layer_network dut (
        .clk(clk), .rst_n(rst_n), .start(start),
        .data_in(data_in),
        .out_f(out_f), .valid_f(valid_f)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    initial begin
        // Reset
        rst_n = 0; start = 0;
        repeat(3) @(posedge clk);
        rst_n = 1;

        // Your 4 inputs — change these to whatever you want
        data_in[0] = 16'd1024;
        data_in[1] = 16'd1024;
        data_in[2] = 16'd3072;
        data_in[3] = 16'd4096;

        // Fire it
        @(negedge clk); start = 1;
        @(negedge clk); start = 0;

        // Wait for result
        wait(valid_f[0] && valid_f[1] && valid_f[2]);
        @(posedge clk);

        $display("out_f[0] = %0d", $signed(out_f[0]));
        $display("out_f[1] = %0d", $signed(out_f[1]));
        $display("out_f[2] = %0d", $signed(out_f[2]));

        $finish;
    end

endmodule