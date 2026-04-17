module tb_layer;

reg clk;
reg rst_n;
reg start;
reg [15:0] data_in;
reg last;

wire [15:0] out0, out1, out2, out3, out4, out5, out6, out7;
wire valid;

layer uut (
    .clk(clk),
    .rst_n(rst_n),
    .start(start),
    .data_in(data_in),
    .last(last),
    .out0(out0),
    .out1(out1),
    .out2(out2),
    .out3(out3),
    .out4(out4),
    .out5(out5),
    .out6(out6),
    .out7(out7),
    .valid(valid)
);

always #5 clk = ~clk;

initial begin
    clk = 0;
    rst_n = 0;
    start = 0;
    data_in = 0;
    last = 0;

    #10 rst_n = 1;

    start = 1;
    #10 start = 0;

    data_in = 16'd2;
    #10;

    data_in = 16'd4;
    last = 1;
    #10;

    last = 0;

    #100;

    $finish;
end

endmodule