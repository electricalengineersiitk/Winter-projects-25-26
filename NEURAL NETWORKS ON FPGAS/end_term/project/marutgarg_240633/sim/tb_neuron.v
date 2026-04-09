module tb_neuron;

reg clk;
reg rst_n;
reg start;
reg [15:0] data_in;
reg [15:0] weight_in;
reg [15:0] bias;
reg last;

wire [15:0] out;
wire valid;

neuron uut (
    .clk(clk),
    .rst_n(rst_n),
    .start(start),
    .data_in(data_in),
    .weight_in(weight_in),
    .bias(bias),
    .last(last),
    .out(out),
    .valid(valid)
);

always #5 clk = ~clk;

initial begin
    clk = 0;
    rst_n = 0;
    start = 0;
    data_in = 0;
    weight_in = 0;
    bias = 0;
    last = 0;

    #10 rst_n = 1;

    start = 1;
    #10 start = 0;

    data_in = 16'd2;
    weight_in = 16'd3;
    #10;

    data_in = 16'd4;
    weight_in = 16'd5;
    last = 1;
    #10;

    last = 0;

    #50;

    $finish;
end

endmodule