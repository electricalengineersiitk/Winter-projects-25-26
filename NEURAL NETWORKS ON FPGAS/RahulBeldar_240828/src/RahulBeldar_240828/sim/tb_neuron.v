module tb_neuron;

reg clk = 0;
always #5 clk = ~clk;

reg rst_n;
reg start;
reg [15:0] data_in, weight_in, bias;
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

initial begin
    rst_n = 0; start = 0; last = 0;
    #10 rst_n = 1;

    bias = 16'd10;

    start = 1; #10 start = 0;
    data_in = 16'd2; weight_in = 16'd3; last = 0; #10;
    data_in = 16'd4; weight_in = 16'd2; last = 0; #10;
    data_in = 16'd1; weight_in = 16'd5; last = 1; #10;

    #20 $finish;
end

endmodule
