module layer(
    input clk,
    input rst_n,
    input start,
    input [15:0] data_in,
    input last,
    output [15:0] out0,
    output [15:0] out1,
    output [15:0] out2,
    output [15:0] out3,
    output [15:0] out4,
    output [15:0] out5,
    output [15:0] out6,
    output [15:0] out7,
    output valid
);

reg [15:0] weight_mem [0:31];

initial begin
    $readmemh("weights.mem", weight_mem);
end

wire [15:0] bias = 16'd0;

neuron n0(clk, rst_n, start, data_in, weight_mem[0], bias, last, out0, valid);
neuron n1(clk, rst_n, start, data_in, weight_mem[1], bias, last, out1, valid);
neuron n2(clk, rst_n, start, data_in, weight_mem[2], bias, last, out2, valid);
neuron n3(clk, rst_n, start, data_in, weight_mem[3], bias, last, out3, valid);
neuron n4(clk, rst_n, start, data_in, weight_mem[4], bias, last, out4, valid);
neuron n5(clk, rst_n, start, data_in, weight_mem[5], bias, last, out5, valid);
neuron n6(clk, rst_n, start, data_in, weight_mem[6], bias, last, out6, valid);
neuron n7(clk, rst_n, start, data_in, weight_mem[7], bias, last, out7, valid);

endmodule
