module nn_top(
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

    wire [15:0] layer_out0;
    wire [15:0] layer_out1;
    wire [15:0] layer_out2;
    wire [15:0] layer_out3;
    wire [15:0] layer_out4;
    wire [15:0] layer_out5;
    wire [15:0] layer_out6;
    wire [15:0] layer_out7;

    wire layer_valid;

    layer u_layer(
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .data_in(data_in),
        .last(last),
        .out0(layer_out0),
        .out1(layer_out1),
        .out2(layer_out2),
        .out3(layer_out3),
        .out4(layer_out4),
        .out5(layer_out5),
        .out6(layer_out6),
        .out7(layer_out7),
        .valid(layer_valid)
    );

    assign out0 = layer_out0;
    assign out1 = layer_out1;
    assign out2 = layer_out2;
    assign out3 = layer_out3;
    assign out4 = layer_out4;
    assign out5 = layer_out5;
    assign out6 = layer_out6;
    assign out7 = layer_out7;

    assign valid = layer_valid;

endmodule