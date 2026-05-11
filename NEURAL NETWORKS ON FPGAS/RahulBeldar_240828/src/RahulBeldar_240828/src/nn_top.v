module nn_top (
    input clk,
    input rst_n,
    input start,
    input [15:0] data_in,
    input last,
    output done
);

wire [15:0] out [7:0];
wire valid;

layer l1 (
    .clk(clk),
    .rst_n(rst_n),
    .start(start),
    .data_in(data_in),
    .last(last),
    .out(out),
    .valid(valid)
);

assign done = valid;

endmodule
