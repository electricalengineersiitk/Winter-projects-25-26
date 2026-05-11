module layer (
    input clk,
    input rst_n,
    input start,
    input [15:0] data_in,
    input last,
    output [15:0] out [7:0],
    output valid
);

genvar i;

generate
    for (i = 0; i < 8; i = i + 1) begin : neurons
        neuron n (
            .clk(clk),
            .rst_n(rst_n),
            .start(start),
            .data_in(data_in),
            .weight_in(16'd1),
            .bias(16'd1),
            .last(last),
            .out(out[i]),
            .valid()
        );
    end
endgenerate

assign valid = last;

endmodule
