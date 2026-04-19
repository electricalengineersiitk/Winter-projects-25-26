module layer (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire signed [15:0] data_in,
    input  wire        last,
    output wire signed [15:0] out0,
    output wire signed [15:0] out1,
    output wire signed [15:0] out2,
    output wire signed [15:0] out3,
    output wire signed [15:0] out4,
    output wire signed [15:0] out5,
    output wire signed [15:0] out6,
    output wire signed [15:0] out7,
    output wire [7:0]  valid
);

    reg signed [15:0] weight_mem [0:31];
    reg signed [15:0] bias_mem [0:7];

    initial begin
        $readmemh("weights.mem", weight_mem);
        $readmemh("w1_bias.mem", bias_mem);
    end

    reg [1:0] input_index;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            input_index <= 0;
        else if (start)
            input_index <= 0;
        else
            input_index <= input_index + 1;
    end

    neuron neuron0 (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .data_in(data_in),
        .weight_in(weight_mem[0*4 + input_index]),
        .bias(bias_mem[0]),
        .last(last),
        .out(out0),
        .valid(valid[0])
    );

    neuron neuron1 (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .data_in(data_in),
        .weight_in(weight_mem[1*4 + input_index]),
        .bias(bias_mem[1]),
        .last(last),
        .out(out1),
        .valid(valid[1])
    );

    neuron neuron2 (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .data_in(data_in),
        .weight_in(weight_mem[2*4 + input_index]),
        .bias(bias_mem[2]),
        .last(last),
        .out(out2),
        .valid(valid[2])
    );

    neuron neuron3 (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .data_in(data_in),
        .weight_in(weight_mem[3*4 + input_index]),
        .bias(bias_mem[3]),
        .last(last),
        .out(out3),
        .valid(valid[3])
    );

    neuron neuron4 (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .data_in(data_in),
        .weight_in(weight_mem[4*4 + input_index]),
        .bias(bias_mem[4]),
        .last(last),
        .out(out4),
        .valid(valid[4])
    );

    neuron neuron5 (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .data_in(data_in),
        .weight_in(weight_mem[5*4 + input_index]),
        .bias(bias_mem[5]),
        .last(last),
        .out(out5),
        .valid(valid[5])
    );

    neuron neuron6 (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .data_in(data_in),
        .weight_in(weight_mem[6*4 + input_index]),
        .bias(bias_mem[6]),
        .last(last),
        .out(out6),
        .valid(valid[6])
    );

    neuron neuron7 (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .data_in(data_in),
        .weight_in(weight_mem[7*4 + input_index]),
        .bias(bias_mem[7]),
        .last(last),
        .out(out7),
        .valid(valid[7])
    );

endmodule
