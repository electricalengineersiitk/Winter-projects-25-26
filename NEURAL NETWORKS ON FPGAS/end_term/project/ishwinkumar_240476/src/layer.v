`timescale 1ns / 1ps

module layer (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [15:0] data_in,
    input  wire        last,

    output wire [15:0] out0,
    output wire [15:0] out1,
    output wire [15:0] out2,
    output wire [15:0] out3,
    output wire [15:0] out4,
    output wire [15:0] out5,
    output wire [15:0] out6,
    output wire [15:0] out7,

    output wire        valid
);

    // 🔹 Memories
    reg [15:0] weight_mem [0:31]; // 8 neurons × 4 inputs
    reg [15:0] bias_mem   [0:7];

    initial begin
        $readmemh("weights.mem", weight_mem);
        $readmemh("biases.mem", bias_mem);
    end

    // 🔹 Input index counter (0→3)
    reg [1:0] idx;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            idx <= 0;
        else if (start)
            idx <= 0;
        else
            idx <= idx + 1;
    end

    // 🔹 Outputs
    wire [15:0] out_w [0:7];
    wire [7:0]  valid_w;

    // 🔥 Instantiate 8 neurons
    genvar j;
    generate
        for (j = 0; j < 8; j = j + 1) begin : NEURONS

            neuron n (
                .clk(clk),
                .rst_n(rst_n),
                .start(start),
                .data_in(data_in),
                .weight_in(weight_mem[j*4 + idx]), // ⭐ key mapping
                .bias(bias_mem[j]),
                .last(last),
                .out(out_w[j]),
                .valid(valid_w[j])
            );

        end
    endgenerate

    // 🔹 Assign outputs
    assign out0 = out_w[0];
    assign out1 = out_w[1];
    assign out2 = out_w[2];
    assign out3 = out_w[3];
    assign out4 = out_w[4];
    assign out5 = out_w[5];
    assign out6 = out_w[6];
    assign out7 = out_w[7];

    // 🔹 Valid when all neurons done
    assign valid = &valid_w;

endmodule