`timescale 1ns / 1ps

module nn_top (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [15:0] data_in,
    input  wire        last,

    output reg  [1:0]  pred_class,
    output reg         done
);

    // 🔹 Hidden layer outputs
    wire [15:0] h0,h1,h2,h3,h4,h5,h6,h7;
    wire h_valid;

    // 🔹 Hidden layer instance (USES your layer.v + neuron.v)
    layer hidden (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .data_in(data_in),
        .last(last),

        .out0(h0), .out1(h1), .out2(h2), .out3(h3),
        .out4(h4), .out5(h5), .out6(h6), .out7(h7),
        .valid(h_valid)
    );

    // 🔹 Shared weight memory (hidden uses 0-31, output uses 32-55)
    reg [15:0] weight_mem [0:100];
    reg [15:0] out_bias   [0:20];

    initial begin
        $readmemh("weights.mem", weight_mem);
        $readmemh("biases.mem", out_bias);
    end

    // 🔹 Accumulators
    reg signed [31:0] acc0, acc1, acc2;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc0 <= 0;
            acc1 <= 0;
            acc2 <= 0;
            done <= 0;
        end else begin
            done <= 0;

            if (h_valid) begin
                // reset accumulators (blocking!)
                acc0 = 0;
                acc1 = 0;
                acc2 = 0;

                // =========================
                // OUTPUT LAYER (3 neurons)
                // =========================

                // neuron 0 → weights[32-39]
                acc0 = acc0 + h0 * weight_mem[32];
                acc0 = acc0 + h1 * weight_mem[33];
                acc0 = acc0 + h2 * weight_mem[34];
                acc0 = acc0 + h3 * weight_mem[35];
                acc0 = acc0 + h4 * weight_mem[36];
                acc0 = acc0 + h5 * weight_mem[37];
                acc0 = acc0 + h6 * weight_mem[38];
                acc0 = acc0 + h7 * weight_mem[39];

                // neuron 1 → weights[40-47]
                acc1 = acc1 + h0 * weight_mem[40];
                acc1 = acc1 + h1 * weight_mem[41];
                acc1 = acc1 + h2 * weight_mem[42];
                acc1 = acc1 + h3 * weight_mem[43];
                acc1 = acc1 + h4 * weight_mem[44];
                acc1 = acc1 + h5 * weight_mem[45];
                acc1 = acc1 + h6 * weight_mem[46];
                acc1 = acc1 + h7 * weight_mem[47];

                // neuron 2 → weights[48-55]
                acc2 = acc2 + h0 * weight_mem[48];
                acc2 = acc2 + h1 * weight_mem[49];
                acc2 = acc2 + h2 * weight_mem[50];
                acc2 = acc2 + h3 * weight_mem[51];
                acc2 = acc2 + h4 * weight_mem[52];
                acc2 = acc2 + h5 * weight_mem[53];
                acc2 = acc2 + h6 * weight_mem[54];
                acc2 = acc2 + h7 * weight_mem[55];

                // 🔹 Add bias (fixed-point scaling)
                acc0 = acc0 + ($signed(out_bias[0]) <<< 8);
                acc1 = acc1 + ($signed(out_bias[1]) <<< 8);
                acc2 = acc2 + ($signed(out_bias[2]) <<< 8);


                if (acc0 >= acc1 && acc0 >= acc2)
                    pred_class <= 0;
                else if (acc1 >= acc2 && acc1>=acc0)
                    pred_class <= 1;
                else
                    pred_class <= 2;
                done <= 1;
            end
        end
    end

    // 🔹 Argmax (final prediction)
//    always @(posedge clk or negedge rst_n) begin
//        if (!rst_n)
//            pred_class <= 0;
//        else if (done) begin
//            if (acc0 >= acc1 && acc0 >= acc2)
//                pred_class <= 0;
//            else if (acc1 >= acc2 && acc1>=acc0)
//                pred_class <= 1;
//            else
//                pred_class <= 2;
//        end
//    end

endmodule