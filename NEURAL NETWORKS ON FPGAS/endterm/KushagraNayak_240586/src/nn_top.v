`timescale 1ns / 1ps

module nn_top (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [15:0] feature_in,
    input  wire [1:0]  feature_idx,
    input  wire        feature_last,
    output reg  [1:0]  predicted_class,
    output reg         done
);

    // Wires from hidden layer
    wire [15:0] h_out [0:7];
    wire hidden_valid;

    // Instantiate the hidden layer
    layer hidden_layer (
        .clk(clk), .rst_n(rst_n), .start(start),
        .data_in(feature_in), .input_idx(feature_idx), .last(feature_last),
        .out_0(h_out[0]), .out_1(h_out[1]), .out_2(h_out[2]), .out_3(h_out[3]),
        .out_4(h_out[4]), .out_5(h_out[5]), .out_6(h_out[6]), .out_7(h_out[7]),
        .valid(hidden_valid)
    );

    // Simplified output layer to ensure logic is not optimized away during synthesis
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            predicted_class <= 2'd0;
            done <= 1'b0;
        end else if (hidden_valid) begin
            // Argmax calculation based on the first 3 neurons for 3 flower classes
            if (h_out[0] >= h_out[1] && h_out[0] >= h_out[2]) begin
                predicted_class <= 2'd0;
            end else if (h_out[1] >= h_out[0] && h_out[1] >= h_out[2]) begin
                predicted_class <= 2'd1;
            end else begin
                predicted_class <= 2'd2;
            end
            done <= 1'b1;
        end else begin
            done <= 1'b0;
        end
    end

endmodule