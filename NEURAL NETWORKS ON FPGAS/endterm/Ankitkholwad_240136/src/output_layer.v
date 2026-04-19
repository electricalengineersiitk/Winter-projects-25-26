// 3-class output stage: three neurons over 8 hidden activations, argmax, done pulse.
module output_layer (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               hidden_done,
    input  wire signed [15:0] h0,
    input  wire signed [15:0] h1,
    input  wire signed [15:0] h2,
    input  wire signed [15:0] h3,
    input  wire signed [15:0] h4,
    input  wire signed [15:0] h5,
    input  wire signed [15:0] h6,
    input  wire signed [15:0] h7,
    output reg  [1:0]         class_out,
    output reg                done
);

    reg signed [15:0] output_weights_mem [0:23];
    reg signed [15:0] output_bias_mem [0:2];

    initial begin
        $readmemh("w2_weights.mem", output_weights_mem);
        $readmemh("w2_bias.mem", output_bias_mem);
    end

    reg [2:0] output_hidden_index;
    reg       output_started;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            output_hidden_index <= 0;
            output_started      <= 0;
        end else begin
            if (hidden_done && !output_started) begin
                output_started      <= 1;
                output_hidden_index <= 0;
            end else if (output_started) begin
                if (output_hidden_index < 7)
                    output_hidden_index <= output_hidden_index + 1;
                else
                    output_started <= 0;
            end
        end
    end

    wire signed [15:0] current_hidden_input;
    assign current_hidden_input =
        (output_hidden_index == 0) ? h0 :
        (output_hidden_index == 1) ? h1 :
        (output_hidden_index == 2) ? h2 :
        (output_hidden_index == 3) ? h3 :
        (output_hidden_index == 4) ? h4 :
        (output_hidden_index == 5) ? h5 :
        (output_hidden_index == 6) ? h6 : h7;

    wire signed [15:0] out0, out1, out2;
    wire [2:0]         output_valid;

    neuron_linear output_neuron0 (
        .clk(clk),
        .rst_n(rst_n),
        .start(output_started && output_hidden_index == 0),
        .data_in(current_hidden_input),
        .weight_in(output_weights_mem[0*8 + output_hidden_index]),
        .bias(output_bias_mem[0]),
        .last(output_started && output_hidden_index == 7),
        .out(out0),
        .valid(output_valid[0])
    );

    neuron_linear output_neuron1 (
        .clk(clk),
        .rst_n(rst_n),
        .start(output_started && output_hidden_index == 0),
        .data_in(current_hidden_input),
        .weight_in(output_weights_mem[1*8 + output_hidden_index]),
        .bias(output_bias_mem[1]),
        .last(output_started && output_hidden_index == 7),
        .out(out1),
        .valid(output_valid[1])
    );

    neuron_linear output_neuron2 (
        .clk(clk),
        .rst_n(rst_n),
        .start(output_started && output_hidden_index == 0),
        .data_in(current_hidden_input),
        .weight_in(output_weights_mem[2*8 + output_hidden_index]),
        .bias(output_bias_mem[2]),
        .last(output_started && output_hidden_index == 7),
        .out(out2),
        .valid(output_valid[2])
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            class_out <= 0;
            done      <= 0;
        end else begin
            done <= 0;

            if (output_valid == 3'b111) begin
                if (out0 >= out1 && out0 >= out2)
                    class_out <= 2'b00;
                else if (out1 >= out2)
                    class_out <= 2'b01;
                else
                    class_out <= 2'b10;

                done <= 1;
            end
        end
    end

endmodule
