`timescale 1ns / 1ps

module layer (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [15:0] data_in,
    input  wire [1:0]  input_idx, // 0 to 3 for the 4 inputs
    input  wire        last,
    output wire [15:0] out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7,
    output wire        valid
);

    // Load the weights using $readmemh
    reg [15:0] weight_mem [0:31]; // 8 neurons x 4 inputs = 32 weights
    initial $readmemh("weights.mem", weight_mem);

    // Biases for the 8 neurons
    reg [15:0] bias_mem [0:7];
    initial $readmemh("biases.mem", bias_mem);

    wire [7:0] valid_flags;
    assign valid = valid_flags[0]; // All finish at the same time

    // Instantiate 8 copies of your neuron module
    genvar i;
    generate
        for (i = 0; i < 8; i = i + 1) begin : hidden_neurons
            neuron n (
                .clk(clk),
                .rst_n(rst_n),
                .start(start),
                .data_in(data_in),
                // Connect each neuron's weight_in to the right entry in weight_mem based on the current input index
                .weight_in(weight_mem[(i * 4) + input_idx]), 
                .bias(bias_mem[i]),
                .last(last),
                // Route outputs
                .out(
                    (i==0) ? out_0 : (i==1) ? out_1 : (i==2) ? out_2 : (i==3) ? out_3 :
                    (i==4) ? out_4 : (i==5) ? out_5 : (i==6) ? out_6 : out_7
                ),
                .valid(valid_flags[i])
            );
        end
    endgenerate

endmodule