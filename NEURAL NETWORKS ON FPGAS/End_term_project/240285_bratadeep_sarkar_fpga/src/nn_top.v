`timescale 1ns / 1ps

module nn_top #(
      parameter INPUT_SIZE = 4,
      parameter HIDDEN_SIZE = 8,
      parameter OUTPUT_SIZE = 3,
      parameter DATA_WIDTH = 16,
      parameter FRAC_WIDTH = 8
)(
      input wire clk,
      input wire rst,
  input wire [DATA_WIDTH*INPUT_SIZE-1:0] input_data,
      input wire valid_in,
  output reg [DATA_WIDTH*OUTPUT_SIZE-1:0] output_data,
      output reg valid_out
);

      // Internal wires for layer connections
  wire [DATA_WIDTH*HIDDEN_SIZE-1:0] hidden_layer_out;
      wire hidden_valid;
  wire [DATA_WIDTH*OUTPUT_SIZE-1:0] output_data_wire;
      wire output_valid_wire;

      // Instantiate Hidden Layer
  layer #(
    .INPUT_SIZE(INPUT_SIZE),
    .OUTPUT_SIZE(HIDDEN_SIZE),
    .DATA_WIDTH(DATA_WIDTH),
    .FRAC_WIDTH(FRAC_WIDTH),
    .WEIGHT_FILE("hidden_weights.mem"),
    .BIAS_FILE("hidden_biases.mem")
  ) hidden_layer (
    .clk(clk),
    .rst(rst),
    .input_data(input_data),
    .valid_in(valid_in),
    .output_data(hidden_layer_out),
    .valid_out(hidden_valid)
  );

      // Instantiate Output Layer
  layer #(
    .INPUT_SIZE(HIDDEN_SIZE),
    .OUTPUT_SIZE(OUTPUT_SIZE),
    .DATA_WIDTH(DATA_WIDTH),
    .FRAC_WIDTH(FRAC_WIDTH),
    .WEIGHT_FILE("output_weights.mem"),
    .BIAS_FILE("output_biases.mem")
  ) output_layer (
    .clk(clk),
    .rst(rst),
    .input_data(hidden_layer_out),
    .valid_in(hidden_valid),
    .output_data(output_data_wire),
    .valid_out(output_valid_wire)
  );

  always @(posedge clk) begin
    if (rst) begin
                  output_data <= 0;
                  valid_out <= 0;
    end else begin
                  output_data <= output_data_wire;
                  valid_out <= output_valid_wire;
    end
  end

endmodule
