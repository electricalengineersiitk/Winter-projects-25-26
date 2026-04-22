
`timescale 1ns / 1ps
`include "layer.v"

module two_layer_network
#(
    parameter n_layer1 = 8,
    parameter n_layer2 = 3,
    parameter n_inputs = 4
)
(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [15:0] data_in    [0:n_inputs-1],

    output reg  [15:0] out_f  [0:n_layer2-1],
    output reg         valid_f[0:n_layer2-1]
);

// ── Internal wires between the two layers ────────────────────────────────────
reg [15:0] weight_mem [0:(n_layer1*n_inputs + n_layer2*n_layer1)-1]; // [0:55]
reg [15:0] bias_mem   [0:(n_layer1 + n_layer2)-1];                  // [0:10]

initial begin
    $readmemh("weights.mem", weight_mem);
    $readmemh("biases.mem",    bias_mem);
end



wire [15:0] out_l1   [0:n_layer1-1];
wire        valid_l1 [0:n_layer1-1];

// Weight/bias slices for each layer
wire [15:0] weight_l1 [0:(n_layer1*n_inputs)-1];
wire [15:0] weight_l2 [0:(n_layer2*n_layer1)-1];
wire [15:0] bias_l1   [0:n_layer1-1];
wire [15:0] bias_l2   [0:n_layer2-1];

genvar i;

// Slice weight_mem and bias_mem for Layer 1
generate
    for (i = 0; i < n_layer1*n_inputs; i = i+1) begin : wl1
        assign weight_l1[i] = weight_mem[i];
    end
    for (i = 0; i < n_layer1; i = i+1) begin : bl1
        assign bias_l1[i] = bias_mem[i];
    end
endgenerate

// Slice weight_mem and bias_mem for Layer 2
generate
    for (i = 0; i < n_layer2*n_layer1; i = i+1) begin : wl2
        assign weight_l2[i] = weight_mem[n_layer1*n_inputs + i];
    end
    for (i = 0; i < n_layer2; i = i+1) begin : bl2
        assign bias_l2[i] = bias_mem[n_layer1 + i];
    end
endgenerate

// ── Layer 1 instantiation ─────────────────────────────────────────────────────

layer_of_neurons #(
    .n_layer  (n_layer1),
    .n_inputs (n_inputs)
) u_layer1 (
    .clk       (clk),
    .rst_n     (rst_n),
    .start     (start),
    .data_in   (data_in),
    .weight_in (weight_l1),
    .bias      (bias_l1),
    .out       (out_l1),
    .valid     (valid_l1)
);

// ── Valid synchronization: wait until ALL Layer 1 outputs are valid ───────────

wire all_valid_l1;

generate
    // AND-reduce all valid_l1 bits into one signal
    wire [n_layer1-1:0] valid_l1_packed;
    for (i = 0; i < n_layer1; i = i+1) begin : vpack
        assign valid_l1_packed[i] = valid_l1[i];
    end
endgenerate

assign all_valid_l1 = &valid_l1_packed; // reduction AND

// Register the start pulse for Layer 2 (one-cycle pulse on rising edge)
reg all_valid_l1_prev;
reg start_l2;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        all_valid_l1_prev <= 1'b0;
        start_l2          <= 1'b0;
    end else begin
        all_valid_l1_prev <= all_valid_l1;
        // Single-cycle start pulse when all of Layer 1 becomes valid
        start_l2 <= all_valid_l1 & ~all_valid_l1_prev;
    end
end

// ── Layer 2 instantiation ─────────────────────────────────────────────────────

wire [15:0] out_l2   [0:n_layer2-1];
wire        valid_l2 [0:n_layer2-1];

layer_of_neurons #(
    .n_layer  (n_layer2),
    .n_inputs (n_layer1)
) u_layer2 (
    .clk       (clk),
    .rst_n     (rst_n),
    .start     (start_l2),
    .data_in   (out_l1),       // Layer 1 outputs feed directly in
    .weight_in (weight_l2),
    .bias      (bias_l2),
    .out       (out_l2),
    .valid     (valid_l2)
);

// ── Register outputs ──────────────────────────────────────────────────────────

integer j;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        for (j = 0; j < n_layer2; j = j+1) begin
            out_f[j]   <= 16'h0;
            valid_f[j] <= 1'b0;
        end
    end else begin
        for (j = 0; j < n_layer2; j = j+1) begin
            out_f[j]   <= out_l2[j];
            valid_f[j] <= valid_l2[j];
        end
    end
end

endmodule