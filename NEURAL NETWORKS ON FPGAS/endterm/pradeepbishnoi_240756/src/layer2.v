module layer2 (
    input  wire         clk,
    input  wire         rst_n,
    input  wire [15:0]  data_in,
    input  wire         start,
    input  wire         last,
    output wire [47:0]  layer_out,   // 3 x 16-bit
    output wire [2:0]   layer_valid  // 3 valid bits
);
    // Full weight file loaded - same files as layer 1
    reg [15:0] weight_mem [0:55];   // all 56 weights  
    reg [15:0] bias_mem   [0:10];   // all 11 biases

    initial begin
        #1;
        $readmemh("weights.mem", weight_mem);
        $readmemh("biases.mem",  bias_mem);
    end

    // Input counter 0-7
    reg [2:0] input_idx;
    reg       active_internal;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            input_idx       <= 3'd0;
            active_internal <= 1'b0;
        end else begin
            if (start) begin
                input_idx       <= 3'd0;
                active_internal <= 1'b1;
            end else if (last) begin
                active_internal <= 1'b0;
            end else if (active_internal) begin
                input_idx <= input_idx + 1'b1;
            end
        end
    end

    // Weight register — 3 neurons, each needs one weight per cycle
    // Layer 2 weights start at index 32 in weight_mem
    // Layout: neuron k, input j → weight_mem[32 + k*8 + j]
    reg [15:0] weight_reg [0:2];
    integer k;

    initial begin
        #1;
        for (k = 0; k < 3; k = k + 1)
            weight_reg[k] = weight_mem[32 + k*8 + 0];
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (k = 0; k < 3; k = k + 1)
                weight_reg[k] <= 16'd0;
        end else begin
            if (start) begin
                for (k = 0; k < 3; k = k + 1)
                    weight_reg[k] <= weight_mem[32 + k*8 + 0];
            end else if (active_internal && !last) begin
                for (k = 0; k < 3; k = k + 1)
                    weight_reg[k] <= weight_mem[32 + k*8 + (input_idx + 1)];
            end
        end
    end

    // 3 parallel output neurons
    // Biases for layer 2 start at index 8 in bias_mem
    genvar i;
    generate
        for (i = 0; i < 3; i = i + 1) begin : neuron_block
            neuron n_inst (
                .clk      (clk),
                .rst_n    (rst_n),
                .data_in  (data_in),
                .weight_in(weight_reg[i]),
                .bias     (bias_mem[8 + i]),   // offset by 8
                .start    (start),
                .last     (last),
                .out      (layer_out[i*16 +: 16]),
                .valid    (layer_valid[i])
            );
        end
    endgenerate

endmodule
