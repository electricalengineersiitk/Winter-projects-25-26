module layer (
    input  wire         clk,
    input  wire         rst_n,
    input  wire [15:0]  data_in,
    input  wire         start,
    input  wire         last,
    output wire [127:0] layer_out, 
    output wire [7:0]   layer_valid
);

    // Declaring memory for Weights and Biases
    reg [15:0] weight_mem [0:31]; // 32 weights for layer 1
    reg [15:0] bias_mem   [0:7]; // 8 biases for layer 1 
    
    initial begin
        $readmemh("weights.mem", weight_mem);
        $readmemh("biases.mem",  bias_mem);
        // Pre-populate weight_reg with index-0 weights
        for (k = 0; k < 8; k = k + 1)
            weight_reg[k] = weight_mem[k*4 + 0];
    end

    // Input Counter Logic (0 to 3)
    reg [1:0] input_idx;
    reg active_internal;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            input_idx       <= 2'd0;
            active_internal <= 1'b0;
        end else begin
            if (start) begin
                input_idx       <= 2'd0;
                active_internal <= 1'b1;
            end else if (last) begin
                active_internal <= 1'b0;
            end else if (active_internal) begin        // FIX 2+3: removed < 3 guard
                input_idx <= input_idx + 1'b1;
            end
        end
    end

    // Weight Register — pre-fetch correct weight for each cycle
    reg [15:0] weight_reg [0:7];

    integer k;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (k = 0; k < 8; k = k + 1)
                weight_reg[k] <= weight_mem[k*4 + 0]; // reset to index 0, not zero
        end else begin
            if (start) begin
                for (k = 0; k < 8; k = k + 1)
                    weight_reg[k] <= weight_mem[k*4 + 0]; // ready for next start
            end else if (active_internal && !last) begin
                // Pre-fetch weight for the NEXT input index
                for (k = 0; k < 8; k = k + 1)
                    weight_reg[k] <= weight_mem[k*4 + (input_idx + 1)];
            end
        end
    end

    // Parallel Neuron Instantiation
    genvar i;
    generate
        for (i = 0; i < 8; i = i + 1) begin : neuron_block
            neuron n_inst (
                .clk       (clk),
                .rst_n     (rst_n),
                .data_in   (data_in),
                .weight_in (weight_reg[i]),   // using registered weight
                .bias      (bias_mem[i]),
                .start     (start),
                .last      (last),
                .out       (layer_out[i*16 +: 16]),
                .valid     (layer_valid[i])
            );
        end
    endgenerate
endmodule
