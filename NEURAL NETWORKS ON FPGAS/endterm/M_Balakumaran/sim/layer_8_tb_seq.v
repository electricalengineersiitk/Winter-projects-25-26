`timescale 1ns / 1ps
`include "layer_8.v"


module layer_8_tb_seq();

    parameter n_layer = 8;
    parameter n_inputs = 4;
   
    reg clk;
    reg rst_n;
    reg start;
    reg [15:0] data_in [0:n_inputs-1];
    reg [15:0] weight_mem [0:55];  // 32 weights for layer 1 + 24 weights for layer 2
    reg [15:0] bias_mem [0:10];    // 8 biases for layer 1 + 3 biases for layer 2
    
    // Arrays sized for the instantiated layer (8 neurons x 4 inputs)
    reg [15:0] weight_in [0:(n_layer*n_inputs)-1];
    reg [15:0] bias [0:n_layer-1];

    wire [15:0] out [0:n_layer-1];
    wire valid [0:n_layer-1];

    // Load weights and biases from files
    initial begin
        $readmemh("weights/weights.mem", weight_mem);
        $readmemh("weights/biases.mem", bias_mem);
        
        // Copy first layer weights and biases to the correctly sized arrays
        for (integer i = 0; i < (n_layer * n_inputs); i = i + 1) begin
            weight_in[i] = weight_mem[i];
        end
        for (integer i = 0; i < n_layer; i = i + 1) begin
            bias[i] = bias_mem[i];
        end
    end

    // Instantiate the layer module
    layer_of_neurons #(.n_layer(n_layer), .n_inputs(n_inputs)) uut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .data_in(data_in),
        .weight_in(weight_in),
        .bias(bias),
        .out(out),
        .valid(valid)
    );

    // Clock Generation (100MHz)
    always #5 clk = ~clk;

    initial begin
        // Create VCD file for waveform viewing
        $dumpfile("layer_8_seq_sim.vcd");
        $dumpvars(0, layer_8_tb_seq); 
    end

    // Stimulus Procedure
    initial begin
        // Initialize Inputs
        clk = 0;
        rst_n = 0;
        start = 0;
        
        // Initialize data_in array with test values (all 1024 in Q8 format)
        for (integer i = 0; i < n_inputs; i = i + 1) begin
            data_in[i] = 16'd1024 * (i+1);
        end

        // Release reset
        #20 rst_n = 1;
        #10;

        // Send start pulse to begin sequential processing
        $display("=== Starting Sequential Layer Processing ===");
        @(posedge clk);
        start = 1;
        
        @(posedge clk);
        start = 0;

        // Wait for processing to complete (all inputs processed)
        // The module will automatically sequence through all 4 inputs
        // and assert generated_last on the 4th input
        
        // Wait for valid signal from last neuron
        wait(valid[0]);
        
        $display("=== Layer 8 Output Results ===");
        for (integer i = 0; i < n_layer; i = i + 1) begin
            $display("Neuron[%0d]: Output = %d (Valid = %b)", i, out[i], valid[i]);
        end

        #100;
        $finish;
    end

endmodule
