`include "neuron.v"
`timescale 1ns / 1ps


module layer_of_neurons
#(
    parameter n_layer = 8,
    parameter n_inputs = 4
)
(
    input  wire        clk,
    input  wire        rst_n,      
    input  wire        start,      
    input  reg [15:0] data_in[0:n_inputs-1],    
    input  reg [15:0] weight_in [0:(n_layer*n_inputs)-1],  
    input  reg [15:0] bias [0:n_layer-1],     
    output reg [15:0] out [0:n_layer-1],        
    output reg         valid [0:n_layer-1]   
);

    reg [15:0] data_in_;
    reg [15:0] data_in_storage [0:n_inputs-1];
    reg [15:0] weight_in_ [0:n_layer-1];
    integer input_index;
    reg processing;
    reg generated_last;
    integer j;

    // Sequential input and weight capture logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            input_index <= 0;
            processing <= 1'b0;
            data_in_ <= 16'd0;
            generated_last <= 1'b0;
            for (j = 0; j < n_layer; j = j + 1) begin
                weight_in_[j] <= 16'd0;
            end
        end else begin
            generated_last <= 1'b0; // Default: last is not asserted
            
            if (start) begin
                // Start pulse: capture first input and weights, reset processing
                processing <= 1'b1;
                input_index <= 0;
                data_in_ <= data_in[0];
                $display("[Layer] START: loading input 0, processing=1");
                
                // Load weights for first input
                for (j = 0; j < n_layer; j = j + 1) begin
                    weight_in_[j] <= weight_in[(0 * n_layer) + j];
                end
                
                if (n_inputs == 1) begin
                    generated_last <= 1'b1; // Only one input
                    $display("[Layer] Single input - setting last");
                end
            end else if (processing) begin
                // Increment to next input on each clock cycle
                input_index <= input_index + 1;
                $display("[Layer] Processing: input_index=%0d", input_index);
                
                if (input_index + 1 < n_inputs - 1) begin
                    // More inputs available
                    data_in_ <= data_in[input_index + 1];
                    
                    // Load weights for next input
                    for (j = 0; j < n_layer; j = j + 1) begin
                        weight_in_[j] <= weight_in[((input_index + 1) * n_layer) + j];
                    end
                    
                    generated_last <= 1'b0;
                    $display("[Layer] Loading next input %0d", input_index + 1);
                end else begin
                    // Last input
                    data_in_ <= data_in[input_index + 1];
                    $display("[Layer] LAST INPUT: assigning input_index=%0d, setting last=1", input_index + 1);
                    
                    // Load weights for last input
                    for (j = 0; j < n_layer; j = j + 1) begin
                        weight_in_[j] <= weight_in[((input_index + 1) * n_layer) + j];
                    end
                    
                    generated_last <= 1'b1;
                    processing <= 1'b0;
                end
            end
        end
    end

    
    genvar i;
    generate
        for (i = 0; i < n_layer; i = i + 1) begin : my_instances
        
            neuron neuron_inst (
                .clk(clk),
                .rst_n(rst_n),
                .start(start),
                .data_in(data_in_),
                .weight_in(weight_in_[i]),
                .bias(bias[i]),
                .last(generated_last),
                .out(out[i]),
                .valid(valid[i])
            );
        end
    endgenerate

endmodule