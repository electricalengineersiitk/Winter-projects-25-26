`timescale 1ns / 1ps

module neuron ( 
    input  wire        clk, 
    input  wire        rst_n,      // active-low reset 
    input  wire        start,      // pulse high for 1 cycle to begin 
    input  wire [15:0] data_in,    // one input value at a time (Q8 format) 
    input  wire [15:0] weight_in,  // matching weight for that input 
    input  wire [15:0] bias,       // bias value 
    input  wire        last,       // pulse high on the final input 
    output reg  [15:0] out,        // result after ReLU 
    output reg         valid       // high for 1 cycle when output is ready 
);

    reg signed [31:0] accumulator;
    wire signed [31:0] product;
    
    // Q8 multiplication
    assign product = $signed(data_in) * $signed(weight_in);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accumulator <= 32'd0;
            out <= 16'd0;
            valid <= 1'b0;
        end else begin
            valid <= 1'b0; // Default state
            
            if (start) begin
                accumulator <= product; // Reset and add first product
            end else begin
                accumulator <= accumulator + product; // Accumulate
            end
            
            if (last) begin
                // Add bias and apply ReLU on the final step
                if ((accumulator + ($signed(bias) <<< 8)) < 0) begin
                    out <= 16'd0; // ReLU: output 0 if negative
                end else begin
                    // Truncate to Q8 format using bits [23:8]
                    out <= (accumulator + ($signed(bias) <<< 8)) >> 8; 
                end
                valid <= 1'b1; // Output is ready
            end
        end
    end
endmodule