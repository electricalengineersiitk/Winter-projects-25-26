`timescale 1ns / 1ps
module neuron (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [15:0] data_in,
    input  wire [15:0] weight_in,
    input  wire [15:0] bias,
    input  wire        last,
    output reg  [15:0] out,
    output reg         valid
);

    reg signed [31:0] acc;
    reg signed [31:0] sum;
    reg last_d;   // 🔥 delay register
    reg last_d2;
    wire signed [31:0] mult;
    wire [15:0] relu_out;
    assign mult = $signed(data_in) * $signed(weight_in);

    assign relu_out = (sum[31]) ? 16'd0 : sum[23:8];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc   <= 0;
            sum   <= 0;
            out   <= 0;
            valid <= 0;
            last_d <= 0;
            last_d2 <= 0;
        end else begin
            valid <= 0;

            // Delay last signal
            last_d <= last;
            last_d2 <= last_d;
            if (start) begin
                acc <= 0;
            end else begin
                acc <= acc + mult;
            end
            
            // Step 1: compute sum
            if (last_d) begin
                sum<= acc + ($signed(bias) <<< 8);
//                out <= sum;
//                valid <=1;
            end
            
            if (last_d2) begin
                out   <= relu_out;;
                valid <= 1;
            end

            // Step 2: output next cycle
//            if (last_d_d) begin
//                out   <= sum;
//                valid <= 1;
//            end
        end
    end

endmodule
