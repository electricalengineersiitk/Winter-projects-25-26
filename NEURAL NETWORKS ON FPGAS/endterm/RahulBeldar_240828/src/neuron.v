module neuron (
    input wire clk,
    input wire rst_n,
    input wire start,
    input wire [15:0] data_in,
    input wire [15:0] weight_in,
    input wire [15:0] bias,
    input wire last,
    output reg [15:0] out,
    output reg valid
);

reg signed [31:0] acc;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        acc <= 0;
        out <= 0;
        valid <= 0;
    end else begin
        if (start)
            acc <= 0;
        else
            acc <= acc + $signed(data_in) * $signed(weight_in);

        if (last) begin
            acc <= acc + $signed(bias);

            if (acc[31])
                out <= 0;
            else
                out <= acc[23:8];

            valid <= 1;
        end else begin
            valid <= 0;
        end
    end
end

endmodule
