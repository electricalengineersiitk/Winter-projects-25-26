module neuron(
    input clk,
    input rst_n,
    input start,
    input [15:0] data_in,
    input [15:0] weight_in,
    input [15:0] bias,
    input last,
    output reg [15:0] out,
    output reg valid
);

reg [31:0] acc;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        acc <= 0;
        out <= 0;
        valid <= 0;
    end
    else begin
        valid <= 0;

        if (start) begin
            acc <= 0;
        end
        else begin
            acc <= acc + data_in * weight_in;
        end

        if (last) begin
            acc <= acc + bias;

            if (acc[31] == 1)
                out <= 0;
            else
                out <= acc[23:8];

            valid <= 1;
        end
    end
end

endmodule