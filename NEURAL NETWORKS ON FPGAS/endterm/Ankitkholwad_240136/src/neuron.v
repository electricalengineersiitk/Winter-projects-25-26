module neuron (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire signed [15:0] data_in,
    input  wire signed [15:0] weight_in,
    input  wire signed [15:0] bias,
    input  wire        last,
    output reg signed [15:0] out,
    output reg         valid
);

    reg signed [31:0] acc;
    wire signed [31:0] product;
    reg signed [31:0] temp;

    assign product = data_in * weight_in;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc   <= 0;
            out   <= 0;
            valid <= 0;
        end else begin
            valid <= 0;

            if (last) begin
                // On last cycle: if start is also high this is a 1-input layer;
                // use product directly rather than adding to stale acc.
                temp = (start ? 32'sd0 : acc) + (product >>> 8) + bias;
                out   <= temp[31] ? 16'd0 : temp[15:0];
                valid <= 1;
                acc   <= 0;
            end else if (start) begin
                acc <= (product >>> 8);
            end else begin
                acc <= acc + (product >>> 8);
            end
        end
    end

endmodule
