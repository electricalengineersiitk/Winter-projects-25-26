module nn_top (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire signed [15:0] data_in,
    input  wire        h_last,
    output wire [1:0]  class_out,
    output wire        done
);

    wire signed [15:0] h_out0, h_out1, h_out2, h_out3, h_out4, h_out5, h_out6, h_out7;
    wire [7:0]         h_valid;
    reg signed [15:0]  h_inputs [0:7];
    reg                hidden_done;

    layer hidden_layer (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .data_in(data_in),
        .last(h_last),
        .out0(h_out0), .out1(h_out1), .out2(h_out2), .out3(h_out3),
        .out4(h_out4), .out5(h_out5), .out6(h_out6), .out7(h_out7),
        .valid(h_valid)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            hidden_done <= 0;
        end else begin
            if (h_valid == 8'hFF) begin
                h_inputs[0] <= h_out0;
                h_inputs[1] <= h_out1;
                h_inputs[2] <= h_out2;
                h_inputs[3] <= h_out3;
                h_inputs[4] <= h_out4;
                h_inputs[5] <= h_out5;
                h_inputs[6] <= h_out6;
                h_inputs[7] <= h_out7;
                hidden_done <= 1;
            end else if (start)
                hidden_done <= 0;
        end
    end

    output_layer out_layer (
        .clk(clk),
        .rst_n(rst_n),
        .hidden_done(hidden_done),
        .h0(h_inputs[0]),
        .h1(h_inputs[1]),
        .h2(h_inputs[2]),
        .h3(h_inputs[3]),
        .h4(h_inputs[4]),
        .h5(h_inputs[5]),
        .h6(h_inputs[6]),
        .h7(h_inputs[7]),
        .class_out(class_out),
        .done(done)
    );

endmodule
