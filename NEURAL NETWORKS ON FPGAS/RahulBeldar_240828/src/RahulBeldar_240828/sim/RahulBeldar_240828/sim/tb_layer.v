module tb_layer;

reg clk = 0;
always #5 clk = ~clk;

reg rst_n, start, last;
reg [15:0] data_in;
wire valid;

layer uut (
    .clk(clk),
    .rst_n(rst_n),
    .start(start),
    .data_in(data_in),
    .last(last),
    .out(),
    .valid(valid)
);

initial begin
    rst_n = 0; 
    #10 rst_n = 1;

    start = 1; 
    #10 start = 0;

    data_in = 16'd5; 
    last = 0; 
    #10;

    data_in = 16'd3; 
    last = 1; 
    #10;

    #50 $finish;
end

endmodule
