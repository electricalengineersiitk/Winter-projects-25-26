`timescale 1ns / 1ps

module tb_neuron;

    reg clk;
    reg rst_n;
    reg start;
    reg [15:0] data_in;
    reg [15:0] weight_in;
    reg [15:0] bias;
    reg last;

    wire [15:0] out;
    wire valid;

    // Instantiate neuron
    neuron uut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .data_in(data_in),
        .weight_in(weight_in),
        .bias(bias),
        .last(last),
        .out(out),
        .valid(valid)
    );

    // Memory
    reg [15:0] weight_mem [0:100];
    reg [15:0] bias_mem   [0:100];
    reg [15:0] test_data  [0:100];

    integer i;
    integer n;
    // Clock
    always #5 clk = ~clk;

    

        initial begin
    clk = 0;
    rst_n = 0;
    start = 0;
    last = 0;
    data_in = 0;
    weight_in = 0;
    bias = 0;
    n=0;
    // Load files
    $readmemh("weights.mem", weight_mem);
    $readmemh("biases.mem", bias_mem);
    $readmemh("test_data.mem", test_data);

    // Reset

    @(posedge clk);
    rst_n = 1;

    // Select neuron
    bias <= bias_mem[n/4];

    // Start pulse
    @(posedge clk);
    start <= 1;

    @(posedge clk);
    start <= 0;

    // Feed inputs
    for (i = n; i < n+4; i = i + 1) begin
        
        @(posedge clk);
        data_in   <= test_data[i];
        weight_in <= weight_mem[i];
        last <= (i==n+3);
        
        @(posedge clk);    
    end

//    @(posedge clk);
    last <= 0;

    // ✅ Wait properly
    wait(valid);
    
    $display("Output = %d (hex=%h)", uut.out, uut.out);
    $display("Valid  = %b", valid);
    $display("FINAL ACC = %d", uut.acc);
    $display("DATA = %d", uut.data_in);
    $display("FINAL SUM = %d", uut.sum);
end

endmodule