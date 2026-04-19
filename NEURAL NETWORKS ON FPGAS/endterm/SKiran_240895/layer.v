module layer(
    input  wire        clk,
    input  wire        rst_n,      
    input  wire        start,      
    input  wire signed [15:0] data_in,    
    input  wire signed [15:0] weight_n1,  
    input  wire signed [15:0] weight_n2,
    input  wire signed [15:0] weight_n3,
    input  wire signed [15:0] weight_n4,
    input  wire signed [15:0] weight_n5,
    input  wire signed [15:0] weight_n6,
    input  wire signed [15:0] weight_n7,
    input  wire signed [15:0] weight_n8,
    input  wire [15:0] bias_1,bias_2,bias_3,bias_4,bias_5,bias_6,bias_7,bias_8,
    input  wire        last,       
    output wire [15:0] out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8,
    output wire        valid_1, valid_2, valid_3, valid_4, valid_5, valid_6, valid_7, valid_8
    );

    // Instances
    neuron neuron1(.clk(clk), .rst_n(rst_n), .start(start), .data_in(data_in), .weight_in(weight_n1), .bias(bias_1),  .last(last), .out(out_1), .valid(valid_1));
    neuron neuron2(.clk(clk), .rst_n(rst_n), .start(start), .data_in(data_in), .weight_in(weight_n2), .bias(bias_2),  .last(last), .out(out_2), .valid(valid_2));
    neuron neuron3(.clk(clk), .rst_n(rst_n), .start(start), .data_in(data_in), .weight_in(weight_n3), .bias(bias_3), .last(last), .out(out_3), .valid(valid_3));
    neuron neuron4(.clk(clk), .rst_n(rst_n), .start(start), .data_in(data_in), .weight_in(weight_n4), .bias(bias_4),  .last(last), .out(out_4), .valid(valid_4));
    neuron neuron5(.clk(clk), .rst_n(rst_n), .start(start), .data_in(data_in), .weight_in(weight_n5), .bias(bias_5), .last(last), .out(out_5), .valid(valid_5));
    neuron neuron6(.clk(clk), .rst_n(rst_n), .start(start), .data_in(data_in), .weight_in(weight_n6), .bias(bias_6),  .last(last), .out(out_6), .valid(valid_6));
    neuron neuron7(.clk(clk), .rst_n(rst_n), .start(start), .data_in(data_in), .weight_in(weight_n7), .bias(bias_7), .last(last), .out(out_7), .valid(valid_7));
    neuron neuron8(.clk(clk), .rst_n(rst_n), .start(start), .data_in(data_in), .weight_in(weight_n8), .bias(bias_8), .last(last), .out(out_8), .valid(valid_8));

endmodule
