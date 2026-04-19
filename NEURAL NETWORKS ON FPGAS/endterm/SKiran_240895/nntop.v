`timescale 1ns / 1ps
module nntop(
    input wire clk,
    input wire rst_n,
    input wire signed data_in,
    output wire signed [15:0]fo_1, fo_2, fo_3,
    output wire fval1,fval2,fval3,
    input wire signed [15:0]fw1,fw2,fw3,
    input wire signed [15:0]fb1,fb2,fb3,
    input wire last,
    output reg signed [15:0] current_data,
    output wire current_last
    );
 wire signed [15:0]out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8;
 wire val1,val2,val3,val4,val5,val6,val7,val8;
 wire start1, start2;
 reg signed [15:0]weights[0:71];
 reg signed [15:0]biases[0:14];
 initial begin
    $readmemb("weights.mem",weights);
    $readmemb("biases.mem",biases);
 end
   reg [2:0] l1_count;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) l1_count <= 0;
        else if (start1 || l1_count > 0) l1_count <= l1_count + 1;
    end
    wire [15:0] w1 = weights[16 + l1_count];
    wire [15:0] w2 = weights[20 + l1_count];
    wire [15:0] w3 = weights[24 + l1_count];
    wire [15:0] w4 = weights[28 + l1_count];
    wire [15:0] w5 = weights[32 + l1_count];
    wire [15:0] w6 = weights[36 + l1_count];
    wire [15:0] w7 = weights[40 + l1_count];
    wire [15:0] w8 = weights[44 + l1_count];
    wire b1 = biases[4];
    wire b2 = biases[5];
    wire b3 = biases[6];
    wire b4 = biases[7];
    wire b5 = biases[8];
    wire b6 = biases[9];
    wire b7 = biases[10];
    wire b8 = biases[11];
   layer deep_layer(
   .clk(clk),.rst_n(rst_n),.start(start1),.data_in(data_in),.last(last),.out_1(out_1),.out_2(out_2),.out_3(out_3),.out_4(out_4),.out_5(out_5),.out_6(out_6),.out_7(out_7),.out_8(out_8),
    .bias_1(b1),.bias_2(b2),.bias_3(b3),.bias_4(b4),.bias_5(b5),.bias_6(b6),.bias_7(b7),.bias_8(b8),
    .valid_1(val1),.valid_2(val2),.valid_3(val3),.valid_4(val4),.valid_5(val5),.valid_6(val6),.valid_7(val7),.valid_8(val8),
    .weight_n1(w1),.weight_n2(w2),.weight_n3(w3),.weight_n4(w4),.weight_n5(w5),.weight_n6(w6),.weight_n7(w7),.weight_n8(w8)
    );
    reg start;
   reg [2:0] count;
   always @(posedge clk) begin
        start <= val1;
        if (!rst_n) begin
            count <= 0;
        end else if (start2 || count > 0) begin
            count <= count + 1;
        end
   end
   always @(*) begin
        case(count)
            3'd0: current_data = out_1;
            3'd1: current_data = out_2;
            3'd2: current_data = out_3;
            3'd3: current_data = out_4;
            3'd4: current_data = out_5;
            3'd5: current_data = out_6;
            3'd6: current_data = out_7;
            3'd7: current_data = out_8;
            default: current_data = 16'd0;
        endcase
    end
    assign current_last = (count == 3'd7);  //T F statement
    neuron fn1(
        .clk(clk), .rst_n(rst_n), .start(start2), .data_in(current_data), 
        .weight_in(fw1), .bias(fb1), .last(current_last), .out(fo_1), .valid(fval1)
    );

    neuron fn2(
        .clk(clk), .rst_n(rst_n), .start(start2), .data_in(current_data), 
        .weight_in(fw2), .bias(fb2), .last(current_last), .out(fo_2), .valid(fval2)
    );

    neuron fn3(
        .clk(clk), .rst_n(rst_n), .start(start2), .data_in(current_data), 
        .weight_in(fw3), .bias(fb3), .last(current_last), .out(fo_3), .valid(fval3)
    );
    //neuron fn1(.clk(clk),.rst_n(rst_n),.data_in(out_1),
endmodule
