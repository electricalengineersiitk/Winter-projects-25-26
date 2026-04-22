`timescale 1ns / 1ps

module layer_tb;
    reg clk;
    reg rst_n;
    reg start;
    reg signed [15:0] data_in;
    reg signed [15:0] weight_n [1:8]; // Array for easier iteration in TB
    reg [15:0] bias[1:8];
    reg last;
    wire [15:0] out [1:8];
    wire valid [1:8];

    reg signed [15:0] w_mem [0:31];
    reg signed [15:0] test_inputs [0:3];
    reg signed [15:0] biasm[0:7];
    layer uut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .data_in(data_in),
        .weight_n1(weight_n[1]), .weight_n2(weight_n[2]),
        .weight_n3(weight_n[3]), .weight_n4(weight_n[4]),
        .weight_n5(weight_n[5]), .weight_n6(weight_n[6]),
        .weight_n7(weight_n[7]), .weight_n8(weight_n[8]),
        .bias_1(bias[1]),.bias_2(bias[2]),.bias_3(bias[3]),.bias_4(bias[4]),.bias_5(bias[5]),.bias_6(bias[6]),.bias_7(bias[7]),.bias_8(bias[8]),
        .last(last),
        .out_1(out[1]), .out_2(out[2]), .out_3(out[3]), .out_4(out[4]),
        .out_5(out[5]), .out_6(out[6]), .out_7(out[7]), .out_8(out[8]),
        .valid_1(valid[1]), .valid_2(valid[2]), .valid_3(valid[3]), .valid_4(valid[4]),
        .valid_5(valid[5]), .valid_6(valid[6]), .valid_7(valid[7]), .valid_8(valid[8])
    );

    always #5 clk = ~clk;
    integer i;
    initial begin
        w_mem[0] = -16'sd122; w_mem[1] = 16'sd138;  w_mem[2] = -16'sd102;  w_mem[3] = 16'sd8; 
        w_mem[4] = -16'sd102;   w_mem[5] = -16'sd129; w_mem[6] = 16'sd27; w_mem[7] = 16'sd34;  
        w_mem[8] = 16'sd19;   w_mem[9] = 16'sd167;   w_mem[10] = -16'sd29; w_mem[11] = -16'sd165; 
        w_mem[12] = -16'sd91;w_mem[13] = 16'sd68; w_mem[14] = -16'sd74;  w_mem[15] = -16'sd11; 
        w_mem[16] = -16'sd104; w_mem[17] = 16'sd79;w_mem[18] = -16'sd144;  w_mem[19] = -16'sd121;
        w_mem[20] = 16'sd20;w_mem[21] = -16'sd58; w_mem[22] = -16'sd32; w_mem[23] = 16'sd71; 
        w_mem[24] = 16'sd8;   w_mem[25] = 16'sd41;  w_mem[26] = -16'sd165;w_mem[27] = -16'sd28; 
        w_mem[28] = -16'sd122;w_mem[29] = 16'sd87;  w_mem[30] = -16'sd110;w_mem[31] = 16'sd132;
        biasm[0] = 16'sd8;biasm[1] = 16'sd0;biasm[2] = -16'sd8;biasm[3] = 16'sd9;biasm[4] = -16'sd2;biasm[5] = 16'sd0;biasm[6] = -16'sd7;biasm[7] = -16'sd8;         
        test_inputs[0] = 16'sd4; // 1.3203
        test_inputs[1] = 16'sd138; // 1.1952
        test_inputs[2] = 16'sd0;  // 0.2168
        test_inputs[3] = 16'sd0;   // 0.0

        clk = 0;
        rst_n = 0;
        start = 0;
        last = 0;
        data_in = 0;
        bias[1] = biasm[0];
        bias[2] = biasm[1];
        bias[3] = biasm[2];
        bias[4] = biasm[3];
        bias[5] = biasm[4];
        bias[6] = biasm[5];
        bias[7] = biasm[6];
        bias[8] = biasm[7];
        #20 rst_n = 1;
        #10;

        for (i = 0; i < 4; i = i + 1) begin
            @(posedge clk);
            start   <= (i == 0);      
            last    <= (i == 3);      
            data_in <= test_inputs[i];
            
            weight_n[1] <= w_mem[0 + i];
            weight_n[2] <= w_mem[4 + i];
            weight_n[3] <= w_mem[8 + i];
            weight_n[4] <= w_mem[12 + i];
            weight_n[5] <= w_mem[16 + i];
            weight_n[6] <= w_mem[20 + i];
            weight_n[7] <= w_mem[24 + i];
            weight_n[8] <= w_mem[28 + i];
        end

        @(posedge clk);
        start <= 0;
        last  <= 0;
        data_in <= 0;

        wait(valid[1]);
        $display("--- Layer 2 Inference Results ---");
        $display("N1: %d | N2: %d | N3: %d | N4: %d", out[1], out[2], out[3], out[4]);
        $display("N5: %d | N6: %d | N7: %d | N8: %d", out[5], out[6], out[7], out[8]);
        
        #100 $finish;
    end

endmodule
