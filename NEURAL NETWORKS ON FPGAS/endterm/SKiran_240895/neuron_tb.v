`timescale 1ns / 1ps

module neuron_tb;

    // Inputs
    reg clk;
    reg rst_n;
    reg start;
    reg signed [15:0] data_in;
    reg signed [15:0] weight_in;
    reg signed [15:0] bias;
    reg last;

    // Outputs
    wire [15:0] out;
    wire valid;

    // Instantiate the Unit Under Test (UUT)
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

    // Clock generation (100MHz)
    always #5 clk = ~clk;

    // Fixed-point conversion values (Q8.8 format)
    // Weights: [0.2764, 0.5804, -0.3521, -0.1618] -> [71, 149, -90, -41]
    reg signed [15:0] weights [0:3];
    // Inputs (Converted to Q8.8: Value * 256)
    reg signed [15:0] inputs [0:2][0:3];
    
    initial begin
    
        weights[0] = 16'h0047; //  71  (0.277)
        weights[1] = 16'h0095; //  149 (0.582)
        weights[2] = 16'hFFA6; // -90 (-0.352)
        weights[3] = 16'hFFD7; // -41 (-0.160)
        // Vector 1: [0.222, 0.625, 0.068, 0.042]
        inputs[0][0] = 16'd57;  inputs[0][1] = 16'd160; 
        inputs[0][2] = 16'd17;  inputs[0][3] = 16'd11;
        
        // Vector 2: [0.167, 0.417, 0.068, 0.042]
        inputs[1][0] = 16'd43;  inputs[1][1] = 16'd107; 
        inputs[1][2] = 16'd17;  inputs[1][3] = 16'd11;
        
        // Vector 3: [0.111, 0.500, 0.051, 0.042]
        inputs[2][0] = 16'd28;  inputs[2][1] = 16'd128; 
        inputs[2][2] = 16'd13;  inputs[2][3] = 16'd11;

        // Initialize
        clk = 0;
        rst_n = 0;
        start = 0;
        data_in = 0;
        weight_in = 0;
        bias = 16'd9; // 0.0367 * 256 ≈ 9
        last = 0;

        // Global Reset
        #20 rst_n = 1;
        #20;

        // Test Sequence
        run_test(0); // Run first vector
        #20;
        run_test(1); // Run second vector
        #20;
        run_test(2); // Run third vector

        #100 $finish;
    end

    // Task to pulse start and feed 4 inputs
    task run_test(input integer vec_idx);
        integer i;
        begin
            @(posedge clk);
            start <= 1;
            @(posedge clk);
            start <= 0;

            for (i = 0; i < 4; i = i + 1) begin
                data_in <= inputs[vec_idx][i];
                weight_in <= weights[i];
                if (i == 3) last <= 1;
                else last <= 0;
                @(posedge clk);
            end
            last <= 0;
            data_in <= 0;
            weight_in <= 0;
            
            // Wait for output valid signal
            wait(valid);
            $display("Vector %0d Output: %h (Float approx: %f)", 
                      vec_idx + 1, out, out / 256.0);
            @(posedge clk);
        end
    endtask

endmodule