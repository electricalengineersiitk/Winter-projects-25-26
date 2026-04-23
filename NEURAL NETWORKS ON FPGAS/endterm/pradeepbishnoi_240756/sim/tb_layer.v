`timescale 1ns / 1ps
module tb_layer();

    // Inputs
    reg        clk;
    reg        rst_n;
    reg [15:0] data_in;
    reg        start;
    reg        last;

    // Outputs
    wire [127:0] layer_out;
    wire [7:0]   layer_valid;

    // Instantiate Layer
    layer uut (
        .clk       (clk),
        .rst_n     (rst_n),
        .data_in   (data_in),
        .start     (start),
        .last      (last),
        .layer_out (layer_out),
        .layer_valid(layer_valid)
    );

    // Clock Generation (100MHz)
    always #5 clk = ~clk;

    // Capture pre-ReLU sums and outputs when valid pulses
    reg signed [31:0] captured_sum [0:7];
    reg        [15:0] captured_out [0:7];
    integer j;

    always @(posedge clk) begin
        for (j = 0; j < 8; j = j + 1) begin
            if (layer_valid[j]) begin
                captured_out[j] <= layer_out[j*16 +: 16];
            end
        end
        // Capture acc + bias for pre-ReLU value at valid pulse
        if (layer_valid[0]) captured_sum[0] <= uut.neuron_block[0].n_inst.acc + ($signed(uut.bias_mem[0]) <<< 8);
        if (layer_valid[1]) captured_sum[1] <= uut.neuron_block[1].n_inst.acc + ($signed(uut.bias_mem[1]) <<< 8);
        if (layer_valid[2]) captured_sum[2] <= uut.neuron_block[2].n_inst.acc + ($signed(uut.bias_mem[2]) <<< 8);
        if (layer_valid[3]) captured_sum[3] <= uut.neuron_block[3].n_inst.acc + ($signed(uut.bias_mem[3]) <<< 8);
        if (layer_valid[4]) captured_sum[4] <= uut.neuron_block[4].n_inst.acc + ($signed(uut.bias_mem[4]) <<< 8);
        if (layer_valid[5]) captured_sum[5] <= uut.neuron_block[5].n_inst.acc + ($signed(uut.bias_mem[5]) <<< 8);
        if (layer_valid[6]) captured_sum[6] <= uut.neuron_block[6].n_inst.acc + ($signed(uut.bias_mem[6]) <<< 8);
        if (layer_valid[7]) captured_sum[7] <= uut.neuron_block[7].n_inst.acc + ($signed(uut.bias_mem[7]) <<< 8);
    end

    integer i;
    integer timeout;
    real    pre_relu_dec;
    real    final_dec;

    initial begin
        $dumpfile("layer_results.vcd");
        $dumpvars(0, tb_layer);

        // Initialise
        clk     = 0;
        rst_n   = 0;
        start   = 0;
        last    = 0;
        data_in = 0;

        // Reset sequence
        #25 rst_n = 1;
        #20;

        // Feed input vector [0.5, 1.0, -0.5, 1.0]
        @(posedge clk); start = 1; data_in = 16'h0080;  // 0.5
        @(posedge clk); start = 0; data_in = 16'h0100;  // 1.0
        @(posedge clk);            data_in = 16'hff80;  // -0.5
        @(posedge clk); last  = 1; data_in = 16'h0100;  // 1.0
        @(posedge clk); last  = 0; data_in = 0;

        // Wait for all 8 neurons with timeout
        timeout = 0;
        while (layer_valid !== 8'hFF && timeout < 100) begin
            @(posedge clk);
            timeout = timeout + 1;
        end

        if (timeout >= 100) begin
            $display("TIMEOUT: not all neurons completed — layer_valid = %b", layer_valid);
            $finish;
        end

        // Extra cycle to let captured registers settle
        @(posedge clk);

        // Display results
        $display("\n============= LAYER 1 HARDWARE VERIFICATION =============");
        $display(" ID | Pre-ReLU (Dec) | Final (Hex) | Final (Dec) | Status");
        $display("---------------------------------------------------------");

        for (i = 0; i < 8; i = i + 1) begin
            pre_relu_dec = $itor(captured_sum[i]) / 65536.0;
            final_dec    = $itor($signed(captured_out[i])) / 256.0;

            $write("%2d |   %10.4f   |    %4h     |  %8.4f   | ", 
                    i, pre_relu_dec, captured_out[i], final_dec);

            if (pre_relu_dec < 0.0 && final_dec == 0.0)
                $display("ReLU OK   — neuron inactive");
            else if (pre_relu_dec >= 0.0 && final_dec >= 0.0)
                $display("ACTIVE    — neuron firing");
            else if (pre_relu_dec > 127.996)
                $display("OVERFLOW  — saturated to max");
            else
                $display("CHECK!    — unexpected result");
        end

        $display("---------------------------------------------------------");
        #50 $finish;
    end

endmodule
