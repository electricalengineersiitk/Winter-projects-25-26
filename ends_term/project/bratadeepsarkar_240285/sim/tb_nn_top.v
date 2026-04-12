`timescale 1ns / 1ps

module tb_nn_top;

    reg        clk;
    reg        btn_rst;
    reg        start;
    reg [3:0]  sw;
    wire [1:0] predicted_class;
    wire       done;

    nn_top uut (
        .clk             (clk),
        .btn_rst         (btn_rst),
        .start           (start),
        .sw              (sw),
        .predicted_class (predicted_class),
        .done            (done)
    );

    // 100 MHz clock
    always #5 clk = ~clk;

    integer i;
    initial begin
        clk     = 0;
        btn_rst = 1;
        start   = 0;
        sw      = 0;

        #100;
        btn_rst = 0;
        #20;

        $display("========================================");
        $display(" nn_top Integration Test - 10 Samples");
        $display("========================================");
        $display("Sample | Expected | Predicted | Status");
        $display("-------|----------|-----------|-------");

        for (i = 0; i < 10; i = i + 1) begin
            sw = i;
            #20;
            
            // Pulse start
            @(posedge clk); #1;
            start = 1;
            @(posedge clk); #1;
            start = 0;

            // Wait for done
            begin : wait_loop
                repeat (1000) begin
                    @(posedge clk); #1;
                    if (done) disable wait_loop;
                end
            end

        begin : display_block
            reg [1:0] expected;
            case(i)
                0: expected = 0;
                1: expected = 2;
                2: expected = 1;
                3: expected = 1;
                4: expected = 0;
                5: expected = 1;
                6: expected = 0;
                7: expected = 0;
                8: expected = 2;
                9: expected = 1;
                default: expected = 0;
            endcase

            if (done) begin
                if (predicted_class == expected)
                    $display("   %0d   |     %0d    |     %0d     |  PASS", i, expected, predicted_class);
                else
                    $display("   %0d   |     %0d    |     %0d     |  FAIL (Mismatch)", i, expected, predicted_class);
            end else begin
                $display("   %0d   |     %0d    |   TIMEOUT |  FAIL (No Done)", i, expected);
            end
        end
        #100;
        end

        $display("========================================");
        $finish;
    end

endmodule
