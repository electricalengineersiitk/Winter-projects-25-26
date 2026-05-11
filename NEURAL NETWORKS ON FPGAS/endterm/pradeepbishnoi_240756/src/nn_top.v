module nn_top (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,       // begin inference
    input  wire [15:0] data_in,     // one Q8 input per cycle
    output reg  [1:0]  class_out,   // predicted class (0, 1, or 2)
    output reg         done         // pulses high when result is ready
);

    // =========================================================================
    // State Machine Encoding
    // =========================================================================
    localparam IDLE    = 3'd0;
    localparam L1_RUN  = 3'd1;   // feeding inputs to layer 1
    localparam L1_WAIT = 3'd2;   // waiting for layer 1 valid
    localparam L2_RUN  = 3'd3;   // feeding layer1 outputs to layer 2
    localparam L2_WAIT = 3'd4;   // waiting for layer 2 valid
    localparam DONE    = 3'd5;   // output result

    reg [2:0] state;

    // =========================================================================
    // Layer 1 Signals
    // =========================================================================
    reg         l1_start;
    reg         l1_last;
    reg  [15:0] l1_data_in;
    wire [127:0] l1_out;          // 8 x 16-bit outputs
    wire [7:0]   l1_valid;        // one valid bit per neuron

    layer layer1 (
        .clk        (clk),
        .rst_n      (rst_n),
        .data_in    (l1_data_in),
        .start      (l1_start),
        .last       (l1_last),
        .layer_out  (l1_out),
        .layer_valid(l1_valid)
    );

    // =========================================================================
    // Layer 1 Output Storage
    // Capture all 8 outputs when layer 1 finishes
    // =========================================================================
    reg [15:0] l1_captured [0:7];
    integer    m;

    always @(posedge clk) begin
        if (l1_valid == 8'hFF) begin
            for (m = 0; m < 8; m = m + 1)
                l1_captured[m] <= l1_out[m*16 +: 16];
        end
    end

    // =========================================================================
    // Layer 2 Signals (3 neurons, 8 inputs each)
    // =========================================================================
    reg         l2_start;
    reg         l2_last;
    reg  [15:0] l2_data_in;
    wire [47:0] l2_out;           // 3 x 16-bit outputs
    wire [2:0]  l2_valid;         // one valid bit per neuron

    layer2 layer2_inst (
        .clk        (clk),
        .rst_n      (rst_n),
        .data_in    (l2_data_in),
        .start      (l2_start),
        .last       (l2_last),
        .layer_out  (l2_out),
        .layer_valid(l2_valid)
    );

    // =========================================================================
    // Input and Layer 2 Feed Counters
    // =========================================================================
    reg [1:0] l1_input_cnt;   // counts 0-3 for layer 1 inputs
    reg [2:0] l2_input_cnt;   // counts 0-7 for layer 2 inputs (l1 outputs)

    // =========================================================================
    // Argmax Logic — finds which of 3 outputs is largest
    // =========================================================================
    reg signed [15:0] l2_out0, l2_out1, l2_out2;

    always @(posedge clk) begin
        if (l2_valid == 3'b111) begin
            l2_out0 <= l2_out[15:0];
            l2_out1 <= l2_out[31:16];
            l2_out2 <= l2_out[47:32];
        end
    end

    // Combinational argmax
    always @(*) begin
        if (l2_out0 >= l2_out1 && l2_out0 >= l2_out2)
            class_out = 2'd0;
        else if (l2_out1 >= l2_out0 && l2_out1 >= l2_out2)
            class_out = 2'd1;
        else
            class_out = 2'd2;
    end

    // =========================================================================
    // Main State Machine
    // =========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= IDLE;
            done        <= 1'b0;
            l1_start    <= 1'b0;
            l1_last     <= 1'b0;
            l1_data_in  <= 16'd0;
            l2_start    <= 1'b0;
            l2_last     <= 1'b0;
            l2_data_in  <= 16'd0;
            l1_input_cnt <= 2'd0;
            l2_input_cnt <= 3'd0;
        end
        else begin
            // Defaults — these only pulse for one cycle unless overridden
            done     <= 1'b0;
            l1_start <= 1'b0;
            l1_last  <= 1'b0;
            l2_start <= 1'b0;
            l2_last  <= 1'b0;

            case (state)

                // -------------------------------------------------------------
                // IDLE: wait for start pulse
                // -------------------------------------------------------------
                IDLE: begin
                    if (start) begin
                        l1_start     <= 1'b1;
                        l1_data_in   <= data_in;   // first input arrives with start
                        l1_input_cnt <= 2'd0;
                        state        <= L1_RUN;
                    end
                end

                // -------------------------------------------------------------
                // L1_RUN: feed remaining 3 inputs to layer 1
                // data_in is driven externally, one per cycle
                // -------------------------------------------------------------
                L1_RUN: begin
                    l1_data_in   <= data_in;
                    l1_input_cnt <= l1_input_cnt + 1'b1;

                    if (l1_input_cnt == 2'd2) begin
                        // Next cycle is the last input
                        l1_last <= 1'b1;
                        state   <= L1_WAIT;
                    end
                end

                // -------------------------------------------------------------
                // L1_WAIT: hold last high for one cycle then wait for valid
                // -------------------------------------------------------------
                L1_WAIT: begin
                    l1_data_in <= data_in;  // last input still being fed
                    if (l1_valid == 8'hFF) begin
                        // Layer 1 done — start feeding its outputs to Layer 2
                        l2_start     <= 1'b1;
                        l2_data_in   <= l1_captured[0];
                        l2_input_cnt <= 3'd0;
                        state        <= L2_RUN;
                    end
                end

                // -------------------------------------------------------------
                // L2_RUN: feed all 8 layer1 outputs serially into layer 2
                // -------------------------------------------------------------
                L2_RUN: begin
                    l2_input_cnt <= l2_input_cnt + 1'b1;
                    l2_data_in   <= l1_captured[l2_input_cnt + 1];

                    if (l2_input_cnt == 3'd6) begin
                        // Next cycle is the last input for layer 2
                        l2_last <= 1'b1;
                        state   <= L2_WAIT;
                    end
                end

                // -------------------------------------------------------------
                // L2_WAIT: wait for all 3 layer 2 neurons to assert valid
                // -------------------------------------------------------------
                L2_WAIT: begin
                    if (l2_valid == 3'b111) begin
                        state <= DONE;
                    end
                end

                // -------------------------------------------------------------
                // DONE: pulse done for one cycle, go back to IDLE
                // -------------------------------------------------------------
                DONE: begin
                    done  <= 1'b1;
                    state <= IDLE;
                end

            endcase
        end
    end

endmodule
