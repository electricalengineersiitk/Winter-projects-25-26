// nn_top.v
// --------
// Top-level wrapper that chains two fully-connected layers:
//
//   Layer 1 (hidden)  : 4 inputs  → 8 ReLU outputs
//   Layer 2 (output)  : 8 inputs  → 3 linear outputs  (no ReLU – softmax
//                       is not needed on hardware; argmax of the raw logits
//                       gives the same class prediction)
//
// Control flow
//   1. Assert 'start' for one cycle.  Feed the 4 input values on
//      data_in[0..3] on that same cycle (all four are latched internally).
//   2. The hidden layer processes them over 4 clock cycles.
//   3. When the hidden layer finishes, the output layer is automatically
//      kicked off using the 8 hidden outputs as its inputs.
//   4. When the output layer finishes, 'done' is asserted for one cycle and
//      'pred_class' holds the index of the neuron with the highest value
//      (0, 1, or 2).

`timescale 1ns / 1ps

module nn_top (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    // All 4 input features presented simultaneously for simplicity
    input  wire [15:0] in0, in1, in2, in3,
    output reg  [1:0]  pred_class,   // 0, 1, or 2
    output reg         done
);

    // ── stage tracking ────────────────────────────────────────────────────────
    // We feed each layer one value per cycle, so we need a small sequencer.

    localparam IDLE      = 2'd0,
               LAYER1    = 2'd1,
               LAYER2    = 2'd2,
               FINISHED  = 2'd3;

    reg [1:0] state;
    reg [2:0] feed_idx;   // current input index being fed (0-7)

    // ── input latch ───────────────────────────────────────────────────────────
    reg [15:0] latched_in [0:3];   // holds the 4 input values

    // ── hidden layer signals ──────────────────────────────────────────────────
    reg         h_start;
    reg  [15:0] h_data_in;
    wire [127:0] h_out;
    wire         h_valid;

    layer #(
        .NUM_NEURONS  (8),
        .NUM_INPUTS   (4),
        .WEIGHTS_FILE ("weights/weights.mem"),
        .BIAS_OFFSET  (0)
    ) u_hidden (
        .clk       (clk),
        .rst_n     (rst_n),
        .start     (h_start),
        .data_in   (h_data_in),
        .layer_out (h_out),
        .all_valid (h_valid)
    );

    // ── output layer signals ──────────────────────────────────────────────────
    // Weight layout in weights.mem: hidden-layer weights occupy lines 0-31,
    // output-layer weights start at line 32.  We pass a separate parameter
    // for the output layer weights file to keep things clean.

    reg         o_start;
    reg  [15:0] o_data_in;
    wire [47:0] o_out;     // 3 × 16 bits
    wire         o_valid;

    // Latch hidden outputs so the output layer can read them sequentially
    reg [15:0] hidden_out [0:7];

    layer #(
        .NUM_NEURONS  (3),
        .NUM_INPUTS   (8),
        .WEIGHTS_FILE ("weights/weights_out.mem"),
        .BIAS_OFFSET  (0)
    ) u_output (
        .clk       (clk),
        .rst_n     (rst_n),
        .start     (o_start),
        .data_in   (o_data_in),
        .layer_out (o_out),
        .all_valid (o_valid)
    );

    // ── main FSM ──────────────────────────────────────────────────────────────
    integer k;

    always @(posedge clk) begin
        if (!rst_n) begin
            state    <= IDLE;
            feed_idx <= 0;
            h_start  <= 0;
            o_start  <= 0;
            h_data_in <= 0;
            o_data_in <= 0;
            done      <= 0;
            pred_class <= 0;
        end else begin
            // Default deasserts
            h_start <= 0;
            o_start <= 0;
            done    <= 0;

            case (state)

                IDLE: begin
                    if (start) begin
                        // Latch all 4 inputs
                        latched_in[0] <= in0;
                        latched_in[1] <= in1;
                        latched_in[2] <= in2;
                        latched_in[3] <= in3;
                        feed_idx  <= 1;          // start=1 sends index 0 now
                        h_start   <= 1;
                        h_data_in <= in0;        // first input with start
                        state     <= LAYER1;
                    end
                end

                LAYER1: begin
                    if (feed_idx < 4) begin
                        h_data_in <= latched_in[feed_idx];
                        feed_idx  <= feed_idx + 1;
                    end
                    if (h_valid) begin
                        // Capture hidden outputs
                        for (k = 0; k < 8; k = k + 1)
                            hidden_out[k] <= h_out[k*16 +: 16];
                        // Start output layer
                        feed_idx  <= 1;
                        o_start   <= 1;
                        o_data_in <= h_out[0 +: 16];  // neuron 0 with start
                        state     <= LAYER2;
                    end
                end

                LAYER2: begin
                    if (feed_idx < 8) begin
                        o_data_in <= hidden_out[feed_idx];
                        feed_idx  <= feed_idx + 1;
                    end
                    if (o_valid) begin
                        // argmax of 3 output logits
                        begin
                            reg signed [15:0] v0, v1, v2;
                            v0 = $signed(o_out[0  +: 16]);
                            v1 = $signed(o_out[16 +: 16]);
                            v2 = $signed(o_out[32 +: 16]);
                            if (v0 >= v1 && v0 >= v2)
                                pred_class <= 2'd0;
                            else if (v1 >= v2)
                                pred_class <= 2'd1;
                            else
                                pred_class <= 2'd2;
                        end
                        done  <= 1;
                        state <= FINISHED;
                    end
                end

                FINISHED: begin
                    // Hold pred_class; wait for next start
                    if (start)
                        state <= IDLE;
                end

            endcase
        end
    end

endmodule
