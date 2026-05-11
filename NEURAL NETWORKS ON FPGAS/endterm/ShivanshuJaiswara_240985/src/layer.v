// layer.v
// -------
// One fully-connected layer containing 8 neuron instances.
// All 8 neurons receive the same sequence of inputs but each uses its own
// row of weights loaded from weights.mem.
//
// Weight memory layout (weights.mem):
//   Lines 0-3   : neuron 0 weights (one per input)
//   Lines 4-7   : neuron 1 weights
//   ...
//   Lines 28-31 : neuron 7 weights
//
// The host controller feeds inputs one cycle at a time via data_in / weight
// strobing is done internally – the layer reads weight_mem using
// a counter so the caller only needs to provide data_in and handshake signals.

`timescale 1ns / 1ps

module layer #(
    parameter NUM_NEURONS  = 8,
    parameter NUM_INPUTS   = 4,
    parameter WEIGHTS_FILE = "weights/weights.mem",
    parameter BIAS_OFFSET  = 0    // first bias index inside biases.mem
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,           // pulse high for 1 cycle to begin
    input  wire [15:0] data_in,         // Q8 input – one value per cycle
    output wire [NUM_NEURONS*16-1:0] layer_out,   // packed output bus
    output wire        all_valid        // high for 1 cycle when all neurons done
);

    // ── weight and bias memories ──────────────────────────────────────────────
    reg [15:0] weight_mem [0 : NUM_NEURONS*NUM_INPUTS - 1];
    reg [15:0] bias_mem   [0 : NUM_NEURONS - 1];

    initial begin
        $readmemh(WEIGHTS_FILE, weight_mem);
        $readmemh("weights/biases.mem", bias_mem, BIAS_OFFSET,
                  BIAS_OFFSET + NUM_NEURONS - 1);
    end

    // ── input sequencing counter ──────────────────────────────────────────────
    reg [2:0]  inp_cnt;   // counts 0 to NUM_INPUTS-1
    reg        running;
    reg [15:0] data_reg;  // registered copy of data_in

    // We latch start on posedge so data_in is stable
    always @(posedge clk) begin
        if (!rst_n) begin
            inp_cnt  <= 0;
            running  <= 0;
            data_reg <= 0;
        end else begin
            data_reg <= data_in;
            if (start) begin
                inp_cnt <= 1;
                running <= 1;
            end else if (running) begin
                if (inp_cnt == NUM_INPUTS - 1) begin
                    inp_cnt <= 0;
                    running <= 0;
                end else begin
                    inp_cnt <= inp_cnt + 1;
                end
            end
        end
    end

    // ── per-neuron wiring ─────────────────────────────────────────────────────
    wire [15:0] neuron_out  [0 : NUM_NEURONS-1];
    wire        neuron_valid[0 : NUM_NEURONS-1];

    genvar n;
    generate
        for (n = 0; n < NUM_NEURONS; n = n + 1) begin : gen_neurons
            // Select the right weight: neuron n, current input index
            wire [15:0] w = weight_mem[n * NUM_INPUTS + inp_cnt];

            wire last_pulse = running && (inp_cnt == NUM_INPUTS - 1);

            neuron u_neuron (
                .clk       (clk),
                .rst_n     (rst_n),
                .start     (start),
                .data_in   (data_reg),
                .weight_in (w),
                .bias      (bias_mem[n]),
                .last      (last_pulse),
                .out       (neuron_out[n]),
                .valid     (neuron_valid[n])
            );

            // Pack into output bus
            assign layer_out[n*16 +: 16] = neuron_out[n];
        end
    endgenerate

    // all_valid goes high when every neuron has raised its valid flag
    // (they should all fire on the same cycle since they share the counter)
    assign all_valid = &{neuron_valid[7], neuron_valid[6],
                         neuron_valid[5], neuron_valid[4],
                         neuron_valid[3], neuron_valid[2],
                         neuron_valid[1], neuron_valid[0]};

endmodule
