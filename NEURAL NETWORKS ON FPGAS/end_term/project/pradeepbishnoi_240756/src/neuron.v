module neuron (
    input  wire        clk,
    input  wire        rst_n,      // Active-low reset
    input  wire        start,      // Start signal - begins a new neuron calculation
    input  wire [15:0] data_in,    // Q8 input data
    input  wire [15:0] weight_in,  // Q8 weight
    input  wire [15:0] bias,       // Q8 bias
    input  wire        last,       // Signals the last input of the current neuron
    output reg  [15:0] out,        // Final 16-bit output
    output reg         valid       // Pulses high for one cycle when output is ready
);

    // -------------------------------------------------------------------------
    // Internal Registers
    // -------------------------------------------------------------------------
    reg signed [31:0] acc;      // 32-bit accumulator for Q16 MAC results
    reg               active;   // Tracks whether a calculation is in progress
    reg               last_d;   // One-cycle delayed version of last signal

    // -------------------------------------------------------------------------
    // Combinational product of current inputs (Q8 x Q8 = Q16)
    // -------------------------------------------------------------------------
    wire signed [31:0] current_product = $signed(data_in) * $signed(weight_in);

    // -------------------------------------------------------------------------
    // Main Logic Block
    // -------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n)
    begin
        if (!rst_n)
        begin
            // Reset all registers to zero
            acc    <= 32'd0;
            out    <= 16'd0;
            valid  <= 1'b0;
            active <= 1'b0;
            last_d <= 1'b0;
        end
        else
        begin
            // Default: valid is low unless explicitly set high this cycle
            valid  <= 1'b0;

            // Delay last by one cycle so bias+ReLU stage sees fully updated acc
            last_d <= last;

            // -----------------------------------------------------------------
            // MAC Stage: runs every cycle while active or starting
            // On start : load first product directly (avoids dual-write bug
            //            where both reset and accumulate target acc together)
            // Otherwise: accumulate — this now also runs on the last cycle,
            //            ensuring the final product is registered into acc
            //            before the bias+ReLU stage reads it
            // -----------------------------------------------------------------
            if (start)
            begin
                acc    <= current_product;  // Load first product, skip reset+add race
                active <= 1'b1;
                valid  <= 1'b0;
            end
            else if (active)
            begin
                acc <= acc + current_product;  // Accumulate every cycle including last
            end

            // -----------------------------------------------------------------
            // Bias + ReLU Stage: triggers one cycle AFTER last goes high
            // By this point acc contains ALL products including the last one
            // Bias is shifted left by 8 to scale from Q8 to Q16 to match acc
            // -----------------------------------------------------------------
            if (last_d && active)
            begin
                // Calculation is complete
                active <= 1'b0;
                valid  <= 1'b1;

                // Add bias (scaled to Q16) to the fully accumulated result
                // Use a local variable for readability inside the always block
                begin : relu_block
                    reg signed [31:0] final_sum;
                    final_sum = acc + ($signed(bias) <<< 8);

                    if (final_sum[31])
                    begin
                        // MSB is 1 → result is negative → ReLU outputs zero
                        out <= 16'd0;
                    end
                    else if (final_sum > 32'h007FFF00)
                    begin
                        // Positive overflow → saturate to max Q8.8 value
                        out <= 16'h7FFF;
                    end
                    else
                    begin
                        // Extract Q8.8 result from Q16 accumulator
                        // Bits [23:8] give the correctly scaled 16-bit output
                        out <= final_sum[23:8];
                    end
                end
            end

        end
    end

endmodule

