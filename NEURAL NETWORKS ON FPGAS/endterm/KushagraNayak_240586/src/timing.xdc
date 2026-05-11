create_clock -period 10.000 [get_ports clk]

set_property BITSTREAM.GENERAL.UNCONSTRAINEDPINS ALLOW [current_design]
