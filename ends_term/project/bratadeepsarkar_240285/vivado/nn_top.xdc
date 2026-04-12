## XDC Constraints — Basys3 (XC7A35T-CPG236-1)
## For nn_top module

## ─── Clock (100 MHz) ──────────────────────────────────────────────────────────
set_property PACKAGE_PIN W5 [get_ports clk]
    set_property IOSTANDARD LVCMOS33 [get_ports clk]
    create_clock -add -name sys_clk_pin -period 10.00 -waveform {0 5} [get_ports clk]

## Buttons
## Center button = reset (active-high externally, directly wired to btn_rst)
set_property PACKAGE_PIN U18 [get_ports btn_rst]
    set_property IOSTANDARD LVCMOS33 [get_ports btn_rst]

## Left button = start inference
set_property PACKAGE_PIN W19 [get_ports start]
    set_property IOSTANDARD LVCMOS33 [get_ports start]

## ─── Switches (Sample Selection) ───────────────────────────────────────────────
## sw[0] = V17
set_property PACKAGE_PIN V17 [get_ports {sw[0]}]					
	set_property IOSTANDARD LVCMOS33 [get_ports {sw[0]}]
## sw[1] = V16
set_property PACKAGE_PIN V16 [get_ports {sw[1]}]					
	set_property IOSTANDARD LVCMOS33 [get_ports {sw[1]}]
## sw[2] = W16
set_property PACKAGE_PIN W16 [get_ports {sw[2]}]					
	set_property IOSTANDARD LVCMOS33 [get_ports {sw[2]}]
## sw[3] = W17
set_property PACKAGE_PIN W17 [get_ports {sw[3]}]					
	set_property IOSTANDARD LVCMOS33 [get_ports {sw[3]}]

## ─── LEDs (predicted class + done) ────────────────────────────────────────────
## LED 0 = predicted_class[0]
set_property PACKAGE_PIN U16 [get_ports {predicted_class[0]}]
    set_property IOSTANDARD LVCMOS33 [get_ports {predicted_class[0]}]

## LED 1 = predicted_class[1]
set_property PACKAGE_PIN E19 [get_ports {predicted_class[1]}]
    set_property IOSTANDARD LVCMOS33 [get_ports {predicted_class[1]}]

## LED 15 = done (rightmost LED, easy to spot)
set_property PACKAGE_PIN L1 [get_ports done]
    set_property IOSTANDARD LVCMOS33 [get_ports done]
