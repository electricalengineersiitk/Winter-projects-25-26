## nn_top.xdc
## Minimal constraints for Basys3 (Artix-7 xc7a35t)

## 100 MHz system clock
create_clock -period 10.000 [get_ports clk]
set_input_jitter [get_clocks -of_objects [get_ports clk]] 0.100

## Clock pin – Basys3 W5
set_property PACKAGE_PIN W5 [get_ports clk]
set_property IOSTANDARD LVCMOS33 [get_ports clk]

## Active-low reset on centre push-button (U18 on Basys3)
set_property PACKAGE_PIN U18 [get_ports rst_n]
set_property IOSTANDARD LVCMOS33 [get_ports rst_n]

## Start on slide-switch SW0 (V17)
set_property PACKAGE_PIN V17 [get_ports start]
set_property IOSTANDARD LVCMOS33 [get_ports start]

## Input features on SW1-SW4 (V16, W16, W17, W15) – 1-bit each for demo
set_property PACKAGE_PIN V16 [get_ports {in0[0]}]
set_property IOSTANDARD LVCMOS33 [get_ports {in0[0]}]

## Done signal on LED0 (U16)
set_property PACKAGE_PIN U16 [get_ports done]
set_property IOSTANDARD LVCMOS33 [get_ports done]

## Predicted class on LED1-LED2 (E19, U19)
set_property PACKAGE_PIN E19 [get_ports {pred_class[0]}]
set_property IOSTANDARD LVCMOS33 [get_ports {pred_class[0]}]
set_property PACKAGE_PIN U19 [get_ports {pred_class[1]}]
set_property IOSTANDARD LVCMOS33 [get_ports {pred_class[1]}]
