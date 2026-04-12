# Vivado Build Script for FPGA Neural Network
# Target: Basys 3 (Artix-7 xc7a35tcpg236-1)

set project_name "fpga_nn"
set project_dir "./vivado/project"
set src_dir "./src"
set constr_dir "./vivado"
set weights_dir "./weights"

# Create project
create_project -force $project_name $project_dir -part xc7a35tcpg236-1

# Add Source Files
add_files [glob $src_dir/*.v]

# Add Constraints
add_files -fileset constrs_1 $constr_dir/nn_top.xdc

# Add Memory Files (.mem)
# We add them to the design sources so Vivado knows to include them in the build
add_files [glob $weights_dir/*.mem]

# Set Top Module
set_property top nn_top [current_fileset]

# --- Synthesis ---
launch_runs synth_1 -jobs 4
wait_on_run synth_1

# --- Implementation (with Post-Route Physical Opt for timing closure) ---
set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]
set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1

puts "Build Complete! Bitstream generated at $project_dir/$project_name.runs/impl_1/nn_top.bit"

# --- Reporting ---
open_run impl_1
report_utilization -file $project_dir/utilization.txt
report_timing_summary -file $project_dir/timing.txt

puts "Reports generated: utilization.txt and timing.txt"
