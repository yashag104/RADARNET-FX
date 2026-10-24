# impl.tcl — out-of-context synth + place + route of rn_top on the ZedBoard part
#
#   vivado -mode batch -source impl.tcl -tclargs <W> <clk_period_ns> <mem_dir> <out_dir>
#
# <mem_dir> holds weights.mem, params.mem and layer_table.vh exported by
# `int_quant.py export --lanes <W>`. Reports land in <out_dir>.
set W      [lindex $argv 0]
set PERIOD [lindex $argv 1]
set MEM    [file normalize [lindex $argv 2]]
set OUT    [file normalize [lindex $argv 3]]
set SRC    [file normalize [file join [file dirname [info script]] .. src]]
set PART   xc7z020clg484-1

file mkdir $OUT
cd $OUT
set_part $PART
read_verilog [glob $SRC/*.v]
read_mem [list $MEM/weights.mem $MEM/params.mem]

synth_design -top rn_top -part $PART -mode out_of_context \
    -include_dirs [list $MEM] -generic W=$W -flatten_hierarchy rebuilt
create_clock -name clk -period $PERIOD [get_ports clk]
set_input_delay  0 -clock clk [get_ports -filter {DIRECTION == IN && NAME != clk}]
set_output_delay 0 -clock clk [all_outputs]

opt_design
place_design
phys_opt_design
route_design

report_utilization    -file util.rpt
report_utilization    -hierarchical -file util_hier.rpt
report_timing_summary -max_paths 5 -file timing.rpt
report_power          -file power.rpt
report_ram_utilization -file ram.rpt

set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
set fmax [expr {1000.0 / ($PERIOD - $wns)}]
set fh [open summary.txt w]
puts $fh "W=$W period=$PERIOD wns=$wns fmax_mhz=[format %.1f $fmax]"
puts $fh "LUT=[llength [get_cells -hier -filter {PRIMITIVE_GROUP == LUT}]]"
close $fh
puts "SUMMARY W=$W period=$PERIOD wns=$wns fmax_mhz=[format %.1f $fmax]"
