# impl.tcl — out-of-context synth + place + route of rn_top on the ZedBoard part
#
#   vivado -mode batch -source impl.tcl -tclargs <W> <clk_period_ns> <mem_dir> <out_dir> \
#          [classes] [part] [pe_dsp]
#
# <mem_dir> holds weights.mem, params.mem and layer_table.vh exported by
# `int_quant.py export --lanes <W>`. Reports land in <out_dir>.
set W      [lindex $argv 0]
set PERIOD [lindex $argv 1]
set MEM    [file normalize [lindex $argv 2]]
set OUT    [file normalize [lindex $argv 3]]
set NC     [expr {$argc > 4 ? [lindex $argv 4] : 4}]
set SRC    [file normalize [file join [file dirname [info script]] .. src]]
set PART   [expr {$argc > 5 ? [lindex $argv 5] : "xc7z020clg484-1"}]
set PEDSP  [expr {$argc > 6 ? [lindex $argv 6] : -1}]

file mkdir $OUT
cd $OUT
set_part $PART
read_verilog [glob $SRC/*.v]
read_mem [list $MEM/weights.mem $MEM/params.mem]

synth_design -top rn_top -part $PART -mode out_of_context \
    -include_dirs [list $MEM] -generic W=$W -generic NUM_CLASSES=$NC \
    {*}[expr {$PEDSP >= 0 ? [list -generic PE_DSP=$PEDSP] : {}}] -flatten_hierarchy rebuilt
create_clock -name clk -period $PERIOD [get_ports clk]
# Out-of-context: tell Vivado the clock arrives on a global buffer, otherwise the
# clock is routed on local fabric and both setup and hold slack are unrealistic.
set bufg [expr {[string match xczu* $PART] || [string match xck* $PART] ? "BUFGCE_X0Y0" : "BUFGCTRL_X0Y0"}]
set_property HD.CLK_SRC $bufg [get_ports clk]
set_input_delay  0 -clock clk [get_ports -filter {DIRECTION == IN && NAME != clk}]
set_output_delay 0 -clock clk [all_outputs]

opt_design
place_design
phys_opt_design
route_design

write_checkpoint -force routed.dcp
report_utilization    -file util.rpt
report_utilization    -hierarchical -file util_hier.rpt
report_timing_summary -max_paths 5 -file timing.rpt
report_power          -file power.rpt
report_ram_utilization -file ram.rpt

# All paths, plus the accelerator's internal register-to-register paths. Port
# paths of a bare OOC module have no real driver (0 ns input delay, no source
# register); the I/O boundary is verified in the full system build instead.
set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
set whs [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -hold]]
set regs [all_registers]
set iwns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup -from $regs -to $regs]]
set iwhs [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -hold -from $regs -to $regs]]
report_timing -setup -from $regs -to $regs -max_paths 5 -file timing_internal.rpt
report_timing -hold  -from $regs -to $regs -max_paths 5 -append -file timing_internal.rpt
set fmax [expr {1000.0 / ($PERIOD - $iwns)}]
set met [expr {$iwns >= 0 && $iwhs >= 0 ? "MET" : "VIOLATED"}]
set fh [open summary.txt w]
puts $fh "W=$W period=$PERIOD internal_wns=$iwns internal_whs=$iwhs internal_timing=$met all_paths_wns=$wns all_paths_whs=$whs fmax_est_mhz=[format %.1f $fmax]"
puts $fh "LUT=[llength [get_cells -hier -filter {PRIMITIVE_GROUP == LUT}]]"
close $fh
puts "SUMMARY W=$W period=$PERIOD internal_wns=$iwns internal_whs=$iwhs internal_timing=$met all_paths_wns=$wns all_paths_whs=$whs fmax_est_mhz=[format %.1f $fmax]"
