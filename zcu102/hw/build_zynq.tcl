# build_zynq.tcl — Zynq system: processing system (board preset) + rn_axi accelerator
#
#   vivado -mode batch -source build_zynq.tcl -tclargs <W> <classes> <mem_dir> <out_dir> \
#          [fclk_mhz] [board: zed | zcu102 | zcu104] [pe_dsp]
#
#   zed    : ZedBoard, Zynq-7020, PS7, accelerator at 0x43C0_0000 on M_AXI_GP0
#   zcu102 : ZCU102, Zynq UltraScale+ ZU9EG, PS8, accelerator at 0xA000_0000 on M_AXI_HPM0_FPD
#   zcu104 : ZCU104, Zynq UltraScale+ ZU7EV, PS8, accelerator at 0xA000_0000 on M_AXI_HPM0_FPD
#
# Produces <out_dir>/rn_zed.xsa (bitstream included) plus utilization, timing
# and power reports of the complete system. <mem_dir> holds weights.mem,
# params.mem and layer_table.vh from `int_quant.py export --lanes <W>`.
set W     [lindex $argv 0]
set NC    [lindex $argv 1]
set MEM   [file normalize [lindex $argv 2]]
set OUT   [file normalize [lindex $argv 3]]
set FCLK  [expr {$argc > 4 ? [lindex $argv 4] : 142}]
set BOARD [expr {$argc > 5 ? [lindex $argv 5] : "zed"}]
set PEDSP [expr {$argc > 6 ? [lindex $argv 6] : -1}]
set HERE  [file normalize [file dirname [info script]]]
set SRC   [file normalize [file join $HERE .. src]]

set_param board.repoPaths [list [file join $HERE board_files]]
file mkdir $OUT
if {$BOARD eq "zcu104"} {
    create_project rn_zed $OUT/proj -part xczu7ev-ffvc1156-2-e -force
    set_property board_part xilinx.com:zcu104:part0:1.1 [current_project]
} elseif {$BOARD eq "zcu102"} {
    create_project rn_zed $OUT/proj -part xczu9eg-ffvb1156-2-e -force
    set_property board_part xilinx.com:zcu102:part0:3.4 [current_project]
} else {
    create_project rn_zed $OUT/proj -part xc7z020clg484-1 -force
    set_property board_part avnet.com:zedboard:part0:1.4 [current_project]
}

add_files [concat [glob $SRC/*.v] [list $HERE/rn_axi.v]]
add_files [list $MEM/weights.mem $MEM/params.mem $MEM/layer_table.vh]
set_property file_type {Verilog Header} [get_files layer_table.vh]
set_property include_dirs [list $MEM] [get_filesets sources_1]
update_compile_order -fileset sources_1

# ── Block design ────────────────────────────────────────────────────
create_bd_design rn_bd
create_bd_cell -type module -reference rn_axi rn
set rn_cfg [list CONFIG.W $W CONFIG.NUM_CLASSES $NC]
if {$PEDSP >= 0} { lappend rn_cfg CONFIG.PE_DSP $PEDSP }
set_property -dict $rn_cfg [get_bd_cells rn]

if {$BOARD in {zcu102 zcu104}} {
    create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e ps
    apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e \
        -config {apply_board_preset "1"} [get_bd_cells ps]
    set_property -dict [list CONFIG.PSU__USE__M_AXI_GP0 1 CONFIG.PSU__USE__M_AXI_GP1 0 \
                             CONFIG.PSU__USE__M_AXI_GP2 0 \
                             CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $FCLK] [get_bd_cells ps]
    set master /ps/M_AXI_HPM0_FPD
    set clk    /ps/pl_clk0
    set base   0xA0000000
} else {
    create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7 ps
    apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
        -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable"} \
        [get_bd_cells ps]
    set_property -dict [list CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ $FCLK \
                             CONFIG.PCW_USE_M_AXI_GP0 1] [get_bd_cells ps]
    set master /ps/M_AXI_GP0
    set clk    /ps/FCLK_CLK0
    set base   0x43C00000
}
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config [list Clk_master $clk Clk_slave {Auto} Clk_xbar {Auto} Master $master \
                  Slave {/rn/s_axi} ddr_seg {Auto} intc_ip {New AXI Interconnect} master_apm {0}] \
    [get_bd_intf_pins rn/s_axi]
assign_bd_address
set seg [get_bd_addr_segs -quiet /ps/Data/SEG_rn_reg0]
if {$seg eq ""} { error "accelerator address segment /ps/Data/SEG_rn_reg0 not found" }
set_property offset $base $seg
set_property range 4K $seg
validate_bd_design
save_bd_design

make_wrapper -files [get_files rn_bd.bd] -top
add_files -norecurse [glob $OUT/proj/rn_zed.gen/sources_1/bd/rn_bd/hdl/rn_bd_wrapper.v]
set_property top rn_bd_wrapper [current_fileset]
update_compile_order -fileset sources_1

# ── Implementation ──────────────────────────────────────────────────
launch_runs synth_1 -jobs 4
wait_on_run synth_1
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
open_run impl_1
report_utilization    -file $OUT/sys_util.rpt
report_timing_summary -max_paths 5 -file $OUT/sys_timing.rpt
report_power          -file $OUT/sys_power.rpt
write_hw_platform -fixed -include_bit -force $OUT/rn_zed.xsa

set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
set f [get_property CONFIG.FREQ_HZ [get_bd_pins $clk]]
puts "SYSTEM_SUMMARY board=$BOARD fclk_hz=$f wns=$wns xsa=$OUT/rn_zed.xsa"
