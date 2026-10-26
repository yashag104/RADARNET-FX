# create_project.tcl — Vivado project for the RadarNet-1D v2 accelerator on ZedBoard
# ==================================================================================
# In Vivado:  Tools → Run Tcl Script… → select this file   (or: source create_project.tcl)
#
# Creates  zedboard_project/vivado/radarnet_zed.xpr  with
#   • design sources : rtl_v2/src/*.v, rtl_v2/zynq/rn_axi.v, ROMs (mem/)
#   • block design   : Zynq PS7 (ZedBoard preset) + rn_axi at 0x43C0_0000, FCLK0 142.86 MHz
#   • simulation     : tb_rn_top (bit-exact check of rn_top against the golden model)
# Nothing is run automatically — use the Flow Navigator buttons afterwards.
set HERE [file dirname [file normalize [info script]]]
set ROOT [file normalize [file join $HERE ..]]
set W    8
set NC   4

set_param board.repoPaths [list [file join $ROOT rtl_v2 zynq board_files]]
create_project radarnet_zed [file join $HERE vivado] -part xc7z020clg484-1 -force
set_property board_part avnet.com:zedboard:part0:1.4 [current_project]
set_property target_language Verilog [current_project]

# ── Design sources ──────────────────────────────────────────────────
add_files -norecurse [concat [glob [file join $ROOT rtl_v2 src *.v]] \
                             [list [file join $ROOT rtl_v2 zynq rn_axi.v]]]
add_files -norecurse [list [file join $HERE mem weights.mem] [file join $HERE mem params.mem] \
                           [file join $HERE mem layer_table.vh]]
set_property file_type {Verilog Header} [get_files layer_table.vh]
set_property include_dirs [list [file join $HERE mem]] [get_filesets sources_1]

# ── Simulation: tb_rn_top, 10 vectors by default ────────────────────
add_files -fileset sim_1 -norecurse [list [file join $ROOT rtl_v2 sim tb_rn_top.v]]
add_files -fileset sim_1 -norecurse [glob [file join $HERE simdata *.mem]]
set_property include_dirs [list [file join $HERE mem]] [get_filesets sim_1]
set_property verilog_define {RN_VIVADO_SIM} [get_filesets sim_1]
set_property top tb_rn_top [get_filesets sim_1]
set_property top_lib xil_defaultlib [get_filesets sim_1]
set_property generic "W=$W NC=$NC NUM_VECTORS=10" [get_filesets sim_1]
set_property -name {xsim.simulate.runtime} -value {-all} -objects [get_filesets sim_1]
update_compile_order -fileset sources_1
update_compile_order -fileset sim_1

# ── Block design: PS7 + accelerator ─────────────────────────────────
create_bd_design rn_bd
create_bd_cell -type module -reference rn_axi rn
set_property -dict [list CONFIG.W $W CONFIG.NUM_CLASSES $NC] [get_bd_cells rn]
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7 ps
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable"} \
    [get_bd_cells ps]
set_property -dict [list CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ 142 \
                         CONFIG.PCW_USE_M_AXI_GP0 1] [get_bd_cells ps]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config {Clk_master {/ps/FCLK_CLK0} Clk_slave {Auto} Clk_xbar {Auto} Master {/ps/M_AXI_GP0} \
             Slave {/rn/s_axi} ddr_seg {Auto} intc_ip {New AXI Interconnect} master_apm {0}} \
    [get_bd_intf_pins rn/s_axi]
assign_bd_address
set_property offset 0x43C00000 [get_bd_addr_segs /ps/Data/SEG_rn_reg0]
set_property range 4K [get_bd_addr_segs /ps/Data/SEG_rn_reg0]
validate_bd_design
regenerate_bd_layout
save_bd_design

make_wrapper -files [get_files rn_bd.bd] -top
add_files -norecurse [glob [file join $HERE vivado radarnet_zed.gen sources_1 bd rn_bd hdl rn_bd_wrapper.v]]
set_property top rn_bd_wrapper [get_filesets sources_1]
update_compile_order -fileset sources_1

puts ""
puts "================================================================"
puts " Project ready: [file join $HERE vivado radarnet_zed.xpr]"
puts "  1. Flow Navigator → SIMULATION → Run Simulation → Run Behavioral"
puts "     (tb_rn_top: expect 'bit-exact logits : 10 / 10' and PASS in the Tcl console)"
puts "  2. Flow Navigator → PROGRAM AND DEBUG → Generate Bitstream"
puts "  3. File → Export → Export Hardware… (include bitstream) for Vitis"
puts "================================================================"
