transcript off
onbreak {quit -force}
onerror {quit -force}
transcript on

vlib work
vlib activehdl/xpm
vlib activehdl/blk_mem_gen_v8_4_12
vlib activehdl/xil_defaultlib

vmap xpm activehdl/xpm
vmap blk_mem_gen_v8_4_12 activehdl/blk_mem_gen_v8_4_12
vmap xil_defaultlib activehdl/xil_defaultlib

vlog -work xpm  -sv2k12 "+incdir+../../../../../../../../../AMDDesignTools/2025.2/Vivado/data/rsb/busdef" -l xpm -l blk_mem_gen_v8_4_12 -l xil_defaultlib \
"C:/AMDDesignTools/2025.2/Vivado/data/ip/xpm/xpm_cdc/hdl/xpm_cdc.sv" \
"C:/AMDDesignTools/2025.2/Vivado/data/ip/xpm/xpm_memory/hdl/xpm_memory.sv" \

vcom -work xpm -93  \
"C:/AMDDesignTools/2025.2/Vivado/data/ip/xpm/xpm_VCOMP.vhd" \

vlog -work blk_mem_gen_v8_4_12  -v2k5 "+incdir+../../../../../../../../../AMDDesignTools/2025.2/Vivado/data/rsb/busdef" -l xpm -l blk_mem_gen_v8_4_12 -l xil_defaultlib \
"../../ipstatic/simulation/blk_mem_gen_v8_4.v" \

vlog -work xil_defaultlib  -v2k5 "+incdir+../../../../../../../../../AMDDesignTools/2025.2/Vivado/data/rsb/busdef" -l xpm -l blk_mem_gen_v8_4_12 -l xil_defaultlib \
"../../../project_fpgacomp.gen/sources_1/ip/blk_mem_gen_0/sim/blk_mem_gen_0.v" \
"../../../project_fpgacomp.srcs/sources_1/new/argmax_out.v" \
"../../../project_fpgacomp.srcs/sources_1/new/input_buffer.v" \
"../../../project_fpgacomp.srcs/sources_1/new/radarnet_top.v" \
"../../../project_fpgacomp.srcs/sources_1/new/weight_rom.v" \
"../../../project_fpgacomp.srcs/sim_1/new/tb_radarnet.v" \

vlog -work xil_defaultlib \
"glbl.v"

