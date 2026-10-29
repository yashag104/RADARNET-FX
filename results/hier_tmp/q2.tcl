open_checkpoint C:/Users/hp/RADARNET-FX/zedboard_project/vivado/radarnet_zed.runs/impl_1/rn_bd_wrapper_routed.dcp
foreach c [get_cells -hier -filter {REF_NAME =~ RAMB*}] { puts "BRAM $c [get_property REF_NAME $c]" }
foreach c [get_cells -hier -filter {REF_NAME =~ DSP48*}] { puts "DSP $c" }
foreach s {fbuf[0].u_buf fbuf[1].u_buf fbuf[2].u_buf u_gap u_pe} {
  set c [get_cells rn_bd_i/rn/inst/u_rn/$s]
  set p [get_cells -hier -filter "NAME =~ rn_bd_i/rn/inst/u_rn/$s/* && IS_PRIMITIVE"]
  puts "MOD $s LUT=[llength [filter $p {PRIMITIVE_GROUP==LUT}]] FF=[llength [filter $p {PRIMITIVE_GROUP==FLOP_LATCH}]] BRAM=[llength [filter $p {REF_NAME=~RAMB*}]] DSP=[llength [filter $p {REF_NAME=~DSP48*}]]"
}
