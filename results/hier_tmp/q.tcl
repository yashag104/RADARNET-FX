open_checkpoint C:/Users/hp/RADARNET-FX/zedboard_project/vivado/radarnet_zed.runs/impl_1/rn_bd_wrapper_routed.dcp
set r rn_bd_i/rn/inst/u_rn
foreach c [get_cells -quiet $r/*] { if {![get_property IS_PRIMITIVE $c]} { puts "SUB $c" } }
foreach c [get_cells -hier -filter {PRIMITIVE_GROUP == BLOCKRAM}] { puts "BRAM $c [get_property REF_NAME $c]" }
foreach c [get_cells -hier -filter {PRIMITIVE_GROUP == ARITHMETIC && REF_NAME =~ DSP*}] { puts "DSP $c" }
# LUT/FF grouped by the first name token below u_rn
array set L {}; array set F {}
foreach c [get_cells -hier -filter "NAME =~ $r/* && IS_PRIMITIVE"] {
  set n [string range $c [expr [string length $r]+1] end]
  regsub {[\[/_].*$} $n {} k
  set g [get_property PRIMITIVE_GROUP $c]
  if {$g eq "LUT"} { incr L($k) } elseif {$g eq "FLOP_LATCH"} { incr F($k) }
}
foreach k [lsort [array names L]] { puts "GRP $k LUT=$L($k) FF=[expr {[info exists F($k)]?$F($k):0}]" }
foreach k [lsort [array names F]] { if {![info exists L($k)]} { puts "GRP $k LUT=0 FF=$F($k)" } }
