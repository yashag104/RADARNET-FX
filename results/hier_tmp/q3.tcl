open_checkpoint C:/Users/hp/RADARNET-FX/zedboard_project/vivado/radarnet_zed.runs/impl_1/rn_bd_wrapper_routed.dcp
set o C:/Users/hp/RADARNET-FX/results/hier_tmp
foreach s {fbuf[1].u_buf fbuf[2].u_buf fbuf[0].u_buf} {
  regsub -all {[\[\]\.]} $s _ f
  report_utilization -cells [get_cells rn_bd_i/rn/inst/u_rn/$s] -file $o/u_$f.rpt
}
