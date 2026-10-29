open_checkpoint C:/Users/hp/RADARNET-FX/zedboard_project/vivado/radarnet_zed.runs/impl_1/rn_bd_wrapper_routed.dcp
set o C:/Users/hp/RADARNET-FX/results/vivado_reports/zedboard_system_142MHz
report_utilization -hierarchical -hierarchical_depth 6 -file $o/rn_bd_wrapper_utilization_hierarchical_routed.rpt
report_power -hierarchical_depth 6 -file $o/rn_bd_wrapper_power_hierarchical_routed.rpt
foreach c [get_cells -hier -filter {REF_NAME =~ rn_* || ORIG_REF_NAME =~ rn_*}] { puts "CELL $c [get_property REF_NAME $c] [get_property ORIG_REF_NAME $c]" }
