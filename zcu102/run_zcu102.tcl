# run_zcu102.tcl — program the ZCU102, run the benchmark, print the results
#   (started by run_zcu102.bat; needs the board's USB-JTAG connection)
set HERE [file dirname [file normalize [info script]]]
set BIT  [file join $HERE rn_zcu102.bit]
set XSA  [file join $HERE rn_zcu102.xsa]
set ELF  [file join $HERE rn_bench.elf]
set INIT [file join $HERE psu_init.tcl]
set RES  0xFFFF8000
set LOG  [open [file join $HERE zcu102_results.txt] w]

proc say {msg} { global LOG; puts $msg; puts $LOG $msg; flush $LOG }
# Read word i of the results struct; fall back to pausing core 0 if the
# debug port refuses memory access while it runs.
proc word {i} {
    global RES
    set addr [expr {$RES + 4 * $i}]
    if {![catch {targets -set -nocase -filter {name =~ "APU*"}; mrd -value $addr} v]} {
        return $v
    }
    targets -set -nocase -filter {name =~ "*A53*#0"}
    stop
    set v [mrd -value $addr]
    con
    return $v
}

say "=== RadarNet-1D on ZCU102 ==="
connect
targets -set -nocase -filter {name =~ "APU*"}
rst -system
after 3000
say "programming FPGA: $BIT"
fpga -file $BIT
targets -set -nocase -filter {name =~ "APU*"}
loadhw -hw $XSA -mem-ranges [list {0x80000000 0xbfffffff} {0x400000000 0x5ffffffff} \
                                  {0x1000000000 0x7fffffffff}] -regs
configparams force-mem-access 1
source $INIT
psu_init
after 1000
psu_ps_pl_isolation_removal
after 1000
psu_ps_pl_reset_config
catch {psu_protection}
targets -set -nocase -filter {name =~ "*A53*#0"}
rst -processor -clear-registers
dow $ELF
configparams force-mem-access 0
say "running benchmark (about 1 minute) ..."
con

set t0 [clock seconds]
while {[word 0] != 0x524E4231} {
    if {[clock seconds] - $t0 > 300} { say "ERROR: no result after 5 min"; exit 1 }
    after 1000
}
set info [word 1]; set n [word 2]; set arm_ok [word 3]; set hw_ok [word 4]
set hz [word 5]; set arm_t [word 6]; set hw_t [word 7]; set cyc [word 8]
set arm_us [expr {1e6 * $arm_t / double($hz)}]
set hw_us  [expr {1e6 * $hw_t / double($hz)}]
say ""
say [format "accelerator INFO        : 0x%08X  (W=%d, classes=%d)" $info [expr {$info & 0xFF}] [expr {($info >> 8) & 0xFF}]]
say [format "bit-exact vs golden     : A53 %d/%d   accelerator %d/%d" $arm_ok $n $hw_ok $n]
say [format "timer                   : %d Hz" $hz]
say [format "ARM Cortex-A53 software : %10.1f us / inference" $arm_us]
say [format "accelerator, end-to-end : %10.1f us / inference  (incl. AXI writes of 128 samples)" $hw_us]
say [format "accelerator PL cycles   : %d" $cyc]
say [format "speed-up (end-to-end)   : %.1fx" [expr {$arm_us / $hw_us}]]
say ""
say "Done. Results saved in zcu102_results.txt — send that file to Claude."
close $LOG
