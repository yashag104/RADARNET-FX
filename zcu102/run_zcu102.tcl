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

# Every step is announced and timed, so a hang shows exactly where it stops.
proc step {n msg cmd} {
    say [format "\[%s\] step %-2s %s ..." [clock format [clock seconds] -format %H:%M:%S] $n $msg]
    if {[catch {uplevel 1 $cmd} res]} {
        say "*** step $n FAILED: $res"
        say "targets now:"; catch {say [targets]}
        exit 1
    }
    return $res
}

say "=== RadarNet-1D on ZCU102 ==="
step 1  "connect to hw_server"            {connect}
say [targets]
step 2  "select APU, system reset"        {targets -set -nocase -filter {name =~ "APU*"}; rst -system; after 3000}
step 3  "program FPGA $BIT"               {fpga -file $BIT}
step 4  "load hardware description (xsa)" {targets -set -nocase -filter {name =~ "APU*"}
    loadhw -hw $XSA -mem-ranges [list {0x80000000 0xbfffffff} {0x400000000 0x5ffffffff} \
                                      {0x1000000000 0x7fffffffff}] -regs
    configparams force-mem-access 1}
step 5  "reset A53 core 0"                {targets -set -nocase -filter {name =~ "*A53*#0"}; rst -processor}
step 6  "source psu_init.tcl"             {source $INIT}
step 7  "psu_init (clocks, DDR, MIO)"     {targets -set -nocase -filter {name =~ "APU*"}; psu_init}
step 8  "remove PS-PL isolation"          {after 1000; psu_ps_pl_isolation_removal; after 1000; psu_ps_pl_reset_config}
step 9  "PS protection config"            {catch {psu_protection}}
step 10 "reset A53 core 0 again"          {targets -set -nocase -filter {name =~ "*A53*#0"}; rst -processor -clear-registers}
step 11 "download rn_bench.elf"           {dow $ELF}
step 12 "check accelerator INFO register" {configparams force-mem-access 0
    targets -set -nocase -filter {name =~ "APU*"}
    say [format "  INFO @0xA0000014 = 0x%08X (expect 0x524E0410)" [mrd -value 0xA0000014]]}
step 13 "start program"                   {targets -set -nocase -filter {name =~ "*A53*#0"}; con}
say "running benchmark (about 1 minute) ..."

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
