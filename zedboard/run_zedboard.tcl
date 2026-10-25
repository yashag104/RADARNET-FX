# run_zedboard.tcl — program the ZedBoard, run the benchmark, print the results
#   (started by run_zedboard.bat; needs only the USB cable on the PROG/JTAG port)
set HERE [file dirname [file normalize [info script]]]
set BIT  [file join $HERE rn_zed.bit]
set ELF  [file join $HERE rn_bench.elf]
set PS7  [file join $HERE ps7_init.tcl]
set RES  0xFFFF8000
set LOG  [open [file join $HERE zedboard_results.txt] w]

proc say {msg} { global LOG; puts $msg; puts $LOG $msg; flush $LOG }
# Read word i of the results struct. Reads go through the APU debug port while
# the core runs; if a setup refuses that, pause core 0 briefly instead.
proc word {i} {
    global RES
    set addr [expr {$RES + 4 * $i}]
    if {![catch {targets -set -nocase -filter {name =~ "APU*"}; mrd -value $addr} v]} {
        return $v
    }
    targets -set -nocase -filter {name =~ "*Cortex-A9*#0"}
    stop
    set v [mrd -value $addr]
    con
    return $v
}

say "=== RadarNet-1D on ZedBoard ==="
connect
targets -set -nocase -filter {name =~ "APU*"}
rst -system
after 1000
say "programming FPGA: $BIT"
fpga -file $BIT
source $PS7
ps7_init
ps7_post_config
targets -set -nocase -filter {name =~ "*Cortex-A9*#0"}
rst -processor
dow $ELF
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
say [format "bit-exact vs golden     : ARM %d/%d   accelerator %d/%d" $arm_ok $n $hw_ok $n]
say [format "ARM Cortex-A9 software  : %10.1f us / inference" $arm_us]
say [format "accelerator, end-to-end : %10.1f us / inference  (incl. AXI writes of 128 samples)" $hw_us]
say [format "accelerator PL cycles   : %d" $cyc]
say [format "speed-up (end-to-end)   : %.1fx" [expr {$arm_us / $hw_us}]]
say ""
say "POWER MEASUREMENT (optional, multimeter on header J21 = 10 mOhm sense resistor):"
say "the board now cycles idle -> ARM loop -> accelerator loop, 20 s each."
say "Read the mV on J21 during each phase (current = mV / 10 mOhm, power = 12 V x current)."
set last -1
set t0 [clock seconds]
while {[clock seconds] - $t0 < 130} {
    set ph [word 9]
    if {$ph != $last} {
        set name [lindex {"-" "IDLE" "ARM inference loop" "ACCELERATOR loop"} $ph]
        say [format "  [clock format [clock seconds] -format %H:%M:%S]  phase %d: %s (20 s)" $ph $name]
        set last $ph
    }
    after 500
}
say ""
say "Done. Results saved in zedboard_results.txt — send that file to Claude."
close $LOG
