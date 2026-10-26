#!/usr/bin/env bash
# run_zedboard.sh — program the ZedBoard over JTAG, run the benchmark, print results.
# Needs rn_zed.bit / ps7_init.tcl and
# rn_bench.elf in this folder; jumpers JP7-JP11 all to GND (JTAG boot).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
for f in rn_zed.bit ps7_init.tcl rn_bench.elf; do
    [ -f "$f" ] || { echo "*** missing $f"; exit 1; }
done
XSDB=""
if command -v xsdb >/dev/null 2>&1; then XSDB=$(command -v xsdb); fi
if [ -z "$XSDB" ]; then
    for f in /tools/Xilinx/Vitis/*/bin/xsdb /tools/Xilinx/*/Vitis/bin/xsdb \
             /opt/Xilinx/Vitis/*/bin/xsdb /opt/Xilinx/*/Vitis/bin/xsdb \
             /tools/AMD/*/Vitis/bin/xsdb "$HOME"/Xilinx/*/Vitis/bin/xsdb \
             /tools/Xilinx/Vivado/*/bin/xsdb /tools/Xilinx/*/Vivado/bin/xsdb; do
        [ -x "$f" ] && { XSDB=$f; break; }
    done
fi
[ -n "$XSDB" ] || { echo "*** xsdb not found. Run:  source <Vitis install>/settings64.sh  and try again."; exit 1; }
echo "Using $XSDB"
"$XSDB" run_zedboard.tcl
