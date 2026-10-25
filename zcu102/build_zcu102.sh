#!/usr/bin/env bash
# build_zcu102.sh — build the ZCU102 bitstream + hardware file on this Linux machine.
# Runs unattended for about 30-60 minutes and produces rn_zcu102.bit,
# rn_zcu102.xsa and psu_init.tcl next to this script (used by run_zcu102.sh).
# Needs a Vivado licence that covers the ZU9EG.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

find_tool() {   # find_tool <name> <glob>...
    local name=$1; shift
    if command -v "$name" >/dev/null 2>&1; then command -v "$name"; return; fi
    local p
    for p in "$@"; do
        for f in $p; do [ -x "$f" ] && { echo "$f"; return; }; done
    done
    return 1
}
VIVADO=$(find_tool vivado \
    "/tools/Xilinx/Vivado/*/bin/vivado" "/tools/Xilinx/*/Vivado/bin/vivado" \
    "/opt/Xilinx/Vivado/*/bin/vivado" "/opt/Xilinx/*/Vivado/bin/vivado" \
    "/tools/AMD/*/Vivado/bin/vivado" "$HOME/Xilinx/*/Vivado/bin/vivado") || {
    echo "*** vivado not found. Run:  source <Vivado install>/settings64.sh  and try again."
    exit 1
}
echo "Using $VIVADO"
echo "Building the ZCU102 hardware (W=16, 250 MHz). This takes 30-60 minutes..."
echo "Full log: $PWD/build/build.log"
mkdir -p build

"$VIVADO" -mode batch -nojournal -log build/build.log -source hw/build_zynq.tcl \
    -tclargs 16 4 "$PWD/hw/mem_w16" "$PWD/build" 250 zcu102 1 > build/vivado_console.txt 2>&1 || true

if [ ! -f build/rn_zed.xsa ]; then
    echo
    echo "*** BUILD FAILED - send build/build.log to Claude."
    grep -E "^ERROR" build/build.log | head -5 || true
    exit 1
fi
cp -f "$(find build/proj -name rn_bd_wrapper.bit | head -1)" rn_zcu102.bit
cp -f build/rn_zed.xsa rn_zcu102.xsa
python3 - <<'EOF'
import zipfile
with zipfile.ZipFile("rn_zcu102.xsa") as z:
    name = next(n for n in z.namelist() if n.endswith("psu_init.tcl"))
    open("psu_init.tcl", "wb").write(z.read(name))
EOF
echo
grep "SYSTEM_SUMMARY" build/build.log || true
if [ -f rn_zcu102.bit ] && [ -f psu_init.tcl ]; then
    echo "BUILD OK - when rn_bench.elf is in this folder, run:  ./run_zcu102.sh"
else
    echo "*** Bitstream or psu_init.tcl missing - send build/build.log to Claude."
fi
