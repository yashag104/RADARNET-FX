#!/usr/bin/env bash
# run_impl.sh <W> <period_ns> — run impl.tcl through Windows Vivado from WSL.
#   VIVADO_BAT : path to vivado.bat (default D:\2025.2\Vivado\bin\vivado.bat)
set -euo pipefail
W=$1; PER=$2
ROOT_WIN=$(wslpath -m "$(cd "$(dirname "$0")/../.." && pwd)")
VIVADO_BAT=${VIVADO_BAT:-'D:\2025.2\Vivado\bin\vivado.bat'}
TAG=w${W}_p${PER}
cmd.exe /c "$VIVADO_BAT -mode batch -nojournal -log $ROOT_WIN/rtl_v2/build/impl_$TAG.log \
  -source $ROOT_WIN/rtl_v2/vivado/impl.tcl \
  -tclargs $W $PER $ROOT_WIN/rtl_v2/build/mem_w$W $ROOT_WIN/rtl_v2/build/impl_$TAG" \
  2>&1 | grep -E "^SUMMARY|^ERROR" || true
