#!/usr/bin/env bash
# run_impl.sh <W> <period_ns> [mem_tag] [classes] [part] [pe_dsp] [tag] — run impl.tcl
# through Windows Vivado from WSL. ROMs come from rtl_v2/build/mem_<mem_tag> (default w<W>).
#   VIVADO_BAT : path to vivado.bat (default D:\2025.2\Vivado\bin\vivado.bat)
set -euo pipefail
W=$1; PER=$2; MEM=${3:-w$W}; NC=${4:-4}; PART=${5:-xc7z020clg484-1}; PEDSP=${6:--1}
ROOT_WIN=$(wslpath -m "$(cd "$(dirname "$0")/../.." && pwd)")
VIVADO_BAT=${VIVADO_BAT:-'D:\2025.2\Vivado\bin\vivado.bat'}
TAG=${7:-${MEM}_p${PER}}
cmd.exe /c "$VIVADO_BAT -mode batch -nojournal -log $ROOT_WIN/rtl_v2/build/impl_$TAG.log \
  -source $ROOT_WIN/rtl_v2/vivado/impl.tcl \
  -tclargs $W $PER $ROOT_WIN/rtl_v2/build/mem_$MEM $ROOT_WIN/rtl_v2/build/impl_$TAG $NC $PART $PEDSP" \
  2>&1 | grep -E "^SUMMARY|^ERROR" || true
