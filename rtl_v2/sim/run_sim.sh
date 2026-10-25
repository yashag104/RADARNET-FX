#!/usr/bin/env bash
# run_sim.sh <W> [num_vectors] — export golden data for W lanes and run the
# bit-exact RTL testbench with Icarus Verilog.
#   PYTHON  : python with numpy + torch (default: python3)
#   INTQ    : integer model .npz (default: ../../checkpoints/radarnet_intq.npz)
set -euo pipefail
W=${1:-8}
N=${2:-50}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
PYTHON=${PYTHON:-python3}
INTQ=${INTQ:-$ROOT/checkpoints/radarnet_intq.npz}
RUN=${RUN_DIR:-$HERE/run_w$W}
TB_INPUTS=${TB_INPUTS:-$ROOT/tb_vectors/input_vectors.hex}
TB_LABELS=${TB_LABELS:-$ROOT/tb_vectors/true_labels.hex}

mkdir -p "$RUN"
(cd "$ROOT/python" && "$PYTHON" int_quant.py export --intq "$INTQ" --lanes "$W" \
    --tb-inputs "$TB_INPUTS" --out-dir "$RUN")
cp "$TB_INPUTS" "$RUN/input_vectors.hex"
cp "$TB_LABELS" "$RUN/true_labels.hex"
NC=$(sed -n 's/^localparam EXPORT_CLASSES = \([0-9]*\);/\1/p' "$RUN/layer_table.vh")
iverilog -g2012 -I "$RUN" -P tb_rn_top.W="$W" -P tb_rn_top.NUM_VECTORS="$N" -P tb_rn_top.NC="$NC" \
    -o "$RUN/sim" "$HERE/tb_rn_top.v" "$ROOT"/rtl_v2/src/*.v
(cd "$RUN" && vvp -n sim)
