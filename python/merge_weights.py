#!/usr/bin/env python3
"""
merge_weights.py — Merge Per-Layer .hex Files into Unified weights_all.hex
===========================================================================
Reads the 16 per-layer INT8 weight .hex files in EXECUTION ORDER and
concatenates them into a single weights_all.hex for weight_rom.v.

The file is padded with 00 to TOTAL_DEPTH (32768) entries.
Also copies weights_all.hex to --sim-dir so all simulation hex files are together.

Usage:
    python merge_weights.py --weights-dir ../weights --total-depth 32768
"""
import argparse
import os
import shutil

# Execution order: bias first, then weights, for each layer
EXECUTION_ORDER = [
    "L1_conv_in_bias",       # 16 values
    "L1_conv_in",            # 224 values
    "L2_conv_down_bias",     # 32 values
    "L2_conv_down",          # 1536 values
    "L3_res1_conv1_bias",    # 32 values
    "L3_res1_conv1",         # 3072 values
    "L4_res1_conv2_bias",    # 32 values
    "L4_res1_conv2",         # 3072 values
    "L5_res2_conv1_bias",    # 64 values
    "L5_res2_conv1",         # 6144 values
    "L6_res2_conv2_bias",    # 64 values
    "L6_res2_conv2",         # 12288 values
    "L5_res2_proj_bias",     # 64 values
    "L5_res2_proj",          # 2048 values
    "L8_fc_bias",            # 4 values
    "L8_fc",                 # 256 values
]


def main():
    parser = argparse.ArgumentParser(description="Merge per-layer .hex into weights_all.hex")
    parser.add_argument("--weights-dir", type=str, default=os.path.join("..", "weights"))
    parser.add_argument("--sim-dir", type=str, default=os.path.join("..", "tb_vectors"),
                        help="Also copy weights_all.hex here (default: ../tb_vectors)")
    parser.add_argument("--total-depth", type=int, default=32768)
    args = parser.parse_args()

    all_lines = []
    addr = 0

    print(f"{'Layer':<30s} {'Start':>8s} {'End':>8s} {'Size':>8s}")
    print("-" * 60)

    for layer_name in EXECUTION_ORDER:
        hex_path = os.path.join(args.weights_dir, f"{layer_name}.hex")
        if not os.path.exists(hex_path):
            print(f"ERROR: Missing {hex_path}")
            return 1

        with open(hex_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        sz = len(lines)
        print(f"{layer_name:<30s} {addr:>8d} {addr + sz - 1:>8d} {sz:>8d}")
        all_lines.extend(lines)
        addr += sz

    total_values = len(all_lines)
    print(f"\nTotal INT8 values: {total_values}")

    # Pad to TOTAL_DEPTH
    pad_count = args.total_depth - total_values
    if pad_count < 0:
        print(f"ERROR: Total values ({total_values}) exceeds TOTAL_DEPTH ({args.total_depth})")
        return 1
    all_lines.extend(["00"] * pad_count)
    print(f"Padded with {pad_count} zeros to depth {args.total_depth}")

    # Write unified hex
    out_path = os.path.join(args.weights_dir, "weights_all.hex")
    with open(out_path, "w") as f:
        for line in all_lines:
            f.write(line + "\n")
    print(f"\nOutput: {out_path}")

    # Write updated address map
    addr_map_path = os.path.join(args.weights_dir, "address_map.txt")
    addr = 0
    with open(addr_map_path, "w") as f:
        f.write(f"{'Layer':<30s} {'Start':>8s} {'End':>8s} {'Size':>8s}\n")
        for layer_name in EXECUTION_ORDER:
            hex_path = os.path.join(args.weights_dir, f"{layer_name}.hex")
            with open(hex_path, "r") as hf:
                sz = sum(1 for line in hf if line.strip())
            f.write(f"{layer_name:<30s} {addr:>8d} {addr + sz - 1:>8d} {sz:>8d}\n")
            addr += sz
    print(f"Address map: {addr_map_path}")

    # Copy weights_all.hex to simulation directory
    if args.sim_dir:
        os.makedirs(args.sim_dir, exist_ok=True)
        sim_dst = os.path.join(args.sim_dir, "weights_all.hex")
        shutil.copy2(out_path, sim_dst)
        print(f"Copied to sim dir: {sim_dst}")

    return 0


if __name__ == "__main__":
    exit(main())
