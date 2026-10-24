# RadarNet-1D accelerator v2

A rewrite of the accelerator in `../rtl/`. The original design is left untouched as the record of the VLSID submission.

The original design has three problems that v2 fixes:

* **It is fully serial.** It issues about one MAC every 2 cycles, which comes to about 2.38 M cycles per inference (≈ 33 ms at 72 MHz).
* **It has no requantization between layers.** Each 24-bit accumulator is saturated straight to INT8. As a result the RTL does not compute the trained network and fails its own testbench.
* **Its weights are read with the wrong BRAM-latency alignment.** The RB2 projection layer also reads the wrong ROM region.

## Architecture

* **PE array.** One **W×W INT8 PE array** (`rn_pe_array.v`) has W output-channel lanes, and each lane computes a W-input dot product through a registered adder tree.
* **Sequencer.** The array is time-multiplexed over all 8 conv/FC layers by a **descriptor-driven sequencer** (`rn_top.v` plus the generated `layer_table.vh`). There are no bubbles between output groups.
* **On-chip buffers.** Three ping-pong feature buffers keep every activation on chip. A word holds W channels at one time position, so one read feeds the whole array.
* **Integer-only requantization, per lane.** Each lane computes

  `y = sat8(relu?(((acc + b)·M[oc] + (skip·Ms << SK) + 2^(SH-1)) >>> SH))`

  with INT32 bias, per-channel weight scales and a fused residual add.
* **GAP.** Global average pooling is lane-parallel: a sum followed by a rounding `>>> 5`. FC logits stay INT32 for the argmax.

The golden model is `../python/int_quant.py`. The RTL matches it **bit-exactly**: all four INT32 logits agree on every test vector.

## Flow

```bash
# 1. Calibrate on the training split and get the honest integer accuracy (needs ../data, e.g. in Colab)
python int_quant.py calib --checkpoint ../checkpoints/radarnet_best.pth --data-dir ../data
python int_quant.py eval  --intq ../checkpoints/radarnet_intq.npz \
                          --checkpoint ../checkpoints/radarnet_best.pth --data-dir ../data

# 2. Bit-exact RTL check (Icarus): exports ROMs for W lanes and runs 50 vectors
rtl_v2/sim/run_sim.sh 8 50

# 3. Implementation on xc7z020clg484-1 (out-of-context), from WSL
python int_quant.py export --intq ../checkpoints/radarnet_intq.npz --lanes 8 --out-dir ../rtl_v2/build/mem_w8
rtl_v2/vivado/run_impl.sh 8 5      # W, clock period in ns → rtl_v2/build/impl_w8_p5/
```

## Results

Setup: xc7z020clg484-1, Vivado 2025.2, out-of-context place & route with a 5 ns constraint.
Fmax = 1 / (5 ns − WNS). Cycles are measured by the DUT counter from `start` to
`result_valid`, including the 128-cycle frame load. Power is the vectorless Vivado estimate,
with dynamic power scaled to Fmax. Energy = power × latency.

| Design | MAC/cycle | Cycles | Fmax (MHz) | Latency | LUT | FF | BRAM | DSP | Power (W) | Energy/inf |
|---|---|---|---|---|---|---|---|---|---|---|
| original `rtl/` | 0.5 | 2,376,786 | 72.2 | 32.9 ms | 10,012 | 5,461 | 15 | 0 | 0.671 | 22.1 mJ |
| v2, W=4  | 16  | 76,121 | 181.6 | 419 µs | 1,563 | 1,394 | 11.5 | 28 | 0.28 | 118 µJ |
| **v2, W=8**  | 64  | 20,313 | 180.9 | **112 µs** | 2,682 | 2,927 | 14 | 104 | 0.58 | **65 µJ** |
| v2, W=16 | 256 | 5,853  | 149.0 | 39 µs  | 25,107 | 14,864 | 41.5 | 32 | 1.53 | 60 µJ |

* **Correctness.** All v2 points match `int_quant.py` bit-exactly: 50/50 test vectors, all INT32 logits.
* **Original design.** It fails its own testbench. Its cycle count is read from its own simulation; Fig. 3 of the submitted paper shows `total_cycles = 0x244452`.
* **W=16** exceeds the DSP budget for the PE multipliers (256 > 220), so it builds them from LUTs. That limits its Fmax and makes it LUT-heavy. W=8 is the efficiency sweet spot.
* **Power figures** are estimates. Before publication, confirm them with SAIF-annotated power or a ZedBoard current measurement.
