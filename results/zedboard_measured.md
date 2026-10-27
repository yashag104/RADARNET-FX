# ZedBoard measurements (first hardware run)

Measured on 26 Oct 2026, 18:14 (local time).

**Setup**
- **Board:** ZedBoard (Zynq-7020).
- **Bitstream:** built by the user from `zedboard_project/create_project.tcl`, Vivado 2025.2. W = 8, 4-class model, PL clock 142.857 MHz. Timing met: WNS +0.385 ns, WHS +0.053 ns (see `vivado_reports/zedboard_system_142MHz/`).
- **Software:** `testing2610.elf`, built by the user in Vitis 2025.2 from `sw/rn_bench.c` and `sw/rn_infer.c`, compiled `-O3`, standalone, Cortex-A9 at 666.67 MHz.
- **Readout:** results struct at 0xFFFF8000, read over JTAG with `mrd` while the program ran.

Raw words (decimal): `1380860465 1380844552 50 50 50 333333343 3262484 53955 23013 3 20`

| Quantity | Value |
|---|---|
| Test vectors | 50 |
| ARM software vs golden logits | 50 / 50 bit-exact |
| **FPGA accelerator vs golden logits** | **50 / 50 bit-exact** |
| Timer frequency | 333,333,343 Hz |
| ARM Cortex-A9 software, mean of 50 × 20 runs | 3,262,484 ticks = **9.787 ms / inference** |
| Accelerator, end to end (start + 128 AXI sample writes + compute + result) | 53,955 ticks = **161.9 µs / inference** |
| Accelerator cycle counter (start → result, including the AXI-paced load) | 23,013 cycles = 161.1 µs at 142.857 MHz |
| Compute-only (simulation, with the frame already loaded) | ≈ 20,187 cycles = 141.3 µs |
| **Speed-up, end to end** | **60.5×** |

Not measured yet: board power (J21 readings during the idle, ARM and accelerator phases).
