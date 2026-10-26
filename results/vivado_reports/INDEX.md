# Vivado reports — index

Every `.rpt` file in this folder was written by **Vivado v2025.2 (win64, build 6299465)** and is
copied unmodified. Each report's own header holds its tool version, date and design name.
This index is the only file added by hand.

| Folder | Run | Clock target | Key values (from the reports) |
|---|---|---|---|
| `zedboard_system_142MHz/` | Full ZedBoard system: PS7 + AXI + rn_axi (W=8, 4 classes). Built from `zedboard_project/create_project.tcl` in the Vivado GUI (impl_1, 26 Oct 2026). Produces the board bitstream. | FCLK0 142.857 MHz | WNS +0.385 ns, WHS +0.053 ns, WPWS +2.520 ns, all constraints met |
| `ooc_timing_closure/W{4,8,16}/` | Accelerator only (`rn_top`), out-of-context, clock on a global buffer (`HD.CLK_SRC`), re-implemented **at** the frequency it is claimed for. `timing_internal.rpt` = register-to-register paths. | W4 5.9 ns (169.5 MHz), W8 5.7 ns (175.4 MHz), W16 7.2 ns (138.9 MHz) | internal setup and hold met at all three. In `timing.rpt`, the all-paths hold violations are port paths of the bare module, which have no driver; the I/O boundary is checked in the system run. |
| `zcu104_system_250MHz/` | Full ZCU104 system: PS8 + AXI + rn_axi (W=16, DSP multipliers). Used to validate the UltraScale+ flow. | pl_clk0 250 MHz | WNS +0.156 ns, WHS +0.013 ns |
| `ooc_design_space_200MHz_target/W{4,8,16}/` | Earlier accelerator-only runs at a 5 ns target, **without** a global-buffer clock model. Kept for the record; superseded by `ooc_timing_closure/`. | 5 ns | WNS / WHS negative (estimate runs only) |

Not here: the ZCU102 build. It ran on the lab machine's Vivado 2026.1, and its reports are in
`~/zcu102/build/` on that machine (`build.log`, `proj/rn_zed.runs/impl_1/*.rpt`).
