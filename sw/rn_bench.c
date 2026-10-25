/*
 * rn_bench.c — ZedBoard benchmark: Cortex-A9 software vs. FPGA accelerator
 * =========================================================================
 * Runs the same integer RadarNet-1D on every test vector
 *   (1) in software on the application CPU (Cortex-A9 on the ZedBoard,
 *       Cortex-A53 on the ZCU104; rn_infer.c, -O3, caches on), and
 *   (2) on the accelerator through AXI4-Lite (rn_axi.v),
 * checks both bit-exactly against the golden logits, and times them with the
 * 64-bit ARM generic/global timer (COUNTS_PER_SECOND from the BSP).
 *
 * Results are written to a struct at RESULTS_ADDR (top of OCM) so the xsdb
 * script can read them over JTAG; they are also printed on the UART.
 * After the benchmark the program cycles through power-measurement phases
 * (idle / ARM busy / accelerator busy, 20 s each) and reports the phase in
 * the same struct, for reading the board current at each phase.
 */
#include <stdint.h>
#include "xil_cache.h"
#include "xil_io.h"
#include "xil_printf.h"
#include "xiltimer.h"           /* XTime_GetTime, COUNTS_PER_SECOND (Vitis 2023.2+) */
#include "rn_infer.h"
#include "rn_vectors.h"

#if defined(__aarch64__)
#define RN_BASE        0xA0000000u          /* ZCU104: M_AXI_HPM0_FPD */
#else
#define RN_BASE        0x43C00000u          /* ZedBoard: M_AXI_GP0 */
#endif
#define RN_CTRL        (RN_BASE + 0x00)
#define RN_STATUS      (RN_BASE + 0x04)
#define RN_IQ          (RN_BASE + 0x08)
#define RN_CLASS       (RN_BASE + 0x0C)
#define RN_CYCLES      (RN_BASE + 0x10)
#define RN_INFO        (RN_BASE + 0x14)
#define RN_LOGIT(c)    (RN_BASE + 0x20 + 4 * (c))

#define RESULTS_ADDR   0xFFFF8000u
#define RESULTS_MAGIC  0x524E4231u          /* "RNB1" */
#define REPEATS        20

typedef struct {
    uint32_t magic;              /* RESULTS_MAGIC once the benchmark is done */
    uint32_t info;               /* rn_axi INFO register */
    uint32_t n_vectors;
    uint32_t arm_exact;          /* vectors whose ARM logits == golden */
    uint32_t hw_exact;           /* vectors whose accelerator logits == golden */
    uint32_t timer_hz;           /* global-timer ticks per second */
    uint32_t arm_ticks;          /* mean ticks per ARM inference */
    uint32_t hw_ticks;           /* mean ticks per accelerator inference, incl. AXI I/O */
    uint32_t hw_cycles;          /* accelerator cycle counter (start → result) */
    uint32_t phase;              /* power phases: 1 idle, 2 ARM loop, 3 accel loop */
    uint32_t phase_seconds;
} rn_results_t;

static volatile rn_results_t *const R = (volatile rn_results_t *)RESULTS_ADDR;

/* make the struct visible to the JTAG reader (bypasses the CPU caches) */
static void publish(void)
{
    Xil_DCacheFlushRange(RESULTS_ADDR, sizeof(rn_results_t));
}

static int hw_infer(const uint16_t *iq, int32_t *logits, uint32_t *cycles)
{
    Xil_Out32(RN_CTRL, 1);
    for (int t = 0; t < 128; t++)
        Xil_Out32(RN_IQ, iq[t]);
    while (!(Xil_In32(RN_STATUS) & 2))
        ;
    for (int c = 0; c < RN_NUM_CLASSES; c++)
        logits[c] = (int32_t)Xil_In32(RN_LOGIT(c));
    if (cycles)
        *cycles = Xil_In32(RN_CYCLES);
    return (int)Xil_In32(RN_CLASS);
}

static int same(const int32_t *a, const int32_t *b)
{
    for (int c = 0; c < RN_NUM_CLASSES; c++)
        if (a[c] != b[c])
            return 0;
    return 1;
}

static void busy_phase(uint32_t phase, uint32_t seconds)
{
    XTime t0, t;
    int32_t lg[RN_NUM_CLASSES];
    R->phase = phase;
    R->phase_seconds = seconds;
    publish();
    xil_printf("power phase %d (%s) for %d s\r\n", (int)phase,
               phase == 1 ? "idle" : phase == 2 ? "ARM inference loop" : "accelerator loop",
               (int)seconds);
    XTime_GetTime(&t0);
    do {
        if (phase == 2)
            rn_infer(rn_vec_in, lg);
        else if (phase == 3)
            hw_infer(rn_vec_iq, lg, 0);
        XTime_GetTime(&t);
    } while ((t - t0) < (XTime)seconds * COUNTS_PER_SECOND);
}

int main(void)
{
    int32_t lg[RN_NUM_CLASSES];
    XTime t0, t1;
    uint64_t arm_total = 0, hw_total = 0;
    uint32_t cycles = 0, arm_ok = 0, hw_ok = 0;

    R->magic = 0;
    R->phase = 0;
    R->info = Xil_In32(RN_INFO);
    publish();
    xil_printf("\r\nRadarNet-1D benchmark  INFO=%08x  vectors=%d\r\n",
               (unsigned)R->info, RN_NUM_VECTORS);

    for (int v = 0; v < RN_NUM_VECTORS; v++) {
        const int8_t *in = rn_vec_in + v * 256;
        const uint16_t *iq = rn_vec_iq + v * 128;
        const int32_t *gold = rn_vec_logits + v * RN_NUM_CLASSES;

        rn_infer(in, lg);                                /* warm caches */
        arm_ok += same(lg, gold);
        XTime_GetTime(&t0);
        for (int r = 0; r < REPEATS; r++)
            rn_infer(in, lg);
        XTime_GetTime(&t1);
        arm_total += (t1 - t0);

        hw_infer(iq, lg, &cycles);
        hw_ok += same(lg, gold);
        XTime_GetTime(&t0);
        for (int r = 0; r < REPEATS; r++)
            hw_infer(iq, lg, 0);
        XTime_GetTime(&t1);
        hw_total += (t1 - t0);
    }

    R->n_vectors = RN_NUM_VECTORS;
    R->arm_exact = arm_ok;
    R->hw_exact = hw_ok;
    R->timer_hz = COUNTS_PER_SECOND;
    R->arm_ticks = (uint32_t)(arm_total / (RN_NUM_VECTORS * REPEATS));
    R->hw_ticks = (uint32_t)(hw_total / (RN_NUM_VECTORS * REPEATS));
    R->hw_cycles = cycles;
    R->magic = RESULTS_MAGIC;
    publish();

    xil_printf("bit-exact: ARM %d/%d  accelerator %d/%d\r\n", (int)arm_ok, RN_NUM_VECTORS,
               (int)hw_ok, RN_NUM_VECTORS);
    xil_printf("ARM: %d ticks/inf   accelerator: %d ticks/inf (incl. AXI), %d PL cycles\r\n",
               (int)R->arm_ticks, (int)R->hw_ticks, (int)cycles);

    for (;;) {                                           /* power measurement phases */
        busy_phase(1, 20);
        busy_phase(2, 20);
        busy_phase(3, 20);
    }
    return 0;
}
