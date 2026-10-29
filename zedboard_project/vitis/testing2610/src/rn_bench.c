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
 * UART output (115200 8N1), per vector: the input I/Q samples, the golden /
 * ARM / FPGA logits, the predicted class vs. the true label, and the timing;
 * then a summary. Printing happens outside the timed regions.
 *
 * Results are also written to a struct at RESULTS_ADDR (top of OCM) so the
 * xsdb script can read them over JTAG. With RN_POWER_PHASES = 1 the program
 * afterwards cycles idle / ARM busy / accelerator busy (20 s each) for
 * measuring board current on header J21.
 */
#include <stdint.h>
#include "xil_cache.h"
#include "xil_io.h"
#include "xil_printf.h"
#include "xiltimer.h"           /* XTime_GetTime, COUNTS_PER_SECOND (Vitis 2023.2+) */
#include "rn_infer.h"
#include "rn_vectors.h"

/* ── Output options ───────────────────────────────────────────────── */
#ifndef PRINT_INPUT_SAMPLES
#define PRINT_INPUT_SAMPLES 16  /* I/Q samples printed per vector (128 = whole frame) */
#endif
#ifndef RN_POWER_PHASES
#define RN_POWER_PHASES     0   /* 1: loop idle / ARM / FPGA phases for power readings */
#endif

#if defined(__aarch64__)
#define RN_BASE        0xA0000000u          /* ZCU102/104: M_AXI_HPM0_FPD */
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

#if RN_NUM_CLASSES == 4
static const char *const CLASS_NAME[] = {"Normal", "Jammer", "Spoofer", "Interf"};
#elif RN_NUM_CLASSES == 7
static const char *const CLASS_NAME[] = {"Normal", "Barrage", "Tone", "Sweep",
                                         "Pulsed", "Replay", "CoChan"};
#else
static const char *const CLASS_NAME[RN_NUM_CLASSES] = {0};
#endif

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

static int argmax(const int32_t *lg)
{
    int best = 0;
    for (int c = 1; c < RN_NUM_CLASSES; c++)
        if (lg[c] > lg[best])
            best = c;
    return best;
}

static const char *cname(int c)
{
    return (c >= 0 && c < RN_NUM_CLASSES && CLASS_NAME[c]) ? CLASS_NAME[c] : "?";
}

/* ticks → microseconds with one decimal, printed as "123.4" */
static void print_us(uint64_t ticks)
{
    /* COUNTS_PER_SECOND expands to "XPAR_CPU_CORE_CLOCK_FREQ_HZ/2" without
     * parentheses, so it must be bracketed here or the result is 4x too small */
    uint64_t tenths = (ticks * 10000000ull) / (COUNTS_PER_SECOND);
    xil_printf("%d.%d", (int)(tenths / 10), (int)(tenths % 10));
}

static void print_logits(const char *who, const int32_t *lg)
{
    xil_printf("  %-10s", who);
    for (int c = 0; c < RN_NUM_CLASSES; c++)
        xil_printf(" %9d", (int)lg[c]);
    xil_printf("\r\n");
}

static void print_vector(int v, const int8_t *in, const int32_t *gold, const int32_t *arm,
                         const int32_t *hw, uint32_t cycles, uint64_t arm_ticks, uint64_t hw_ticks)
{
    xil_printf("\r\n------------------------------ vector %d / %d ------------------------------\r\n",
               v + 1, RN_NUM_VECTORS);
    xil_printf("input I/Q (INT8, first %d of 128 samples):\r\n", PRINT_INPUT_SAMPLES);
    for (int base = 0; base < PRINT_INPUT_SAMPLES; base += 16) {
        xil_printf("  samples %d-%d\r\n    I:", base,
                   (base + 15 < PRINT_INPUT_SAMPLES ? base + 15 : PRINT_INPUT_SAMPLES - 1));
        for (int t = base; t < base + 16 && t < PRINT_INPUT_SAMPLES; t++)
            xil_printf(" %4d", in[t]);
        xil_printf("\r\n    Q:");
        for (int t = base; t < base + 16 && t < PRINT_INPUT_SAMPLES; t++)
            xil_printf(" %4d", in[128 + t]);
        xil_printf("\r\n");
    }
    xil_printf("logits (INT32)\r\n  %-10s", "");
    for (int c = 0; c < RN_NUM_CLASSES; c++)
        xil_printf(" %9s", cname(c));
    xil_printf("\r\n");
    print_logits("golden", gold);
    print_logits("ARM (C)", arm);
    print_logits("FPGA", hw);

    int g = argmax(gold), a = argmax(arm), h = argmax(hw);
    xil_printf("prediction: FPGA %d (%s)  ARM %d (%s)  golden %d (%s)", h, cname(h), a, cname(a),
               g, cname(g));
#ifdef RN_HAVE_LABELS
    int y = rn_vec_label[v];
    xil_printf("  | true label %d (%s)", y, cname(y));
#endif
    xil_printf("\r\n  FPGA logits %s golden, anomaly flag %d",
               same(hw, gold) ? "== (bit-exact)" : "!= (MISMATCH)", h != 0);
#ifdef RN_HAVE_LABELS
    xil_printf(", %s", h == y ? "correct" : "wrong");
#endif
    xil_printf("\r\ntiming: ARM ");
    print_us(arm_ticks);
    xil_printf(" us   FPGA ");
    print_us(hw_ticks);
    xil_printf(" us (%d PL cycles)\r\n", (int)cycles);
}

__attribute__((unused)) static void busy_phase(uint32_t phase, uint32_t seconds)
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
    int32_t arm_lg[RN_NUM_CLASSES], hw_lg[RN_NUM_CLASSES];
    XTime t0, t1;
    uint64_t arm_total = 0, hw_total = 0;
    uint32_t cycles = 0, arm_ok = 0, hw_ok = 0, hw_correct = 0, arm_correct = 0;

    R->magic = 0;
    R->phase = 0;
    R->info = Xil_In32(RN_INFO);
    publish();
    xil_printf("\r\n==================================================================\r\n");
    xil_printf(" RadarNet-1D benchmark: Cortex-A9 software vs FPGA accelerator\r\n");
    xil_printf(" accelerator INFO = 0x%08x (W=%d, classes=%d), %d test vectors\r\n",
               (unsigned)R->info, (int)(R->info & 0xFF), (int)((R->info >> 8) & 0xFF),
               RN_NUM_VECTORS);
    xil_printf("==================================================================\r\n");

    for (int v = 0; v < RN_NUM_VECTORS; v++) {
        const int8_t *in = rn_vec_in + v * 256;
        const uint16_t *iq = rn_vec_iq + v * 128;
        const int32_t *gold = rn_vec_logits + v * RN_NUM_CLASSES;

        /* ARM: one checked run (also warms the caches), then REPEATS timed runs */
        rn_infer(in, arm_lg);
        arm_ok += same(arm_lg, gold);
        XTime_GetTime(&t0);
        for (int r = 0; r < REPEATS; r++)
            rn_infer(in, arm_lg);
        XTime_GetTime(&t1);
        uint64_t arm_t = (t1 - t0) / REPEATS;
        arm_total += (t1 - t0);

        /* FPGA: one checked run, then REPEATS timed runs */
        hw_infer(iq, hw_lg, &cycles);
        hw_ok += same(hw_lg, gold);
        XTime_GetTime(&t0);
        for (int r = 0; r < REPEATS; r++)
            hw_infer(iq, hw_lg, 0);
        XTime_GetTime(&t1);
        uint64_t hw_t = (t1 - t0) / REPEATS;
        hw_total += (t1 - t0);

#ifdef RN_HAVE_LABELS
        hw_correct += (argmax(hw_lg) == rn_vec_label[v]);
        arm_correct += (argmax(arm_lg) == rn_vec_label[v]);
#endif
        print_vector(v, in, gold, arm_lg, hw_lg, cycles, arm_t, hw_t);
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

    uint64_t speedup10 = (10ull * R->arm_ticks) / (R->hw_ticks ? R->hw_ticks : 1);
    xil_printf("\r\n============================== SUMMARY ==============================\r\n");
    xil_printf(" bit-exact vs golden : ARM %d/%d   FPGA %d/%d\r\n", (int)arm_ok, RN_NUM_VECTORS,
               (int)hw_ok, RN_NUM_VECTORS);
#ifdef RN_HAVE_LABELS
    xil_printf(" correct vs label    : ARM %d/%d   FPGA %d/%d   (%d vectors only; the model's\r\n"
               "                       accuracy is measured on the 30,000-sample test set)\r\n",
               (int)arm_correct, RN_NUM_VECTORS, (int)hw_correct, RN_NUM_VECTORS, RN_NUM_VECTORS);
#endif
    xil_printf(" mean ARM time       : ");
    print_us(R->arm_ticks);
    xil_printf(" us / inference  (%d timer ticks)\r\n", (int)R->arm_ticks);
    xil_printf(" mean FPGA time      : ");
    print_us(R->hw_ticks);
    xil_printf(" us / inference  (%d ticks, incl. 128 AXI sample writes)\r\n", (int)R->hw_ticks);
    xil_printf(" FPGA cycle counter  : %d PL cycles (start -> result)\r\n", (int)cycles);
    xil_printf(" speed-up            : %d.%dx\r\n", (int)(speedup10 / 10), (int)(speedup10 % 10));
    xil_printf(" timer               : %d Hz\r\n", (int)COUNTS_PER_SECOND);
    xil_printf("=====================================================================\r\n");

#if RN_POWER_PHASES
    for (;;) {                                           /* power measurement phases */
        busy_phase(1, 20);
        busy_phase(2, 20);
        busy_phase(3, 20);
    }
#else
    xil_printf("done.\r\n");
    for (;;)
        ;
#endif
    return 0;
}
