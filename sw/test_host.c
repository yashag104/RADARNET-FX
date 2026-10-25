/*
 * test_host.c — check rn_infer() against the golden logits from int_quant.py
 *
 *   gcc -O2 -o test_host test_host.c rn_infer.c
 *   ./test_host input_vectors.hex golden_logits.hex
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "rn_infer.h"

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s input_vectors.hex golden_logits.hex\n", argv[0]);
        return 2;
    }
    FILE *fi = fopen(argv[1], "r"), *fg = fopen(argv[2], "r");
    if (!fi || !fg) {
        perror("open");
        return 2;
    }
    int n = 0, exact = 0;
    double secs = 0;
    for (;;) {
        int8_t in[256];
        unsigned w;
        int t;
        for (t = 0; t < 128 && fscanf(fi, "%x", &w) == 1; t++) {
            in[t] = (int8_t)(w >> 8);            /* I */
            in[128 + t] = (int8_t)(w & 0xFF);    /* Q */
        }
        if (t < 128)
            break;
        int32_t lg[RN_NUM_CLASSES], gold;
        clock_t c0 = clock();
        rn_infer(in, lg);
        secs += (double)(clock() - c0) / CLOCKS_PER_SEC;
        int ok = 1;
        for (int o = 0; o < RN_NUM_CLASSES; o++) {
            unsigned g;
            if (fscanf(fg, "%x", &g) != 1)
                return 3;
            gold = (int32_t)g;
            if (lg[o] != gold) {
                ok = 0;
                printf("vec %d logit %d: c=%d gold=%d\n", n, o, lg[o], gold);
            }
        }
        exact += ok;
        n++;
    }
    printf("bit-exact logits: %d / %d   (host: %.1f us / inference)\n", exact, n, 1e6 * secs / n);
    return exact == n ? 0 : 1;
}
