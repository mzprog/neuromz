/* bench_matmul.c — the blocked matrix product against the naive loops, on
 * the shapes a small character Transformer uses (rows = batch x context,
 * D = 64, feed-forward 256). Prints milliseconds and GFLOP/s.
 * Usage: build/bench_matmul [repeats]   (--target bench runs it) */
#include "neuromz.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    static const struct {
        const char *what;
        int M, N, K;
        unsigned flags;
    } cases[] = {{"x*W_qkv  4096x192x64", 4096, 192, 64, 0},
                 {"a*W2     4096x64x256", 4096, 64, 256, 0},
                 {"x*W1     4096x256x64", 4096, 256, 64, 0},
                 {"dW1 = x^T*dz  64x256x4096", 64, 256, 4096, NMZ_MM_TRANS_A | NMZ_MM_ACCUMULATE},
                 {"dx = dz*W1^T  4096x64x256", 4096, 64, 256, NMZ_MM_TRANS_B}};
    int reps = argc > 1 ? atoi(argv[1]) : 5, c, r;
    if (reps < 1) reps = 1;
    printf("matmul, %d thread(s), best of %d:\n", nmz_threads(), reps);
    printf("  %-28s %10s %10s %8s %8s\n", "shape (MxNxK)", "naive ms", "blocked ms", "speedup", "GFLOP/s");
    for (c = 0; c < (int)(sizeof cases / sizeof cases[0]); c++) {
        int M = cases[c].M, N = cases[c].N, K = cases[c].K;
        float *A = malloc((size_t)M * K * sizeof *A), *B = malloc((size_t)K * N * sizeof *B);
        float *C = malloc((size_t)M * N * sizeof *C);
        double tn = 1e9, tb = 1e9, t;
        size_t e;
        if (!A || !B || !C) return 1;
        for (e = 0; e < (size_t)M * K; e++) A[e] = (float)((e * 7919 % 1000) / 1000.0 - 0.5);
        for (e = 0; e < (size_t)K * N; e++) B[e] = (float)((e * 104729 % 1000) / 1000.0 - 0.5);
        for (r = 0; r < reps; r++) {
            memset(C, 0, (size_t)M * N * sizeof *C);
            t = now();
            nmz_matmul_naive(M, N, K, A, B, C, cases[c].flags);
            t = now() - t;
            if (t < tn) tn = t;
            memset(C, 0, (size_t)M * N * sizeof *C);
            t = now();
            nmz_matmul(M, N, K, A, B, C, cases[c].flags);
            t = now() - t;
            if (t < tb) tb = t;
        }
        printf("  %-28s %10.2f %10.2f %7.1fx %8.2f\n", cases[c].what, tn * 1e3, tb * 1e3, tn / tb,
               2.0 * M * N * K / tb * 1e-9);
        free(A);
        free(B);
        free(C);
    }
    return 0;
}
