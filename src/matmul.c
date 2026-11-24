/* matmul.c — the matrix product the Transformer layers spend their time in.
 *
 *     C[M,N] = A[M,K]·B[K,N]          (NMZ_MM_TRANS_A: A is stored [K,M],
 *                                      NMZ_MM_TRANS_B: B is stored [N,K],
 *                                      NMZ_MM_ACCUMULATE: C += instead of =)
 *
 * Two implementations that give bit-identical results:
 *
 *  - nmz_matmul_naive: the formula as three loops, c_ij = Σ_k a_ik·b_kj.
 *    The readable reference the tests compare against.
 *  - nmz_matmul: the same sums, cache-blocked, with a 1 × 32 register tile
 *    of C (below). Its inner loop is
 *        acc[0..31] += a_ik · b[k][j..j+31]
 *    which GCC vectorises at -O3 without -ffast-math, because no sum is
 *    reordered: every c_ij still receives a_i0·b_0j, a_i1·b_1j, … in
 *    increasing k, one float addition at a time — exactly what the naive
 *    loop does. Blocking only changes WHICH c_ij is worked on when, so
 *    that a block of B (KB × NB floats) stays in the cache while every row
 *    of A uses it, and the tile keeps 32 sums in registers.
 *
 * Because each c_ij depends only on row i of A (and on B), the result for a
 * row never depends on how many rows are computed at once, on the block
 * sizes, or on the number of threads. The KV cache relies on this:
 * computing one new position gives the same bits as recomputing the whole
 * window (DESIGN decision 98).
 *
 * The sums are in float (like the im2col convolution, decision 46): a
 * double accumulator would make the loop twice as slow and stop it from
 * being one vector operation per k. The layers that use it keep their
 * other reductions (softmax, LayerNorm statistics, dot products of the
 * attention scores) in double.
 *
 * With OpenMP (make OMP=1) the row blocks of C are shared among threads;
 * each c_ij is still computed by one thread in the same order. */
#include "internal.h"

#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef MB
#define MB 32  /* rows of C per block (and per thread task)     */
#endif
#ifndef KB
#define KB 128 /* depth of a block: rows of B kept in cache      */
#endif
#ifndef NB
#define NB 256 /* columns of C / B per block (1 KB per row)      */
#endif

int nmz_threads(void) {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

static nmz_status check(int M, int N, int K, const float *A, const float *B, float *C, unsigned flags) {
    if (M < 0 || N < 0 || K < 0 || (flags & ~(NMZ_MM_TRANS_A | NMZ_MM_TRANS_B | NMZ_MM_ACCUMULATE)))
        return nmz__fail(NMZ_ERR_ARG, "matmul: bad sizes or flags");
    if ((flags & NMZ_MM_TRANS_A) && (flags & NMZ_MM_TRANS_B))
        return nmz__fail(NMZ_ERR_UNSUPPORTED, "matmul: A and B cannot both be transposed");
    if ((size_t)M * N > 0 && (!C || (K > 0 && (!A || !B)))) return nmz__fail(NMZ_ERR_ARG, "matmul: NULL matrix");
    return NMZ_OK;
}

nmz_status nmz_matmul_naive(int M, int N, int K, const float *A, const float *B, float *C, unsigned flags) {
    int i, j, k, ta = (flags & NMZ_MM_TRANS_A) != 0, tb = (flags & NMZ_MM_TRANS_B) != 0;
    nmz_status s = check(M, N, K, A, B, C, flags);
    if (s != NMZ_OK) return s;
    for (i = 0; i < M; i++)
        for (j = 0; j < N; j++) {
            float c = (flags & NMZ_MM_ACCUMULATE) ? C[(size_t)i * N + j] : 0.0f;
            for (k = 0; k < K; k++) {
                float a = ta ? A[(size_t)k * M + i] : A[(size_t)i * K + k];
                float b = tb ? B[(size_t)j * K + k] : B[(size_t)k * N + j];
                c += a * b;
            }
            C[(size_t)i * N + j] = c;
        }
    return NMZ_OK;
}

/* Register tiles: TW consecutive sums of one row of C stay in registers for
 * a whole block of k (TW = 32 → 8 SSE registers), so each step loads one A
 * value and TW values of B and does TW multiply-adds, instead of loading
 * and storing C for every k. Each c_ij still gets its products in
 * increasing k, one float addition at a time. Measured on the shapes of a
 * small Transformer (examples/bench_matmul.c, one core of an i7-7500U):
 * a 1×32 tile ≈ 17–21 GFLOP/s, 1×16 ≈ 15, the plain row loop ≈ 12, and a
 * 4×8 tile only 4–6 (GCC spilled it to the stack). A(i, k) = A[i·sa + k·sk]. */
#define TILE(TW)                                                                                                     \
    static void tile##TW(const float *a, size_t sk, const float *B, int N, float *c, int k0, int k1) {               \
        float acc[TW];                                                                                               \
        int k, j;                                                                                                    \
        for (j = 0; j < TW; j++) acc[j] = c[j];                                                                      \
        for (k = k0; k < k1; k++) {                                                                                  \
            const float av = a[(size_t)k * sk], *b = B + (size_t)k * N;                                              \
            for (j = 0; j < TW; j++) acc[j] += av * b[j];                                                            \
        }                                                                                                            \
        for (j = 0; j < TW; j++) c[j] = acc[j];                                                                      \
    }
TILE(32)
TILE(8)

/* rows [i0, i1) of C; A is [M,K] (or [K,M] when ta) */
static void block_rows(int i0, int i1, int M, int N, int K, const float *A, int ta, const float *B, float *C) {
    const size_t sa = ta ? 1 : (size_t)K, sk = ta ? (size_t)M : 1;
    int k0, j0, i, k, j;
    for (k0 = 0; k0 < K; k0 += KB) {
        int k1 = k0 + KB < K ? k0 + KB : K;
        for (j0 = 0; j0 < N; j0 += NB) {
            int j1 = j0 + NB < N ? j0 + NB : N;
            for (i = i0; i < i1; i++) {
                const float *a = A + (size_t)i * sa;
                float *c = C + (size_t)i * N;
                for (j = j0; j + 32 <= j1; j += 32) tile32(a, sk, B + j, N, c + j, k0, k1);
                for (; j + 8 <= j1; j += 8) tile8(a, sk, B + j, N, c + j, k0, k1);
                for (; j < j1; j++) {
                    float cj = c[j];
                    for (k = k0; k < k1; k++) cj += a[(size_t)k * sk] * B[(size_t)k * N + j];
                    c[j] = cj;
                }
            }
        }
    }
}

nmz_status nmz_matmul(int M, int N, int K, const float *A, const float *B, float *C, unsigned flags) {
    int ta = (flags & NMZ_MM_TRANS_A) != 0, nblocks, ib;
    float *Bt = NULL, *At = NULL;
    const float *Bn = B;
    nmz_status s = check(M, N, K, A, B, C, flags);
    if (s != NMZ_OK) return s;
    if ((size_t)M * N == 0) return NMZ_OK;
    if (!(flags & NMZ_MM_ACCUMULATE))
        memset(C, 0, (size_t)M * N * sizeof *C);
    if (K == 0) return NMZ_OK;
    if (flags & NMZ_MM_TRANS_B) {
        /* C = A·Bᵀ: copy Bᵀ once ([N,K] → [K,N]) so the inner loop runs over
         * contiguous rows; the sums are the same as with B read strided */
        int j, k;
        Bt = malloc((size_t)K * N * sizeof *Bt);
        if (!Bt) return nmz__fail(NMZ_ERR_NOMEM, "out of memory");
        for (j = 0; j < N; j++)
            for (k = 0; k < K; k++) Bt[(size_t)k * N + j] = B[(size_t)j * K + k];
        Bn = Bt;
    }
    if (ta && M > 1) {
        /* C = Aᵀ·B (weight gradients: A = the layer input [rows, M]): copy A
         * once into [M, K] so each row of C reads A contiguously */
        int i, k;
        At = malloc((size_t)K * M * sizeof *At);
        if (!At) {
            free(Bt);
            return nmz__fail(NMZ_ERR_NOMEM, "out of memory");
        }
        for (k = 0; k < K; k++)
            for (i = 0; i < M; i++) At[(size_t)i * K + k] = A[(size_t)k * M + i];
        A = At;
        ta = 0;
    }
    nblocks = (M + MB - 1) / MB;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if ((double)M * N * K > 2e5 && nblocks > 1)
#endif
    for (ib = 0; ib < nblocks; ib++) {
        int i0 = ib * MB, i1 = i0 + MB < M ? i0 + MB : M;
        block_rows(i0, i1, M, N, K, A, ta, Bn, C);
    }
    free(Bt);
    free(At);
    return NMZ_OK;
}
