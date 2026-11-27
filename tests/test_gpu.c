/* test_gpu.c — the CUDA backend against the CPU (DESIGN §11.7, decision 141).
 *
 * Op level: every required op of the table, CPU vs CUDA, on random inputs
 * and many shapes (odd sizes, strides, flags), within the tolerances below.
 * Layer level: every layer type forward + backward on both, the gradient
 * checks of every layer type on the GPU, traces, the KV cache, moving
 * models between devices, determinism, running out of memory.
 * --parity: training on both devices reaches the same quality (XOR, Iris,
 * the MNIST CNN, the names RNN, the Alice LSTM, a small Transformer).
 *
 * Tolerances (largest |gpu − cpu| / largest |cpu| of a result):
 *   0      data movement, element-wise +, ×, one-hot, gathers, im2col,
 *          max-pool winners, normalisation, counts (the same float ops)
 *   1e-6   reductions summed in double on both sides (column/row sums,
 *          squared norms), the optimizers (double math on both)
 *   2e-6   transcendental functions (expf/tanhf/erff in float on the GPU,
 *          the double versions rounded on the CPU), softmax, losses,
 *          LayerNorm, recurrent cells, col2im (another summation order)
 *   1e-5   matrix products and attention (cuBLAS sums in its own order;
 *          the CPU's attention dot products are double)
 *   1e-4   whole layers and models: all of the above compounded */
#include "test.h"
#include "neuromz_backend.h"
#include "neuromz_layer.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>

static const nmz_backend *cpu, *gpu;

static int backends(void) {
    if (cpu && gpu) return 1;
    nmz_backend_select("cpu");
    cpu = nmz_backend_current();
    if (nmz_backend_select("cuda") != NMZ_OK) return 0;
    gpu = nmz_backend_current();
    nmz_backend_select("cpu");
    return 1;
}

static float *hrand(size_t n, nmz_rng *r, double lo, double hi) {
    float *p = malloc((n ? n : 1) * sizeof *p);
    size_t i;
    for (i = 0; i < n; i++) p[i] = (float)(lo + (hi - lo) * nmz_rng_uniform(r));
    return p;
}

static float *hcopy(const float *h, size_t n) {
    float *p = malloc((n ? n : 1) * sizeof *p);
    if (n) memcpy(p, h, n * sizeof *p);
    return p;
}

static float *dup_to_gpu(const float *h, size_t n) {
    float *d = gpu->alloc(gpu, (n ? n : 1) * sizeof(float));
    if (d && n && h) gpu->upload(gpu, d, h, n * sizeof(float));
    return d;
}

static void from_gpu(float *h, const float *d, size_t n) {
    if (n) gpu->download(gpu, h, d, n * sizeof(float));
}

/* largest |a − b| relative to the largest |a| */
static double rel(const float *a, const float *b, size_t n) {
    double worst = 0, scale = 1e-30;
    size_t i;
    for (i = 0; i < n; i++) {
        double d = fabs((double)a[i] - b[i]);
        if (fabs(a[i]) > scale) scale = fabs(a[i]);
        if (!(d <= worst)) worst = d; /* NaN counts */
    }
    return worst / scale;
}

static double g_worst[16];
static const char *g_class[16];
static void note_worst(int cls, const char *name, double e) {
    if (e > g_worst[cls]) g_worst[cls] = e, g_class[cls] = name;
}

#define SAME(name, cpu_buf, gpu_dev, n, tol, cls)                                                                     \
    do {                                                                                                             \
        float *h_ = malloc(((n) ? (n) : 1) * sizeof(float));                                                         \
        double e_;                                                                                                   \
        from_gpu(h_, gpu_dev, n);                                                                                    \
        e_ = rel(cpu_buf, h_, n);                                                                                    \
        note_worst(cls, name, e_);                                                                                   \
        CHECK_MSG(e_ <= (tol), "%s: relative difference %.3g > %.1g (n = %zu)", name, e_, (double)(tol), (size_t)(n)); \
        free(h_);                                                                                                    \
    } while (0)

static void report_worst(void) {
    int i;
    for (i = 0; i < 16; i++)
        if (g_class[i]) test_note("worst %-36s %.2e", g_class[i], g_worst[i]), g_class[i] = NULL, g_worst[i] = 0;
}

/* ------------------------------------------------------- element-wise */

static void test_ops_elementwise(void) {
    static const size_t sizes[] = {1, 7, 31, 1000, 4097, 100003};
    nmz_rng r;
    size_t si;
    if (!backends()) return;
    nmz_rng_seed(&r, 1, 1);
    for (si = 0; si < sizeof sizes / sizeof sizes[0]; si++) {
        size_t n = sizes[si];
        float *a = hrand(n, &r, -2, 2), *b = hrand(n, &r, -2, 2), *o = malloc(n * sizeof *o);
        float *da = dup_to_gpu(a, n), *db = dup_to_gpu(b, n), *dout = dup_to_gpu(NULL, n);
        CHECK(cpu->fill(cpu, n, 1.5f, o) == NMZ_OK && gpu->fill(gpu, n, 1.5f, dout) == NMZ_OK);
        SAME("fill", o, dout, n, 0, 0);
        CHECK(cpu->fill(cpu, n, 0.0f, o) == NMZ_OK && gpu->fill(gpu, n, 0.0f, dout) == NMZ_OK);
        SAME("fill 0", o, dout, n, 0, 0);
        CHECK(cpu->add(cpu, n, a, b, o) == NMZ_OK && gpu->add(gpu, n, da, db, dout) == NMZ_OK);
        SAME("add", o, dout, n, 0, 0);
        CHECK(cpu->mul(cpu, n, a, b, o) == NMZ_OK && gpu->mul(gpu, n, da, db, dout) == NMZ_OK);
        SAME("mul", o, dout, n, 0, 0);
        CHECK(cpu->copy(cpu, 1, n, a, n, o, n) == NMZ_OK && gpu->copy(gpu, 1, n, da, n, dout, n) == NMZ_OK);
        SAME("copy", o, dout, n, 0, 0);
        free(a), free(b), free(o);
        gpu->release(gpu, da), gpu->release(gpu, db), gpu->release(gpu, dout);
    }
    { /* a strided block: 7 rows of 13 out of rows of 20, into rows of 17 */
        float *s = hrand(7 * 20, &r, -1, 1), *o = hrand(7 * 17, &r, -1, 1), *ds = dup_to_gpu(s, 140),
              *dout = dup_to_gpu(o, 119);
        CHECK(cpu->copy(cpu, 7, 13, s, 20, o, 17) == NMZ_OK && gpu->copy(gpu, 7, 13, ds, 20, dout, 17) == NMZ_OK);
        SAME("copy (strided)", o, dout, 119, 0, 0);
        free(s), free(o);
        gpu->release(gpu, ds), gpu->release(gpu, dout);
    }
    report_worst();
}

/* ------------------------------------------------------------- gemm */

static void test_ops_gemm(void) {
    static const int shapes[][3] = {{1, 1, 1},     {3, 5, 7},     {33, 65, 17},  {128, 64, 513}, {257, 3, 129},
                                    {2, 600, 3},   {64, 256, 1},  {1, 300, 200}, {1000, 31, 64}, {17, 17, 0}};
    static const unsigned flags[] = {0, NMZ_MM_TRANS_A, NMZ_MM_TRANS_B, NMZ_MM_ACCUMULATE,
                                     NMZ_MM_TRANS_A | NMZ_MM_ACCUMULATE, NMZ_MM_TRANS_B | NMZ_MM_ACCUMULATE};
    nmz_rng r;
    size_t si, fi;
    int pad, cases = 0;
    if (!backends()) return;
    nmz_rng_seed(&r, 2, 1);
    for (si = 0; si < sizeof shapes / sizeof shapes[0]; si++)
        for (fi = 0; fi < sizeof flags / sizeof flags[0]; fi++)
            for (pad = 0; pad < 2; pad++) { /* pad: every operand inside wider rows */
                const int M = shapes[si][0], N = shapes[si][1], K = shapes[si][2];
                const unsigned f = flags[fi];
                const int ta = (f & NMZ_MM_TRANS_A) != 0, tb = (f & NMZ_MM_TRANS_B) != 0;
                int lda = (ta ? M : K) + 3 * pad, ldb = (tb ? K : N) + 2 * pad, ldc = N + 5 * pad;
                size_t na = (size_t)(ta ? K : M) * lda + 1, nb = (size_t)(tb ? N : K) * ldb + 1, nc = (size_t)M * ldc;
                float *A = hrand(na, &r, -1, 1), *B = hrand(nb, &r, -1, 1), *C = hrand(nc, &r, -1, 1);
                float *dA = dup_to_gpu(A, na), *dB = dup_to_gpu(B, nb), *dC = dup_to_gpu(C, nc);
                char name[64];
                snprintf(name, sizeof name, "gemm %dx%dx%d f%u%s", M, N, K, f, pad ? " strided" : "");
                CHECK_OK(cpu->gemm(cpu, f, M, N, K, A, lda, B, ldb, C, ldc));
                CHECK_OK(gpu->gemm(gpu, f, M, N, K, dA, lda, dB, ldb, dC, ldc));
                SAME(name, C, dC, nc, 1e-5, 1);
                cases++;
                free(A), free(B), free(C);
                gpu->release(gpu, dA), gpu->release(gpu, dB), gpu->release(gpu, dC);
            }
    test_note("%d products (10 shapes x 6 flag sets x contiguous/strided)", cases);
    report_worst();
}

/* ------------------------------------------------- bias and reductions */

static void test_ops_reductions(void) {
    static const int shapes[][2] = {{1, 1}, {7, 3}, {300, 65}, {3000, 17}, {5, 1000}, {2049, 256}};
    nmz_rng r;
    size_t si;
    if (!backends()) return;
    nmz_rng_seed(&r, 3, 1);
    for (si = 0; si < sizeof shapes / sizeof shapes[0]; si++) {
        int rows = shapes[si][0], cols = shapes[si][1];
        size_t n = (size_t)rows * cols;
        float *a = hrand(n, &r, -1, 1), *bias = hrand((size_t)cols, &r, -1, 1), *o = hcopy(a, n);
        float *cs = hrand((size_t)cols, &r, -1, 1), *rs = hrand((size_t)rows, &r, -1, 1), *t = malloc(n * sizeof *t);
        float *da = dup_to_gpu(a, n), *dbias = dup_to_gpu(bias, (size_t)cols), *dout = dup_to_gpu(a, n);
        float *dcs = dup_to_gpu(cs, (size_t)cols), *drs = dup_to_gpu(rs, (size_t)rows), *dt = dup_to_gpu(NULL, n);
        CHECK_OK(cpu->add_bias(cpu, (size_t)rows, cols, o, bias));
        CHECK_OK(gpu->add_bias(gpu, (size_t)rows, cols, dout, dbias));
        SAME("add_bias", o, dout, n, 0, 0);
        CHECK_OK(cpu->colsum_add(cpu, (size_t)rows, cols, a, cs));
        CHECK_OK(gpu->colsum_add(gpu, (size_t)rows, cols, da, dcs));
        SAME("colsum_add", cs, dcs, (size_t)cols, 1e-6, 2);
        CHECK_OK(cpu->rowsum_add(cpu, rows, (size_t)cols, a, rs));
        CHECK_OK(gpu->rowsum_add(gpu, rows, (size_t)cols, da, drs));
        SAME("rowsum_add", rs, drs, (size_t)rows, 1e-6, 2);
        if (rows % 3 == 0 || rows == 7) { /* [A, B, C] views of the same numbers */
            int A = rows == 7 ? 7 : 3, B = rows / A, C = cols;
            float *bb = hrand((size_t)B, &r, -1, 1), *dbb = dup_to_gpu(bb, (size_t)B);
            CHECK_OK(cpu->transpose3(cpu, A, B, C, a, t));
            CHECK_OK(gpu->transpose3(gpu, A, B, C, da, dt));
            SAME("transpose3", t, dt, n, 0, 0);
            CHECK_OK(cpu->add_bias3(cpu, A, B, C, t, bb));
            CHECK_OK(gpu->add_bias3(gpu, A, B, C, dt, dbb));
            SAME("add_bias3", t, dt, n, 0, 0);
            free(bb);
            gpu->release(gpu, dbb);
        }
        free(a), free(bias), free(o), free(cs), free(rs), free(t);
        gpu->release(gpu, da), gpu->release(gpu, dbias), gpu->release(gpu, dout), gpu->release(gpu, dcs);
        gpu->release(gpu, drs), gpu->release(gpu, dt);
    }
    { /* the clipping norm over several tensors, then the scale */
        size_t sz[3] = {1, 1000, 77777};
        float *h[3], *d[3];
        double sc = 0, sg = -1;
        int i;
        for (i = 0; i < 3; i++) h[i] = hrand(sz[i], &r, -3, 3), d[i] = dup_to_gpu(h[i], sz[i]);
        CHECK_OK(cpu->sumsq(cpu, 3, h, sz, &sc));
        CHECK_OK(gpu->sumsq(gpu, 3, d, sz, &sg));
        CHECK_OK(gpu->sync(gpu));
        CHECK_MSG(fabs(sc - sg) <= 1e-12 * sc, "sumsq %.17g vs %.17g", sc, sg);
        CHECK_OK(cpu->scale(cpu, 3, h, sz, 0.37));
        CHECK_OK(gpu->scale(gpu, 3, d, sz, 0.37));
        for (i = 0; i < 3; i++) {
            SAME("scale", h[i], d[i], sz[i], 0, 0);
            free(h[i]);
            gpu->release(gpu, d[i]);
        }
    }
    report_worst();
}

/* ------------------------------------- activations, softmax, losses */

static void test_ops_activations(void) {
    static const size_t sizes[] = {1, 31, 1000, 4097};
    nmz_rng r;
    size_t si;
    int a;
    if (!backends()) return;
    nmz_rng_seed(&r, 4, 1);
    for (a = 0; a < NMZ_ACT_COUNT; a++) {
        if (a == NMZ_ACT_SOFTMAX) continue;
        for (si = 0; si < sizeof sizes / sizeof sizes[0]; si++) {
            size_t n = sizes[si];
            float *z = hrand(n, &r, -6, 6), *dy = hrand(n, &r, -1, 1), *y = malloc(n * 4), *dz = malloc(n * 4),
                  *fp = malloc(n * 4);
            float *dz_ = dup_to_gpu(z, n), *ddy = dup_to_gpu(dy, n), *dyo = dup_to_gpu(NULL, n),
                  *ddz = dup_to_gpu(NULL, n), *dfp = dup_to_gpu(NULL, n);
            char name[64];
            z[0] = 0.0f; /* the kink of relu and leaky relu */
            gpu->upload(gpu, dz_, z, n * 4);
            CHECK_OK(cpu->act_forward(cpu, (nmz_act)a, 0.05f, n, z, y));
            CHECK_OK(gpu->act_forward(gpu, (nmz_act)a, 0.05f, n, dz_, dyo));
            snprintf(name, sizeof name, "%s forward", nmz_act_name((nmz_act)a));
            SAME(name, y, dyo, n, 2e-6, 3);
            CHECK_OK(cpu->act_backward(cpu, (nmz_act)a, 0.05f, n, z, dy, dz, fp));
            CHECK_OK(gpu->act_backward(gpu, (nmz_act)a, 0.05f, n, dz_, ddy, ddz, dfp));
            snprintf(name, sizeof name, "%s backward", nmz_act_name((nmz_act)a));
            SAME(name, dz, ddz, n, 2e-6, 3);
            SAME(name, fp, dfp, n, 2e-6, 3);
            free(z), free(dy), free(y), free(dz), free(fp);
            gpu->release(gpu, dz_), gpu->release(gpu, ddy), gpu->release(gpu, dyo), gpu->release(gpu, ddz);
            gpu->release(gpu, dfp);
        }
    }
    {
        static const int shapes[][2] = {{1, 1}, {3, 7}, {64, 65}, {5, 1000}, {333, 2}};
        for (si = 0; si < sizeof shapes / sizeof shapes[0]; si++) {
            size_t rows = (size_t)shapes[si][0], n;
            int cols = shapes[si][1], k;
            float *z, *p, *dy, *dz, *t, *gr, *dz_, *dp, *ddy, *ddz, *dt, *dgr;
            double lc = 0, lg = 0;
            n = rows * cols;
            z = hrand(n, &r, -5, 5), dy = hrand(n, &r, -1, 1), p = malloc(n * 4), dz = malloc(n * 4), gr = malloc(n * 4);
            t = calloc(n, 4);
            for (k = 0; k < (int)rows; k++) t[(size_t)k * cols + nmz_rng_below(&r, (uint32_t)cols)] = 1.0f;
            dz_ = dup_to_gpu(z, n), ddy = dup_to_gpu(dy, n), dp = dup_to_gpu(NULL, n), ddz = dup_to_gpu(NULL, n);
            dt = dup_to_gpu(t, n), dgr = dup_to_gpu(NULL, n);
            CHECK_OK(cpu->softmax_forward(cpu, rows, cols, z, p));
            CHECK_OK(gpu->softmax_forward(gpu, rows, cols, dz_, dp));
            SAME("softmax", p, dp, n, 2e-6, 3);
            CHECK_OK(cpu->softmax_backward(cpu, rows, cols, p, dy, dz));
            CHECK_OK(gpu->softmax_backward(gpu, rows, cols, dp, ddy, ddz));
            SAME("softmax backward", dz, ddz, n, 2e-6, 3);
            CHECK_OK(cpu->loss_value(cpu, NMZ_LOSS_CROSS_ENTROPY, rows, cols, p, t, z, &lc));
            CHECK_OK(gpu->loss_value(gpu, NMZ_LOSS_CROSS_ENTROPY, rows, cols, dp, dt, dz_, &lg));
            CHECK_OK(gpu->sync(gpu));
            CHECK_MSG(fabs(lc - lg) <= 2e-6 * fabs(lc) + 1e-12, "cross-entropy (logits) %.9g vs %.9g", lc, lg);
            CHECK_OK(cpu->loss_value(cpu, NMZ_LOSS_CROSS_ENTROPY, rows, cols, p, t, NULL, &lc));
            CHECK_OK(gpu->loss_value(gpu, NMZ_LOSS_CROSS_ENTROPY, rows, cols, dp, dt, NULL, &lg));
            CHECK_OK(gpu->sync(gpu));
            CHECK_MSG(fabs(lc - lg) <= 2e-6 * fabs(lc) + 1e-12, "cross-entropy (probabilities) %.9g vs %.9g", lc, lg);
            CHECK_OK(cpu->loss_value(cpu, NMZ_LOSS_MSE, rows, cols, z, dy, NULL, &lc));
            CHECK_OK(gpu->loss_value(gpu, NMZ_LOSS_MSE, rows, cols, dz_, ddy, NULL, &lg));
            CHECK_OK(gpu->sync(gpu));
            CHECK_MSG(fabs(lc - lg) <= 1e-9 * fabs(lc), "mse %.12g vs %.12g", lc, lg);
            CHECK_OK(cpu->loss_grad(cpu, NMZ_LOSS_CROSS_ENTROPY, rows, cols, p, t, gr));
            CHECK_OK(gpu->loss_grad(gpu, NMZ_LOSS_CROSS_ENTROPY, rows, cols, dp, dt, dgr));
            SAME("cross-entropy gradient", gr, dgr, n, 2e-6, 3);
            CHECK_OK(cpu->loss_grad(cpu, NMZ_LOSS_MSE, rows, cols, z, dy, gr));
            CHECK_OK(gpu->loss_grad(gpu, NMZ_LOSS_MSE, rows, cols, dz_, ddy, dgr));
            SAME("mse gradient", gr, dgr, n, 0, 0);
            { /* accuracy counts: argmax (with ties) and thresholds */
                double cc = 0, cg = 0;
                if (rows > 2) z[cols] = z[cols + (cols > 1)] = 9.0f; /* a tie in row 1: the first index wins */
                gpu->upload(gpu, dz_, z, n * 4);
                CHECK_OK(cpu->count_correct(cpu, 1, rows, cols, z, t, &cc));
                CHECK_OK(gpu->count_correct(gpu, 1, rows, cols, dz_, dt, &cg));
                CHECK_OK(gpu->sync(gpu));
                CHECK_MSG(cc == cg, "count_correct (argmax) %g vs %g", cc, cg);
                CHECK_OK(cpu->count_correct(cpu, 2, rows, cols, p, t, &cc));
                CHECK_OK(gpu->count_correct(gpu, 2, rows, cols, dp, dt, &cg));
                CHECK_OK(gpu->sync(gpu));
                CHECK_MSG(cc == cg, "count_correct (threshold) %g vs %g", cc, cg);
            }
            free(z), free(p), free(dy), free(dz), free(t), free(gr);
            gpu->release(gpu, dz_), gpu->release(gpu, dp), gpu->release(gpu, ddy), gpu->release(gpu, ddz);
            gpu->release(gpu, dt), gpu->release(gpu, dgr);
        }
    }
    report_worst();
}

/* ------------------------------------------------ convolution, pooling */

static void test_ops_conv_pool(void) {
    static const int convs[][9] = {/* B C H W kh kw stride pad */
                                   {2, 3, 7, 6, 3, 3, 1, 1, 0}, {3, 2, 9, 8, 2, 3, 2, 1, 0}, {1, 1, 28, 28, 3, 3, 1, 0, 0},
                                   {4, 5, 5, 5, 5, 5, 1, 2, 0}, {2, 8, 13, 13, 3, 3, 1, 0, 0}, {1, 2, 4, 9, 1, 1, 3, 0, 0}};
    static const int pools[][5] = {/* planes H W size stride */
                                   {6, 7, 6, 2, 1}, {3, 8, 8, 2, 2}, {2, 9, 7, 3, 2}, {16, 26, 26, 2, 2}, {1, 5, 5, 5, 1}};
    nmz_rng r;
    size_t i;
    if (!backends()) return;
    nmz_rng_seed(&r, 5, 1);
    for (i = 0; i < sizeof convs / sizeof convs[0]; i++) {
        nmz_conv_shape s;
        size_t nx, nc;
        float *x, *col, *dx, *dxd, *dcol;
        s.B = convs[i][0], s.C = convs[i][1], s.H = convs[i][2], s.W = convs[i][3], s.kh = convs[i][4];
        s.kw = convs[i][5], s.stride = convs[i][6], s.pad = convs[i][7], s.F = 1;
        s.OH = (s.H + 2 * s.pad - s.kh) / s.stride + 1;
        s.OW = (s.W + 2 * s.pad - s.kw) / s.stride + 1;
        nx = (size_t)s.B * s.C * s.H * s.W;
        nc = (size_t)s.C * s.kh * s.kw * s.B * s.OH * s.OW;
        x = hrand(nx, &r, -1, 1), col = malloc(nc * 4), dx = malloc(nx * 4);
        dxd = dup_to_gpu(x, nx), dcol = dup_to_gpu(NULL, nc);
        CHECK_OK(cpu->im2col(cpu, &s, x, col));
        CHECK_OK(gpu->im2col(gpu, &s, dxd, dcol));
        SAME("im2col", col, dcol, nc, 0, 0);
        free(col);
        col = hrand(nc, &r, -1, 1);
        gpu->upload(gpu, dcol, col, nc * 4);
        CHECK_OK(cpu->col2im(cpu, &s, col, dx));
        CHECK_OK(gpu->col2im(gpu, &s, dcol, dxd));
        SAME("col2im", dx, dxd, nx, 2e-6, 3);
        free(x), free(col), free(dx);
        gpu->release(gpu, dxd), gpu->release(gpu, dcol);
    }
    for (i = 0; i < sizeof pools / sizeof pools[0]; i++) {
        int P = pools[i][0], H = pools[i][1], W = pools[i][2], k = pools[i][3], st = pools[i][4];
        int OH = (H - k) / st + 1, OW = (W - k) / st + 1;
        size_t nx = (size_t)P * H * W, ny = (size_t)P * OH * OW;
        float *x = hrand(nx, &r, -1, 1), *y = malloc(ny * 4), *am = malloc(ny * 4), *dy = hrand(ny, &r, -1, 1),
              *dx = malloc(nx * 4);
        float *dxx = dup_to_gpu(x, nx), *dyy = dup_to_gpu(NULL, ny), *dam = dup_to_gpu(NULL, ny),
              *ddy = dup_to_gpu(dy, ny), *ddx = dup_to_gpu(NULL, nx);
        CHECK_OK(cpu->maxpool_forward(cpu, P, H, W, k, st, OH, OW, x, y, am));
        CHECK_OK(gpu->maxpool_forward(gpu, P, H, W, k, st, OH, OW, dxx, dyy, dam));
        SAME("maxpool", y, dyy, ny, 0, 0);
        SAME("maxpool winners", am, dam, ny, 0, 0);
        CHECK_OK(cpu->maxpool_backward(cpu, P, H, W, k, st, OH, OW, dy, am, dx));
        CHECK_OK(gpu->maxpool_backward(gpu, P, H, W, k, st, OH, OW, ddy, dam, ddx));
        SAME("maxpool backward", dx, ddx, nx, 2e-6, 3);
        free(x), free(y), free(am), free(dy), free(dx);
        gpu->release(gpu, dxx), gpu->release(gpu, dyy), gpu->release(gpu, dam), gpu->release(gpu, ddy);
        gpu->release(gpu, ddx);
    }
    report_worst();
}

/* ------------------------------------ dropout, embeddings, positions */

static void test_ops_lookups(void) {
    nmz_rng r, r1, r2;
    if (!backends()) return;
    nmz_rng_seed(&r, 6, 1);
    { /* dropout: the GPU draws its own masks — check what a mask must be */
        size_t n = 200001, i, zeros = 0, bad = 0;
        float *m = malloc(n * 4), *m2 = malloc(n * 4), *dm = dup_to_gpu(NULL, n);
        const float keep = 1.0f / (1.0f - 0.3f);
        nmz_rng_seed(&r1, 9, 9);
        r2 = r1;
        CHECK_OK(gpu->dropout_mask(gpu, n, 0.3f, &r1, dm));
        from_gpu(m, dm, n);
        CHECK_OK(gpu->dropout_mask(gpu, n, 0.3f, &r2, dm));
        from_gpu(m2, dm, n);
        for (i = 0; i < n; i++) {
            zeros += m[i] == 0.0f;
            bad += m[i] != 0.0f && m[i] != keep;
        }
        CHECK_MSG(fabs((double)zeros / n - 0.3) < 0.005, "dropout: %.4f of the mask is 0", (double)zeros / n);
        CHECK(bad == 0);
        CHECK(memcmp(m, m2, n * 4) == 0); /* the same PRNG state, the same mask */
        CHECK(r1.state == r2.state);      /* and the net's PRNG moved on */
        CHECK_OK(gpu->dropout_mask(gpu, n, 0.3f, &r1, dm));
        from_gpu(m2, dm, n);
        CHECK(memcmp(m, m2, n * 4) != 0);
        test_note("dropout: %.4f zeros at rate 0.3, values 0 or 1/(1-p)", (double)zeros / n);
        free(m), free(m2);
        gpu->release(gpu, dm);
    }
    {
        static const int cases[][3] = {{17, 5, 40}, {65, 32, 4096}, {5000, 8, 4000}}; /* V D n; the last adds atomically */
        size_t ci;
        for (ci = 0; ci < sizeof cases / sizeof cases[0]; ci++) {
            int V = cases[ci][0], D = cases[ci][1];
            size_t n = (size_t)cases[ci][2], i;
            float *ids = malloc(n * 4), *W = hrand((size_t)V * D, &r, -1, 1), *y = malloc(n * D * 4),
                  *dy = hrand(n * D, &r, -1, 1), *dW = hrand((size_t)V * D, &r, -1, 1);
            float *dids, *dWd, *dyd, *ddy, *ddW;
            for (i = 0; i < n; i++) ids[i] = (float)nmz_rng_below(&r, (uint32_t)(i % 3 ? V : 3)); /* repeats */
            dids = dup_to_gpu(ids, n), dWd = dup_to_gpu(W, (size_t)V * D), dyd = dup_to_gpu(NULL, n * D);
            ddy = dup_to_gpu(dy, n * D), ddW = dup_to_gpu(dW, (size_t)V * D);
            CHECK_OK(cpu->embed_forward(cpu, n, V, D, ids, W, y));
            CHECK_OK(gpu->embed_forward(gpu, n, V, D, dids, dWd, dyd));
            SAME("embedding", y, dyd, n * D, 0, 0);
            CHECK_OK(cpu->embed_backward(cpu, n, V, D, ids, dy, dW));
            CHECK_OK(gpu->embed_backward(gpu, n, V, D, dids, ddy, ddW));
            SAME(V > 1000 ? "embedding backward (atomic)" : "embedding backward", dW, ddW, (size_t)V * D, 2e-6, 3);
            free(ids), free(W), free(y), free(dy), free(dW);
            gpu->release(gpu, dids), gpu->release(gpu, dWd), gpu->release(gpu, dyd), gpu->release(gpu, ddy);
            gpu->release(gpu, ddW);
        }
    }
    {
        int offs[2][3] = {{3, 3, 3}, {0, 2, 5}}, k;
        for (k = 0; k < 2; k++) {
            const int B = 3, T = 4, D = 6, L = 12;
            size_t n = (size_t)B * T * D;
            float *x = hrand(n, &r, -1, 1), *P = hrand((size_t)L * D, &r, -1, 1), *y = malloc(n * 4),
                  *dP = hrand((size_t)L * D, &r, -1, 1);
            float *dx = dup_to_gpu(x, n), *dPP = dup_to_gpu(P, (size_t)L * D), *dy = dup_to_gpu(NULL, n),
                  *ddP = dup_to_gpu(dP, (size_t)L * D);
            CHECK_OK(cpu->posenc_forward(cpu, B, T, D, offs[k], x, P, y));
            CHECK_OK(gpu->posenc_forward(gpu, B, T, D, offs[k], dx, dPP, dy));
            SAME("position encoding", y, dy, n, 0, 0);
            CHECK_OK(cpu->posenc_backward(cpu, B, T, D, offs[k], x, dP));
            CHECK_OK(gpu->posenc_backward(gpu, B, T, D, offs[k], dx, ddP));
            SAME(k ? "position encoding backward (offsets)" : "position encoding backward", dP, ddP, (size_t)L * D, 2e-6,
                 3);
            free(x), free(P), free(y), free(dP);
            gpu->release(gpu, dx), gpu->release(gpu, dPP), gpu->release(gpu, dy), gpu->release(gpu, ddP);
        }
    }
    report_worst();
}

/* --------------------------------------------- LayerNorm, attention */

static void test_ops_layernorm_attention(void) {
    static const int lns[][2] = {{1, 5}, {7, 64}, {300, 129}, {64, 1}, {9, 512}};
    static const int attns[][5] = {/* B H T Tp d */ {2, 3, 5, 0, 4}, {2, 3, 5, 3, 4}, {1, 2, 1, 7, 16},
                                   {1, 2, 70, 0, 16}, {3, 4, 33, 0, 8}, {2, 1, 40, 25, 64},
                                   {5, 2, 20, 4, 8}};
    nmz_rng r;
    size_t i;
    if (!backends()) return;
    nmz_rng_seed(&r, 7, 1);
    for (i = 0; i < sizeof lns / sizeof lns[0]; i++) {
        size_t rows = (size_t)lns[i][0], D = (size_t)lns[i][1], n = rows * D;
        float *x = hrand(n, &r, -3, 3), *gn = hrand(D, &r, 0.5, 1.5), *b = hrand(D, &r, -1, 1), *y = malloc(n * 4),
              *xh = malloc(n * 4), *mu = malloc(rows * 4), *rs = malloc(rows * 4), *dy = hrand(n, &r, -1, 1),
              *dx = malloc(n * 4), *dg = hrand(D, &r, -1, 1), *db = hrand(D, &r, -1, 1), *dxh = malloc(n * 4);
        float *dX = dup_to_gpu(x, n), *dG = dup_to_gpu(gn, D), *dB = dup_to_gpu(b, D), *dY = dup_to_gpu(NULL, n),
              *dXH = dup_to_gpu(NULL, n), *dMU = dup_to_gpu(NULL, rows), *dRS = dup_to_gpu(NULL, rows),
              *dDY = dup_to_gpu(dy, n), *dDX = dup_to_gpu(NULL, n), *dDG = dup_to_gpu(dg, D), *dDB = dup_to_gpu(db, D),
              *dDXH = dup_to_gpu(NULL, n);
        CHECK_OK(cpu->layernorm_forward(cpu, rows, (int)D, 1e-5f, x, gn, b, y, xh, mu, rs));
        CHECK_OK(gpu->layernorm_forward(gpu, rows, (int)D, 1e-5f, dX, dG, dB, dY, dXH, dMU, dRS));
        SAME("layernorm", y, dY, n, 2e-6, 3);
        SAME("layernorm x_hat", xh, dXH, n, 2e-6, 3);
        SAME("layernorm mean", mu, dMU, rows, 2e-6, 3);
        SAME("layernorm rstd", rs, dRS, rows, 2e-6, 3);
        CHECK_OK(cpu->layernorm_backward(cpu, rows, (int)D, dy, xh, rs, gn, dx, dg, db, dxh));
        CHECK_OK(gpu->layernorm_backward(gpu, rows, (int)D, dDY, dXH, dRS, dG, dDX, dDG, dDB, dDXH));
        SAME("layernorm backward dx", dx, dDX, n, 2e-6, 3);
        SAME("layernorm backward dg", dg, dDG, D, 2e-6, 3);
        SAME("layernorm backward db", db, dDB, D, 2e-6, 3);
        SAME("layernorm backward dx_hat", dxh, dDXH, n, 2e-6, 3);
        free(x), free(gn), free(b), free(y), free(xh), free(mu), free(rs), free(dy), free(dx), free(dg), free(db);
        free(dxh);
        gpu->release(gpu, dX), gpu->release(gpu, dG), gpu->release(gpu, dB), gpu->release(gpu, dY);
        gpu->release(gpu, dXH), gpu->release(gpu, dMU), gpu->release(gpu, dRS), gpu->release(gpu, dDY);
        gpu->release(gpu, dDX), gpu->release(gpu, dDG), gpu->release(gpu, dDB), gpu->release(gpu, dDXH);
    }
    for (i = 0; i < sizeof attns / sizeof attns[0]; i++) {
        const int B = attns[i][0], H = attns[i][1], T = attns[i][2], Tp = attns[i][3], d = attns[i][4], E = H * d,
                  Tk = Tp + T;
        size_t nq = (size_t)B * T * 3 * E, nkv = (size_t)B * Tk * 2 * E, nP = (size_t)B * H * T * Tk,
               nc = (size_t)B * T * E;
        float *qkv = hrand(nq, &r, -1, 1), *kv = hrand(nkv, &r, -1, 1), *P = malloc(nP * 4), *S = calloc(nP, 4),
              *cv = malloc(nc * 4), *dc = hrand(nc, &r, -1, 1), *dP = malloc(nP * 4), *dq = calloc(nq, 4),
              *dkv = calloc(nkv, 4);
        float *gq = dup_to_gpu(qkv, nq), *gkv = dup_to_gpu(kv, nkv), *gP = dup_to_gpu(NULL, nP), *gS = dup_to_gpu(S, nP),
              *gcv = dup_to_gpu(NULL, nc), *gdc = dup_to_gpu(dc, nc), *gdP = dup_to_gpu(NULL, nP),
              *gdq = dup_to_gpu(dq, nq), *gdkv = dup_to_gpu(dkv, nkv);
        char name[64];
        snprintf(name, sizeof name, "attention B%d H%d T%d Tp%d d%d", B, H, T, Tp, d);
        CHECK_OK(cpu->attention_forward(cpu, B, H, T, Tk, Tp, d, E, qkv, kv, P, S, cv));
        CHECK_OK(gpu->attention_forward(gpu, B, H, T, Tk, Tp, d, E, gq, gkv, gP, gS, gcv));
        SAME(name, P, gP, nP, 1e-5, 4);
        SAME(name, S, gS, nP, 1e-5, 4);
        SAME(name, cv, gcv, nc, 1e-5, 4);
        CHECK_OK(cpu->attention_backward(cpu, B, H, T, Tk, Tp, d, E, qkv, kv, P, dc, dP, dq, dkv));
        CHECK_OK(gpu->attention_backward(gpu, B, H, T, Tk, Tp, d, E, gq, gkv, gP, gdc, gdP, gdq, gdkv));
        snprintf(name, sizeof name, "attention backward T%d Tp%d d%d", T, Tp, d);
        SAME(name, dP, gdP, nP, 1e-5, 4);
        SAME(name, dq, gdq, nq, 1e-5, 4);
        SAME(name, dkv, gdkv, nkv, 1e-5, 4);
        free(qkv), free(kv), free(P), free(S), free(cv), free(dc), free(dP), free(dq), free(dkv);
        gpu->release(gpu, gq), gpu->release(gpu, gkv), gpu->release(gpu, gP), gpu->release(gpu, gS);
        gpu->release(gpu, gcv), gpu->release(gpu, gdc), gpu->release(gpu, gdP), gpu->release(gpu, gdq);
        gpu->release(gpu, gdkv);
    }
    report_worst();
}

/* -------------------------------------------------- recurrent cells */

typedef struct pair {
    float *h, *d;
    size_t n;
} pair;

static pair mk(size_t n, nmz_rng *r, double lo, double hi) {
    pair p;
    p.n = n;
    p.h = hrand(n, r, lo, hi);
    p.d = dup_to_gpu(p.h, n);
    return p;
}
static void drop(pair *p) {
    free(p->h);
    gpu->release(gpu, p->d);
}
static nmz_rows rh(pair p, size_t off, int ld) {
    nmz_rows x;
    x.p = p.h + off;
    x.ld = ld;
    return x;
}
static nmz_rows rd(pair p, size_t off, int ld) {
    nmz_rows x;
    x.p = p.d + off;
    x.ld = ld;
    return x;
}

static void test_ops_cells(void) {
    const int B = 3, H = 5, T = 4, t = 2;
    nmz_rng r;
    int G;
    if (!backends()) return;
    nmz_rng_seed(&r, 8, 1);
    for (G = 1; G <= 4; G++) { /* 1 = rnn, 3 = gru, 4 = lstm (2: skipped) */
        const int GH = G * H;
        pair a, prev, c, cp, h, dy, dn, dc, da, rec_buf[5], dir, rhh, drh;
        nmz_seq_record rc, rg;
        int k;
        if (G == 2) continue;
        a = mk((size_t)B * T * GH, &r, -2, 2);
        prev = mk((size_t)B * H, &r, -1, 1);
        c = mk((size_t)B * T * H, &r, -1, 1);
        cp = mk((size_t)B * H, &r, -1, 1);
        h = mk((size_t)B * T * H, &r, -1, 1);
        dy = mk((size_t)B * T * H, &r, -1, 1);
        dn = mk((size_t)B * H, &r, -1, 1);
        dc = mk((size_t)B * H, &r, -1, 1);
        da = mk((size_t)B * T * GH, &r, -1, 1);
        dir = mk((size_t)B * H, &r, -1, 1);
        rhh = mk((size_t)B * T * H, &r, -1, 1);
        drh = mk((size_t)B * H, &r, -1, 1);
        for (k = 0; k < 5; k++) rec_buf[k] = mk((size_t)B * T * H, &r, 0, 0);
        memset(&rc, 0, sizeof rc);
        memset(&rg, 0, sizeof rg);
        rc.T = rg.T = T;
        rc.dh_next = rec_buf[0].h, rc.dh = rec_buf[1].h, rc.dc_next = rec_buf[2].h, rc.dc = rec_buf[3].h;
        rc.dc_prev = rec_buf[4].h;
        rg.dh_next = rec_buf[0].d, rg.dh = rec_buf[1].d, rg.dc_next = rec_buf[2].d, rg.dc = rec_buf[3].d;
        rg.dc_prev = rec_buf[4].d;
        if (G == 3) rc.dc_next = rg.dc_next = NULL, rc.direct = rec_buf[2].h, rg.direct = rec_buf[2].d;
#define AH(p) rh(p, (size_t)t * GH, T * GH)
#define AD(p) rd(p, (size_t)t * GH, T * GH)
#define SH(p) rh(p, (size_t)t * H, T * H)
#define SD(p) rd(p, (size_t)t * H, T * H)
#define FH(p) rh(p, 0, H)
#define FD(p) rd(p, 0, H)
        if (G == 1) {
            CHECK_OK(cpu->rnn_cell_forward(cpu, B, H, AH(a), SH(h)));
            CHECK_OK(gpu->rnn_cell_forward(gpu, B, H, AD(a), SD(h)));
            SAME("rnn cell", h.h, h.d, h.n, 2e-6, 3);
            CHECK_OK(cpu->rnn_cell_backward(cpu, B, H, SH(h), SH(dy), FH(dn), AH(da), &rc, t));
            CHECK_OK(gpu->rnn_cell_backward(gpu, B, H, SD(h), SD(dy), FD(dn), AD(da), &rg, t));
            SAME("rnn cell backward", da.h, da.d, da.n, 2e-6, 3);
        } else if (G == 4) {
            CHECK_OK(cpu->lstm_cell_forward(cpu, B, H, AH(a), FH(cp), SH(c), SH(h)));
            CHECK_OK(gpu->lstm_cell_forward(gpu, B, H, AD(a), FD(cp), SD(c), SD(h)));
            SAME("lstm cell gates", a.h, a.d, a.n, 2e-6, 3);
            SAME("lstm cell c", c.h, c.d, c.n, 2e-6, 3);
            SAME("lstm cell h", h.h, h.d, h.n, 2e-6, 3);
            CHECK_OK(cpu->lstm_cell_backward(cpu, B, H, AH(a), SH(c), FH(cp), SH(dy), FH(dn), FH(dc), AH(da), &rc, t));
            CHECK_OK(gpu->lstm_cell_backward(gpu, B, H, AD(a), SD(c), FD(cp), SD(dy), FD(dn), FD(dc), AD(da), &rg, t));
            SAME("lstm cell backward da", da.h, da.d, da.n, 2e-6, 3);
            SAME("lstm cell backward dc", dc.h, dc.d, dc.n, 2e-6, 3);
        } else {
            CHECK_OK(cpu->gru_cell_forward1(cpu, B, H, AH(a), FH(prev), FH(drh)));
            CHECK_OK(gpu->gru_cell_forward1(gpu, B, H, AD(a), FD(prev), FD(drh)));
            SAME("gru cell (z, r)", a.h, a.d, a.n, 2e-6, 3);
            SAME("gru cell r*h", drh.h, drh.d, drh.n, 2e-6, 3);
            CHECK_OK(cpu->gru_cell_forward2(cpu, B, H, AH(a), FH(prev), SH(h)));
            CHECK_OK(gpu->gru_cell_forward2(gpu, B, H, AD(a), FD(prev), SD(h)));
            SAME("gru cell h", h.h, h.d, h.n, 2e-6, 3);
            CHECK_OK(cpu->gru_cell_backward1(cpu, B, H, AH(a), FH(prev), SH(dy), FH(dn), AH(da), FH(dir), SH(rhh), &rc, t));
            CHECK_OK(gpu->gru_cell_backward1(gpu, B, H, AD(a), FD(prev), SD(dy), FD(dn), AD(da), FD(dir), SD(rhh), &rg, t));
            SAME("gru cell backward 1", da.h, da.d, da.n, 2e-6, 3);
            SAME("gru cell backward direct", dir.h, dir.d, dir.n, 2e-6, 3);
            SAME("gru cell backward r*h", rhh.h, rhh.d, rhh.n, 2e-6, 3);
            CHECK_OK(cpu->gru_cell_backward2(cpu, B, H, AH(a), FH(prev), FH(drh), FH(dir), AH(da), FH(dn)));
            CHECK_OK(gpu->gru_cell_backward2(gpu, B, H, AD(a), FD(prev), FD(drh), FD(dir), AD(da), FD(dn)));
            SAME("gru cell backward 2", da.h, da.d, da.n, 2e-6, 3);
            SAME("gru cell backward dh", dn.h, dn.d, dn.n, 2e-6, 3);
        }
        for (k = 0; k < 5; k++) SAME("cell trace record", rec_buf[k].h, rec_buf[k].d, rec_buf[k].n, 2e-6, 3);
        drop(&a), drop(&prev), drop(&c), drop(&cp), drop(&h), drop(&dy), drop(&dn), drop(&dc), drop(&da), drop(&dir);
        drop(&rhh), drop(&drh);
        for (k = 0; k < 5; k++) drop(&rec_buf[k]);
    }
    report_worst();
}

/* ------------------------------------ optimizers, the training loop */

static void test_ops_optim_loop(void) {
    nmz_rng r;
    size_t n = 1001;
    int fresh;
    if (!backends()) return;
    nmz_rng_seed(&r, 9, 1);
    for (fresh = 0; fresh < 2; fresh++) {
        pair g = mk(n, &r, -1, 1), v = mk(n, &r, -1, 1), w = mk(n, &r, -1, 1), m = mk(n, &r, -1, 1),
             vv = mk(n, &r, 0, 1), o[5];
        int k;
        for (k = 0; k < 5; k++) o[k] = mk(n, &r, 0, 0);
        CHECK_OK(cpu->sgd_velocity(cpu, n, 0.1f, 0.9f, fresh, g.h, v.h, o[0].h));
        CHECK_OK(gpu->sgd_velocity(gpu, n, 0.1f, 0.9f, fresh, g.d, v.d, o[0].d));
        SAME("sgd velocity", o[0].h, o[0].d, n, 0, 0);
        CHECK_OK(cpu->adam_preview(cpu, n, 0.01, 0.9, 0.999, 1e-8, 0.19, 0.002, 0.1, fresh, w.h, g.h, m.h, vv.h, o[0].h,
                                   o[1].h, o[2].h, o[3].h, o[4].h));
        CHECK_OK(gpu->adam_preview(gpu, n, 0.01, 0.9, 0.999, 1e-8, 0.19, 0.002, 0.1, fresh, w.d, g.d, m.d, vv.d, o[0].d,
                                   o[1].d, o[2].d, o[3].d, o[4].d));
        for (k = 0; k < 5; k++) SAME("adam preview", o[k].h, o[k].d, n, 1e-6, 2);
        CHECK_OK(cpu->adam_update(cpu, n, 0.01, 0.9, 0.999, 1e-8, 0.19, 0.002, 0.1, w.h, g.h, m.h, vv.h));
        CHECK_OK(gpu->adam_update(gpu, n, 0.01, 0.9, 0.999, 1e-8, 0.19, 0.002, 0.1, w.d, g.d, m.d, vv.d));
        SAME("adam w", w.h, w.d, n, 1e-6, 2);
        SAME("adam m", m.h, m.d, n, 1e-6, 2);
        SAME("adam v", vv.h, vv.d, n, 1e-6, 2);
        drop(&g), drop(&v), drop(&w), drop(&m), drop(&vv);
        for (k = 0; k < 5; k++) drop(&o[k]);
    }
    { /* normalisation: per channel ([3, 4, 5] samples) and per feature */
        float off[3] = {0.5f, -1.0f, 2.0f}, sc[3] = {2.0f, 0.25f, 1.5f};
        pair x = mk(2 * 60, &r, -3, 3), po = {off, dup_to_gpu(off, 3), 3}, ps = {sc, dup_to_gpu(sc, 3), 3};
        CHECK_OK(cpu->norm_apply(cpu, 120, 60, 20, 3, off, sc, x.h, 0));
        CHECK_OK(gpu->norm_apply(gpu, 120, 60, 20, 3, po.d, ps.d, x.d, 0));
        SAME("normalize (channels)", x.h, x.d, 120, 0, 0);
        CHECK_OK(cpu->norm_apply(cpu, 120, 3, 1, 3, off, sc, x.h, 1));
        CHECK_OK(gpu->norm_apply(gpu, 120, 3, 1, 3, po.d, ps.d, x.d, 1));
        SAME("normalize gradient (features)", x.h, x.d, 120, 0, 0);
        drop(&x);
        gpu->release(gpu, po.d), gpu->release(gpu, ps.d);
    }
    { /* gathers and one-hot targets */
        const int rows = 37, V = 11;
        const size_t width = 13, ns = 50;
        pair src = mk(ns * width, &r, -1, 1), idx = mk((size_t)rows, &r, 0, 0), dst = mk((size_t)rows * width, &r, 0, 0);
        pair ids = mk(ns, &r, 0, 0), oh = mk(ns * V, &r, 0, 0);
        int i;
        for (i = 0; i < rows; i++) idx.h[i] = (float)nmz_rng_below(&r, (uint32_t)ns);
        for (i = 0; i < (int)ns; i++) ids.h[i] = (float)nmz_rng_below(&r, (uint32_t)V);
        gpu->upload(gpu, idx.d, idx.h, rows * 4u);
        gpu->upload(gpu, ids.d, ids.h, ns * 4);
        CHECK_OK(cpu->gather_rows(cpu, rows, width, idx.h, src.h, dst.h));
        CHECK_OK(gpu->gather_rows(gpu, rows, width, idx.d, src.d, dst.d));
        SAME("gather_rows", dst.h, dst.d, dst.n, 0, 0);
        CHECK_OK(cpu->onehot(cpu, ns, V, ids.h, oh.h));
        CHECK_OK(gpu->onehot(gpu, ns, V, ids.d, oh.d));
        SAME("onehot", oh.h, oh.d, oh.n, 0, 0);
        drop(&src), drop(&idx), drop(&dst), drop(&ids), drop(&oh);
    }
    report_worst();
}

/* ------------------------------------------------- layers and models */

/* Z_WIDE (not in the loops over the zoo: too many parameters for a full
 * gradient check): recurrent layers wider than the CUDA sequence kernels
 * take (lstm H > 256, gru H > 512, rnn H > 1024), which the layers then
 * compose step by step on the GPU */
enum { Z_DENSE, Z_ACTS, Z_CONV, Z_RNN, Z_LSTM, Z_GRU, Z_GPT, Z_PARTS, Z_CNN_BIG, Z_COUNT, Z_WIDE = Z_COUNT };
static const char *const k_zoo[Z_COUNT] = {"dense+softmax/ce", "activations/mse", "conv+pool+dropout",
                                           "embed+rnn (carry)", "lstm+dropout+lstm", "gru (carry)",
                                           "transformer (tied)", "sinusoid+ln+attention+ffn", "mnist-like cnn"};

typedef struct model {
    nmz_net *net;
    nmz_tensor x, t;
    int carry;
} model;

static void fill_r(nmz_tensor *t, nmz_rng *r, double lo, double hi) {
    size_t i;
    for (i = 0; i < t->size; i++) t->data[i] = (float)(lo + (hi - lo) * nmz_rng_uniform(r));
}

static int make(int k, model *m) {
    nmz_rng r;
    int i, j, V = 11, shape[4];
    memset(m, 0, sizeof *m);
    nmz_rng_seed(&r, 77 + (uint64_t)k, 3);
    switch (k) {
    case Z_DENSE:
        shape[0] = 5;
        if (nmz_net_create(&m->net, 1, shape, 5) != NMZ_OK) return 0;
        nmz_net_add_dense(m->net, 7, NMZ_ACT_TANH, NMZ_INIT_AUTO);
        nmz_net_add_dense(m->net, 3, NMZ_ACT_SOFTMAX, NMZ_INIT_AUTO);
        nmz_net_set_loss(m->net, NMZ_LOSS_CROSS_ENTROPY);
        nmz_net_set_normalization(m->net, NMZ_NORM_ZSCORE);
        break;
    case Z_ACTS:
        shape[0] = 4;
        if (nmz_net_create(&m->net, 1, shape, 6) != NMZ_OK) return 0;
        nmz_net_add_dense(m->net, 6, NMZ_ACT_SIGMOID, NMZ_INIT_AUTO);
        nmz_net_add_dense(m->net, 6, NMZ_ACT_LEAKY_RELU, NMZ_INIT_AUTO);
        nmz_net_add_dense(m->net, 6, NMZ_ACT_GELU, NMZ_INIT_AUTO);
        nmz_net_add_dense(m->net, 5, NMZ_ACT_RELU, NMZ_INIT_AUTO);
        nmz_net_add_dense(m->net, 3, NMZ_ACT_LINEAR, NMZ_INIT_AUTO);
        nmz_net_add(m->net, nmz_layer_activation(NMZ_ACT_SOFTMAX, 0));
        break;
    case Z_CONV:
    case Z_CNN_BIG:
        if (k == Z_CONV)
            shape[0] = 2, shape[1] = 7, shape[2] = 6;
        else
            shape[0] = 1, shape[1] = 28, shape[2] = 28;
        if (nmz_net_create(&m->net, 3, shape, 7) != NMZ_OK) return 0;
        if (k == Z_CONV) {
            nmz_net_add(m->net, nmz_layer_conv2d(3, 3, 3, 1, NMZ_PAD_SAME, NMZ_INIT_AUTO));
            nmz_net_add(m->net, nmz_layer_activation(NMZ_ACT_TANH, 0));
            nmz_net_add(m->net, nmz_layer_maxpool2d(2, 1));
            nmz_net_add(m->net, nmz_layer_conv2d(4, 2, 3, 2, 1, NMZ_INIT_XAVIER));
            nmz_net_add(m->net, nmz_layer_activation(NMZ_ACT_SIGMOID, 0));
        } else {
            nmz_net_add(m->net, nmz_layer_conv2d(8, 3, 3, 1, 0, NMZ_INIT_AUTO));
            nmz_net_add(m->net, nmz_layer_activation(NMZ_ACT_TANH, 0));
            nmz_net_add(m->net, nmz_layer_maxpool2d(2, 0));
            nmz_net_add(m->net, nmz_layer_conv2d(16, 3, 3, 1, 0, NMZ_INIT_AUTO));
            nmz_net_add(m->net, nmz_layer_activation(NMZ_ACT_TANH, 0));
            nmz_net_add(m->net, nmz_layer_maxpool2d(2, 0));
        }
        nmz_net_add(m->net, nmz_layer_flatten());
        nmz_net_add(m->net, nmz_layer_dropout(0.3f));
        nmz_net_add_dense(m->net, 3, NMZ_ACT_SOFTMAX, NMZ_INIT_AUTO);
        nmz_net_set_loss(m->net, NMZ_LOSS_CROSS_ENTROPY);
        break;
    default:
        shape[0] = NMZ_ANY_LENGTH;
        if (nmz_net_create(&m->net, 1, shape, 8 + (uint64_t)k) != NMZ_OK) return 0;
        nmz_net_add(m->net, nmz_layer_embedding(V, 6, NMZ_INIT_AUTO));
        if (k == Z_RNN) {
            nmz_net_add(m->net, nmz_layer_rnn(7, NMZ_INIT_AUTO));
            m->carry = 1;
        } else if (k == Z_WIDE) {
            nmz_net_add(m->net, nmz_layer_lstm(260, NMZ_INIT_AUTO, 1.0f));
            nmz_net_add(m->net, nmz_layer_gru(520, NMZ_INIT_AUTO));
            nmz_net_add(m->net, nmz_layer_rnn(1030, NMZ_INIT_AUTO));
            m->carry = 1;
        } else if (k == Z_LSTM) {
            nmz_net_add(m->net, nmz_layer_lstm(5, NMZ_INIT_AUTO, 1.0f));
            nmz_net_add(m->net, nmz_layer_dropout(0.25f));
            nmz_net_add(m->net, nmz_layer_lstm(4, NMZ_INIT_XAVIER, 0.5f));
        } else if (k == Z_GRU) {
            nmz_net_add(m->net, nmz_layer_gru(6, NMZ_INIT_AUTO));
            m->carry = 1;
        } else if (k == Z_GPT) {
            nmz_net_add(m->net, nmz_layer_posenc(NMZ_POS_LEARNED, 16));
            nmz_net_add(m->net, nmz_layer_block(2, 0, 12, NMZ_ACT_GELU, 0.2f));
            nmz_net_add(m->net, nmz_layer_layernorm());
            nmz_net_add(m->net, nmz_layer_tied_output(V));
        } else {
            nmz_net_add(m->net, nmz_layer_posenc(NMZ_POS_SINUSOIDAL, 16));
            nmz_net_add(m->net, nmz_layer_layernorm());
            nmz_net_add(m->net, nmz_layer_attention(2, 3));
            nmz_net_add(m->net, nmz_layer_ffn(10, NMZ_ACT_RELU));
            nmz_net_add(m->net, nmz_layer_dense(V, NMZ_INIT_AUTO));
        }
        if (k != Z_GPT && k != Z_PARTS) nmz_net_add(m->net, nmz_layer_dense(V, NMZ_INIT_AUTO));
        nmz_net_add(m->net, nmz_layer_activation(NMZ_ACT_SOFTMAX, 0));
        nmz_net_set_loss(m->net, NMZ_LOSS_CROSS_ENTROPY);
        break;
    }
    if (nmz_net_build(m->net) != NMZ_OK) return 0;
    for (i = 0; k != Z_WIDE && i < nmz_net_layer_count(m->net); i++) { /* wide: tanh would saturate */
        nmz_layer *l = nmz_net_layer(m->net, i);
        for (j = 0; j < nmz_layer_param_count(l); j++) {
            nmz_tensor *v = nmz_layer_param_value(l, j);
            size_t e;
            for (e = 0; e < v->size; e++) v->data[e] += (float)(0.3 * (2.0 * nmz_rng_uniform(&r) - 1.0));
        }
    }
    if (k >= Z_RNN && k != Z_CNN_BIG) {
        int s3[3] = {3, 6, V};
        size_t e;
        shape[0] = 3, shape[1] = 6;
        nmz_tensor_init(&m->x, 2, shape);
        nmz_tensor_init(&m->t, 3, s3);
        for (e = 0; e < m->x.size; e++) m->x.data[e] = (float)nmz_rng_below(&r, (uint32_t)V);
        for (e = 0; e < 18; e++) m->t.data[e * V + nmz_rng_below(&r, (uint32_t)V)] = 1.0f;
    } else {
        const nmz_layer *l0 = nmz_net_layer(m->net, 0);
        const int out = nmz_net_output_size(m->net);
        int s2[2];
        size_t rr;
        shape[0] = 4;
        for (i = 0; i < l0->in_rank; i++) shape[i + 1] = l0->in_shape[i];
        nmz_tensor_init(&m->x, l0->in_rank + 1, shape);
        fill_r(&m->x, &r, -1.0, 2.0);
        s2[0] = 4;
        s2[1] = out;
        nmz_tensor_init(&m->t, 2, s2);
        for (rr = 0; rr < 4; rr++) m->t.data[rr * out + nmz_rng_below(&r, (uint32_t)out)] = 1.0f;
        if (k == Z_DENSE) {
            nmz_dataset *ds = NULL;
            nmz_dataset_create(&ds, 4, 5, 3, m->x.data, m->t.data);
            nmz_net_fit_normalization(m->net, ds);
            nmz_dataset_free(ds);
        }
    }
    return 1;
}

static void unmake(model *m) {
    nmz_net_free(m->net);
    nmz_tensor_free(&m->x);
    nmz_tensor_free(&m->t);
}

typedef struct result {
    double loss;
    float *v;
    size_t n;
} result;

static void push(result *r, const nmz_tensor *t) {
    if (!t || !t->size) return;
    r->v = realloc(r->v, (r->n + t->size) * sizeof *r->v);
    memcpy(r->v + r->n, t->data, t->size * sizeof(float));
    r->n += t->size;
}

/* evaluation mode (dropout off: the GPU draws other masks) */
static int run_pass(int k, const char *backend, result *out) {
    model m;
    const nmz_tensor *y;
    int i, j;
    memset(out, 0, sizeof *out);
    if (nmz_backend_select(backend) != NMZ_OK || !make(k, &m)) return 0;
    if (m.carry) {
        nmz_net_set_carry(m.net, 1);
        if (nmz_net_forward(m.net, &m.x, NULL) != NMZ_OK) return 0;
    }
    if (nmz_net_forward_mode(m.net, &m.x, 0, &y) != NMZ_OK) return 0;
    push(out, y);
    if (nmz_net_backward(m.net, &m.t, &out->loss) != NMZ_OK) return 0;
    for (i = 0; i < nmz_net_layer_count(m.net); i++)
        for (j = 0; j < nmz_layer_param_count(nmz_net_layer(m.net, i)); j++)
            push(out, nmz_layer_param_grad(nmz_net_layer(m.net, i), j));
    push(out, nmz_net_input_grad(m.net));
    unmake(&m);
    nmz_backend_select("cpu");
    return 1;
}

static void test_layers_match_cpu(void) {
    int k;
    for (k = 0; k < Z_COUNT; k++) {
        result a, b;
        double d;
        CHECK_MSG(run_pass(k, "cpu", &a), "%s on the CPU", k_zoo[k]);
        CHECK_MSG(run_pass(k, "cuda", &b), "%s on the GPU: %s", k_zoo[k], nmz_last_error());
        d = rel(a.v, b.v, a.n < b.n ? a.n : b.n);
        CHECK_MSG(a.n == b.n && d < 1e-4 && fabs(a.loss - b.loss) < 1e-5 * (1 + fabs(a.loss)),
                  "%s: GPU vs CPU relative difference %.3g, loss %.9g vs %.9g", k_zoo[k], d, a.loss, b.loss);
        test_note("%-28s %6zu numbers, max rel diff %.1e", k_zoo[k], a.n, d);
        free(a.v);
        free(b.v);
    }
    { /* recurrent layers too wide for the sequence kernels: composed on the GPU */
        result a, b;
        double d;
        CHECK_MSG(run_pass(Z_WIDE, "cpu", &a), "wide recurrent layers on the CPU");
        CHECK_MSG(run_pass(Z_WIDE, "cuda", &b), "wide recurrent layers on the GPU: %s", nmz_last_error());
        d = rel(a.v, b.v, a.n < b.n ? a.n : b.n);
        CHECK_MSG(a.n == b.n && d < 1e-4 && fabs(a.loss - b.loss) < 1e-5 * (1 + fabs(a.loss)),
                  "wide lstm/gru/rnn: GPU vs CPU relative difference %.3g, loss %.9g vs %.9g", d, a.loss, b.loss);
        test_note("%-28s %6zu numbers, max rel diff %.1e", "wide lstm/gru/rnn (composed)", a.n, d);
        free(a.v);
        free(b.v);
    }
}

static void test_gradcheck_gpu(void) {
    int k, training;
    for (k = 0; k < Z_COUNT; k++)
        for (training = 0; training < 2; training++) {
            model m;
            nmz_gradcheck_result res;
            memset(&res, 0, sizeof res);
            nmz_backend_select("cuda");
            if (!make(k, &m)) {
                CHECK_MSG(0, "%s: %s", k_zoo[k], nmz_last_error());
                continue;
            }
            if (m.carry) {
                nmz_net_set_carry(m.net, 1);
                CHECK_OK(nmz_net_forward(m.net, &m.x, NULL));
            }
            CHECK_OK(nmz_net_gradcheck_mode(m.net, &m.x, &m.t, 1e-3, 3e-4, 3e-2, training, &res));
            CHECK_MSG(res.failures == 0 && res.checked > 0,
                      "%s (%s): %zu/%zu mismatches; worst layer %d %s[%zu]: %.6g vs %.6g", k_zoo[k],
                      training ? "training" : "evaluation", res.failures, res.checked, res.worst_layer, res.worst_param,
                      res.worst_index, res.worst_analytic, res.worst_numeric);
            if (training)
                test_note("%-28s %5zu grads, max |err| %.2e%s", k_zoo[k], res.checked, res.max_abs_err,
                          res.kinks ? " (kinks skipped)" : "");
            unmake(&m);
        }
    nmz_backend_select("cpu");
}

/* traced numbers on the GPU: the same events, the numbers within float rounding */
static void test_trace_gpu(void) {
    int k;
    for (k = 0; k < Z_COUNT; k++) {
        nmz_trace_recorder *rec[2] = {NULL, NULL};
        int pass;
        for (pass = 0; pass < 2; pass++) {
            model m;
            nmz_train_opts o = nmz_train_defaults();
            nmz_trace_recorder_create(&rec[pass]);
            nmz_backend_select(pass ? "cuda" : "cpu");
            if (!make(k, &m)) break;
            o.optimizer = NMZ_OPT_ADAM;
            nmz_net_set_trace(m.net, nmz_trace_recorder_fn, rec[pass]);
            CHECK_OK(nmz_net_forward_mode(m.net, &m.x, 0, NULL));
            CHECK_OK(nmz_net_backward(m.net, &m.t, NULL));
            CHECK_OK(nmz_net_clip_gradients(m.net, 0.5, NULL, NULL));
            CHECK_OK(nmz_net_optimizer_step(m.net, &o, 0.01f));
            unmake(&m);
        }
        {
            size_t i, n = nmz_trace_recorder_count(rec[0]), bad = 0;
            double worst = 0;
            CHECK_MSG(n > 0 && n == nmz_trace_recorder_count(rec[1]), "%s: %zu events on the CPU, %zu on the GPU",
                      k_zoo[k], n, nmz_trace_recorder_count(rec[1]));
            for (i = 0; i < n && n == nmz_trace_recorder_count(rec[1]); i++) {
                const nmz_trace_event *a = nmz_trace_recorder_get(rec[0], i), *b = nmz_trace_recorder_get(rec[1], i);
                int j;
                bad += a->n_items != b->n_items || strcmp(a->layer_type, b->layer_type) || a->time_step != b->time_step;
                for (j = 0; j < a->n_items && j < b->n_items; j++) {
                    const nmz_tensor *x = a->items[j].t, *y = b->items[j].t;
                    double d;
                    if (!x || !y || x->size != y->size || strcmp(a->items[j].name, b->items[j].name)) {
                        bad++;
                        continue;
                    }
                    /* Adam's Δ = m̂/(√v̂ + ε) is ±η wherever the gradient is
                     * rounding noise (the key bias of attention: exactly 0 in
                     * theory, ~1e-9 in practice), whatever its sign: skipped */
                    if (a->phase == NMZ_TRACE_UPDATE && !strcmp(a->items[j].name, "delta")) continue;
                    d = rel(x->data, y->data, x->size);
                    if (d > 1e-4)
                        test_note("event %zu (%s %s/%s, t %d): item %s differs by %.3g", i, nmz_trace_phase_name(a->phase),
                                  a->layer_type, a->detail, a->time_step, a->items[j].name, d);
                    if (d > worst) worst = d;
                }
            }
            CHECK_MSG(bad == 0 && worst < 1e-4, "%s: traces differ (%zu structural, max rel %.3g)", k_zoo[k], bad, worst);
        }
        nmz_trace_recorder_free(rec[0]);
        nmz_trace_recorder_free(rec[1]);
    }
    nmz_backend_select("cpu");
}

static unsigned long long file_hash(const char *path) {
    FILE *f = fopen(path, "rb");
    unsigned long long h = 14695981039346656037ull;
    int c;
    if (!f) return 0;
    while ((c = fgetc(f)) != EOF) h = (h ^ (unsigned char)c) * 1099511628211ull;
    fclose(f);
    return h;
}

/* a small text model trained on the GPU: twice the same bits; saved there,
 * loaded on the CPU it predicts the same; trained further on the CPU */
static nmz_dataset *names(int n) {
    nmz_dataset *ds = NULL;
    nmz_text_opts to = nmz_text_defaults();
    to.items = 1;
    to.seq_len = 16;
    if (nmz_dataset_load_text(&ds, "examples/names_en.txt", &to) != NMZ_OK) return NULL;
    if (n) nmz_dataset_truncate(ds, n);
    return ds;
}

static nmz_net *text_net(const nmz_dataset *ds, int kind) {
    nmz_net *net = NULL;
    int shape[1] = {NMZ_ANY_LENGTH}, V = ds->vocab->size;
    nmz_net_create(&net, 1, shape, 21);
    nmz_net_set_vocab(net, ds->vocab);
    nmz_net_add(net, nmz_layer_embedding(V, 16, NMZ_INIT_AUTO));
    if (kind == 0) {
        nmz_net_add(net, nmz_layer_lstm(24, NMZ_INIT_AUTO, 1.0f));
        nmz_net_add(net, nmz_layer_dropout(0.1f));
        nmz_net_add(net, nmz_layer_gru(16, NMZ_INIT_AUTO));
        nmz_net_add_dense(net, V, NMZ_ACT_SOFTMAX, NMZ_INIT_AUTO);
    } else {
        nmz_net_add(net, nmz_layer_posenc(NMZ_POS_LEARNED, 16));
        nmz_net_add(net, nmz_layer_block(4, 0, 0, NMZ_ACT_GELU, 0.1f));
        nmz_net_add(net, nmz_layer_block(4, 0, 0, NMZ_ACT_RELU, 0.1f));
        nmz_net_add(net, nmz_layer_layernorm());
        nmz_net_add(net, nmz_layer_tied_output(V));
        nmz_net_add(net, nmz_layer_activation(NMZ_ACT_SOFTMAX, 0));
    }
    nmz_net_set_loss(net, NMZ_LOSS_CROSS_ENTROPY);
    return net;
}

static nmz_train_opts text_opts(int kind) {
    nmz_train_opts o = nmz_train_defaults();
    o.optimizer = kind ? NMZ_OPT_ADAMW : NMZ_OPT_ADAM;
    o.weight_decay = 0.1f;
    o.learning_rate = 0.01f;
    o.batch_size = 16;
    o.max_epochs = 3;
    o.target_loss = 0;
    o.clip_norm = 1.0f;
    o.val_fraction = 0.2f;
    if (kind) o.schedule = NMZ_SCHED_WARMUP_COSINE, o.warmup_steps = 10;
    return o;
}

static void test_determinism_and_files(void) {
    nmz_dataset *ds = names(600);
    int kind;
    if (!ds) {
        CHECK_MSG(0, "examples/names_en.txt: %s", nmz_last_error());
        return;
    }
    for (kind = 0; kind < 2; kind++) {
        unsigned long long h[2] = {0, 0};
        nmz_train_result r[2];
        int pass;
        for (pass = 0; pass < 2; pass++) {
            nmz_net *net;
            nmz_train_opts o = text_opts(kind);
            nmz_backend_select("cuda");
            net = text_net(ds, kind);
            CHECK_OK(nmz_net_train(net, ds, &o, &r[pass]));
            CHECK_OK(nmz_net_save(net, test_path("gpu_model.nmz")));
            h[pass] = file_hash(test_path("gpu_model.nmz"));
            nmz_net_free(net);
        }
        CHECK_MSG(h[0] && h[0] == h[1], "%s: two GPU runs differ (val %.9g vs %.9g)", kind ? "transformer" : "lstm+gru",
                  r[0].val_loss, r[1].val_loss);
        { /* the file from the GPU, on the CPU: the same predictions */
            nmz_net *net = NULL;
            const nmz_tensor *y;
            float *a;
            size_t n;
            int shape[2] = {2, 10}, i;
            nmz_tensor x = NMZ_TENSOR_EMPTY;
            nmz_tensor_init(&x, 2, shape);
            for (i = 0; i < 20; i++) x.data[i] = ds->x[i];
            nmz_backend_select("cpu");
            CHECK_OK(nmz_net_load(&net, test_path("gpu_model.nmz")));
            CHECK_OK(nmz_net_forward(net, &x, &y));
            n = y->size;
            a = hcopy(y->data, n);
            nmz_backend_select("cuda");
            CHECK_OK(nmz_net_forward(net, &x, &y));
            CHECK_MSG(rel(a, y->data, n) < 1e-5, "%s: CPU and GPU predictions of a loaded model differ by %.3g",
                      kind ? "transformer" : "lstm+gru", rel(a, y->data, n));
            { /* and training continues on the CPU, from the GPU's optimizer state */
                nmz_train_opts o = text_opts(kind);
                nmz_train_result tr;
                o.max_epochs = 1;
                nmz_backend_select("cpu");
                CHECK_OK(nmz_net_train(net, ds, &o, &tr));
                CHECK(tr.val_loss < r[0].val_loss + 0.05);
            }
            free(a);
            nmz_tensor_free(&x);
            nmz_net_free(net);
        }
    }
    nmz_dataset_free(ds);
    nmz_backend_select("cpu");
}

/* generation on the GPU: the KV cache gives the window's logits again */
static void test_kv_cache_gpu(void) {
    nmz_net *net = NULL;
    nmz_tensor x = NMZ_TENSOR_EMPTY, x1 = NMZ_TENSOR_EMPTY;
    const nmz_tensor *y;
    int shape[1] = {NMZ_ANY_LENGTH}, V = 12, s2[2], i;
    float *full, *step;
    nmz_rng r;
    nmz_backend_select("cuda");
    nmz_rng_seed(&r, 4, 4);
    nmz_net_create(&net, 1, shape, 5);
    nmz_net_add(net, nmz_layer_embedding(V, 16, NMZ_INIT_AUTO));
    nmz_net_add(net, nmz_layer_posenc(NMZ_POS_LEARNED, 20));
    nmz_net_add(net, nmz_layer_block(4, 0, 0, NMZ_ACT_GELU, 0));
    nmz_net_add(net, nmz_layer_block(2, 0, 0, NMZ_ACT_GELU, 0));
    nmz_net_add(net, nmz_layer_tied_output(V));
    nmz_net_add(net, nmz_layer_activation(NMZ_ACT_SOFTMAX, 0));
    s2[0] = 3, s2[1] = 9;
    nmz_tensor_init(&x, 2, s2);
    for (i = 0; i < 27; i++) x.data[i] = (float)nmz_rng_below(&r, (uint32_t)V);
    CHECK_OK(nmz_net_forward(net, &x, &y)); /* the whole window */
    full = hcopy(y->data, y->size);
    nmz_net_set_carry(net, 1); /* 8 positions, then the 9th from the cache */
    s2[1] = 8;
    {
        nmz_tensor x8 = NMZ_TENSOR_EMPTY;
        nmz_tensor_init(&x8, 2, s2);
        for (i = 0; i < 3; i++) memcpy(x8.data + i * 8, x.data + i * 9, 8 * sizeof(float));
        CHECK_OK(nmz_net_forward(net, &x8, NULL));
        nmz_tensor_free(&x8);
    }
    s2[1] = 1;
    nmz_tensor_init(&x1, 2, s2);
    for (i = 0; i < 3; i++) x1.data[i] = x.data[i * 9 + 8];
    CHECK_OK(nmz_net_forward(net, &x1, &y));
    step = hcopy(y->data, y->size);
    for (i = 0; i < 3; i++) {
        double d = rel(full + ((size_t)i * 9 + 8) * V, step + (size_t)i * V, (size_t)V);
        CHECK_MSG(d < 1e-5, "row %d: cached and recomputed probabilities differ by %.3g", i, d);
    }
    free(full);
    free(step);
    nmz_tensor_free(&x);
    nmz_tensor_free(&x1);
    nmz_net_free(net);
    nmz_backend_select("cpu");
}

static void test_info_and_oom(void) {
    nmz_device_info di;
    nmz_net *net = NULL;
    nmz_dataset *ds = NULL;
    nmz_train_opts o = nmz_train_defaults();
    nmz_train_result r;
    int shape[1] = {1024}, i;
    float *x, *y;
    void *big;
    nmz_status s;
    if (!backends()) return;
    CHECK_OK(nmz_backend_select("auto"));
    CHECK_OK(nmz_backend_info(&di));
    CHECK(strcmp(di.backend, "cuda") == 0 && di.device[0] && di.cc_major >= 7 && di.mem_total > ((size_t)1 << 30));
    test_note("%s: %s, compute %d.%d, %.0f MiB, %.0f MiB free, %.1f MiB held by the library", di.backend, di.device,
              di.cc_major, di.cc_minor, di.mem_total / 1048576.0, di.mem_free / 1048576.0, di.mem_used / 1048576.0);
    big = gpu->alloc(gpu, (size_t)64 << 30); /* 64 GB */
    CHECK(big == NULL);
    CHECK(strstr(nmz_last_error(), "smaller batch") != NULL);
    test_note("%s", nmz_last_error());
    /* fill the device but for 48 MiB, then train with a batch that needs more */
    x = malloc(sizeof(float) * 4096 * 1024);
    y = calloc(4096 * 10, sizeof(float));
    for (i = 0; i < 4096 * 1024; i++) x[i] = (float)(i % 13) * 0.1f;
    for (i = 0; i < 4096; i++) y[i * 10 + i % 10] = 1.0f;
    CHECK_OK(nmz_dataset_create(&ds, 4096, 1024, 10, x, y));
    CHECK_OK(nmz_net_create(&net, 1, shape, 1));
    CHECK_OK(nmz_net_add_dense(net, 4096, NMZ_ACT_RELU, NMZ_INIT_AUTO));
    CHECK_OK(nmz_net_add_dense(net, 10, NMZ_ACT_SOFTMAX, NMZ_INIT_AUTO));
    CHECK_OK(nmz_net_set_loss(net, NMZ_LOSS_CROSS_ENTROPY));
    o.optimizer = NMZ_OPT_ADAM;
    o.learning_rate = 0.001f;
    o.max_epochs = 1;
    o.batch_size = 4096;
    /* the library may use what is free (WDDM would page beyond it); 48 MiB more
     * than it holds now is less than a batch of 4096 needs */
    test_note("memory limit %.0f MiB (free when CUDA started)", di.mem_limit / 1048576.0);
    CHECK(di.mem_limit > 0 && di.mem_limit <= di.mem_total);
    CHECK_OK(nmz_backend_memory_limit(di.mem_used + ((size_t)48 << 20)));
    s = nmz_net_train(net, ds, &o, &r);
    CHECK_STATUS(s, NMZ_ERR_NOMEM);
    CHECK(strstr(nmz_last_error(), "smaller batch") != NULL);
    test_note("training with a batch of 4096: %s", nmz_last_error());
    CHECK_OK(nmz_backend_memory_limit(0));
    CHECK_OK(nmz_net_train(net, ds, &o, &r)); /* room again: the same net trains */
    nmz_net_free(net);
    nmz_dataset_free(ds);
    free(x);
    free(y);
    CHECK_OK(nmz_backend_info(&di));
    test_note("afterwards: %.1f MiB held by the library (at most %.1f MiB)", di.mem_used / 1048576.0,
              di.mem_peak / 1048576.0);
    nmz_backend_select("cpu");
}

const test_case gpu_tests[] = {{"gpu/ops_elementwise", test_ops_elementwise},
                               {"gpu/ops_gemm", test_ops_gemm},
                               {"gpu/ops_reductions", test_ops_reductions},
                               {"gpu/ops_activations_losses", test_ops_activations},
                               {"gpu/ops_conv_pool", test_ops_conv_pool},
                               {"gpu/ops_dropout_embed_posenc", test_ops_lookups},
                               {"gpu/ops_layernorm_attention", test_ops_layernorm_attention},
                               {"gpu/ops_recurrent_cells", test_ops_cells},
                               {"gpu/ops_optimizers_loop", test_ops_optim_loop},
                               {"gpu/layers_match_cpu", test_layers_match_cpu},
                               {"gpu/gradcheck_every_layer", test_gradcheck_gpu},
                               {"gpu/trace", test_trace_gpu},
                               {"gpu/determinism_and_files", test_determinism_and_files},
                               {"gpu/kv_cache", test_kv_cache_gpu},
                               {"gpu/info_and_out_of_memory", test_info_and_oom},
                               {NULL, NULL}};

/* --------------------------------------------------------------- parity */

static double now(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}

typedef nmz_status (*build_fn)(nmz_net **net, const nmz_dataset *ds);

/* train the same net on both devices; returns the two results */
static int both(const char *what, build_fn build, const nmz_dataset *ds, const nmz_dataset *test,
                const nmz_train_opts *o, double out[2][2]) {
    int d;
    for (d = 0; d < 2; d++) {
        nmz_net *net = NULL;
        nmz_train_result r;
        double t0 = now(), loss = 0, acc = 0;
        nmz_backend_select(d ? "cuda" : "cpu");
        if (build(&net, ds) != NMZ_OK || nmz_net_train(net, ds, o, &r) != NMZ_OK) {
            CHECK_MSG(0, "%s on %s: %s", what, d ? "the GPU" : "the CPU", nmz_last_error());
            nmz_net_free(net);
            return 0;
        }
        if (test)
            CHECK_OK(nmz_net_evaluate(net, test, &loss, &acc));
        else
            loss = o->val_fraction > 0 ? r.val_loss : r.loss, acc = o->val_fraction > 0 ? r.val_accuracy : r.accuracy;
        out[d][0] = loss;
        out[d][1] = acc;
        test_note("%-5s %-14s %3d epochs %7.2f s   loss %.4f   accuracy %.2f%%", d ? "GPU" : "CPU", what, r.epochs,
                  now() - t0, loss, 100.0 * acc);
        nmz_net_free(net);
    }
    nmz_backend_select("cpu");
    return 1;
}

static nmz_status b_xor(nmz_net **net, const nmz_dataset *ds) {
    int sizes[3] = {2, 4, 1};
    (void)ds;
    return nmz_net_create_quick(net, 3, sizes, 42, NULL);
}

static void test_parity_xor(void) {
    nmz_dataset *ds = NULL;
    nmz_csv_opts co = nmz_csv_defaults();
    nmz_quick q;
    int sizes[3] = {2, 4, 1};
    double v[2][2];
    CHECK_OK(nmz_dataset_load_csv(&ds, "examples/xor.csv", &co));
    CHECK_OK(nmz_quick_defaults(3, sizes, &q));
    if (ds && both("xor (quick)", b_xor, ds, NULL, &q.train, v))
        CHECK_MSG(v[0][0] <= q.train.target_loss && v[1][0] <= q.train.target_loss && v[1][1] == 1.0,
                  "XOR: loss %.5f (CPU) %.5f (GPU)", v[0][0], v[1][0]);
    nmz_dataset_free(ds);
}

static nmz_status b_iris(nmz_net **net, const nmz_dataset *ds) {
    int shape[1] = {4};
    nmz_status s;
    (void)ds;
    if ((s = nmz_net_create(net, 1, shape, 7)) != NMZ_OK) return s;
    nmz_net_add_dense(*net, 16, NMZ_ACT_RELU, NMZ_INIT_AUTO);
    nmz_net_add_dense(*net, 3, NMZ_ACT_SOFTMAX, NMZ_INIT_AUTO);
    nmz_net_set_loss(*net, NMZ_LOSS_CROSS_ENTROPY);
    return nmz_net_set_normalization(*net, NMZ_NORM_ZSCORE);
}

static void test_parity_iris(void) {
    nmz_dataset *tr = NULL, *te = NULL;
    nmz_csv_opts co = nmz_csv_defaults();
    nmz_train_opts o = nmz_train_defaults();
    double v[2][2];
    CHECK_OK(nmz_dataset_load_csv(&tr, "examples/iris_train.csv", &co));
    CHECK_OK(nmz_dataset_load_csv(&te, "examples/iris_test.csv", &co));
    o.optimizer = NMZ_OPT_ADAM;
    o.learning_rate = 0.01f;
    o.max_epochs = 100;
    o.batch_size = 16;
    o.target_loss = 0;
    if (tr && te && both("iris", b_iris, tr, te, &o, v))
        CHECK_MSG(fabs(v[0][1] - v[1][1]) <= 0.034 && v[1][1] >= 0.9 && fabs(v[0][0] - v[1][0]) < 0.05,
                  "Iris test accuracy %.3f (CPU) vs %.3f (GPU), loss %.4f vs %.4f", v[0][1], v[1][1], v[0][0], v[1][0]);
    nmz_dataset_free(tr);
    nmz_dataset_free(te);
}

static nmz_status b_cnn(nmz_net **net, const nmz_dataset *ds) {
    int shape[3] = {1, 28, 28};
    nmz_status s;
    (void)ds;
    if ((s = nmz_net_create(net, 3, shape, 1)) != NMZ_OK) return s;
    nmz_net_add(*net, nmz_layer_conv2d(8, 3, 3, 1, 0, NMZ_INIT_AUTO));
    nmz_net_add(*net, nmz_layer_activation(NMZ_ACT_RELU, 0));
    nmz_net_add(*net, nmz_layer_maxpool2d(2, 2));
    nmz_net_add(*net, nmz_layer_conv2d(16, 3, 3, 1, 0, NMZ_INIT_AUTO));
    nmz_net_add(*net, nmz_layer_activation(NMZ_ACT_RELU, 0));
    nmz_net_add(*net, nmz_layer_maxpool2d(2, 2));
    nmz_net_add(*net, nmz_layer_flatten());
    nmz_net_add_dense(*net, 64, NMZ_ACT_RELU, NMZ_INIT_AUTO);
    nmz_net_add_dense(*net, 10, NMZ_ACT_SOFTMAX, NMZ_INIT_AUTO);
    nmz_net_set_loss(*net, NMZ_LOSS_CROSS_ENTROPY);
    return nmz_net_set_normalization(*net, NMZ_NORM_ZSCORE);
}

static void test_parity_mnist(void) {
    nmz_dataset *tr = NULL, *te = NULL;
    nmz_train_opts o = nmz_train_defaults();
    double v[2][2];
    FILE *f = fopen("data/train-images-idx3-ubyte", "rb");
    if (!f) {
        test_note("MNIST not downloaded (cmake --build build --target data): skipped");
        return;
    }
    fclose(f);
    CHECK_OK(nmz_dataset_load_idx(&tr, "data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte", 20000));
    CHECK_OK(nmz_dataset_load_idx(&te, "data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte", 0));
    o.optimizer = NMZ_OPT_ADAM;
    o.learning_rate = 0.001f;
    o.batch_size = 32;
    o.max_epochs = 1;
    o.target_loss = 0;
    if (tr && te && both("mnist cnn", b_cnn, tr, te, &o, v))
        CHECK_MSG(v[1][1] >= 0.95 && fabs(v[0][1] - v[1][1]) <= 0.01,
                  "MNIST test accuracy %.4f (CPU) vs %.4f (GPU)", v[0][1], v[1][1]);
    nmz_dataset_free(tr);
    nmz_dataset_free(te);
}

static int g_text_kind;
static nmz_status b_text(nmz_net **net, const nmz_dataset *ds) {
    int shape[1] = {NMZ_ANY_LENGTH}, V = ds->vocab->size, b;
    nmz_status s;
    if ((s = nmz_net_create(net, 1, shape, 42)) != NMZ_OK) return s;
    nmz_net_set_vocab(*net, ds->vocab);
    if (g_text_kind == 0) { /* the names preset: embed 16, rnn 128 */
        nmz_net_add(*net, nmz_layer_embedding(V, 16, NMZ_INIT_AUTO));
        nmz_net_add(*net, nmz_layer_rnn(128, NMZ_INIT_XAVIER));
        nmz_net_add_dense(*net, V, NMZ_ACT_SOFTMAX, NMZ_INIT_AUTO);
    } else if (g_text_kind == 1) { /* char-lstm */
        nmz_net_add(*net, nmz_layer_embedding(V, 32, NMZ_INIT_AUTO));
        nmz_net_add(*net, nmz_layer_lstm(64, NMZ_INIT_AUTO, 1.0f));
        nmz_net_add_dense(*net, V, NMZ_ACT_SOFTMAX, NMZ_INIT_AUTO);
    } else { /* char-gpt */
        nmz_net_add(*net, nmz_layer_embedding(V, 32, NMZ_INIT_AUTO));
        nmz_net_add(*net, nmz_layer_posenc(NMZ_POS_LEARNED, 128));
        for (b = 0; b < 2; b++) nmz_net_add(*net, nmz_layer_block(4, 0, 0, NMZ_ACT_GELU, 0));
        nmz_net_add(*net, nmz_layer_layernorm());
        nmz_net_add(*net, nmz_layer_tied_output(V));
        nmz_net_add(*net, nmz_layer_activation(NMZ_ACT_SOFTMAX, 0));
    }
    return nmz_net_set_loss(*net, NMZ_LOSS_CROSS_ENTROPY);
}

static void parity_text(int kind) {
    static const char *names_[3] = {"names rnn", "alice lstm", "alice gpt"};
    nmz_dataset *ds = NULL;
    nmz_text_opts to = nmz_text_defaults();
    nmz_train_opts o = nmz_train_defaults();
    double v[2][2], tol = kind == 0 ? 0.05 : 0.04;
    const char *file = kind == 0 ? "examples/names_en.txt" : "data/alice.txt";
    FILE *f = fopen(file, "rb");
    if (!f) {
        test_note("%s not found (cmake --build build --target data-text): skipped", file);
        return;
    }
    fclose(f);
    g_text_kind = kind;
    to.items = kind == 0;
    to.seq_len = kind == 0 ? 16 : kind == 1 ? 64 : 128;
    CHECK_OK(nmz_dataset_load_text(&ds, file, &to));
    o.optimizer = kind == 2 ? NMZ_OPT_ADAMW : NMZ_OPT_ADAM;
    o.weight_decay = 0.1f;
    o.learning_rate = kind == 0 ? 0.003f : 0.01f;
    o.batch_size = kind == 0 ? 32 : kind == 1 ? 16 : 8;
    o.max_epochs = kind == 0 ? 8 : 3;
    o.target_loss = 0;
    o.val_fraction = 0.1f;
    o.clip_norm = kind == 2 ? 1.0f : 5.0f;
    o.carry_state = kind == 1;
    o.shuffle = kind != 1;
    if (kind == 2) o.schedule = NMZ_SCHED_WARMUP_COSINE, o.warmup_steps = 20, o.min_lr = 0.001f;
    if (ds && both(names_[kind], b_text, ds, NULL, &o, v))
        CHECK_MSG(fabs(v[0][0] - v[1][0]) <= tol, "%s: validation loss %.4f (CPU) vs %.4f (GPU)", names_[kind], v[0][0],
                  v[1][0]);
    nmz_dataset_free(ds);
}

static void test_parity_names(void) { parity_text(0); }
static void test_parity_alice_lstm(void) { parity_text(1); }
static void test_parity_alice_gpt(void) { parity_text(2); }

const test_case gpu_parity_tests[] = {{"parity/xor", test_parity_xor},
                                      {"parity/iris", test_parity_iris},
                                      {"parity/mnist_cnn", test_parity_mnist},
                                      {"parity/names_rnn", test_parity_names},
                                      {"parity/alice_lstm", test_parity_alice_lstm},
                                      {"parity/alice_transformer", test_parity_alice_gpt},
                                      {NULL, NULL}};
