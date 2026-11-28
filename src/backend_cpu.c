/* backend_cpu.c — the CPU backend, the reference every other backend is
 * compared with (DESIGN §11.2).
 *
 * Its ops are the reference loops (the library's CPU code from before the
 * backends, decision 128), moved here or left next to their layer (the
 * layer kernels nmz__cpu_*), so a net on the CPU computes bit for bit what
 * that code computed. Two tables share them:
 *
 *   "cpu"          the ops plus the fused layer kernels (dense, conv2d,
 *                  rnn, lstm, gru): the reference arithmetic, the default;
 *   "cpu-generic"  the same ops without fused kernels, so the layers
 *                  compose their math from gemm and the cell ops exactly
 *                  as on a GPU. It runs the device code path on the CPU,
 *                  where CI can test it, and shows how the work is split.
 *
 * The ops that only the generic path uses (im2col in batch layout, the
 * recurrent cell steps, strided gemm, the training-loop gathers) are new
 * and written as plainly as the old ones. */
#include "internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ------------------------------------------------------------- lifecycle */

static nmz_status cpu_init(const nmz_backend *be, char *why, size_t n) {
    (void)be;
    if (why && n) why[0] = '\0';
    return NMZ_OK;
}

static void cpu_info(const nmz_backend *be, nmz_device_info *out) {
    out->threads = nmz_threads();
    snprintf(out->device, sizeof out->device, "host CPU (%d thread%s%s)", out->threads, out->threads == 1 ? "" : "s",
             be == &nmz__cpu_generic_backend ? ", generic kernels" : "");
}

static nmz_status cpu_sync(const nmz_backend *be) {
    (void)be;
    return NMZ_OK;
}

/* ------------------------------------------------------ element-wise ops */

static nmz_status cpu_fill(const nmz_backend *be, size_t n, float v, float *x) {
    size_t i;
    (void)be;
    for (i = 0; i < n; i++) x[i] = v;
    return NMZ_OK;
}

static nmz_status cpu_copy(const nmz_backend *be, size_t rows, size_t cols, const float *src, size_t lds, float *dst,
                           size_t ldd) {
    size_t r;
    (void)be;
    if (rows == 1 || (lds == cols && ldd == cols)) {
        if (rows && cols) memmove(dst, src, rows * cols * sizeof(float));
        return NMZ_OK;
    }
    for (r = 0; r < rows; r++) memmove(dst + r * ldd, src + r * lds, cols * sizeof(float));
    return NMZ_OK;
}

static nmz_status cpu_add(const nmz_backend *be, size_t n, const float *a, const float *b, float *out) {
    size_t i;
    (void)be;
    for (i = 0; i < n; i++) out[i] = a[i] + b[i];
    return NMZ_OK;
}

static nmz_status cpu_mul(const nmz_backend *be, size_t n, const float *a, const float *b, float *out) {
    size_t i;
    (void)be;
    for (i = 0; i < n; i++) out[i] = a[i] * b[i];
    return NMZ_OK;
}

/* ------------------------------------------------------------ products */

/* contiguous operands go to nmz_matmul (blocked, the reference products);
 * strided ones to the same sums as three loops — nmz_matmul is
 * bit-identical to those (decision 92), so both give the same bits */
static nmz_status cpu_gemm(const nmz_backend *be, unsigned flags, int M, int N, int K, const float *A, int lda,
                           const float *B, int ldb, float *C, int ldc) {
    int ta = (flags & NMZ_MM_TRANS_A) != 0, tb = (flags & NMZ_MM_TRANS_B) != 0, i, j, k;
    (void)be;
    if (lda == (ta ? M : K) && ldb == (tb ? K : N) && ldc == N) return nmz_matmul(M, N, K, A, B, C, flags);
    if (ta && tb) return nmz__fail(NMZ_ERR_UNSUPPORTED, "gemm: A and B cannot both be transposed");
    for (i = 0; i < M; i++)
        for (j = 0; j < N; j++) {
            float c = (flags & NMZ_MM_ACCUMULATE) ? C[(size_t)i * ldc + j] : 0.0f;
            for (k = 0; k < K; k++) {
                float a = ta ? A[(size_t)k * lda + i] : A[(size_t)i * lda + k];
                float b = tb ? B[(size_t)j * ldb + k] : B[(size_t)k * ldb + j];
                c += a * b;
            }
            C[(size_t)i * ldc + j] = c;
        }
    return NMZ_OK;
}

/* the reference add_rows (attention.c) */
static nmz_status cpu_add_bias(const nmz_backend *be, size_t rows, int cols, float *Y, const float *bias) {
    size_t r;
    int j;
    (void)be;
    for (r = 0; r < rows; r++) {
        float *y = Y + r * (size_t)cols;
        for (j = 0; j < cols; j++) y[j] += bias[j];
    }
    return NMZ_OK;
}

static nmz_status cpu_add_bias3(const nmz_backend *be, int A, int B, int C, float *y, const float *bias) {
    int a, b, c;
    (void)be;
    for (a = 0; a < A; a++)
        for (b = 0; b < B; b++) {
            float *row = y + ((size_t)a * B + b) * C;
            for (c = 0; c < C; c++) row[c] += bias[b];
        }
    return NMZ_OK;
}

/* the reference colsum_add (attention.c): sums in double */
static nmz_status cpu_colsum_add(const nmz_backend *be, size_t rows, int cols, const float *A, float *out) {
    double stack[256], *acc = cols <= 256 ? stack : calloc((size_t)cols, sizeof *acc);
    size_t r;
    int j;
    (void)be;
    if (!acc) return nmz__fail(NMZ_ERR_NOMEM, "out of memory");
    if (acc == stack)
        for (j = 0; j < cols; j++) acc[j] = 0.0;
    for (r = 0; r < rows; r++) {
        const float *ar = A + r * (size_t)cols;
        for (j = 0; j < cols; j++) acc[j] += ar[j];
    }
    for (j = 0; j < cols; j++) out[j] += (float)acc[j];
    if (acc != stack) free(acc);
    return NMZ_OK;
}

static nmz_status cpu_rowsum_add(const nmz_backend *be, int rows, size_t cols, const float *A, float *out) {
    int r;
    size_t c;
    (void)be;
    for (r = 0; r < rows; r++) {
        const float *ar = A + (size_t)r * cols;
        double s = 0.0;
        for (c = 0; c < cols; c++) s += ar[c];
        out[r] += (float)s;
    }
    return NMZ_OK;
}

static nmz_status cpu_transpose3(const nmz_backend *be, int A, int B, int C, const float *in, float *out) {
    int a, b;
    (void)be;
    for (a = 0; a < A; a++)
        for (b = 0; b < B; b++)
            memcpy(out + ((size_t)b * A + a) * C, in + ((size_t)a * B + b) * C, (size_t)C * sizeof(float));
    return NMZ_OK;
}

/* ------------------------------------------------- convolution (generic) */

static nmz_status cpu_im2col(const nmz_backend *be, const nmz_conv_shape *s, const float *x, float *col) {
    const size_t P = (size_t)s->OH * s->OW, N = (size_t)s->B * P;
    int ch, ky, kx, b, oy, ox;
    size_t k = 0;
    (void)be;
    for (ch = 0; ch < s->C; ch++)
        for (ky = 0; ky < s->kh; ky++)
            for (kx = 0; kx < s->kw; kx++, k++) {
                float *row = col + k * N;
                for (b = 0; b < s->B; b++) {
                    const float *xc = x + ((size_t)b * s->C + ch) * s->H * s->W;
                    float *rb = row + (size_t)b * P;
                    for (oy = 0; oy < s->OH; oy++) {
                        int iy = oy * s->stride + ky - s->pad;
                        for (ox = 0; ox < s->OW; ox++) {
                            int ix = ox * s->stride + kx - s->pad;
                            rb[oy * s->OW + ox] = (iy < 0 || iy >= s->H || ix < 0 || ix >= s->W) ? 0.0f : xc[iy * s->W + ix];
                        }
                    }
                }
            }
    return NMZ_OK;
}

static nmz_status cpu_col2im(const nmz_backend *be, const nmz_conv_shape *s, const float *col, float *dx) {
    const size_t P = (size_t)s->OH * s->OW, N = (size_t)s->B * P;
    int ch, ky, kx, b, oy, ox;
    size_t k = 0;
    (void)be;
    memset(dx, 0, (size_t)s->B * s->C * s->H * s->W * sizeof(float));
    for (ch = 0; ch < s->C; ch++)
        for (ky = 0; ky < s->kh; ky++)
            for (kx = 0; kx < s->kw; kx++, k++) {
                const float *row = col + k * N;
                for (b = 0; b < s->B; b++) {
                    float *dxc = dx + ((size_t)b * s->C + ch) * s->H * s->W;
                    const float *rb = row + (size_t)b * P;
                    for (oy = 0; oy < s->OH; oy++) {
                        int iy = oy * s->stride + ky - s->pad;
                        if (iy < 0 || iy >= s->H) continue;
                        for (ox = 0; ox < s->OW; ox++) {
                            int ix = ox * s->stride + kx - s->pad;
                            if (ix >= 0 && ix < s->W) dxc[iy * s->W + ix] += rb[oy * s->OW + ox];
                        }
                    }
                }
            }
    return NMZ_OK;
}

/* ---------------------------------------------------------------- dropout */

/* the reference masks: one draw of the net's PCG32 per element */
static nmz_status cpu_dropout_mask(const nmz_backend *be, size_t n, float rate, nmz_rng *rng, float *mask) {
    const float keep = 1.0f / (1.0f - rate);
    size_t i;
    (void)be;
    for (i = 0; i < n; i++) mask[i] = nmz_rng_uniform(rng) < rate ? 0.0f : keep;
    return NMZ_OK;
}

/* ------------------------------------------------- recurrent cells (generic) */

static float sigm(double a) { return (float)(1.0 / (1.0 + exp(-a))); }
#define AT(r, b, j) ((r).p[(size_t)(b) * (r).ld + (j)])
#define REC(field, b, j, H)                                                                                          \
    if (rec && rec->field) rec->field[((size_t)(b) * rec->T + t) * (H) + (j)]

static nmz_status cpu_rnn_cell_forward(const nmz_backend *be, int B, int H, nmz_rows z, nmz_rows h) {
    int b, j;
    (void)be;
    for (b = 0; b < B; b++)
        for (j = 0; j < H; j++) AT(h, b, j) = (float)tanh(AT(z, b, j));
    return NMZ_OK;
}

static nmz_status cpu_rnn_cell_backward(const nmz_backend *be, int B, int H, nmz_rows h, nmz_rows dy,
                                        nmz_rows dh_next, nmz_rows dz, const nmz_seq_record *rec, int t) {
    int b, j;
    (void)be;
    for (b = 0; b < B; b++)
        for (j = 0; j < H; j++) {
            float hv = AT(h, b, j), dn = AT(dh_next, b, j), dh = AT(dy, b, j) + dn;
            AT(dz, b, j) = dh * (1.0f - hv * hv);
            REC(dh_next, b, j, H) = dn;
            REC(dh, b, j, H) = dh;
        }
    return NMZ_OK;
}

static nmz_status cpu_lstm_cell_forward(const nmz_backend *be, int B, int H, nmz_rows a, nmz_rows c_prev, nmz_rows c,
                                        nmz_rows h) {
    int b, j;
    (void)be;
    for (b = 0; b < B; b++)
        for (j = 0; j < H; j++) {
            float ig = sigm(AT(a, b, j)), fg = sigm(AT(a, b, H + j));
            float gg = (float)tanh(AT(a, b, 2 * H + j)), og = sigm(AT(a, b, 3 * H + j));
            float cv = fg * AT(c_prev, b, j) + ig * gg;
            AT(a, b, j) = ig;
            AT(a, b, H + j) = fg;
            AT(a, b, 2 * H + j) = gg;
            AT(a, b, 3 * H + j) = og;
            AT(c, b, j) = cv;
            AT(h, b, j) = og * tanhf(cv);
        }
    return NMZ_OK;
}

static nmz_status cpu_lstm_cell_backward(const nmz_backend *be, int B, int H, nmz_rows gates, nmz_rows c,
                                         nmz_rows c_prev, nmz_rows dy, nmz_rows dh_next, nmz_rows dc_next,
                                         nmz_rows da, const nmz_seq_record *rec, int t) {
    int b, j;
    (void)be;
    for (b = 0; b < B; b++)
        for (j = 0; j < H; j++) {
            float ig = AT(gates, b, j), fg = AT(gates, b, H + j), gg = AT(gates, b, 2 * H + j),
                  og = AT(gates, b, 3 * H + j);
            float tc = tanhf(AT(c, b, j)), dhn = AT(dh_next, b, j), dcn = AT(dc_next, b, j);
            float dh = AT(dy, b, j) + dhn;
            float dc = dcn + dh * og * (1.0f - tc * tc);
            AT(da, b, j) = dc * gg * ig * (1.0f - ig);
            AT(da, b, H + j) = dc * AT(c_prev, b, j) * fg * (1.0f - fg);
            AT(da, b, 2 * H + j) = dc * ig * (1.0f - gg * gg);
            AT(da, b, 3 * H + j) = dh * tc * og * (1.0f - og);
            REC(dh_next, b, j, H) = dhn;
            REC(dh, b, j, H) = dh;
            REC(dc_next, b, j, H) = dcn;
            REC(dc, b, j, H) = dc;
            REC(dc_prev, b, j, H) = dc * fg;
            AT(dc_next, b, j) = dc * fg;
        }
    return NMZ_OK;
}

static nmz_status cpu_gru_cell_forward1(const nmz_backend *be, int B, int H, nmz_rows a, nmz_rows h_prev,
                                        nmz_rows rh) {
    int b, j;
    (void)be;
    for (b = 0; b < B; b++)
        for (j = 0; j < H; j++) {
            float r;
            AT(a, b, j) = sigm(AT(a, b, j));
            r = AT(a, b, H + j) = sigm(AT(a, b, H + j));
            AT(rh, b, j) = r * AT(h_prev, b, j);
        }
    return NMZ_OK;
}

static nmz_status cpu_gru_cell_forward2(const nmz_backend *be, int B, int H, nmz_rows a, nmz_rows h_prev,
                                        nmz_rows h) {
    int b, j;
    (void)be;
    for (b = 0; b < B; b++)
        for (j = 0; j < H; j++) {
            float z = AT(a, b, j), n = (float)tanh(AT(a, b, 2 * H + j));
            AT(a, b, 2 * H + j) = n;
            AT(h, b, j) = z * AT(h_prev, b, j) + (1.0f - z) * n;
        }
    return NMZ_OK;
}

static nmz_status cpu_gru_cell_backward1(const nmz_backend *be, int B, int H, nmz_rows gates, nmz_rows h_prev,
                                         nmz_rows dy, nmz_rows dh_next, nmz_rows da, nmz_rows direct, nmz_rows rh,
                                         const nmz_seq_record *rec, int t) {
    int b, j;
    (void)be;
    for (b = 0; b < B; b++)
        for (j = 0; j < H; j++) {
            float z = AT(gates, b, j), r = AT(gates, b, H + j), n = AT(gates, b, 2 * H + j);
            float hp = AT(h_prev, b, j), dn = AT(dh_next, b, j), dh = AT(dy, b, j) + dn;
            AT(da, b, j) = dh * (hp - n) * z * (1.0f - z);
            AT(da, b, 2 * H + j) = dh * (1.0f - z) * (1.0f - n * n);
            AT(direct, b, j) = dh * z;
            AT(rh, b, j) = r * hp;
            REC(dh_next, b, j, H) = dn;
            REC(dh, b, j, H) = dh;
            REC(direct, b, j, H) = dh * z;
        }
    return NMZ_OK;
}

static nmz_status cpu_gru_cell_backward2(const nmz_backend *be, int B, int H, nmz_rows gates, nmz_rows h_prev,
                                         nmz_rows drh, nmz_rows direct, nmz_rows da, nmz_rows dh_next) {
    int b, j;
    (void)be;
    for (b = 0; b < B; b++)
        for (j = 0; j < H; j++) {
            float r = AT(gates, b, H + j), d = AT(drh, b, j);
            AT(da, b, H + j) = d * AT(h_prev, b, j) * r * (1.0f - r);
            AT(dh_next, b, j) = AT(direct, b, j) + d * r;
        }
    return NMZ_OK;
}

/* ------------------------------------------------------------- optimizers
 * the reference update loops of net.c */

static nmz_status cpu_sgd_velocity(const nmz_backend *be, size_t n, float lr, float mu, int fresh, const float *g,
                                   const float *v_prev, float *v_out) {
    size_t e;
    (void)be;
    for (e = 0; e < n; e++) v_out[e] = (fresh ? 0.0f : mu * v_prev[e]) - lr * g[e];
    return NMZ_OK;
}

static nmz_status cpu_adam_update(const nmz_backend *be, size_t n, double lr, double b1, double b2, double eps,
                                  double bc1, double bc2, double decay, float *w, const float *g, float *m, float *v) {
    size_t e;
    (void)be;
    for (e = 0; e < n; e++) {
        double gi = g[e];
        double mi = b1 * m[e] + (1.0 - b1) * gi, vi = b2 * v[e] + (1.0 - b2) * gi * gi;
        m[e] = (float)mi;
        v[e] = (float)vi;
        w[e] += (float)(-lr * ((mi / bc1) / (sqrt(vi / bc2) + eps) + decay * w[e]));
    }
    return NMZ_OK;
}

static nmz_status cpu_adam_preview(const nmz_backend *be, size_t n, double lr, double b1, double b2, double eps,
                                   double bc1, double bc2, double decay, int fresh, const float *w, const float *g,
                                   const float *m, const float *v, float *m_out, float *v_out, float *mhat_out,
                                   float *vhat_out, float *delta) {
    size_t e;
    (void)be;
    for (e = 0; e < n; e++) {
        double gi = g[e], m0 = fresh ? 0.0 : m[e], v0 = fresh ? 0.0 : v[e];
        double mi = b1 * m0 + (1.0 - b1) * gi, vi = b2 * v0 + (1.0 - b2) * gi * gi;
        double mhat = mi / bc1, vhat = vi / bc2;
        m_out[e] = (float)mi;
        v_out[e] = (float)vi;
        mhat_out[e] = (float)mhat;
        vhat_out[e] = (float)vhat;
        delta[e] = (float)(-lr * (mhat / (sqrt(vhat) + eps) + decay * w[e]));
    }
    return NMZ_OK;
}

/* the reference clipping loops: one double accumulator over every tensor */
static nmz_status cpu_sumsq(const nmz_backend *be, int k, float *const *x, const size_t *n, double *out) {
    double sq = 0.0;
    int i;
    size_t e;
    (void)be;
    for (i = 0; i < k; i++)
        for (e = 0; e < n[i]; e++) sq += (double)x[i][e] * x[i][e];
    *out = sq;
    return NMZ_OK;
}

static nmz_status cpu_scale(const nmz_backend *be, int k, float *const *x, const size_t *n, double f) {
    int i;
    size_t e;
    (void)be;
    for (i = 0; i < k; i++)
        for (e = 0; e < n[i]; e++) x[i][e] = (float)(x[i][e] * f);
    return NMZ_OK;
}

/* ------------------------------------------------------- the training loop */

/* the reference nmz__norm_apply / nmz__norm_scale_grad (normalize.c) */
static nmz_status cpu_norm_apply(const nmz_backend *be, size_t n, size_t per, size_t div, int groups,
                                 const float *offset, const float *scale, float *x, int grad) {
    size_t e;
    (void)be;
    if (per == 0) return NMZ_OK;
    for (e = 0; e < n; e++) {
        int g = (int)((e % per) / div % (size_t)groups);
        if (grad)
            x[e] *= scale[g];
        else
            x[e] = (x[e] - offset[g]) * scale[g];
    }
    return NMZ_OK;
}

static nmz_status cpu_gather_rows(const nmz_backend *be, int rows, size_t width, const float *idx, const float *src,
                                  float *dst) {
    int r;
    (void)be;
    for (r = 0; r < rows; r++) memcpy(dst + (size_t)r * width, src + (size_t)idx[r] * width, width * sizeof(float));
    return NMZ_OK;
}

static nmz_status cpu_onehot(const nmz_backend *be, size_t n, int V, const float *ids, float *out) {
    size_t i;
    (void)be;
    memset(out, 0, n * (size_t)V * sizeof(float));
    for (i = 0; i < n; i++) {
        float id = ids[i];
        if (!(id >= 0.0f && id < (float)V))
            return nmz__fail(NMZ_ERR_FORMAT, "target id %g out of range 0..%d", (double)id, V - 1);
        out[i * (size_t)V + (size_t)id] = 1.0f;
    }
    return NMZ_OK;
}

static int argmax_row(const float *v, int n) {
    int i, best = 0;
    for (i = 1; i < n; i++)
        if (v[i] > v[best]) best = i;
    return best;
}

static nmz_status cpu_count_correct(const nmz_backend *be, int mode, size_t rows, int cols, const float *y,
                                    const float *t, double *out) {
    size_t r;
    long correct = 0;
    int k;
    (void)be;
    for (r = 0; r < rows; r++) {
        const float *yr = y + r * (size_t)cols, *tr = t + r * (size_t)cols;
        if (mode == 1) {
            correct += argmax_row(yr, cols) == argmax_row(tr, cols);
        } else if (mode == 2) {
            int ok = 1;
            for (k = 0; k < cols; k++) ok &= (yr[k] >= 0.5f) == (tr[k] >= 0.5f);
            correct += ok;
        }
    }
    *out = (double)correct;
    return NMZ_OK;
}

/* ------------------------------------------------------------- the tables */

#define CPU_OPS                                                                                                      \
    .abi = NMZ_BACKEND_ABI, .host_memory = 1, .impl = NULL, .init = cpu_init, .info = cpu_info, .alloc = NULL,         \
    .release = NULL, .upload = NULL, .download = NULL, .sync = cpu_sync, .fill = cpu_fill, .copy = cpu_copy,           \
    .add = cpu_add, .mul = cpu_mul, .gemm = cpu_gemm, .add_bias = cpu_add_bias, .add_bias3 = cpu_add_bias3,            \
    .colsum_add = cpu_colsum_add, .rowsum_add = cpu_rowsum_add, .transpose3 = cpu_transpose3,                          \
    .act_forward = nmz__cpu_act_forward, .act_backward = nmz__cpu_act_backward,                                        \
    .softmax_forward = nmz__cpu_softmax_forward, .softmax_backward = nmz__cpu_softmax_backward,                        \
    .loss_value = nmz__cpu_loss_value, .loss_grad = nmz__cpu_loss_grad, .im2col = cpu_im2col, .col2im = cpu_col2im,    \
    .maxpool_forward = nmz__cpu_maxpool_forward, .maxpool_backward = nmz__cpu_maxpool_backward,                        \
    .dropout_mask = cpu_dropout_mask, .embed_forward = nmz__cpu_embed_forward,                                         \
    .embed_backward = nmz__cpu_embed_backward, .posenc_forward = nmz__cpu_posenc_forward,                              \
    .posenc_backward = nmz__cpu_posenc_backward, .layernorm_forward = nmz__cpu_layernorm_forward,                      \
    .layernorm_backward = nmz__cpu_layernorm_backward, .attention_forward = nmz__cpu_attention_forward,                \
    .attention_backward = nmz__cpu_attention_backward, .rnn_cell_forward = cpu_rnn_cell_forward,                       \
    .rnn_cell_backward = cpu_rnn_cell_backward, .lstm_cell_forward = cpu_lstm_cell_forward,                            \
    .lstm_cell_backward = cpu_lstm_cell_backward, .gru_cell_forward1 = cpu_gru_cell_forward1,                          \
    .gru_cell_forward2 = cpu_gru_cell_forward2, .gru_cell_backward1 = cpu_gru_cell_backward1,                          \
    .gru_cell_backward2 = cpu_gru_cell_backward2, .sgd_velocity = cpu_sgd_velocity, .adam_update = cpu_adam_update,   \
    .adam_preview = cpu_adam_preview, .sumsq = cpu_sumsq, .scale = cpu_scale, .norm_apply = cpu_norm_apply,           \
    .gather_rows = cpu_gather_rows, .onehot = cpu_onehot, .count_correct = cpu_count_correct

const nmz_backend nmz__cpu_backend = {
    CPU_OPS,
    .name = "cpu",
    .summary = "the CPU (reference loops, optional OpenMP)",
    .dense_forward = nmz__cpu_dense_forward,
    .dense_backward = nmz__cpu_dense_backward,
    .conv_forward = nmz__cpu_conv_forward,
    .conv_backward = nmz__cpu_conv_backward,
    .rnn_forward = nmz__cpu_rnn_forward,
    .rnn_backward = nmz__cpu_rnn_backward,
    .lstm_forward = nmz__cpu_lstm_forward,
    .lstm_backward = nmz__cpu_lstm_backward,
    .gru_forward = nmz__cpu_gru_forward,
    .gru_backward = nmz__cpu_gru_backward,
};

const nmz_backend nmz__cpu_generic_backend = {
    CPU_OPS,
    .name = "cpu-generic",
    .summary = "the CPU running the generic compositions a GPU runs (for testing)",
};
