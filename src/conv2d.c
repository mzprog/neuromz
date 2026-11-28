/* conv2d.c — 2-D convolution (strictly: cross-correlation, as in every DL
 * library), naive loops, zero padding.
 *
 * Per sample: x [C, H, W] → z [F, OH, OW], weights W [F, C, KH, KW], bias b [F]
 *   OH = (H + 2·pad − KH) / stride + 1         (integer division)
 *
 *   forward:   z[f,oy,ox] = b[f] + Σ_{c,ky,kx} x[c, oy·s+ky−p, ox·s+kx−p] · W[f,c,ky,kx]
 *   backward:  δ = ∂L/∂z (given)
 *              ∂L/∂W[f,c,ky,kx] += Σ_{oy,ox} δ[f,oy,ox] · x[c, oy·s+ky−p, ox·s+kx−p]
 *              ∂L/∂b[f]         += Σ_{oy,ox} δ[f,oy,ox]
 *              ∂L/∂x[c,iy,ix]    = Σ over every (f,oy,ox,ky,kx) that read x[c,iy,ix]: δ·W
 * Positions that fall into the zero padding contribute nothing. */
#include "internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct conv_layer {
    nmz_conv_algo algo;
    int filters, kh, kw, stride, pad;
    int same; /* pad was requested as "same" */
    nmz_init init, resolved;
    int C, H, W, OH, OW;
    nmz_tensor Wt, b, dW, db;
    double *acc; /* CPU kernel: dW accumulator (one batch), same size as W */
    float *col;  /* CPU kernel: im2col [C·KH·KW, OH·OW] for one sample */
    float *dcol; /* CPU kernel: gradient of col */
    nmz_tensor gcol, gz, gdcol; /* generic path: [K, B·P], [F, B·P], [K, B·P] */
} conv_layer;

static nmz_status conv_setup(nmz_layer *l, int in_rank, const int *in_shape, int *out_rank,
                             int *out_shape) {
    conv_layer *c = l->self;
    int ws[4];
    nmz_status s;
    if (in_rank != 3)
        return nmz__fail(NMZ_ERR_SHAPE, "conv2d needs [channels, height, width] input (got rank %d); "
                                        "create the net with an input shape like 1x28x28", in_rank);
    c->C = in_shape[0];
    c->H = in_shape[1];
    c->W = in_shape[2];
    if (c->same) {
        if (c->stride != 1 || c->kh % 2 == 0 || c->kw % 2 == 0)
            return nmz__fail(NMZ_ERR_ARG, "pad=same needs stride 1 and an odd kernel size");
        c->pad = c->kh / 2;
        if (c->kw / 2 != c->pad) return nmz__fail(NMZ_ERR_ARG, "pad=same needs a square kernel");
    }
    if (c->H + 2 * c->pad < c->kh || c->W + 2 * c->pad < c->kw)
        return nmz__fail(NMZ_ERR_SHAPE, "conv2d kernel %dx%d is larger than the padded input %dx%d", c->kh, c->kw,
                         c->H + 2 * c->pad, c->W + 2 * c->pad);
    c->OH = (c->H + 2 * c->pad - c->kh) / c->stride + 1;
    c->OW = (c->W + 2 * c->pad - c->kw) / c->stride + 1;
    ws[0] = c->filters;
    ws[1] = c->C;
    ws[2] = c->kh;
    ws[3] = c->kw;
    if ((s = nmz_tensor_init(&c->Wt, 4, ws)) != NMZ_OK) return s;
    if ((s = nmz_tensor_init(&c->dW, 4, ws)) != NMZ_OK) return s;
    if ((s = nmz_tensor_init(&c->b, 1, &c->filters)) != NMZ_OK) return s;
    if ((s = nmz_tensor_init(&c->db, 1, &c->filters)) != NMZ_OK) return s;
    free(c->acc);
    c->acc = malloc(c->Wt.size * sizeof *c->acc);
    if (!c->acc) return nmz__fail(NMZ_ERR_NOMEM, "out of memory");
    {
        size_t K = (size_t)c->C * c->kh * c->kw;
        size_t P = (size_t)((c->H + 2 * c->pad - c->kh) / c->stride + 1) * (size_t)((c->W + 2 * c->pad - c->kw) / c->stride + 1);
        free(c->col);
        free(c->dcol);
        c->col = malloc(K * P * sizeof *c->col);
        c->dcol = malloc(K * P * sizeof *c->dcol);
        if (!c->col || !c->dcol) return nmz__fail(NMZ_ERR_NOMEM, "out of memory");
    }
    *out_rank = 3;
    out_shape[0] = c->filters;
    out_shape[1] = c->OH;
    out_shape[2] = c->OW;
    return NMZ_OK;
}

/* He unless a tanh/sigmoid activation follows (then Glorot). */
static nmz_status conv_init(nmz_layer *l, nmz_rng *rng, const nmz_layer *next) {
    conv_layer *c = l->self;
    nmz_init init = c->init;
    double fan_in = (double)c->C * c->kh * c->kw, fan_out = (double)c->filters * c->kh * c->kw, limit;
    size_t i;
    if (init == NMZ_INIT_AUTO) {
        int k = next ? nmz_layer_activation_kind(next) : -1;
        init = (k == NMZ_ACT_TANH || k == NMZ_ACT_SIGMOID) ? NMZ_INIT_XAVIER : NMZ_INIT_HE;
    }
    c->resolved = init;
    limit = init == NMZ_INIT_HE ? sqrt(6.0 / fan_in) : sqrt(6.0 / (fan_in + fan_out));
    {
        float *w = nmz__hw(&c->Wt);
        for (i = 0; i < c->Wt.size; i++) w[i] = (float)((2.0 * nmz_rng_uniform(rng) - 1.0) * limit);
    }
    nmz_tensor_zero(&c->b);
    return NMZ_OK;
}

static void emit(const nmz_ctx *ctx, conv_layer *c, nmz_trace_phase ph, int n, const nmz_trace_item *it) {
    nmz_trace_scalar sc[2];
    sc[0].name = "stride";
    sc[0].value = c->stride;
    sc[1].name = "pad";
    sc[1].value = c->pad;
    nmz_trace_emit(ctx, ph, "conv2d", "", 2, sc, n, it);
}

/* ------------------------------------------------------------- im2col
 * The same math as the loops below, rearranged: every receptive field is
 * copied into one column of `col` ([C·KH·KW rows, OH·OW columns]), so the
 * convolution becomes a matrix product z[f,:] = b[f] + W[f,:]·col whose
 * inner loop runs over contiguous memory. Backward: dW += dz·colᵀ,
 * dcol = Wᵀ·dz, then dcol is scattered back ("col2im") into dx.
 *
 * These are the CPU kernels (the reference loops, one sample at a time). A
 * backend without them gets the generic path further down: one im2col for
 * the whole batch and one matrix product per pass. */

static void im2col(const nmz_conv_shape *c, const float *xs, float *col) {
    int ch, ky, kx, oy, ox;
    size_t P = (size_t)c->OH * c->OW, k = 0;
    for (ch = 0; ch < c->C; ch++)
        for (ky = 0; ky < c->kh; ky++)
            for (kx = 0; kx < c->kw; kx++, k++) {
                float *row = col + k * P;
                const float *xc = xs + (size_t)ch * c->H * c->W;
                for (oy = 0; oy < c->OH; oy++) {
                    int iy = oy * c->stride + ky - c->pad;
                    for (ox = 0; ox < c->OW; ox++) {
                        int ix = ox * c->stride + kx - c->pad;
                        row[oy * c->OW + ox] = (iy < 0 || iy >= c->H || ix < 0 || ix >= c->W) ? 0.0f : xc[iy * c->W + ix];
                    }
                }
            }
}

static void col2im_add(const nmz_conv_shape *c, const float *dcol, float *dxs) {
    int ch, ky, kx, oy, ox;
    size_t P = (size_t)c->OH * c->OW, k = 0;
    for (ch = 0; ch < c->C; ch++)
        for (ky = 0; ky < c->kh; ky++)
            for (kx = 0; kx < c->kw; kx++, k++) {
                const float *row = dcol + k * P;
                float *dxc = dxs + (size_t)ch * c->H * c->W;
                for (oy = 0; oy < c->OH; oy++) {
                    int iy = oy * c->stride + ky - c->pad;
                    if (iy < 0 || iy >= c->H) continue;
                    for (ox = 0; ox < c->OW; ox++) {
                        int ix = ox * c->stride + kx - c->pad;
                        if (ix >= 0 && ix < c->W) dxc[iy * c->W + ix] += row[oy * c->OW + ox];
                    }
                }
            }
}

static void forward_im2col(const nmz_conv_shape *c, const float *x, const float *Wt, const float *bias, float *y,
                           float *col) {
    int B = c->B, bi, f;
    size_t K = (size_t)c->C * c->kh * c->kw, P = (size_t)c->OH * c->OW, k, p;
    size_t in_sample = (size_t)c->C * c->H * c->W;
    for (bi = 0; bi < B; bi++) {
        im2col(c, x + bi * in_sample, col);
        for (f = 0; f < c->F; f++) {
            const float *wf = Wt + (size_t)f * K;
            float *yf = y + ((size_t)bi * c->F + (size_t)f) * P;
            for (p = 0; p < P; p++) yf[p] = bias[f];
            for (k = 0; k < K; k++) {
                const float w = wf[k], *ck = col + k * P;
                for (p = 0; p < P; p++) yf[p] += w * ck[p];
            }
        }
    }
}

static void backward_im2col(const nmz_conv_shape *c, const float *x, const float *Wt, const float *dy, float *db,
                            float *dx, double *acc, float *col, float *dcol) {
    int B = c->B, bi, f;
    size_t K = (size_t)c->C * c->kh * c->kw, P = (size_t)c->OH * c->OW, k, p;
    size_t in_sample = (size_t)c->C * c->H * c->W;
    for (bi = 0; bi < B; bi++) {
        const float *gs = dy + (size_t)bi * c->F * P;
        im2col(c, x + bi * in_sample, col);
        if (dx) memset(dcol, 0, K * P * sizeof *dcol);
        for (f = 0; f < c->F; f++) {
            const float *gf = gs + (size_t)f * P, *wf = Wt + (size_t)f * K;
            double bsum = 0.0;
            for (p = 0; p < P; p++) bsum += gf[p];
            db[f] += (float)bsum;
            for (k = 0; k < K; k++) {
                const float *ck = col + k * P;
                float part[8] = {0, 0, 0, 0, 0, 0, 0, 0}, s;
                int j;
                /* 8 independent partial sums: a dot product the compiler can vectorise
                 * without reordering a single running sum */
                for (p = 0; p + 8 <= P; p += 8)
                    for (j = 0; j < 8; j++) part[j] += gf[p + (size_t)j] * ck[p + (size_t)j];
                s = ((part[0] + part[1]) + (part[2] + part[3])) + ((part[4] + part[5]) + (part[6] + part[7]));
                for (; p < P; p++) s += gf[p] * ck[p];
                acc[(size_t)f * K + k] += s; /* per-sample sums are short; the batch total is double */
                if (dx) {
                    float w = wf[k], *dk = dcol + k * P;
                    for (p = 0; p < P; p++) dk[p] += w * gf[p];
                }
            }
        }
        if (dx) col2im_add(c, dcol, dx + bi * in_sample);
    }
}

nmz_status nmz__cpu_conv_forward(const nmz_backend *be, const nmz_conv_shape *c, int naive, const float *x,
                                 const float *Wt, const float *bias, float *y, float *col) {
    int B = c->B, bi, f, oy, ox, ch, ky, kx;
    const size_t in_plane = (size_t)c->H * c->W, in_sample = in_plane * c->C;
    const size_t out_plane = (size_t)c->OH * c->OW, out_sample = out_plane * c->F;
    const size_t k_plane = (size_t)c->kh * c->kw, k_filter = k_plane * c->C;
    (void)be;
    if (!naive) {
        forward_im2col(c, x, Wt, bias, y, col);
        return NMZ_OK;
    }
    for (bi = 0; bi < B; bi++) {
        const float *xs = x + bi * in_sample;
        float *ys = y + bi * out_sample;
        for (f = 0; f < c->F; f++) {
            const float *wf = Wt + f * k_filter;
            for (oy = 0; oy < c->OH; oy++)
                for (ox = 0; ox < c->OW; ox++) {
                    double acc = bias[f];
                    int y0 = oy * c->stride - c->pad, x0 = ox * c->stride - c->pad;
                    for (ch = 0; ch < c->C; ch++) {
                        const float *xc = xs + ch * in_plane, *wc = wf + ch * k_plane;
                        for (ky = 0; ky < c->kh; ky++) {
                            int iy = y0 + ky;
                            if (iy < 0 || iy >= c->H) continue;
                            for (kx = 0; kx < c->kw; kx++) {
                                int ix = x0 + kx;
                                if (ix < 0 || ix >= c->W) continue;
                                acc += (double)xc[iy * c->W + ix] * wc[ky * c->kw + kx];
                            }
                        }
                    }
                    ys[f * out_plane + (size_t)oy * c->OW + ox] = (float)acc;
                }
        }
    }
    return NMZ_OK;
}

nmz_status nmz__cpu_conv_backward(const nmz_backend *be, const nmz_conv_shape *c, int naive, const float *x,
                                  const float *Wt, const float *dy, float *dW, float *db, float *dx, double *acc,
                                  float *col, float *dcol) {
    int B = c->B, bi, f, oy, ox, ch, ky, kx;
    const size_t in_plane = (size_t)c->H * c->W, in_sample = in_plane * c->C;
    const size_t out_plane = (size_t)c->OH * c->OW, out_sample = out_plane * c->F;
    const size_t k_plane = (size_t)c->kh * c->kw, k_filter = k_plane * c->C, wsize = k_filter * c->F;
    size_t i;
    (void)be;
    memset(acc, 0, wsize * sizeof *acc);
    if (dx) memset(dx, 0, in_sample * B * sizeof(float));
    if (!naive) {
        backward_im2col(c, x, Wt, dy, db, dx, acc, col, dcol);
        B = 0; /* skip the naive loop below */
    }
    for (bi = 0; bi < B; bi++) {
        const float *xs = x + bi * in_sample;
        const float *gs = dy + bi * out_sample;
        float *dxs = dx ? dx + bi * in_sample : NULL;
        for (f = 0; f < c->F; f++) {
            const float *wf = Wt + f * k_filter;
            double *af = acc + f * k_filter, bsum = 0.0;
            for (oy = 0; oy < c->OH; oy++)
                for (ox = 0; ox < c->OW; ox++) {
                    double g = gs[f * out_plane + (size_t)oy * c->OW + ox];
                    int y0 = oy * c->stride - c->pad, x0 = ox * c->stride - c->pad;
                    bsum += g;
                    if (g == 0.0) continue; /* e.g. behind a relu or a pool loser */
                    for (ch = 0; ch < c->C; ch++) {
                        const float *xc = xs + ch * in_plane, *wc = wf + ch * k_plane;
                        double *ac = af + ch * k_plane;
                        float *dxc = dxs ? dxs + ch * in_plane : NULL;
                        for (ky = 0; ky < c->kh; ky++) {
                            int iy = y0 + ky;
                            if (iy < 0 || iy >= c->H) continue;
                            for (kx = 0; kx < c->kw; kx++) {
                                int ix = x0 + kx;
                                if (ix < 0 || ix >= c->W) continue;
                                ac[ky * c->kw + kx] += g * xc[iy * c->W + ix];
                                if (dxc) dxc[iy * c->W + ix] += (float)(g * wc[ky * c->kw + kx]);
                            }
                        }
                    }
                }
            db[f] += (float)bsum;
        }
    }
    for (i = 0; i < wsize; i++) dW[i] += (float)acc[i];
    return NMZ_OK;
}

static nmz_conv_shape conv_shape(const conv_layer *c, int B) {
    nmz_conv_shape s;
    s.B = B;
    s.C = c->C;
    s.H = c->H;
    s.W = c->W;
    s.F = c->filters;
    s.kh = c->kh;
    s.kw = c->kw;
    s.stride = c->stride;
    s.pad = c->pad;
    s.OH = c->OH;
    s.OW = c->OW;
    return s;
}

/* generic: col [K, B·P] = im2col(x); z [F, B·P] = W·col; y = z as [B, F, P] + b */
static nmz_status conv_forward(nmz_layer *l, const nmz_tensor *x, nmz_tensor *y, const nmz_ctx *ctx) {
    conv_layer *c = l->self;
    const nmz_backend *be = ctx->backend;
    nmz_conv_shape sh = conv_shape(c, x->shape[0]);
    const float *px = nmz__r(be, x), *pW = nmz__r(be, &c->Wt), *pb = nmz__r(be, &c->b);
    float *py = nmz__w(be, y);
    nmz_status s;
    if (!px || !pW || !pb || !py) return nmz__op_failed();
    if (NMZ_FUSED(be, conv_forward)) {
        s = be->conv_forward(be, &sh, c->algo == NMZ_CONV_NAIVE, px, pW, pb, py, c->col);
    } else {
        int K = c->C * c->kh * c->kw, N = sh.B * c->OH * c->OW, cs[2], zs[2];
        float *pcol, *pz;
        cs[0] = K;
        cs[1] = N;
        zs[0] = c->filters;
        zs[1] = N;
        if ((s = nmz_tensor_reshape(&c->gcol, 2, cs)) != NMZ_OK || (s = nmz_tensor_reshape(&c->gz, 2, zs)) != NMZ_OK)
            return s;
        pcol = nmz__w(be, &c->gcol);
        pz = nmz__w(be, &c->gz);
        if (!pcol || !pz) return nmz__op_failed();
        if ((s = be->im2col(be, &sh, px, pcol)) == NMZ_OK &&
            (s = be->gemm(be, 0, c->filters, N, K, pW, K, pcol, N, pz, N)) == NMZ_OK &&
            (s = be->transpose3(be, c->filters, sh.B, c->OH * c->OW, pz, py)) == NMZ_OK)
            s = be->add_bias3(be, sh.B, c->filters, c->OH * c->OW, py, pb);
    }
    if (s != NMZ_OK) return s;
    if (NMZ_TRACING(ctx)) {
        nmz_trace_item it[4] = {{"x", x}, {"W", &c->Wt}, {"b", &c->b}, {"z", y}};
        emit(ctx, c, NMZ_TRACE_FORWARD, 4, it);
    }
    return NMZ_OK;
}

/* generic: dz [F, B·P] = dy as [F, B, P]; db += row sums; dW += dz·colᵀ;
 * dcol = Wᵀ·dz; dx = col2im(dcol) */
static nmz_status conv_backward(nmz_layer *l, const nmz_tensor *x, const nmz_tensor *y, const nmz_tensor *dy,
                                nmz_tensor *dx, const nmz_ctx *ctx) {
    conv_layer *c = l->self;
    const nmz_backend *be = ctx->backend;
    nmz_conv_shape sh = conv_shape(c, x->shape[0]);
    const float *px = nmz__r(be, x), *pW = nmz__r(be, &c->Wt), *pdy = nmz__r(be, dy);
    float *pdW = nmz__u(be, &c->dW), *pdb = nmz__u(be, &c->db), *pdx = dx ? nmz__w(be, dx) : NULL;
    nmz_status s;
    (void)y;
    if (!px || !pW || !pdy || !pdW || !pdb || (dx && !pdx)) return nmz__op_failed();
    if (NMZ_FUSED(be, conv_backward)) {
        s = be->conv_backward(be, &sh, c->algo == NMZ_CONV_NAIVE, px, pW, pdy, pdW, pdb, pdx, c->acc, c->col,
                              c->dcol);
    } else {
        int K = c->C * c->kh * c->kw, P = c->OH * c->OW, N = sh.B * P, cs[2], zs[2];
        float *pcol, *pz, *pdc = NULL;
        cs[0] = K;
        cs[1] = N;
        zs[0] = c->filters;
        zs[1] = N;
        if ((s = nmz_tensor_reshape(&c->gcol, 2, cs)) != NMZ_OK || (s = nmz_tensor_reshape(&c->gz, 2, zs)) != NMZ_OK ||
            (dx && (s = nmz_tensor_reshape(&c->gdcol, 2, cs)) != NMZ_OK))
            return s;
        pcol = nmz__w(be, &c->gcol);
        pz = nmz__w(be, &c->gz);
        if (dx) pdc = nmz__w(be, &c->gdcol);
        if (!pcol || !pz || (dx && !pdc)) return nmz__op_failed();
        s = be->im2col(be, &sh, px, pcol);
        if (s == NMZ_OK) s = be->transpose3(be, sh.B, c->filters, P, pdy, pz);
        if (s == NMZ_OK) s = be->rowsum_add(be, c->filters, (size_t)N, pz, pdb);
        if (s == NMZ_OK) s = be->gemm(be, NMZ_MM_TRANS_B | NMZ_MM_ACCUMULATE, c->filters, K, N, pz, N, pcol, N, pdW, K);
        if (s == NMZ_OK && dx) s = be->gemm(be, NMZ_MM_TRANS_A, K, N, c->filters, pW, K, pz, N, pdc, N);
        if (s == NMZ_OK && dx) s = be->col2im(be, &sh, pdc, pdx);
    }
    if (s != NMZ_OK) return s;
    if (NMZ_TRACING(ctx)) {
        nmz_trace_item it[6] = {{"dL/dz", dy}, {"x", x}, {"W", &c->Wt}, {"dL/dW", &c->dW}, {"dL/db", &c->db},
                                {"dL/dx", dx}};
        emit(ctx, c, NMZ_TRACE_BACKWARD, dx ? 6 : 5, it);
    }
    return NMZ_OK;
}

static int conv_param_count(const nmz_layer *l) {
    (void)l;
    return 2;
}

static nmz_param conv_param(nmz_layer *l, int i) {
    conv_layer *c = l->self;
    nmz_param p = {NULL, NULL, NULL};
    if (i == 0) {
        p.name = "W";
        p.value = &c->Wt;
        p.grad = &c->dW;
    } else if (i == 1) {
        p.name = "b";
        p.value = &c->b;
        p.grad = &c->db;
    }
    return p;
}

static nmz_status conv_save(const nmz_layer *l, nmz_writer *w) {
    const conv_layer *c = l->self;
    nmz_write_u32(w, (uint32_t)c->filters);
    nmz_write_u32(w, (uint32_t)c->kh);
    nmz_write_u32(w, (uint32_t)c->kw);
    nmz_write_u32(w, (uint32_t)c->stride);
    nmz_write_u32(w, (uint32_t)c->pad);
    nmz_write_u32(w, (uint32_t)c->same);
    nmz_write_u32(w, (uint32_t)c->init);
    return nmz_write_u32(w, (uint32_t)c->resolved);
}

static nmz_status conv_load(nmz_layer **out, nmz_reader *r) {
    uint32_t v[8];
    int i;
    for (i = 0; i < 8; i++) nmz_read_u32(r, &v[i]);
    if (r->err) return r->err;
    if (v[0] == 0 || v[0] > 100000 || v[1] == 0 || v[1] > 1000 || v[2] == 0 || v[2] > 1000 || v[3] == 0 ||
        v[3] > 1000 || v[4] > 1000 || v[5] > 1 || v[6] >= NMZ_INIT_COUNT || v[7] >= NMZ_INIT_COUNT)
        return nmz__fail(NMZ_ERR_FORMAT, "corrupt conv2d layer config");
    *out = nmz_layer_conv2d((int)v[0], (int)v[1], (int)v[2], (int)v[3], v[5] ? NMZ_PAD_SAME : (int)v[4],
                            (nmz_init)v[6]);
    if (!*out) return NMZ_ERR_NOMEM;
    ((conv_layer *)(*out)->self)->resolved = (nmz_init)v[7];
    return NMZ_OK;
}

static void conv_describe(const nmz_layer *l, char *buf, size_t n) {
    const conv_layer *c = l->self;
    snprintf(buf, n, "conv2d %dx%dx%d->%dx%dx%d (%d filters %dx%d, stride %d, pad %d, %s init)", c->C, c->H, c->W,
             c->filters, c->OH, c->OW, c->filters, c->kh, c->kw, c->stride, c->pad, nmz_init_name(c->resolved));
}

static void conv_destroy(nmz_layer *l) {
    conv_layer *c = l->self;
    if (!c) return;
    nmz_tensor_free(&c->Wt);
    nmz_tensor_free(&c->b);
    nmz_tensor_free(&c->dW);
    nmz_tensor_free(&c->db);
    free(c->acc);
    free(c->col);
    free(c->dcol);
    nmz_tensor_free(&c->gcol);
    nmz_tensor_free(&c->gz);
    nmz_tensor_free(&c->gdcol);
    free(c);
}

static int conv_buffers(nmz_layer *l, nmz_tensor **out, int max) {
    conv_layer *c = l->self;
    nmz_tensor *all[] = {&c->gcol, &c->gz, &c->gdcol};
    return nmz__list(out, max, (int)(sizeof all / sizeof all[0]), all);
}

const nmz_layer_ops nmz__conv2d_ops = {"conv2d",        conv_setup,   conv_init,     conv_forward,
                                       conv_backward,   conv_param_count, conv_param, conv_save,
                                       conv_load,       conv_describe, conv_destroy, NULL, NMZ_LAYER_BACKEND, NULL,
                                       conv_buffers};

nmz_status nmz_layer_conv2d_set_algo(nmz_layer *l, nmz_conv_algo algo) {
    if (!l || l->ops != &nmz__conv2d_ops) return nmz__fail(NMZ_ERR_ARG, "not a conv2d layer");
    if ((int)algo < 0 || algo > NMZ_CONV_NAIVE) return nmz__fail(NMZ_ERR_ARG, "invalid conv algorithm");
    ((conv_layer *)l->self)->algo = algo;
    return NMZ_OK;
}

int nmz__conv2d_init_used(const nmz_layer *l) { return (int)((const conv_layer *)l->self)->resolved; }

nmz_layer *nmz_layer_conv2d(int filters, int kh, int kw, int stride, int pad, nmz_init init) {
    conv_layer *c;
    nmz_layer *l;
    if (filters <= 0 || filters > 100000 || kh <= 0 || kw <= 0 || kh > 1000 || kw > 1000 || stride <= 0 ||
        stride > 1000 || pad > 1000 || (pad < 0 && pad != NMZ_PAD_SAME)) {
        nmz__fail(NMZ_ERR_ARG, "conv2d needs filters > 0, kernel > 0, stride > 0 and pad >= 0 (or same)");
        return NULL;
    }
    if ((int)init < 0 || init >= NMZ_INIT_COUNT || init == NMZ_INIT_ORTHOGONAL) {
        nmz__fail(NMZ_ERR_ARG, "invalid init %d for conv2d (orthogonal is for recurrent layers)", (int)init);
        return NULL;
    }
    c = calloc(1, sizeof *c);
    if (!c) {
        nmz__fail(NMZ_ERR_NOMEM, "out of memory");
        return NULL;
    }
    c->filters = filters;
    c->kh = kh;
    c->kw = kw;
    c->stride = stride;
    c->same = pad == NMZ_PAD_SAME;
    c->pad = c->same ? 0 : pad;
    c->init = c->resolved = init;
    l = nmz_layer_alloc(&nmz__conv2d_ops, c);
    if (!l) free(c);
    return l;
}
