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
    int filters, kh, kw, stride, pad;
    int same; /* pad was requested as "same" */
    nmz_init init, resolved;
    int C, H, W, OH, OW;
    nmz_tensor Wt, b, dW, db;
    double *acc; /* dW accumulator (one batch), same size as W */
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
    for (i = 0; i < c->Wt.size; i++) c->Wt.data[i] = (float)((2.0 * nmz_rng_uniform(rng) - 1.0) * limit);
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

static nmz_status conv_forward(nmz_layer *l, const nmz_tensor *x, nmz_tensor *y, const nmz_ctx *ctx) {
    conv_layer *c = l->self;
    int B = x->shape[0], bi, f, oy, ox, ch, ky, kx;
    const size_t in_plane = (size_t)c->H * c->W, in_sample = in_plane * c->C;
    const size_t out_plane = (size_t)c->OH * c->OW, out_sample = out_plane * c->filters;
    const size_t k_plane = (size_t)c->kh * c->kw, k_filter = k_plane * c->C;
    for (bi = 0; bi < B; bi++) {
        const float *xs = x->data + bi * in_sample;
        float *ys = y->data + bi * out_sample;
        for (f = 0; f < c->filters; f++) {
            const float *wf = c->Wt.data + f * k_filter;
            for (oy = 0; oy < c->OH; oy++)
                for (ox = 0; ox < c->OW; ox++) {
                    double acc = c->b.data[f];
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
    if (NMZ_TRACING(ctx)) {
        nmz_trace_item it[4] = {{"x", x}, {"W", &c->Wt}, {"b", &c->b}, {"z", y}};
        emit(ctx, c, NMZ_TRACE_FORWARD, 4, it);
    }
    return NMZ_OK;
}

static nmz_status conv_backward(nmz_layer *l, const nmz_tensor *x, const nmz_tensor *y, const nmz_tensor *dy,
                                nmz_tensor *dx, const nmz_ctx *ctx) {
    conv_layer *c = l->self;
    int B = x->shape[0], bi, f, oy, ox, ch, ky, kx;
    const size_t in_plane = (size_t)c->H * c->W, in_sample = in_plane * c->C;
    const size_t out_plane = (size_t)c->OH * c->OW, out_sample = out_plane * c->filters;
    const size_t k_plane = (size_t)c->kh * c->kw, k_filter = k_plane * c->C;
    size_t i;
    (void)y;
    memset(c->acc, 0, c->Wt.size * sizeof *c->acc);
    if (dx) nmz_tensor_zero(dx);
    for (bi = 0; bi < B; bi++) {
        const float *xs = x->data + bi * in_sample;
        const float *gs = dy->data + bi * out_sample;
        float *dxs = dx ? dx->data + bi * in_sample : NULL;
        for (f = 0; f < c->filters; f++) {
            const float *wf = c->Wt.data + f * k_filter;
            double *af = c->acc + f * k_filter, bsum = 0.0;
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
            c->db.data[f] += (float)bsum;
        }
    }
    for (i = 0; i < c->Wt.size; i++) c->dW.data[i] += (float)c->acc[i];
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
    free(c);
}

const nmz_layer_ops nmz__conv2d_ops = {"conv2d",        conv_setup,   conv_init,     conv_forward,
                                       conv_backward,   conv_param_count, conv_param, conv_save,
                                       conv_load,       conv_describe, conv_destroy};

int nmz__conv2d_init_used(const nmz_layer *l) { return (int)((const conv_layer *)l->self)->resolved; }

nmz_layer *nmz_layer_conv2d(int filters, int kh, int kw, int stride, int pad, nmz_init init) {
    conv_layer *c;
    nmz_layer *l;
    if (filters <= 0 || filters > 100000 || kh <= 0 || kw <= 0 || kh > 1000 || kw > 1000 || stride <= 0 ||
        stride > 1000 || pad > 1000 || (pad < 0 && pad != NMZ_PAD_SAME)) {
        nmz__fail(NMZ_ERR_ARG, "conv2d needs filters > 0, kernel > 0, stride > 0 and pad >= 0 (or same)");
        return NULL;
    }
    if ((int)init < 0 || init >= NMZ_INIT_COUNT) {
        nmz__fail(NMZ_ERR_ARG, "invalid init %d", (int)init);
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
