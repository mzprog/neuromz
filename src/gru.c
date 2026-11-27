/* gru.c — Gated Recurrent Unit (Cho et al. 2014). x [B, T, D] → h [B, T, H].
 *
 * Gates side by side: W_x [D, 3H], W_h [H, 3H], b [3H], in the order
 * z (update), r (reset), ĥ (candidate). For t = 0 … T−1:
 *     z   = σ(x_t·W_xz + h_{t−1}·W_hz + b_z)
 *     r   = σ(x_t·W_xr + h_{t−1}·W_hr + b_r)
 *     ĥ   = tanh(x_t·W_xĥ + (r ⊙ h_{t−1})·W_hĥ + b_ĥ)
 *     h_t = z ⊙ h_{t−1} + (1 − z) ⊙ ĥ
 * This is the variant of the original paper: the reset gate scales h_{t−1}
 * BEFORE the product with W_hĥ. (PyTorch/cuDNN apply it after:
 * r ⊙ (h_{t−1}·W_hĥ + b_hĥ), with a second bias; it trains the same but
 * reads less directly as "r decides how much of the past the candidate
 * sees".) z = 1 keeps the old state, z = 0 replaces it by the candidate.
 *
 * Backward through time, t = T−1 … 0, with ∂L/∂h_t = (from above) + (from t+1):
 *     ∂L/∂ĥ   = ∂L/∂h_t ⊙ (1 − z)          ∂L/∂z = ∂L/∂h_t ⊙ (h_{t−1} − ĥ)
 *     ∂L/∂a_ĥ = ∂L/∂ĥ ⊙ (1 − ĥ²)            ∂L/∂a_z = ∂L/∂z ⊙ z(1−z)
 *     ∂L/∂(r⊙h) = ∂L/∂a_ĥ·W_hĥᵀ            ∂L/∂a_r = ∂L/∂(r⊙h) ⊙ h_{t−1} ⊙ r(1−r)
 *     ∂L/∂h_{t−1} = ∂L/∂h_t ⊙ z                       (the direct path)
 *                 + ∂L/∂(r⊙h) ⊙ r + [∂L/∂a_z ∂L/∂a_r]·[W_hz W_hr]ᵀ
 * The direct path is the GRU's gradient highway: only multiplied by z. */
#include "internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct gru_layer {
    int units, D;
    nmz_init init, resolved;
    nmz_tensor Wx, Wh, b, dWx, dWh, db;
    nmz_tensor h0;      /* initial state of the last forward, [B, H]   */
    nmz_tensor state;   /* carried state [B, H], or empty              */
    nmz_tensor gates;   /* [B, T, 3H]: z r ĥ after σ / tanh             */
    nmz_tensor dh_next; /* ∂L/∂h_t arriving from step t+1, [B, H]      */
    nmz_tensor tmp[8];  /* trace-only slices                           */
    double *acc;        /* 3H */
    float *da, *rh, *drh, *direct;
} gru_layer;

static float sigmoid(double a) { return (float)(1.0 / (1.0 + exp(-a))); }

static nmz_status gru_setup(nmz_layer *l, int in_rank, const int *in_shape, int *out_rank, int *out_shape) {
    gru_layer *g = l->self;
    int s2[2], G = 3 * g->units, H = g->units;
    nmz_status s;
    if (in_rank != 2)
        return nmz__fail(NMZ_ERR_SHAPE, "gru needs [steps, features] per sample (e.g. after an embedding), got rank %d",
                         in_rank);
    g->D = in_shape[1];
    s2[0] = g->D;
    s2[1] = G;
    if ((s = nmz_tensor_init(&g->Wx, 2, s2)) != NMZ_OK || (s = nmz_tensor_init(&g->dWx, 2, s2)) != NMZ_OK) return s;
    s2[0] = H;
    if ((s = nmz_tensor_init(&g->Wh, 2, s2)) != NMZ_OK || (s = nmz_tensor_init(&g->dWh, 2, s2)) != NMZ_OK) return s;
    if ((s = nmz_tensor_init(&g->b, 1, &G)) != NMZ_OK || (s = nmz_tensor_init(&g->db, 1, &G)) != NMZ_OK) return s;
    free(g->acc);
    free(g->da);
    free(g->rh);
    free(g->drh);
    free(g->direct);
    g->acc = malloc((size_t)(G > g->D ? G : g->D) * sizeof *g->acc);
    g->da = malloc((size_t)G * sizeof *g->da);
    g->rh = malloc((size_t)H * sizeof *g->rh);
    g->drh = malloc((size_t)H * sizeof *g->drh);
    g->direct = malloc((size_t)H * sizeof *g->direct);
    if (!g->acc || !g->da || !g->rh || !g->drh || !g->direct) return nmz__fail(NMZ_ERR_NOMEM, "out of memory");
    *out_rank = 2;
    out_shape[0] = in_shape[0];
    out_shape[1] = H;
    return NMZ_OK;
}

static nmz_status gru_init(nmz_layer *l, nmz_rng *rng, const nmz_layer *next) {
    gru_layer *g = l->self;
    (void)next;
    nmz_tensor_zero(&g->b);
    return nmz__recurrent_init(g->init, &g->resolved, &g->Wx, &g->Wh, g->D, g->units, 3, rng);
}

static nmz_status gru_forward(nmz_layer *l, const nmz_tensor *x, nmz_tensor *y, const nmz_ctx *ctx) {
    gru_layer *g = l->self;
    int B = x->shape[0], T = x->shape[1], H = g->units, D = g->D, G = 3 * H, bi, t, i, j;
    int shape[3];
    nmz_status s;
    shape[0] = B;
    shape[1] = H;
    if ((s = nmz_tensor_reshape(&g->h0, 2, shape)) != NMZ_OK) return s;
    if (ctx->carry && g->state.size == (size_t)B * H)
        memcpy(g->h0.data, g->state.data, g->h0.size * sizeof(float)); /* continue the stream */
    else
        nmz_tensor_zero(&g->h0);
    shape[1] = T;
    shape[2] = G;
    if ((s = nmz_tensor_reshape(&g->gates, 3, shape)) != NMZ_OK) return s;
    for (bi = 0; bi < B; bi++)
        for (t = 0; t < T; t++) {
            size_t row = (size_t)bi * T + (size_t)t;
            const float *xt = x->data + row * D;
            const float *hp = t ? y->data + (row - 1) * H : g->h0.data + (size_t)bi * H;
            float *gt = g->gates.data + row * G, *ht = y->data + row * H;
            for (j = 0; j < G; j++) g->acc[j] = g->b.data[j];
            for (i = 0; i < D; i++) {
                const float *w = g->Wx.data + (size_t)i * G;
                double xi = xt[i];
                if (xi != 0.0)
                    for (j = 0; j < G; j++) g->acc[j] += xi * w[j];
            }
            for (i = 0; i < H; i++) { /* z and r see h_{t−1} */
                const float *w = g->Wh.data + (size_t)i * G;
                double hi = hp[i];
                if (hi != 0.0)
                    for (j = 0; j < 2 * H; j++) g->acc[j] += hi * w[j];
            }
            for (j = 0; j < H; j++) {
                gt[j] = sigmoid(g->acc[j]);
                gt[H + j] = sigmoid(g->acc[H + j]);
                g->rh[j] = gt[H + j] * hp[j];
            }
            for (i = 0; i < H; i++) { /* the candidate sees r ⊙ h_{t−1} */
                const float *w = g->Wh.data + (size_t)i * G + 2 * H;
                double v = g->rh[i];
                if (v != 0.0)
                    for (j = 0; j < H; j++) g->acc[2 * H + j] += v * w[j];
            }
            for (j = 0; j < H; j++) {
                float z = gt[j], n = (float)tanh(g->acc[2 * H + j]);
                gt[2 * H + j] = n;
                ht[j] = z * hp[j] + (1.0f - z) * n;
            }
        }
    if (ctx->carry) { /* remember h_{T−1} for the next batch of the same streams */
        shape[0] = B;
        shape[1] = H;
        if ((s = nmz_tensor_reshape(&g->state, 2, shape)) != NMZ_OK) return s;
        for (bi = 0; bi < B; bi++)
            memcpy(g->state.data + (size_t)bi * H, y->data + ((size_t)bi * T + (size_t)T - 1) * H,
                   (size_t)H * sizeof(float));
    }
    if (NMZ_TRACING(ctx)) {
        nmz_trace_scalar sc[1] = {{"steps", T}};
        nmz_trace_item all[7] = {{"x", x},        {"h0", &g->h0}, {"W_x", &g->Wx},       {"W_h", &g->Wh},
                                 {"b", &g->b},    {"gates", &g->gates}, {"h", y}};
        nmz_tensor *tp = g->tmp;
        nmz_trace_emit(ctx, NMZ_TRACE_FORWARD, "gru", "z r h~", 1, sc, 7, all);
        for (t = 0; t < T; t++) {
            if ((s = nmz__slice_t(x, t, &tp[0])) != NMZ_OK || (s = nmz__slice_tc(&g->gates, t, 0, H, &tp[2])) != NMZ_OK ||
                (s = nmz__slice_tc(&g->gates, t, H, H, &tp[3])) != NMZ_OK ||
                (s = nmz__slice_tc(&g->gates, t, 2 * H, H, &tp[4])) != NMZ_OK ||
                (s = nmz__slice_t(y, t, &tp[5])) != NMZ_OK)
                return s;
            if (t == 0) {
                if ((s = nmz_tensor_copy(&tp[1], &g->h0)) != NMZ_OK) return s;
            } else if ((s = nmz__slice_t(y, t - 1, &tp[1])) != NMZ_OK) {
                return s;
            }
            {
                nmz_trace_item it[6] = {{"x_t", &tp[0]}, {"h_prev", &tp[1]}, {"z", &tp[2]},
                                        {"r", &tp[3]},   {"h~", &tp[4]},     {"h_t", &tp[5]}};
                nmz_trace_emit_step(ctx, NMZ_TRACE_FORWARD, "gru", "z r h~", t, 6, it);
            }
        }
    }
    return NMZ_OK;
}

static nmz_status gru_backward(nmz_layer *l, const nmz_tensor *x, const nmz_tensor *y, const nmz_tensor *dy,
                               nmz_tensor *dx, const nmz_ctx *ctx) {
    gru_layer *g = l->self;
    int B = x->shape[0], T = x->shape[1], H = g->units, D = g->D, G = 3 * H, bi, t, i, j;
    int tracing = NMZ_TRACING(ctx), shape[2];
    nmz_tensor *tp = g->tmp;
    nmz_status s;
    float *da = g->da;
    shape[0] = B;
    shape[1] = H;
    if ((s = nmz_tensor_reshape(&g->dh_next, 2, shape)) != NMZ_OK) return s;
    nmz_tensor_zero(&g->dh_next);
    if (tracing) {
        for (i = 0; i < 6; i++) {
            shape[1] = i == 3 ? G : H;
            if ((s = nmz_tensor_reshape(&tp[i], 2, shape)) != NMZ_OK) return s;
        }
    }
    for (t = T - 1; t >= 0; t--) {
        for (bi = 0; bi < B; bi++) {
            size_t row = (size_t)bi * T + (size_t)t, rb = (size_t)bi * H;
            const float *xt = x->data + row * D;
            const float *hp = t ? y->data + (row - 1) * H : g->h0.data + rb;
            const float *gt = g->gates.data + row * G;
            const float *gout = dy->data + row * H;
            float *dhn = g->dh_next.data + rb;
            for (j = 0; j < H; j++) {
                float z = gt[j], n = gt[2 * H + j];
                float dh = gout[j] + dhn[j];
                da[j] = dh * (hp[j] - n) * z * (1.0f - z);     /* ∂L/∂a_z */
                da[2 * H + j] = dh * (1.0f - z) * (1.0f - n * n); /* ∂L/∂a_ĥ */
                g->direct[j] = dh * z;
                g->rh[j] = gt[H + j] * hp[j];
                if (tracing) {
                    tp[0].data[rb + j] = gout[j];
                    tp[1].data[rb + j] = dhn[j];
                    tp[2].data[rb + j] = dh;
                    tp[4].data[rb + j] = g->direct[j];
                }
            }
            for (i = 0; i < H; i++) /* ∂L/∂(r⊙h_{t−1}) = ∂L/∂a_ĥ·W_hĥᵀ */
                g->drh[i] = (float)nmz__dot(g->Wh.data + (size_t)i * G + 2 * H, da + 2 * H, H);
            for (j = 0; j < H; j++) {
                float r = gt[H + j];
                da[H + j] = g->drh[j] * hp[j] * r * (1.0f - r); /* ∂L/∂a_r */
            }
            if (tracing) memcpy(tp[3].data + (size_t)bi * G, da, (size_t)G * sizeof(float));
            for (j = 0; j < G; j++) g->db.data[j] += da[j];
            for (i = 0; i < D; i++) {
                float *dw = g->dWx.data + (size_t)i * G, xi = xt[i];
                if (xi != 0.0f)
                    for (j = 0; j < G; j++) dw[j] += xi * da[j];
            }
            for (i = 0; i < H; i++) {
                float *dw = g->dWh.data + (size_t)i * G, hi = hp[i], ri = g->rh[i];
                if (hi != 0.0f)
                    for (j = 0; j < 2 * H; j++) dw[j] += hi * da[j];
                if (ri != 0.0f)
                    for (j = 0; j < H; j++) dw[2 * H + j] += ri * da[2 * H + j];
            }
            if (dx) {
                float *dxt = dx->data + row * D;
                for (i = 0; i < D; i++) dxt[i] = (float)nmz__dot(g->Wx.data + (size_t)i * G, da, G);
            }
            for (i = 0; i < H; i++) { /* ∂L/∂h_{t−1}: direct + through r⊙h + through the z, r products */
                double acc = (double)g->direct[i] + (double)g->drh[i] * gt[H + i];
                acc += nmz__dot(g->Wh.data + (size_t)i * G, da, 2 * H);
                dhn[i] = (float)acc;
                if (tracing) tp[5].data[rb + i] = dhn[i];
            }
        }
        if (tracing) {
            nmz_trace_item it[6] = {{"dL/dh_out", &tp[0]}, {"dL/dh_next", &tp[1]},   {"dL/dh", &tp[2]},
                                    {"dL/da", &tp[3]},     {"dL/dh_direct", &tp[4]}, {"dL/dh_prev", &tp[5]}};
            nmz_trace_emit_step(ctx, NMZ_TRACE_BACKWARD, "gru", "z r h~", t, 6, it);
        }
    }
    if (tracing) {
        nmz_trace_scalar sc[1] = {{"steps", T}};
        nmz_trace_item all[5] = {{"dL/dW_x", &g->dWx}, {"dL/dW_h", &g->dWh}, {"dL/db", &g->db},
                                 {"W_h", &g->Wh},      {"dL/dx", dx}};
        nmz_trace_emit(ctx, NMZ_TRACE_BACKWARD, "gru", "z r h~", 1, sc, dx ? 5 : 4, all);
    }
    return NMZ_OK;
}

static int gru_param_count(const nmz_layer *l) {
    (void)l;
    return 3;
}

static nmz_param gru_param(nmz_layer *l, int i) {
    gru_layer *g = l->self;
    nmz_param p = {NULL, NULL, NULL};
    if (i == 0) {
        p.name = "W_x";
        p.value = &g->Wx;
        p.grad = &g->dWx;
    } else if (i == 1) {
        p.name = "W_h";
        p.value = &g->Wh;
        p.grad = &g->dWh;
    } else if (i == 2) {
        p.name = "b";
        p.value = &g->b;
        p.grad = &g->db;
    }
    return p;
}

static nmz_status gru_save(const nmz_layer *l, nmz_writer *w) {
    const gru_layer *g = l->self;
    nmz_write_u32(w, (uint32_t)g->units);
    nmz_write_u32(w, (uint32_t)g->init);
    return nmz_write_u32(w, (uint32_t)g->resolved);
}

static nmz_status gru_load(nmz_layer **out, nmz_reader *rd) {
    uint32_t units = 0, init = 0, resolved = 0;
    nmz_read_u32(rd, &units);
    nmz_read_u32(rd, &init);
    if (nmz_read_u32(rd, &resolved) != NMZ_OK) return rd->err;
    if (units == 0 || units > 100000 || init >= NMZ_INIT_COUNT || resolved >= NMZ_INIT_COUNT)
        return nmz__fail(NMZ_ERR_FORMAT, "corrupt gru config");
    *out = nmz_layer_gru((int)units, (nmz_init)init);
    if (!*out) return NMZ_ERR_NOMEM;
    ((gru_layer *)(*out)->self)->resolved = (nmz_init)resolved;
    return NMZ_OK;
}

static void gru_describe(const nmz_layer *l, char *buf, size_t n) {
    const gru_layer *g = l->self;
    snprintf(buf, n, "gru %d->%d (gates z r h~, %s init)", g->D, g->units, nmz_init_name(g->resolved));
}

static nmz_tensor *gru_state(nmz_layer *l) { return &((gru_layer *)l->self)->state; }

static void gru_destroy(nmz_layer *l) {
    gru_layer *g = l->self;
    int i;
    if (!g) return;
    nmz_tensor_free(&g->Wx);
    nmz_tensor_free(&g->Wh);
    nmz_tensor_free(&g->b);
    nmz_tensor_free(&g->dWx);
    nmz_tensor_free(&g->dWh);
    nmz_tensor_free(&g->db);
    nmz_tensor_free(&g->h0);
    nmz_tensor_free(&g->state);
    nmz_tensor_free(&g->gates);
    nmz_tensor_free(&g->dh_next);
    for (i = 0; i < 8; i++) nmz_tensor_free(&g->tmp[i]);
    free(g->acc);
    free(g->da);
    free(g->rh);
    free(g->drh);
    free(g->direct);
    free(g);
}

const nmz_layer_ops nmz__gru_ops = {"gru",          gru_setup, gru_init, gru_forward, gru_backward, gru_param_count,
                                    gru_param,      gru_save,  gru_load, gru_describe, gru_destroy, gru_state,
                                    0, NULL};

nmz_layer *nmz_layer_gru(int units, nmz_init init) {
    gru_layer *g;
    nmz_layer *l;
    if (units < 1 || units > 100000 || (int)init < 0 || init >= NMZ_INIT_COUNT) {
        nmz__fail(NMZ_ERR_ARG, "gru needs a positive number of units");
        return NULL;
    }
    g = calloc(1, sizeof *g);
    if (!g) {
        nmz__fail(NMZ_ERR_NOMEM, "out of memory");
        return NULL;
    }
    g->units = units;
    g->init = g->resolved = init;
    l = nmz_layer_alloc(&nmz__gru_ops, g);
    if (!l) free(g);
    return l;
}

int nmz__gated_init_used(const nmz_layer *l) {
    if (l->ops == &nmz__gru_ops) return (int)((const gru_layer *)l->self)->resolved;
    return nmz__lstm_init_used(l);
}
