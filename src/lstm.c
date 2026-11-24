/* lstm.c — Long Short-Term Memory (Hochreiter & Schmidhuber 1997, with the
 * forget gate of Gers et al. 2000). x [B, T, D] → h [B, T, H].
 *
 * Gates side by side: W_x [D, 4H], W_h [H, 4H], b [4H], in the order
 * i (input), f (forget), g (candidate), o (output). For t = 0 … T−1:
 *     a_t = x_t·W_x + h_{t−1}·W_h + b                      (4H numbers)
 *     i = σ(a_i)   f = σ(a_f)   g = tanh(a_g)   o = σ(a_o)
 *     c_t = f ⊙ c_{t−1} + i ⊙ g                  the cell state
 *     h_t = o ⊙ tanh(c_t)                        the output / hidden state
 *
 * Backward through time, t = T−1 … 0 (⊙ = element-wise):
 *     ∂L/∂h_t = (from the layer above at t) + (from step t+1, through W_h)
 *     ∂L/∂c_t = (from step t+1: ∂L/∂c_{t+1} ⊙ f_{t+1}) + ∂L/∂h_t ⊙ o ⊙ (1 − tanh²c_t)
 *     ∂L/∂a_i = ∂L/∂c_t ⊙ g ⊙ i(1−i)        ∂L/∂a_f = ∂L/∂c_t ⊙ c_{t−1} ⊙ f(1−f)
 *     ∂L/∂a_g = ∂L/∂c_t ⊙ i ⊙ (1−g²)        ∂L/∂a_o = ∂L/∂h_t ⊙ tanh(c_t) ⊙ o(1−o)
 *     ∂L/∂W_x += x_tᵀ·∂L/∂a   ∂L/∂W_h += h_{t−1}ᵀ·∂L/∂a   ∂L/∂b += Σ ∂L/∂a
 *     ∂L/∂x_t = ∂L/∂a·W_xᵀ    ∂L/∂h_{t−1} = ∂L/∂a·W_hᵀ    ∂L/∂c_{t−1} = ∂L/∂c_t ⊙ f
 *
 * The last line is the point of the LSTM: along the cell state the gradient
 * is only multiplied by the forget gate — no weight matrix, no squashing
 * derivative — so with f ≈ 1 it survives many steps (the "gradient
 * highway"). The carried state (truncated BPTT) is [B, 2H]: h | c. */
#include "internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct lstm_layer {
    int units, D;
    nmz_init init, resolved;
    float forget_bias;
    nmz_tensor Wx, Wh, b, dWx, dWh, db;
    nmz_tensor h0, c0;           /* initial state of the last forward, [B, H] */
    nmz_tensor state;            /* carried state [B, 2H] = h | c, or empty   */
    nmz_tensor gates;            /* [B, T, 4H]: i f g o after σ / tanh         */
    nmz_tensor c;                /* [B, T, H] cell states                     */
    nmz_tensor dh_next, dc_next; /* gradients arriving from step t+1, [B, H]  */
    nmz_tensor tmp[12];          /* trace-only slices                         */
    double *acc;                 /* 4H */
    float *da;                   /* 4H */
} lstm_layer;

static float sigmoid(double a) { return (float)(1.0 / (1.0 + exp(-a))); }

static nmz_status lstm_setup(nmz_layer *l, int in_rank, const int *in_shape, int *out_rank, int *out_shape) {
    lstm_layer *m = l->self;
    int s2[2], G = 4 * m->units;
    nmz_status s;
    if (in_rank != 2)
        return nmz__fail(NMZ_ERR_SHAPE, "lstm needs [steps, features] per sample (e.g. after an embedding), got rank %d",
                         in_rank);
    m->D = in_shape[1];
    s2[0] = m->D;
    s2[1] = G;
    if ((s = nmz_tensor_init(&m->Wx, 2, s2)) != NMZ_OK || (s = nmz_tensor_init(&m->dWx, 2, s2)) != NMZ_OK) return s;
    s2[0] = m->units;
    if ((s = nmz_tensor_init(&m->Wh, 2, s2)) != NMZ_OK || (s = nmz_tensor_init(&m->dWh, 2, s2)) != NMZ_OK) return s;
    if ((s = nmz_tensor_init(&m->b, 1, &G)) != NMZ_OK || (s = nmz_tensor_init(&m->db, 1, &G)) != NMZ_OK) return s;
    free(m->acc);
    free(m->da);
    m->acc = malloc((size_t)(G > m->D ? G : m->D) * sizeof *m->acc);
    m->da = malloc((size_t)G * sizeof *m->da);
    if (!m->acc || !m->da) return nmz__fail(NMZ_ERR_NOMEM, "out of memory");
    *out_rank = 2;
    out_shape[0] = in_shape[0];
    out_shape[1] = m->units;
    return NMZ_OK;
}

static nmz_status lstm_init(nmz_layer *l, nmz_rng *rng, const nmz_layer *next) {
    lstm_layer *m = l->self;
    int j;
    (void)next;
    nmz_tensor_zero(&m->b);
    for (j = 0; j < m->units; j++) m->b.data[m->units + j] = m->forget_bias; /* f starts ≈ σ(1) = 0.73 */
    return nmz__recurrent_init(m->init, &m->resolved, &m->Wx, &m->Wh, m->D, m->units, 4, rng);
}

static nmz_status lstm_forward(nmz_layer *l, const nmz_tensor *x, nmz_tensor *y, const nmz_ctx *ctx) {
    lstm_layer *m = l->self;
    int B = x->shape[0], T = x->shape[1], H = m->units, D = m->D, G = 4 * H, bi, t, i, j;
    int shape[3];
    nmz_status s;
    shape[0] = B;
    shape[1] = H;
    if ((s = nmz_tensor_reshape(&m->h0, 2, shape)) != NMZ_OK || (s = nmz_tensor_reshape(&m->c0, 2, shape)) != NMZ_OK)
        return s;
    if (ctx->carry && m->state.size == (size_t)B * 2 * H) { /* continue the streams: h | c */
        for (bi = 0; bi < B; bi++) {
            memcpy(m->h0.data + (size_t)bi * H, m->state.data + (size_t)bi * 2 * H, (size_t)H * sizeof(float));
            memcpy(m->c0.data + (size_t)bi * H, m->state.data + (size_t)bi * 2 * H + H, (size_t)H * sizeof(float));
        }
    } else {
        nmz_tensor_zero(&m->h0);
        nmz_tensor_zero(&m->c0);
    }
    shape[0] = B;
    shape[1] = T;
    shape[2] = G;
    if ((s = nmz_tensor_reshape(&m->gates, 3, shape)) != NMZ_OK) return s;
    shape[2] = H;
    if ((s = nmz_tensor_reshape(&m->c, 3, shape)) != NMZ_OK) return s;
    for (bi = 0; bi < B; bi++)
        for (t = 0; t < T; t++) {
            size_t row = (size_t)bi * T + (size_t)t;
            const float *xt = x->data + row * D;
            const float *hp = t ? y->data + (row - 1) * H : m->h0.data + (size_t)bi * H;
            const float *cp = t ? m->c.data + (row - 1) * H : m->c0.data + (size_t)bi * H;
            float *gt = m->gates.data + row * G, *ct = m->c.data + row * H, *ht = y->data + row * H;
            for (j = 0; j < G; j++) m->acc[j] = m->b.data[j];
            for (i = 0; i < D; i++) {
                const float *w = m->Wx.data + (size_t)i * G;
                double xi = xt[i];
                if (xi != 0.0)
                    for (j = 0; j < G; j++) m->acc[j] += xi * w[j];
            }
            for (i = 0; i < H; i++) {
                const float *w = m->Wh.data + (size_t)i * G;
                double hi = hp[i];
                if (hi != 0.0)
                    for (j = 0; j < G; j++) m->acc[j] += hi * w[j];
            }
            for (j = 0; j < H; j++) {
                float ig = sigmoid(m->acc[j]), fg = sigmoid(m->acc[H + j]);
                float gg = (float)tanh(m->acc[2 * H + j]), og = sigmoid(m->acc[3 * H + j]);
                float cv = fg * cp[j] + ig * gg;
                gt[j] = ig;
                gt[H + j] = fg;
                gt[2 * H + j] = gg;
                gt[3 * H + j] = og;
                ct[j] = cv;
                ht[j] = og * tanhf(cv);
            }
        }
    if (ctx->carry) { /* remember h_{T−1} | c_{T−1} for the next batch of the same streams */
        shape[0] = B;
        shape[1] = 2 * H;
        if ((s = nmz_tensor_reshape(&m->state, 2, shape)) != NMZ_OK) return s;
        for (bi = 0; bi < B; bi++) {
            size_t last = (size_t)bi * T + (size_t)T - 1;
            memcpy(m->state.data + (size_t)bi * 2 * H, y->data + last * H, (size_t)H * sizeof(float));
            memcpy(m->state.data + (size_t)bi * 2 * H + H, m->c.data + last * H, (size_t)H * sizeof(float));
        }
    }
    if (NMZ_TRACING(ctx)) {
        nmz_trace_scalar sc[2] = {{"steps", T}, {"forget_bias", m->forget_bias}};
        nmz_trace_item all[9] = {{"x", x},       {"h0", &m->h0},       {"c0", &m->c0},
                                 {"W_x", &m->Wx}, {"W_h", &m->Wh},      {"b", &m->b},
                                 {"gates", &m->gates}, {"c", &m->c}, {"h", y}};
        nmz_tensor *tp = m->tmp;
        nmz_trace_emit(ctx, NMZ_TRACE_FORWARD, "lstm", "i f g o", 2, sc, 9, all);
        for (t = 0; t < T; t++) { /* one event per time step: the unrolled network */
            if ((s = nmz__slice_t(x, t, &tp[0])) != NMZ_OK || (s = nmz__slice_tc(&m->gates, t, 0, H, &tp[3])) != NMZ_OK ||
                (s = nmz__slice_tc(&m->gates, t, H, H, &tp[4])) != NMZ_OK ||
                (s = nmz__slice_tc(&m->gates, t, 2 * H, H, &tp[5])) != NMZ_OK ||
                (s = nmz__slice_tc(&m->gates, t, 3 * H, H, &tp[6])) != NMZ_OK ||
                (s = nmz__slice_t(&m->c, t, &tp[7])) != NMZ_OK || (s = nmz__slice_t(y, t, &tp[8])) != NMZ_OK)
                return s;
            if (t == 0) {
                if ((s = nmz_tensor_copy(&tp[1], &m->h0)) != NMZ_OK || (s = nmz_tensor_copy(&tp[2], &m->c0)) != NMZ_OK)
                    return s;
            } else if ((s = nmz__slice_t(y, t - 1, &tp[1])) != NMZ_OK || (s = nmz__slice_t(&m->c, t - 1, &tp[2])) != NMZ_OK) {
                return s;
            }
            {
                nmz_trace_item it[9] = {{"x_t", &tp[0]}, {"h_prev", &tp[1]}, {"c_prev", &tp[2]},
                                        {"i", &tp[3]},   {"f", &tp[4]},      {"g", &tp[5]},
                                        {"o", &tp[6]},   {"c_t", &tp[7]},    {"h_t", &tp[8]}};
                nmz_trace_emit_step(ctx, NMZ_TRACE_FORWARD, "lstm", "i f g o", t, 9, it);
            }
        }
    }
    return NMZ_OK;
}

static nmz_status lstm_backward(nmz_layer *l, const nmz_tensor *x, const nmz_tensor *y, const nmz_tensor *dy,
                                nmz_tensor *dx, const nmz_ctx *ctx) {
    lstm_layer *m = l->self;
    int B = x->shape[0], T = x->shape[1], H = m->units, D = m->D, G = 4 * H, bi, t, i, j;
    int tracing = NMZ_TRACING(ctx), shape[2];
    nmz_tensor *tp = m->tmp;
    nmz_status s;
    float *da = m->da;
    shape[0] = B;
    shape[1] = H;
    if ((s = nmz_tensor_reshape(&m->dh_next, 2, shape)) != NMZ_OK || (s = nmz_tensor_reshape(&m->dc_next, 2, shape)) != NMZ_OK)
        return s;
    nmz_tensor_zero(&m->dh_next); /* nothing comes from beyond the last step */
    nmz_tensor_zero(&m->dc_next);
    if (tracing) {
        for (i = 0; i < 8; i++) {
            shape[1] = i == 5 ? G : H;
            if ((s = nmz_tensor_reshape(&tp[i], 2, shape)) != NMZ_OK) return s;
        }
    }
    for (t = T - 1; t >= 0; t--) {
        for (bi = 0; bi < B; bi++) {
            size_t row = (size_t)bi * T + (size_t)t, rb = (size_t)bi * H;
            const float *xt = x->data + row * D;
            const float *hp = t ? y->data + (row - 1) * H : m->h0.data + rb;
            const float *cp = t ? m->c.data + (row - 1) * H : m->c0.data + rb;
            const float *gt = m->gates.data + row * G, *ct = m->c.data + row * H;
            const float *gout = dy->data + row * H;
            float *dhn = m->dh_next.data + rb, *dcn = m->dc_next.data + rb;
            for (j = 0; j < H; j++) {
                float ig = gt[j], fg = gt[H + j], gg = gt[2 * H + j], og = gt[3 * H + j];
                float tc = tanhf(ct[j]);
                float dh = gout[j] + dhn[j];               /* from above at t + from step t+1 */
                float dc = dcn[j] + dh * og * (1.0f - tc * tc); /* along c + through h_t = o·tanh(c_t) */
                da[j] = dc * gg * ig * (1.0f - ig);
                da[H + j] = dc * cp[j] * fg * (1.0f - fg);
                da[2 * H + j] = dc * ig * (1.0f - gg * gg);
                da[3 * H + j] = dh * tc * og * (1.0f - og);
                if (tracing) {
                    tp[0].data[rb + j] = gout[j];
                    tp[1].data[rb + j] = dhn[j];
                    tp[2].data[rb + j] = dh;
                    tp[3].data[rb + j] = dcn[j];
                    tp[4].data[rb + j] = dc;
                    tp[7].data[rb + j] = dc * fg;
                }
                dcn[j] = dc * fg; /* ∂L/∂c_{t−1}: only the forget gate in between */
            }
            if (tracing) memcpy(tp[5].data + (size_t)bi * G, da, (size_t)G * sizeof(float));
            for (j = 0; j < G; j++) m->db.data[j] += da[j];
            for (i = 0; i < D; i++) {
                float *dw = m->dWx.data + (size_t)i * G, xi = xt[i];
                if (xi != 0.0f)
                    for (j = 0; j < G; j++) dw[j] += xi * da[j];
            }
            for (i = 0; i < H; i++) {
                float *dw = m->dWh.data + (size_t)i * G, hi = hp[i];
                if (hi != 0.0f)
                    for (j = 0; j < G; j++) dw[j] += hi * da[j];
            }
            if (dx) {
                float *dxt = dx->data + row * D;
                for (i = 0; i < D; i++) dxt[i] = (float)nmz__dot(m->Wx.data + (size_t)i * G, da, G);
            }
            for (i = 0; i < H; i++) { /* ∂L/∂h_{t−1} = ∂L/∂a·W_hᵀ, overwrites dh_next */
                dhn[i] = (float)nmz__dot(m->Wh.data + (size_t)i * G, da, G);
                if (tracing) tp[6].data[rb + i] = dhn[i];
            }
        }
        if (tracing) {
            nmz_trace_item it[8] = {{"dL/dh_out", &tp[0]},  {"dL/dh_next", &tp[1]}, {"dL/dh", &tp[2]},
                                    {"dL/dc_next", &tp[3]}, {"dL/dc", &tp[4]},      {"dL/da", &tp[5]},
                                    {"dL/dh_prev", &tp[6]}, {"dL/dc_prev", &tp[7]}};
            nmz_trace_emit_step(ctx, NMZ_TRACE_BACKWARD, "lstm", "i f g o", t, 8, it);
        }
    }
    if (tracing) {
        nmz_trace_scalar sc[1] = {{"steps", T}};
        nmz_trace_item all[5] = {{"dL/dW_x", &m->dWx}, {"dL/dW_h", &m->dWh}, {"dL/db", &m->db},
                                 {"W_h", &m->Wh},      {"dL/dx", dx}};
        nmz_trace_emit(ctx, NMZ_TRACE_BACKWARD, "lstm", "i f g o", 1, sc, dx ? 5 : 4, all);
    }
    return NMZ_OK;
}

static int lstm_param_count(const nmz_layer *l) {
    (void)l;
    return 3;
}

static nmz_param lstm_param(nmz_layer *l, int i) {
    lstm_layer *m = l->self;
    nmz_param p = {NULL, NULL, NULL};
    if (i == 0) {
        p.name = "W_x";
        p.value = &m->Wx;
        p.grad = &m->dWx;
    } else if (i == 1) {
        p.name = "W_h";
        p.value = &m->Wh;
        p.grad = &m->dWh;
    } else if (i == 2) {
        p.name = "b";
        p.value = &m->b;
        p.grad = &m->db;
    }
    return p;
}

static nmz_status lstm_save(const nmz_layer *l, nmz_writer *w) {
    const lstm_layer *m = l->self;
    nmz_write_u32(w, (uint32_t)m->units);
    nmz_write_u32(w, (uint32_t)m->init);
    nmz_write_u32(w, (uint32_t)m->resolved);
    return nmz_write_f32(w, m->forget_bias);
}

static nmz_status lstm_load(nmz_layer **out, nmz_reader *rd) {
    uint32_t units = 0, init = 0, resolved = 0;
    float fb = 0;
    nmz_read_u32(rd, &units);
    nmz_read_u32(rd, &init);
    nmz_read_u32(rd, &resolved);
    if (nmz_read_f32(rd, &fb) != NMZ_OK) return rd->err;
    if (units == 0 || units > 100000 || init >= NMZ_INIT_COUNT || resolved >= NMZ_INIT_COUNT || !isfinite(fb) ||
        fabsf(fb) > 100.0f)
        return nmz__fail(NMZ_ERR_FORMAT, "corrupt lstm config");
    *out = nmz_layer_lstm((int)units, (nmz_init)init, fb);
    if (!*out) return NMZ_ERR_NOMEM;
    ((lstm_layer *)(*out)->self)->resolved = (nmz_init)resolved;
    return NMZ_OK;
}

static void lstm_describe(const nmz_layer *l, char *buf, size_t n) {
    const lstm_layer *m = l->self;
    snprintf(buf, n, "lstm %d->%d (gates i f g o, forget bias %g, %s init)", m->D, m->units, (double)m->forget_bias,
             nmz_init_name(m->resolved));
}

static nmz_tensor *lstm_state(nmz_layer *l) { return &((lstm_layer *)l->self)->state; }

static void lstm_destroy(nmz_layer *l) {
    lstm_layer *m = l->self;
    int i;
    if (!m) return;
    nmz_tensor_free(&m->Wx);
    nmz_tensor_free(&m->Wh);
    nmz_tensor_free(&m->b);
    nmz_tensor_free(&m->dWx);
    nmz_tensor_free(&m->dWh);
    nmz_tensor_free(&m->db);
    nmz_tensor_free(&m->h0);
    nmz_tensor_free(&m->c0);
    nmz_tensor_free(&m->state);
    nmz_tensor_free(&m->gates);
    nmz_tensor_free(&m->c);
    nmz_tensor_free(&m->dh_next);
    nmz_tensor_free(&m->dc_next);
    for (i = 0; i < 12; i++) nmz_tensor_free(&m->tmp[i]);
    free(m->acc);
    free(m->da);
    free(m);
}

const nmz_layer_ops nmz__lstm_ops = {"lstm",          lstm_setup,  lstm_init,  lstm_forward,  lstm_backward,
                                     lstm_param_count, lstm_param, lstm_save,  lstm_load,     lstm_describe,
                                     lstm_destroy,    lstm_state,  0, NULL};

nmz_layer *nmz_layer_lstm(int units, nmz_init init, float forget_bias) {
    lstm_layer *m;
    nmz_layer *l;
    if (units < 1 || units > 100000 || (int)init < 0 || init >= NMZ_INIT_COUNT) {
        nmz__fail(NMZ_ERR_ARG, "lstm needs a positive number of units");
        return NULL;
    }
    if (!isfinite(forget_bias) || fabsf(forget_bias) > 100.0f) {
        nmz__fail(NMZ_ERR_ARG, "lstm forget bias must be a number in -100..100");
        return NULL;
    }
    m = calloc(1, sizeof *m);
    if (!m) {
        nmz__fail(NMZ_ERR_NOMEM, "out of memory");
        return NULL;
    }
    m->units = units;
    m->init = m->resolved = init;
    m->forget_bias = forget_bias;
    l = nmz_layer_alloc(&nmz__lstm_ops, m);
    if (!l) free(m);
    return l;
}

float nmz__lstm_forget_bias(const nmz_layer *l) { return ((const lstm_layer *)l->self)->forget_bias; }

int nmz__lstm_init_used(const nmz_layer *l) { return (int)((const lstm_layer *)l->self)->resolved; }
