/* recurrent.c — Embedding and the vanilla RNN layer.
 *
 * Embedding   ids [B, T] → vectors [B, T, D]:  e = W[id]          (W is [V, D])
 *             backward: ∂L/∂W[id] += ∂L/∂e  — only rows of tokens that occur
 *             (a sparse gradient); token ids have no gradient.
 *
 * RNN         x [B, T, D] → h [B, T, H], for t = 0 … T−1:
 *                 z_t = x_t·W_xh + h_{t−1}·W_hh + b        h_t = tanh(z_t)
 *             (row vectors, as in Dense: x·W ≡ Wᵀx). h_{−1} = 0, or the state
 *             carried over from the previous batch (truncated BPTT).
 *             backward through time, t = T−1 … 0:
 *                 ∂L/∂h_t  = (from the layer above at t) + (from step t+1)
 *                 δ_t      = ∂L/∂h_t ⊙ (1 − h_t²)                    (tanh')
 *                 ∂L/∂W_xh += x_tᵀ·δ_t   ∂L/∂W_hh += h_{t−1}ᵀ·δ_t   ∂L/∂b += Σ δ_t
 *                 ∂L/∂x_t  = δ_t·W_xhᵀ   ∂L/∂h_{t−1} = δ_t·W_hhᵀ  (→ step t−1)
 *             The gradient that would flow into the carried state h_{−1} is
 *             dropped: that is the truncation. The time loop and BPTT live
 *             entirely inside this layer (DESIGN §3); the engine sees one
 *             forward and one backward call.
 *
 * Also the helpers shared with lstm.c and gru.c: orthogonal initialisation,
 * slicing one time step out of [B, T, F], and the recurrent init rules. */
#include "internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================== Embedding */

typedef struct embed_layer {
    int vocab, dim;
    nmz_init init, resolved;
    nmz_tensor W, dW;
} embed_layer;

static nmz_status embed_setup(nmz_layer *l, int in_rank, const int *in_shape, int *out_rank, int *out_shape) {
    embed_layer *e = l->self;
    int ws[2];
    nmz_status s;
    if (in_rank != 1)
        return nmz__fail(NMZ_ERR_SHAPE, "embedding takes token ids [steps] per sample (got rank %d)", in_rank);
    ws[0] = e->vocab;
    ws[1] = e->dim;
    if ((s = nmz_tensor_init(&e->W, 2, ws)) != NMZ_OK) return s;
    if ((s = nmz_tensor_init(&e->dW, 2, ws)) != NMZ_OK) return s;
    *out_rank = 2;
    out_shape[0] = in_shape[0];
    out_shape[1] = e->dim;
    return NMZ_OK;
}

static nmz_status embed_init(nmz_layer *l, nmz_rng *rng, const nmz_layer *next) {
    embed_layer *e = l->self;
    double limit;
    size_t i;
    (void)next;
    /* AUTO = Glorot with fan_in = V, fan_out = D: exactly what a dense layer
     * on one-hot vectors would get (the embedding IS that layer) */
    e->resolved = e->init == NMZ_INIT_AUTO ? NMZ_INIT_XAVIER : e->init;
    limit = e->resolved == NMZ_INIT_HE ? sqrt(6.0 / e->vocab) : sqrt(6.0 / (e->vocab + e->dim));
    for (i = 0; i < e->W.size; i++) e->W.data[i] = (float)((2.0 * nmz_rng_uniform(rng) - 1.0) * limit);
    return NMZ_OK;
}

static nmz_status token_of(const embed_layer *e, float v, int *id) {
    if (!(v >= 0.0f && v < (float)e->vocab) || v != floorf(v))
        return nmz__fail(NMZ_ERR_ARG, "token id %g is not an integer in 0..%d (vocabulary size %d)", (double)v,
                         e->vocab - 1, e->vocab);
    *id = (int)v;
    return NMZ_OK;
}

static nmz_status embed_forward(nmz_layer *l, const nmz_tensor *x, nmz_tensor *y, const nmz_ctx *ctx) {
    embed_layer *e = l->self;
    size_t i;
    nmz_status s;
    for (i = 0; i < x->size; i++) {
        int id = 0;
        if ((s = token_of(e, x->data[i], &id)) != NMZ_OK) return s;
        memcpy(y->data + i * (size_t)e->dim, e->W.data + (size_t)id * e->dim, (size_t)e->dim * sizeof(float));
    }
    if (NMZ_TRACING(ctx)) {
        nmz_trace_item it[3] = {{"ids", x}, {"W", &e->W}, {"e", y}};
        nmz_trace_emit(ctx, NMZ_TRACE_FORWARD, "embedding", "", 0, NULL, 3, it);
    }
    return NMZ_OK;
}

static nmz_status embed_backward(nmz_layer *l, const nmz_tensor *x, const nmz_tensor *y, const nmz_tensor *dy,
                                 nmz_tensor *dx, const nmz_ctx *ctx) {
    embed_layer *e = l->self;
    size_t i;
    int k;
    (void)y;
    for (i = 0; i < x->size; i++) { /* sparse: only the rows of tokens that occur */
        float *row = e->dW.data + (size_t)x->data[i] * e->dim;
        const float *g = dy->data + i * (size_t)e->dim;
        for (k = 0; k < e->dim; k++) row[k] += g[k];
    }
    if (dx) nmz_tensor_zero(dx); /* token ids are not differentiable */
    if (NMZ_TRACING(ctx)) {
        nmz_trace_item it[3] = {{"dL/de", dy}, {"ids", x}, {"dL/dW", &e->dW}};
        nmz_trace_emit(ctx, NMZ_TRACE_BACKWARD, "embedding", "", 0, NULL, 3, it);
    }
    return NMZ_OK;
}

static int embed_param_count(const nmz_layer *l) {
    (void)l;
    return 1;
}

static nmz_param embed_param(nmz_layer *l, int i) {
    embed_layer *e = l->self;
    nmz_param p = {NULL, NULL, NULL};
    if (i == 0) {
        p.name = "W";
        p.value = &e->W;
        p.grad = &e->dW;
    }
    return p;
}

static nmz_status embed_save(const nmz_layer *l, nmz_writer *w) {
    const embed_layer *e = l->self;
    nmz_write_u32(w, (uint32_t)e->vocab);
    nmz_write_u32(w, (uint32_t)e->dim);
    nmz_write_u32(w, (uint32_t)e->init);
    return nmz_write_u32(w, (uint32_t)e->resolved);
}

static nmz_status embed_load(nmz_layer **out, nmz_reader *r) {
    uint32_t v[4];
    int i;
    for (i = 0; i < 4; i++) nmz_read_u32(r, &v[i]);
    if (r->err) return r->err;
    if (v[0] == 0 || v[0] > NMZ_MAX_VOCAB || v[1] == 0 || v[1] > 100000 || v[2] >= NMZ_INIT_COUNT ||
        v[3] >= NMZ_INIT_COUNT)
        return nmz__fail(NMZ_ERR_FORMAT, "corrupt embedding config");
    *out = nmz_layer_embedding((int)v[0], (int)v[1], (nmz_init)v[2]);
    if (!*out) return NMZ_ERR_NOMEM;
    ((embed_layer *)(*out)->self)->resolved = (nmz_init)v[3];
    return NMZ_OK;
}

static void embed_describe(const nmz_layer *l, char *buf, size_t n) {
    const embed_layer *e = l->self;
    snprintf(buf, n, "embedding %d tokens -> %d (%s init)", e->vocab, e->dim, nmz_init_name(e->resolved));
}

static void embed_destroy(nmz_layer *l) {
    embed_layer *e = l->self;
    if (!e) return;
    nmz_tensor_free(&e->W);
    nmz_tensor_free(&e->dW);
    free(e);
}

const nmz_layer_ops nmz__embedding_ops = {"embedding",  embed_setup,       embed_init,   embed_forward,
                                          embed_backward, embed_param_count, embed_param, embed_save,
                                          embed_load,   embed_describe,    embed_destroy, NULL,
                                          NMZ_LAYER_DISCRETE_INPUT};

nmz_layer *nmz_layer_embedding(int vocab, int dim, nmz_init init) {
    embed_layer *e;
    nmz_layer *l;
    if (vocab < 1 || vocab > NMZ_MAX_VOCAB || dim < 1 || dim > 100000 || (int)init < 0 || init >= NMZ_INIT_COUNT ||
        init == NMZ_INIT_ORTHOGONAL) {
        nmz__fail(NMZ_ERR_ARG, "embedding needs a vocabulary size 1..%d, a dimension >= 1 and init auto, xavier or he",
                  NMZ_MAX_VOCAB);
        return NULL;
    }
    e = calloc(1, sizeof *e);
    if (!e) {
        nmz__fail(NMZ_ERR_NOMEM, "out of memory");
        return NULL;
    }
    e->vocab = vocab;
    e->dim = dim;
    e->init = e->resolved = init;
    l = nmz_layer_alloc(&nmz__embedding_ops, e);
    if (!l) free(e);
    return l;
}

/* ===================================================================== RNN */

typedef struct rnn_layer {
    int units, D;
    nmz_init init, resolved;
    nmz_tensor Wxh, Whh, b, dWxh, dWhh, db;
    nmz_tensor h0;    /* initial state used by the last forward [B, H] */
    nmz_tensor state; /* carried state (last h of the previous forward), or empty */
    nmz_tensor z;     /* pre-activations [B, T, H], kept only while tracing */
    nmz_tensor dnext; /* ∂L/∂h_{t} arriving from step t+1, [B, H] */
    nmz_tensor tmp[6];
    double *acc;
} rnn_layer;

static nmz_status rnn_setup(nmz_layer *l, int in_rank, const int *in_shape, int *out_rank, int *out_shape) {
    rnn_layer *r = l->self;
    int s2[2];
    nmz_status s;
    if (in_rank != 2)
        return nmz__fail(NMZ_ERR_SHAPE, "rnn needs [steps, features] per sample (e.g. after an embedding), got rank %d",
                         in_rank);
    r->D = in_shape[1];
    s2[0] = r->D;
    s2[1] = r->units;
    if ((s = nmz_tensor_init(&r->Wxh, 2, s2)) != NMZ_OK || (s = nmz_tensor_init(&r->dWxh, 2, s2)) != NMZ_OK) return s;
    s2[0] = r->units;
    if ((s = nmz_tensor_init(&r->Whh, 2, s2)) != NMZ_OK || (s = nmz_tensor_init(&r->dWhh, 2, s2)) != NMZ_OK) return s;
    if ((s = nmz_tensor_init(&r->b, 1, &r->units)) != NMZ_OK || (s = nmz_tensor_init(&r->db, 1, &r->units)) != NMZ_OK)
        return s;
    free(r->acc);
    r->acc = malloc((size_t)(r->units > r->D ? r->units : r->D) * sizeof *r->acc);
    if (!r->acc) return nmz__fail(NMZ_ERR_NOMEM, "out of memory");
    *out_rank = 2;
    out_shape[0] = in_shape[0];
    out_shape[1] = r->units;
    return NMZ_OK;
}

/* Orthogonal n×n block: columns col0 … col0+n−1 of a row-major [n, ld]
 * matrix. A Gaussian matrix is orthonormalised column by column (modified
 * Gram–Schmidt in double, two passes for float accuracy): that is the Q of
 * its QR decomposition with a positive diagonal R, a uniformly random
 * (Haar) orthogonal matrix. All its singular values are 1, so repeated
 * multiplication by it neither shrinks nor grows a vector. */
nmz_status nmz__orthogonal_block(float *W, int n, int ld, int col0, nmz_rng *rng) {
    double *q = malloc((size_t)n * (size_t)n * sizeof *q); /* q[c·n + r] = column c */
    int c, k, r, pass;
    if (!q) return nmz__fail(NMZ_ERR_NOMEM, "out of memory");
    for (c = 0; c < n; c++)
        for (r = 0; r < n; r++) q[(size_t)c * n + r] = nmz_rng_normal(rng);
    for (c = 0; c < n; c++) {
        double *qc = q + (size_t)c * n, len;
        for (pass = 0; pass < 2; pass++)
            for (k = 0; k < c; k++) {
                const double *qk = q + (size_t)k * n;
                double dot = 0;
                for (r = 0; r < n; r++) dot += qk[r] * qc[r];
                for (r = 0; r < n; r++) qc[r] -= dot * qk[r];
            }
        len = 0;
        for (r = 0; r < n; r++) len += qc[r] * qc[r];
        len = sqrt(len);
        if (len < 1e-12) { /* (probability 0) a dependent column: draw it again */
            for (r = 0; r < n; r++) qc[r] = nmz_rng_normal(rng);
            c--;
            continue;
        }
        for (r = 0; r < n; r++) qc[r] /= len;
    }
    for (r = 0; r < n; r++)
        for (c = 0; c < n; c++) W[(size_t)r * ld + col0 + c] = (float)q[(size_t)c * n + r];
    free(q);
    return NMZ_OK;
}

/* Recurrent weights: input matrix Wx [D, G·H] and recurrent matrix Wh
 * [H, G·H] for G gates (1 for the RNN). AUTO/ORTHOGONAL: Wx Glorot (fan
 * D + H per gate), each H×H gate block of Wh orthogonal. XAVIER: Glorot for
 * both (Wh entries with variance 1/H: spectral radius ≈ 1, but singular
 * values spread from ≈0 to ≈2). HE: ±sqrt(6/fan_in). */
nmz_status nmz__recurrent_init(nmz_init init, nmz_init *resolved, nmz_tensor *Wx, nmz_tensor *Wh, int D, int H,
                               int gates, nmz_rng *rng) {
    double lx, lh;
    size_t i;
    int g;
    nmz_status s;
    *resolved = init == NMZ_INIT_AUTO ? NMZ_INIT_ORTHOGONAL : init;
    if (*resolved == NMZ_INIT_HE) {
        lx = sqrt(6.0 / D);
        lh = sqrt(6.0 / H);
    } else {
        lx = sqrt(6.0 / (D + H));
        lh = sqrt(6.0 / (2.0 * H));
    }
    for (i = 0; i < Wx->size; i++) Wx->data[i] = (float)((2.0 * nmz_rng_uniform(rng) - 1.0) * lx);
    if (*resolved == NMZ_INIT_ORTHOGONAL) {
        for (g = 0; g < gates; g++)
            if ((s = nmz__orthogonal_block(Wh->data, H, gates * H, g * H, rng)) != NMZ_OK) return s;
    } else {
        for (i = 0; i < Wh->size; i++) Wh->data[i] = (float)((2.0 * nmz_rng_uniform(rng) - 1.0) * lh);
    }
    return NMZ_OK;
}

static nmz_status rnn_init(nmz_layer *l, nmz_rng *rng, const nmz_layer *next) {
    rnn_layer *r = l->self;
    (void)next;
    nmz_tensor_zero(&r->b);
    return nmz__recurrent_init(r->init, &r->resolved, &r->Wxh, &r->Whh, r->D, r->units, 1, rng);
}

nmz_status nmz__slice_tc(const nmz_tensor *src, int t, int col0, int n, nmz_tensor *dst) {
    int B = src->shape[0], T = src->shape[1], F = src->shape[2], bi, shape[2];
    nmz_status s;
    shape[0] = B;
    shape[1] = n;
    if ((s = nmz_tensor_reshape(dst, 2, shape)) != NMZ_OK) return s;
    for (bi = 0; bi < B; bi++)
        memcpy(dst->data + (size_t)bi * n, src->data + ((size_t)bi * T + (size_t)t) * F + col0,
               (size_t)n * sizeof(float));
    return NMZ_OK;
}

nmz_status nmz__slice_t(const nmz_tensor *src, int t, nmz_tensor *dst) {
    return nmz__slice_tc(src, t, 0, src->shape[2], dst);
}

static nmz_status rnn_forward(nmz_layer *l, const nmz_tensor *x, nmz_tensor *y, const nmz_ctx *ctx) {
    rnn_layer *r = l->self;
    int B = x->shape[0], T = x->shape[1], H = r->units, D = r->D, bi, t, i, j, tracing = NMZ_TRACING(ctx);
    int shape[2];
    nmz_status s;
    shape[0] = B;
    shape[1] = H;
    if ((s = nmz_tensor_reshape(&r->h0, 2, shape)) != NMZ_OK) return s;
    if (ctx->carry && r->state.size == (size_t)B * H)
        memcpy(r->h0.data, r->state.data, r->h0.size * sizeof(float)); /* continue the stream */
    else
        nmz_tensor_zero(&r->h0);
    if (tracing && (s = nmz_tensor_reshape(&r->z, 3, y->shape)) != NMZ_OK) return s;
    for (bi = 0; bi < B; bi++)
        for (t = 0; t < T; t++) {
            const float *xt = x->data + ((size_t)bi * T + (size_t)t) * D;
            const float *hp = t ? y->data + ((size_t)bi * T + (size_t)t - 1) * H : r->h0.data + (size_t)bi * H;
            float *ht = y->data + ((size_t)bi * T + (size_t)t) * H;
            for (j = 0; j < H; j++) r->acc[j] = r->b.data[j];
            for (i = 0; i < D; i++) {
                const float *w = r->Wxh.data + (size_t)i * H;
                double xi = xt[i];
                if (xi != 0.0)
                    for (j = 0; j < H; j++) r->acc[j] += xi * w[j];
            }
            for (i = 0; i < H; i++) {
                const float *w = r->Whh.data + (size_t)i * H;
                double hi = hp[i];
                if (hi != 0.0)
                    for (j = 0; j < H; j++) r->acc[j] += hi * w[j];
            }
            for (j = 0; j < H; j++) {
                if (tracing) r->z.data[((size_t)bi * T + (size_t)t) * H + (size_t)j] = (float)r->acc[j];
                ht[j] = (float)tanh(r->acc[j]);
            }
        }
    if (ctx->carry) { /* remember h_{T−1} for the next batch of the same streams */
        if ((s = nmz_tensor_reshape(&r->state, 2, shape)) != NMZ_OK) return s;
        for (bi = 0; bi < B; bi++)
            memcpy(r->state.data + (size_t)bi * H, y->data + ((size_t)bi * T + (size_t)T - 1) * H,
                   (size_t)H * sizeof(float));
    }
    if (tracing) {
        nmz_trace_scalar sc[1] = {{"steps", T}};
        nmz_trace_item all[6] = {{"x", x}, {"h0", &r->h0}, {"W_xh", &r->Wxh}, {"W_hh", &r->Whh}, {"b", &r->b}, {"h", y}};
        nmz_ctx c = *ctx;
        nmz_trace_emit(ctx, NMZ_TRACE_FORWARD, "rnn", "tanh", 1, sc, 6, all);
        for (t = 0; t < T; t++) { /* one event per time step: the unrolled network */
            nmz_tensor *xt = &r->tmp[0], *hp = &r->tmp[1], *zt = &r->tmp[2], *ht = &r->tmp[3];
            if ((s = nmz__slice_t(x, t, xt)) != NMZ_OK || (s = nmz__slice_t(&r->z, t, zt)) != NMZ_OK ||
                (s = nmz__slice_t(y, t, ht)) != NMZ_OK)
                return s;
            if (t == 0) {
                if ((s = nmz_tensor_copy(hp, &r->h0)) != NMZ_OK) return s;
            } else if ((s = nmz__slice_t(y, t - 1, hp)) != NMZ_OK) {
                return s;
            }
            {
                nmz_trace_item it[4] = {{"x_t", xt}, {"h_prev", hp}, {"z_t", zt}, {"h_t", ht}};
                nmz_trace_emit_step(&c, NMZ_TRACE_FORWARD, "rnn", "tanh", t, 4, it);
            }
        }
    }
    return NMZ_OK;
}

static nmz_status rnn_backward(nmz_layer *l, const nmz_tensor *x, const nmz_tensor *y, const nmz_tensor *dy,
                               nmz_tensor *dx, const nmz_ctx *ctx) {
    rnn_layer *r = l->self;
    int B = x->shape[0], T = x->shape[1], H = r->units, D = r->D, bi, t, i, j, tracing = NMZ_TRACING(ctx);
    int shape[2];
    nmz_status s;
    float *dz = NULL;
    shape[0] = B;
    shape[1] = H;
    if ((s = nmz_tensor_reshape(&r->dnext, 2, shape)) != NMZ_OK) return s;
    nmz_tensor_zero(&r->dnext); /* nothing comes from beyond the last step */
    if (tracing) {
        for (i = 0; i < 5; i++)
            if ((s = nmz_tensor_reshape(&r->tmp[i], 2, shape)) != NMZ_OK) return s;
    }
    dz = malloc((size_t)H * sizeof *dz);
    if (!dz) return nmz__fail(NMZ_ERR_NOMEM, "out of memory");
    for (t = T - 1; t >= 0; t--) {
        for (bi = 0; bi < B; bi++) {
            const float *xt = x->data + ((size_t)bi * T + (size_t)t) * D;
            const float *ht = y->data + ((size_t)bi * T + (size_t)t) * H;
            const float *hp = t ? y->data + ((size_t)bi * T + (size_t)t - 1) * H : r->h0.data + (size_t)bi * H;
            const float *gout = dy->data + ((size_t)bi * T + (size_t)t) * H;
            float *gnext = r->dnext.data + (size_t)bi * H;
            for (j = 0; j < H; j++) {
                float dh = gout[j] + gnext[j]; /* from above at t + from step t+1 */
                dz[j] = dh * (1.0f - ht[j] * ht[j]);
                if (tracing) {
                    r->tmp[0].data[(size_t)bi * H + j] = gout[j];
                    r->tmp[1].data[(size_t)bi * H + j] = gnext[j];
                    r->tmp[2].data[(size_t)bi * H + j] = dh;
                    r->tmp[3].data[(size_t)bi * H + j] = dz[j];
                }
            }
            for (j = 0; j < H; j++) r->db.data[j] += dz[j];
            for (i = 0; i < D; i++) {
                float *dw = r->dWxh.data + (size_t)i * H, xi = xt[i];
                if (xi != 0.0f)
                    for (j = 0; j < H; j++) dw[j] += xi * dz[j];
            }
            for (i = 0; i < H; i++) {
                float *dw = r->dWhh.data + (size_t)i * H, hi = hp[i];
                if (hi != 0.0f)
                    for (j = 0; j < H; j++) dw[j] += hi * dz[j];
            }
            if (dx) {
                float *dxt = dx->data + ((size_t)bi * T + (size_t)t) * D;
                for (i = 0; i < D; i++) {
                    const float *w = r->Wxh.data + (size_t)i * H;
                    double acc = 0.0;
                    for (j = 0; j < H; j++) acc += (double)w[j] * dz[j];
                    dxt[i] = (float)acc;
                }
            }
            for (i = 0; i < H; i++) { /* ∂L/∂h_{t−1} = δ_t·W_hhᵀ, overwrites gnext */
                const float *w = r->Whh.data + (size_t)i * H;
                double acc = 0.0;
                for (j = 0; j < H; j++) acc += (double)w[j] * dz[j];
                r->acc[i] = acc;
            }
            for (i = 0; i < H; i++) {
                gnext[i] = (float)r->acc[i];
                if (tracing) r->tmp[4].data[(size_t)bi * H + i] = gnext[i];
            }
        }
        if (tracing) {
            nmz_trace_item it[5] = {{"dL/dh_out", &r->tmp[0]}, {"dL/dh_next", &r->tmp[1]}, {"dL/dh", &r->tmp[2]},
                                    {"dL/dz", &r->tmp[3]},     {"dL/dh_prev", &r->tmp[4]}};
            nmz_trace_emit_step(ctx, NMZ_TRACE_BACKWARD, "rnn", "tanh", t, 5, it);
        }
    }
    free(dz);
    if (tracing) {
        nmz_trace_scalar sc[1] = {{"steps", T}};
        nmz_trace_item all[5] = {{"dL/dW_xh", &r->dWxh}, {"dL/dW_hh", &r->dWhh}, {"dL/db", &r->db},
                                 {"W_hh", &r->Whh},      {"dL/dx", dx}};
        nmz_trace_emit(ctx, NMZ_TRACE_BACKWARD, "rnn", "tanh", 1, sc, dx ? 5 : 4, all);
    }
    return NMZ_OK;
}

static int rnn_param_count(const nmz_layer *l) {
    (void)l;
    return 3;
}

static nmz_param rnn_param(nmz_layer *l, int i) {
    rnn_layer *r = l->self;
    nmz_param p = {NULL, NULL, NULL};
    if (i == 0) {
        p.name = "W_xh";
        p.value = &r->Wxh;
        p.grad = &r->dWxh;
    } else if (i == 1) {
        p.name = "W_hh";
        p.value = &r->Whh;
        p.grad = &r->dWhh;
    } else if (i == 2) {
        p.name = "b";
        p.value = &r->b;
        p.grad = &r->db;
    }
    return p;
}

static nmz_status rnn_save(const nmz_layer *l, nmz_writer *w) {
    const rnn_layer *r = l->self;
    nmz_write_u32(w, (uint32_t)r->units);
    nmz_write_u32(w, (uint32_t)r->init);
    return nmz_write_u32(w, (uint32_t)r->resolved);
}

static nmz_status rnn_load(nmz_layer **out, nmz_reader *rd) {
    uint32_t units = 0, init = 0, resolved = 0;
    nmz_read_u32(rd, &units);
    nmz_read_u32(rd, &init);
    if (nmz_read_u32(rd, &resolved) != NMZ_OK) return rd->err;
    if (units == 0 || units > 100000 || init >= NMZ_INIT_COUNT || resolved >= NMZ_INIT_COUNT)
        return nmz__fail(NMZ_ERR_FORMAT, "corrupt rnn config");
    *out = nmz_layer_rnn((int)units, (nmz_init)init);
    if (!*out) return NMZ_ERR_NOMEM;
    ((rnn_layer *)(*out)->self)->resolved = (nmz_init)resolved;
    return NMZ_OK;
}

static void rnn_describe(const nmz_layer *l, char *buf, size_t n) {
    const rnn_layer *r = l->self;
    snprintf(buf, n, "rnn %d->%d tanh (%s init)", r->D, r->units, nmz_init_name(r->resolved));
}

static nmz_tensor *rnn_state(nmz_layer *l) { return &((rnn_layer *)l->self)->state; }

static void rnn_destroy(nmz_layer *l) {
    rnn_layer *r = l->self;
    int i;
    if (!r) return;
    nmz_tensor_free(&r->Wxh);
    nmz_tensor_free(&r->Whh);
    nmz_tensor_free(&r->b);
    nmz_tensor_free(&r->dWxh);
    nmz_tensor_free(&r->dWhh);
    nmz_tensor_free(&r->db);
    nmz_tensor_free(&r->h0);
    nmz_tensor_free(&r->state);
    nmz_tensor_free(&r->z);
    nmz_tensor_free(&r->dnext);
    for (i = 0; i < 6; i++) nmz_tensor_free(&r->tmp[i]);
    free(r->acc);
    free(r);
}

const nmz_layer_ops nmz__rnn_ops = {"rnn",        rnn_setup,       rnn_init,  rnn_forward, rnn_backward,
                                    rnn_param_count, rnn_param,    rnn_save,  rnn_load,    rnn_describe,
                                    rnn_destroy,  rnn_state,       0};

nmz_layer *nmz_layer_rnn(int units, nmz_init init) {
    rnn_layer *r;
    nmz_layer *l;
    if (units < 1 || units > 100000 || (int)init < 0 || init >= NMZ_INIT_COUNT) {
        nmz__fail(NMZ_ERR_ARG, "rnn needs a positive number of units");
        return NULL;
    }
    r = calloc(1, sizeof *r);
    if (!r) {
        nmz__fail(NMZ_ERR_NOMEM, "out of memory");
        return NULL;
    }
    r->units = units;
    r->init = r->resolved = init;
    l = nmz_layer_alloc(&nmz__rnn_ops, r);
    if (!l) free(r);
    return l;
}

int nmz__recurrent_init_used(const nmz_layer *l) {
    if (l->ops == &nmz__embedding_ops) return (int)((const embed_layer *)l->self)->resolved;
    if (l->ops == &nmz__rnn_ops) return (int)((const rnn_layer *)l->self)->resolved;
    if (l->ops == &nmz__lstm_ops || l->ops == &nmz__gru_ops) return nmz__gated_init_used(l);
    return -1;
}

int nmz_layer_is_recurrent(const nmz_layer *l) {
    return l && (l->ops == &nmz__rnn_ops || l->ops == &nmz__lstm_ops || l->ops == &nmz__gru_ops);
}
