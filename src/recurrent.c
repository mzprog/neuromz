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
    {
        float *w = nmz__hw(&e->W);
        for (i = 0; i < e->W.size; i++) w[i] = (float)((2.0 * nmz_rng_uniform(rng) - 1.0) * limit);
    }
    return NMZ_OK;
}

static nmz_status token_of(int V, float v, int *id) {
    if (!(v >= 0.0f && v < (float)V) || v != floorf(v))
        return nmz__fail(NMZ_ERR_ARG, "token id %g is not an integer in 0..%d (vocabulary size %d)", (double)v, V - 1, V);
    *id = (int)v;
    return NMZ_OK;
}

/* The CPU kernels (the v2.0.1 loops). */
nmz_status nmz__cpu_embed_forward(const nmz_backend *be, size_t n, int V, int D, const float *ids, const float *W,
                                  float *y) {
    size_t i;
    nmz_status s;
    (void)be;
    for (i = 0; i < n; i++) {
        int id = 0;
        if ((s = token_of(V, ids[i], &id)) != NMZ_OK) return s;
        memcpy(y + i * (size_t)D, W + (size_t)id * D, (size_t)D * sizeof(float));
    }
    return NMZ_OK;
}

nmz_status nmz__cpu_embed_backward(const nmz_backend *be, size_t n, int V, int D, const float *ids, const float *dy,
                                   float *dW) {
    size_t i;
    int k;
    (void)be;
    (void)V;
    for (i = 0; i < n; i++) { /* sparse: only the rows of tokens that occur */
        float *row = dW + (size_t)ids[i] * D;
        const float *g = dy + i * (size_t)D;
        for (k = 0; k < D; k++) row[k] += g[k];
    }
    return NMZ_OK;
}

nmz_status nmz__check_ids(const nmz_layer *l, const float *ids, size_t n) {
    const embed_layer *e;
    size_t i;
    nmz_status s;
    int id;
    if (!l || l->ops != &nmz__embedding_ops) return NMZ_OK;
    e = l->self;
    for (i = 0; i < n; i++)
        if ((s = token_of(e->vocab, ids[i], &id)) != NMZ_OK) return s;
    return NMZ_OK;
}

static nmz_status embed_forward(nmz_layer *l, const nmz_tensor *x, nmz_tensor *y, const nmz_ctx *ctx) {
    embed_layer *e = l->self;
    const nmz_backend *be = ctx->backend;
    const float *px = nmz__r(be, x), *pW = nmz__r(be, &e->W);
    float *py = nmz__w(be, y);
    nmz_status s;
    if (!px || !pW || !py) return nmz__op_failed();
    if ((s = be->embed_forward(be, x->size, e->vocab, e->dim, px, pW, py)) != NMZ_OK) return s;
    if (NMZ_TRACING(ctx)) {
        nmz_trace_item it[3] = {{"ids", x}, {"W", &e->W}, {"e", y}};
        nmz_trace_emit(ctx, NMZ_TRACE_FORWARD, "embedding", "", 0, NULL, 3, it);
    }
    return NMZ_OK;
}

static nmz_status embed_backward(nmz_layer *l, const nmz_tensor *x, const nmz_tensor *y, const nmz_tensor *dy,
                                 nmz_tensor *dx, const nmz_ctx *ctx) {
    embed_layer *e = l->self;
    const nmz_backend *be = ctx->backend;
    const float *px = nmz__r(be, x), *pdy = nmz__r(be, dy);
    float *pdW = nmz__u(be, &e->dW);
    nmz_status s;
    (void)y;
    if (!px || !pdy || !pdW) return nmz__op_failed();
    if ((s = be->embed_backward(be, x->size, e->vocab, e->dim, px, pdy, pdW)) != NMZ_OK) return s;
    if (dx && (s = nmz__zero_on(be, dx)) != NMZ_OK) return s; /* token ids are not differentiable */
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
                                          NMZ_LAYER_DISCRETE_INPUT | NMZ_LAYER_BACKEND, NULL, NULL};

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
    nmz_tensor z;     /* pre-activations [B, T, H] (CPU kernel: only while tracing) */
    nmz_tensor dnext; /* ∂L/∂h_{t} arriving from step t+1, [B, H] */
    nmz_tensor tmp[6];
    nmz_tensor rec[4]; /* trace: dh_next, dh, dz, dh_prev per step [B, T, H] */
    nmz_tensor gdz, gprev; /* generic path: ∂L/∂z and h_{t−1} for every step [B, T, H] */
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
    {
        float *wx = nmz__hw(Wx), *wh = nmz__hw(Wh);
        for (i = 0; i < Wx->size; i++) wx[i] = (float)((2.0 * nmz_rng_uniform(rng) - 1.0) * lx);
        if (*resolved == NMZ_INIT_ORTHOGONAL) {
            for (g = 0; g < gates; g++)
                if ((s = nmz__orthogonal_block(wh, H, gates * H, g * H, rng)) != NMZ_OK) return s;
        } else {
            for (i = 0; i < Wh->size; i++) wh[i] = (float)((2.0 * nmz_rng_uniform(rng) - 1.0) * lh);
        }
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
    const float *sp = nmz__hr(src);
    float *dp;
    shape[0] = B;
    shape[1] = n;
    if ((s = nmz_tensor_reshape(dst, 2, shape)) != NMZ_OK) return s;
    dp = nmz__hw(dst);
    for (bi = 0; bi < B; bi++)
        memcpy(dp + (size_t)bi * n, sp + ((size_t)bi * T + (size_t)t) * F + col0, (size_t)n * sizeof(float));
    return NMZ_OK;
}

nmz_status nmz__slice_t(const nmz_tensor *src, int t, nmz_tensor *dst) {
    return nmz__slice_tc(src, t, 0, src->shape[2], dst);
}

/* The CPU kernels: the v2.0.1 loops (each row through all its steps; the
 * sums in double). With rec, the trace's per-step values are recorded
 * instead of being emitted from inside the loop. */
nmz_status nmz__cpu_rnn_forward(const nmz_backend *be, int B, int T, int D, int H, const float *x, const float *h0,
                                const float *Wxh, const float *Whh, const float *b, float *y, float *z) {
    double stack[1024], *acc = H <= 1024 ? stack : malloc((size_t)H * sizeof *acc);
    int bi, t, i, j;
    (void)be;
    if (!acc) return nmz__fail(NMZ_ERR_NOMEM, "out of memory");
    for (bi = 0; bi < B; bi++)
        for (t = 0; t < T; t++) {
            const float *xt = x + ((size_t)bi * T + (size_t)t) * D;
            const float *hp = t ? y + ((size_t)bi * T + (size_t)t - 1) * H : h0 + (size_t)bi * H;
            float *ht = y + ((size_t)bi * T + (size_t)t) * H;
            for (j = 0; j < H; j++) acc[j] = b[j];
            for (i = 0; i < D; i++) {
                const float *w = Wxh + (size_t)i * H;
                double xi = xt[i];
                if (xi != 0.0)
                    for (j = 0; j < H; j++) acc[j] += xi * w[j];
            }
            for (i = 0; i < H; i++) {
                const float *w = Whh + (size_t)i * H;
                double hi = hp[i];
                if (hi != 0.0)
                    for (j = 0; j < H; j++) acc[j] += hi * w[j];
            }
            for (j = 0; j < H; j++) {
                if (z) z[((size_t)bi * T + (size_t)t) * H + (size_t)j] = (float)acc[j];
                ht[j] = (float)tanh(acc[j]);
            }
        }
    if (acc != stack) free(acc);
    return NMZ_OK;
}

nmz_status nmz__cpu_rnn_backward(const nmz_backend *be, int B, int T, int D, int H, const float *x, const float *y,
                                 const float *h0, const float *Wxh, const float *Whh, const float *dy, float *dWxh,
                                 float *dWhh, float *db, float *dx, float *dh_next, const nmz_seq_record *rec) {
    double stack[1024], *acc = H <= 1024 ? stack : malloc((size_t)H * sizeof *acc);
    float *dz = malloc((size_t)H * sizeof *dz);
    int bi, t, i, j;
    (void)be;
    if (!acc || !dz) {
        if (acc != stack) free(acc);
        free(dz);
        return nmz__fail(NMZ_ERR_NOMEM, "out of memory");
    }
    memset(dh_next, 0, (size_t)B * H * sizeof(float)); /* nothing comes from beyond the last step */
    for (t = T - 1; t >= 0; t--) {
        for (bi = 0; bi < B; bi++) {
            const float *xt = x + ((size_t)bi * T + (size_t)t) * D;
            const float *ht = y + ((size_t)bi * T + (size_t)t) * H;
            const float *hp = t ? y + ((size_t)bi * T + (size_t)t - 1) * H : h0 + (size_t)bi * H;
            const float *gout = dy + ((size_t)bi * T + (size_t)t) * H;
            float *gnext = dh_next + (size_t)bi * H;
            size_t ro = ((size_t)bi * T + (size_t)t) * H; /* this step's row in the record */
            for (j = 0; j < H; j++) {
                float dh = gout[j] + gnext[j]; /* from above at t + from step t+1 */
                dz[j] = dh * (1.0f - ht[j] * ht[j]);
                if (rec) {
                    rec->dh_next[ro + j] = gnext[j];
                    rec->dh[ro + j] = dh;
                    rec->da[ro + j] = dz[j];
                }
            }
            for (j = 0; j < H; j++) db[j] += dz[j];
            for (i = 0; i < D; i++) {
                float *dw = dWxh + (size_t)i * H, xi = xt[i];
                if (xi != 0.0f)
                    for (j = 0; j < H; j++) dw[j] += xi * dz[j];
            }
            for (i = 0; i < H; i++) {
                float *dw = dWhh + (size_t)i * H, hi = hp[i];
                if (hi != 0.0f)
                    for (j = 0; j < H; j++) dw[j] += hi * dz[j];
            }
            if (dx) {
                float *dxt = dx + ((size_t)bi * T + (size_t)t) * D;
                for (i = 0; i < D; i++) dxt[i] = (float)nmz__dot(Wxh + (size_t)i * H, dz, H);
            }
            for (i = 0; i < H; i++) /* ∂L/∂h_{t−1} = δ_t·W_hhᵀ, overwrites gnext */
                acc[i] = nmz__dot(Whh + (size_t)i * H, dz, H);
            for (i = 0; i < H; i++) {
                gnext[i] = (float)acc[i];
                if (rec) rec->dh_prev[ro + i] = gnext[i];
            }
        }
    }
    free(dz);
    if (acc != stack) free(acc);
    return NMZ_OK;
}

/* h0 = the carried state (carry on and the batch size fits) or zeros */
static nmz_status initial_state(const nmz_backend *be, const nmz_ctx *ctx, nmz_tensor *h0, nmz_tensor *state, int B,
                                int H, size_t per_row, int col0) {
    int shape[2];
    nmz_status s;
    shape[0] = B;
    shape[1] = H;
    if ((s = nmz_tensor_reshape(h0, 2, shape)) != NMZ_OK) return s;
    if (ctx->carry && state->size == (size_t)B * per_row) { /* continue the stream */
        const float *ps = nmz__r(be, state);
        float *ph = nmz__w(be, h0);
        if (!ps || !ph) return nmz__op_failed();
        return be->copy(be, (size_t)B, (size_t)H, ps + col0, per_row, ph, (size_t)H);
    }
    return nmz__zero_on(be, h0);
}

/* the recurrent layers' generic forward step product: a_t += h_{t−1}·W_h (cols c0 … c0+n) */
nmz_status nmz__rec_product(const nmz_backend *be, int B, int T, int t, int H, int G, int c0, int n, const float *h0,
                            const float *y, const float *Wh, float *a) {
    const float *prev = t ? y + (size_t)(t - 1) * H : h0;
    int ldp = t ? T * H : H;
    return be->gemm(be, NMZ_MM_ACCUMULATE, B, n, H, prev, ldp, Wh + c0, G, a + (size_t)t * G + c0, T * G);
}

/* generic backward: h_{t−1} for every step, [B, T, H] (h0 at t = 0) */
nmz_status nmz__rec_prev(const nmz_backend *be, int B, int T, int H, const float *h0, const float *y, float *prev) {
    nmz_status s = NMZ_OK;
    if (T > 1) s = be->copy(be, (size_t)B, (size_t)(T - 1) * H, y, (size_t)T * H, prev + H, (size_t)T * H);
    if (s == NMZ_OK) s = be->copy(be, (size_t)B, (size_t)H, h0, (size_t)H, prev, (size_t)T * H);
    return s;
}

static nmz_status rnn_forward(nmz_layer *l, const nmz_tensor *x, nmz_tensor *y, const nmz_ctx *ctx) {
    rnn_layer *r = l->self;
    const nmz_backend *be = ctx->backend;
    int B = x->shape[0], T = x->shape[1], H = r->units, D = r->D, t, tracing = NMZ_TRACING(ctx);
    int shape[2];
    const float *px, *pWx, *pWh, *pb, *ph0;
    float *py, *pz = NULL;
    nmz_status s;
    if ((s = initial_state(be, ctx, &r->h0, &r->state, B, H, (size_t)H, 0)) != NMZ_OK) return s;
    if (tracing && (s = nmz_tensor_reshape(&r->z, 3, y->shape)) != NMZ_OK) return s;
    px = nmz__r(be, x);
    pWx = nmz__r(be, &r->Wxh);
    pWh = nmz__r(be, &r->Whh);
    pb = nmz__r(be, &r->b);
    ph0 = nmz__r(be, &r->h0);
    py = nmz__w(be, y);
    if (tracing) pz = nmz__w(be, &r->z);
    if (!px || !pWx || !pWh || !pb || !ph0 || !py || (tracing && !pz)) return nmz__op_failed();
    /* a fused kernel may decline a shape it does not handle (NMZ_ERR_UNSUPPORTED) */
    s = NMZ_FUSED(be, rnn_forward) ? be->rnn_forward(be, B, T, D, H, px, ph0, pWx, pWh, pb, py, pz)
                                   : NMZ_ERR_UNSUPPORTED;
    if (s == NMZ_ERR_UNSUPPORTED) { /* z = x·W_xh + b for all steps at once; then per step z_t += h_{t−1}·W_hh, h_t = tanh(z_t) */
        if (!pz && ((s = nmz_tensor_reshape(&r->z, 3, y->shape)) != NMZ_OK || !(pz = nmz__w(be, &r->z))))
            return s != NMZ_OK ? s : nmz__op_failed();
        s = be->gemm(be, 0, B * T, H, D, px, D, pWx, H, pz, H);
        if (s == NMZ_OK) s = be->add_bias(be, (size_t)B * T, H, pz, pb);
        for (t = 0; t < T && s == NMZ_OK; t++) {
            nmz_rows zt, ht;
            zt.p = pz + (size_t)t * H;
            zt.ld = T * H;
            ht.p = py + (size_t)t * H;
            ht.ld = T * H;
            if ((s = nmz__rec_product(be, B, T, t, H, H, 0, H, ph0, py, pWh, pz)) == NMZ_OK)
                s = be->rnn_cell_forward(be, B, H, zt, ht);
        }
    }
    if (s != NMZ_OK) return s;
    if (ctx->carry) { /* remember h_{T−1} for the next batch of the same streams */
        float *pst;
        shape[0] = B;
        shape[1] = H;
        if ((s = nmz_tensor_reshape(&r->state, 2, shape)) != NMZ_OK) return s;
        if (!(pst = nmz__w(be, &r->state))) return nmz__op_failed();
        if ((s = be->copy(be, (size_t)B, (size_t)H, py + (size_t)(T - 1) * H, (size_t)T * H, pst, (size_t)H)) != NMZ_OK)
            return s;
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

/* trace: the record tensors in be's memory, [B, T, n] each */
nmz_status nmz__rec_alloc(const nmz_backend *be, nmz_tensor *t, int n, int B, int T, int G, nmz_seq_record *rec,
                          float **out[], const int *widths) {
    int k, shape[3];
    nmz_status s;
    memset(rec, 0, sizeof *rec);
    rec->T = T;
    shape[0] = B;
    shape[1] = T;
    for (k = 0; k < n; k++) {
        shape[2] = widths[k] ? widths[k] : G;
        if ((s = nmz_tensor_reshape(&t[k], 3, shape)) != NMZ_OK) return s;
        if (!(*out[k] = nmz__w(be, &t[k]))) return nmz__op_failed();
    }
    return NMZ_OK;
}

static nmz_status rnn_backward(nmz_layer *l, const nmz_tensor *x, const nmz_tensor *y, const nmz_tensor *dy,
                               nmz_tensor *dx, const nmz_ctx *ctx) {
    rnn_layer *r = l->self;
    const nmz_backend *be = ctx->backend;
    int B = x->shape[0], T = x->shape[1], H = r->units, D = r->D, t, i, tracing = NMZ_TRACING(ctx), shape[2];
    nmz_seq_record rec = {0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    const float *px, *py, *ph0, *pWx, *pWh, *pdy;
    float *pdWx, *pdWh, *pdb, *pdx = NULL, *pdn;
    nmz_status s;
    shape[0] = B;
    shape[1] = H;
    if ((s = nmz_tensor_reshape(&r->dnext, 2, shape)) != NMZ_OK) return s;
    if (tracing) {
        float **slots[4];
        int widths[4] = {H, H, H, H};
        slots[0] = &rec.dh_next;
        slots[1] = &rec.dh;
        slots[2] = &rec.da;
        slots[3] = &rec.dh_prev;
        if ((s = nmz__rec_alloc(be, r->rec, 4, B, T, H, &rec, slots, widths)) != NMZ_OK) return s;
    }
    px = nmz__r(be, x);
    py = nmz__r(be, y);
    ph0 = nmz__r(be, &r->h0);
    pWx = nmz__r(be, &r->Wxh);
    pWh = nmz__r(be, &r->Whh);
    pdy = nmz__r(be, dy);
    pdWx = nmz__u(be, &r->dWxh);
    pdWh = nmz__u(be, &r->dWhh);
    pdb = nmz__u(be, &r->db);
    pdn = nmz__w(be, &r->dnext);
    if (dx) pdx = nmz__w(be, dx);
    if (!px || !py || !ph0 || !pWx || !pWh || !pdy || !pdWx || !pdWh || !pdb || !pdn || (dx && !pdx))
        return nmz__op_failed();
    s = NMZ_FUSED(be, rnn_backward) ? be->rnn_backward(be, B, T, D, H, px, py, ph0, pWx, pWh, pdy, pdWx, pdWh, pdb, pdx,
                                                       pdn, tracing ? &rec : NULL)
                                    : NMZ_ERR_UNSUPPORTED;
    if (s == NMZ_ERR_UNSUPPORTED) {
        int s3[3];
        float *pgz, *pgp;
        s3[0] = B;
        s3[1] = T;
        s3[2] = H;
        if ((s = nmz_tensor_reshape(&r->gdz, 3, s3)) != NMZ_OK || (s = nmz_tensor_reshape(&r->gprev, 3, s3)) != NMZ_OK)
            return s;
        pgz = nmz__w(be, &r->gdz);
        pgp = nmz__w(be, &r->gprev);
        if (!pgz || !pgp) return nmz__op_failed();
        s = be->fill(be, (size_t)B * H, 0.0f, pdn); /* nothing comes from beyond the last step */
        for (t = T - 1; t >= 0 && s == NMZ_OK; t--) {
            nmz_rows h, d, n, z;
            h.p = (float *)py + (size_t)t * H;
            h.ld = T * H;
            d.p = (float *)pdy + (size_t)t * H;
            d.ld = T * H;
            n.p = pdn;
            n.ld = H;
            z.p = pgz + (size_t)t * H;
            z.ld = T * H;
            s = be->rnn_cell_backward(be, B, H, h, d, n, z, tracing ? &rec : NULL, t);
            if (s == NMZ_OK) s = be->gemm(be, NMZ_MM_TRANS_B, B, H, H, z.p, T * H, pWh, H, pdn, H); /* ∂L/∂h_{t−1} */
            if (s == NMZ_OK && tracing)
                s = be->copy(be, (size_t)B, (size_t)H, pdn, (size_t)H, rec.dh_prev + (size_t)t * H, (size_t)T * H);
        }
        /* the weight gradients of all steps at once */
        if (s == NMZ_OK) s = be->gemm(be, NMZ_MM_TRANS_A | NMZ_MM_ACCUMULATE, D, H, B * T, px, D, pgz, H, pdWx, H);
        if (s == NMZ_OK) s = nmz__rec_prev(be, B, T, H, ph0, py, pgp);
        if (s == NMZ_OK) s = be->gemm(be, NMZ_MM_TRANS_A | NMZ_MM_ACCUMULATE, H, H, B * T, pgp, H, pgz, H, pdWh, H);
        if (s == NMZ_OK) s = be->colsum_add(be, (size_t)B * T, H, pgz, pdb);
        if (s == NMZ_OK && dx) s = be->gemm(be, NMZ_MM_TRANS_B, B * T, D, H, pgz, H, pWx, H, pdx, D);
        if (s == NMZ_OK && tracing) s = be->copy(be, 1, (size_t)B * T * H, pgz, (size_t)B * T * H, rec.da, (size_t)B * T * H);
    }
    if (s != NMZ_OK) return s;
    if (tracing) {
        for (t = T - 1; t >= 0; t--) {
            if ((s = nmz__slice_t(dy, t, &r->tmp[0])) != NMZ_OK) return s;
            for (i = 0; i < 4; i++)
                if ((s = nmz__slice_t(&r->rec[i], t, &r->tmp[1 + i])) != NMZ_OK) return s;
            {
                nmz_trace_item it[5] = {{"dL/dh_out", &r->tmp[0]}, {"dL/dh_next", &r->tmp[1]}, {"dL/dh", &r->tmp[2]},
                                        {"dL/dz", &r->tmp[3]},     {"dL/dh_prev", &r->tmp[4]}};
                nmz_trace_emit_step(ctx, NMZ_TRACE_BACKWARD, "rnn", "tanh", t, 5, it);
            }
        }
        {
            nmz_trace_scalar sc[1] = {{"steps", T}};
            nmz_trace_item all[5] = {{"dL/dW_xh", &r->dWxh}, {"dL/dW_hh", &r->dWhh}, {"dL/db", &r->db},
                                     {"W_hh", &r->Whh},      {"dL/dx", dx}};
            nmz_trace_emit(ctx, NMZ_TRACE_BACKWARD, "rnn", "tanh", 1, sc, dx ? 5 : 4, all);
        }
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
    for (i = 0; i < 4; i++) nmz_tensor_free(&r->rec[i]);
    nmz_tensor_free(&r->gdz);
    nmz_tensor_free(&r->gprev);
    free(r);
}

static int rnn_buffers(nmz_layer *l, nmz_tensor **out, int max) {
    rnn_layer *r = l->self;
    nmz_tensor *all[] = {&r->h0, &r->state, &r->z, &r->dnext, &r->tmp[0], &r->tmp[1], &r->tmp[2], &r->tmp[3], &r->tmp[4], &r->tmp[5], &r->rec[0], &r->rec[1], &r->rec[2], &r->rec[3], &r->gdz, &r->gprev};
    return nmz__list(out, max, (int)(sizeof all / sizeof all[0]), all);
}

const nmz_layer_ops nmz__rnn_ops = {"rnn",        rnn_setup,       rnn_init,  rnn_forward, rnn_backward,
                                    rnn_param_count, rnn_param,    rnn_save,  rnn_load,    rnn_describe,
                                    rnn_destroy,  rnn_state,       NMZ_LAYER_BACKEND, NULL,
                                    rnn_buffers};

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
