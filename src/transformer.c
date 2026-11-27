/* transformer.c — the other building blocks of a decoder-only Transformer
 * (attention itself is in attention.c). Every layer works on [B, T, D].
 *
 * posenc        y_t = x_t + P[offset + t]
 *               learned: P [max_len, D] is a parameter (starts ~N(0, 0.02²));
 *               sinusoidal: P[t][2i] = sin(t/10000^(2i/D)),
 *                           P[t][2i+1] = cos(t/10000^(2i/D)), fixed.
 *               ∂L/∂x = ∂L/∂y;  ∂L/∂P[offset+t] += Σ_b ∂L/∂y_t.
 *               Without it attention has no idea of order: a weighted sum
 *               does not care where a value came from.
 *
 * layernorm     per row of D numbers: μ = mean, σ² = mean((x − μ)²),
 *               x̂ = (x − μ)/√(σ² + ε),  y = g ⊙ x̂ + b
 *               backward: ∂L/∂x̂ = ∂L/∂y ⊙ g,  ∂L/∂g += Σ ∂L/∂y ⊙ x̂,
 *               ∂L/∂b += Σ ∂L/∂y,
 *               ∂L/∂x = (∂L/∂x̂ − mean(∂L/∂x̂) − x̂ ⊙ mean(∂L/∂x̂ ⊙ x̂)) / √(σ² + ε)
 *
 * ffn           z = x·W1 + b1, a = act(z) (gelu or relu), y = a·W2 + b2,
 *               applied to every position on its own.
 *
 * block         pre-LayerNorm Transformer block, one composite layer that
 *               owns four sub-layers and calls them through their ops:
 *                   h = x + dropout(attention(LN1(x)))
 *                   y = h + dropout(ffn(LN2(h)))
 *               backward: ∂L/∂h = ∂L/∂y + (back through ffn and LN2)
 *                         ∂L/∂x = ∂L/∂h + (back through attention and LN1)
 *               The "+ ∂L/∂y" and "+ ∂L/∂h" terms are the residual path:
 *               the gradient reaches every block unchanged, whatever the
 *               sub-layers do (DESIGN decision 93 on pre- vs post-LN).
 *
 * tied_output   z = x·Eᵀ + b where E is the embedding matrix [V, D]: the
 *               same parameter turns characters into vectors at the bottom
 *               and vectors into character scores at the top.
 *               ∂L/∂E += ∂L/∂zᵀ·x (added to the embedding's own gradient),
 *               ∂L/∂x = ∂L/∂z·E. */
#include "internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static nmz_status shape_like(nmz_tensor *t, const nmz_tensor *ref) { return nmz_tensor_reshape(t, ref->rank, ref->shape); }

static nmz_status need_rank2(int in_rank, const char *what) {
    if (in_rank != 2)
        return nmz__fail(NMZ_ERR_SHAPE, "%s needs [steps, features] per sample (e.g. after an embedding), got rank %d",
                         what, in_rank);
    return NMZ_OK;
}

static void uniform(nmz_tensor *W, double limit, nmz_rng *rng) {
    float *w = nmz__hw(W);
    size_t i;
    for (i = 0; i < W->size; i++) w[i] = (float)((2.0 * nmz_rng_uniform(rng) - 1.0) * limit);
}

/* a 1-row copy in the backend's memory */
static nmz_status copy_all(const nmz_backend *be, const nmz_tensor *src, nmz_tensor *dst) {
    const float *ps = nmz__r(be, src);
    float *pd = nmz__w(be, dst);
    if (!src->size) return NMZ_OK;
    if (!ps || !pd) return nmz__op_failed();
    return be->copy(be, 1, src->size, ps, src->size, pd, src->size);
}

static nmz_param mkparam(const char *name, nmz_tensor *v, nmz_tensor *g) {
    nmz_param p;
    p.name = name;
    p.value = v;
    p.grad = g;
    return p;
}

/* ================================================================ posenc */

typedef struct pos_layer {
    nmz_posenc kind;
    int max_len, D;
    nmz_tensor P, dP;  /* learned: the parameter; sinusoidal: the fixed table */
    nmz_tensor state;  /* [B, 1]: positions already seen per stream (carry)   */
    int *off;          /* offsets of the last forward, one per row            */
    int n_off;
    nmz_tensor tmp;
} pos_layer;

const char *nmz_posenc_name(nmz_posenc k) {
    return k == NMZ_POS_LEARNED ? "learned" : k == NMZ_POS_SINUSOIDAL ? "sinusoidal" : "?";
}

static nmz_status pos_setup(nmz_layer *l, int in_rank, const int *in_shape, int *out_rank, int *out_shape) {
    pos_layer *p = l->self;
    int s2[2], t, i;
    nmz_status s;
    if ((s = need_rank2(in_rank, "posenc")) != NMZ_OK) return s;
    p->D = in_shape[1];
    s2[0] = p->max_len;
    s2[1] = p->D;
    if ((s = nmz_tensor_init(&p->P, 2, s2)) != NMZ_OK) return s;
    if (p->kind == NMZ_POS_LEARNED && (s = nmz_tensor_init(&p->dP, 2, s2)) != NMZ_OK) return s;
    if (p->kind == NMZ_POS_SINUSOIDAL) {
        float *P = nmz__hw(&p->P);
        for (t = 0; t < p->max_len; t++)
            for (i = 0; i < p->D; i++) {
                double freq = pow(10000.0, -(double)(i - i % 2) / p->D); /* 1/10000^(2k/D) for columns 2k, 2k+1 */
                P[(size_t)t * p->D + i] = (float)(i % 2 ? cos(t * freq) : sin(t * freq));
            }
    }
    *out_rank = 2;
    out_shape[0] = in_shape[0];
    out_shape[1] = p->D;
    return NMZ_OK;
}

static nmz_status pos_init(nmz_layer *l, nmz_rng *rng, const nmz_layer *next) {
    pos_layer *p = l->self;
    size_t i;
    (void)next;
    if (p->kind == NMZ_POS_LEARNED) { /* small: positions start as a hint, tokens dominate (GPT-2 uses 0.01–0.02) */
        float *P = nmz__hw(&p->P);
        for (i = 0; i < p->P.size; i++) P[i] = (float)(0.02 * nmz_rng_normal(rng));
    }
    return NMZ_OK;
}

/* The CPU kernels (the v2.0.1 loops). */
nmz_status nmz__cpu_posenc_forward(const nmz_backend *be, int B, int T, int D, const int *off, const float *x,
                                   const float *P, float *y) {
    int b, t, k;
    (void)be;
    for (b = 0; b < B; b++)
        for (t = 0; t < T; t++) {
            const float *xt = x + ((size_t)b * T + t) * D, *pt = P + (size_t)(off[b] + t) * D;
            float *yt = y + ((size_t)b * T + t) * D;
            for (k = 0; k < D; k++) yt[k] = xt[k] + pt[k];
        }
    return NMZ_OK;
}

nmz_status nmz__cpu_posenc_backward(const nmz_backend *be, int B, int T, int D, const int *off, const float *dy,
                                    float *dP) {
    int b, t, k;
    (void)be;
    for (b = 0; b < B; b++)
        for (t = 0; t < T; t++) {
            const float *g = dy + ((size_t)b * T + t) * D;
            float *dp = dP + (size_t)(off[b] + t) * D;
            for (k = 0; k < D; k++) dp[k] += g[k];
        }
    return NMZ_OK;
}

static nmz_status pos_forward(nmz_layer *l, const nmz_tensor *x, nmz_tensor *y, const nmz_ctx *ctx) {
    pos_layer *p = l->self;
    const nmz_backend *be = ctx->backend;
    const int B = x->shape[0], T = x->shape[1], D = p->D;
    int b, shape[2];
    const float *px, *pP;
    float *py;
    nmz_status s;
    if (B > p->n_off) {
        int *o = realloc(p->off, (size_t)B * sizeof *o);
        if (!o) return nmz__fail(NMZ_ERR_NOMEM, "out of memory");
        p->off = o;
        p->n_off = B;
    }
    {
        const float *st = ctx->carry && p->state.size == (size_t)B ? nmz__hr(&p->state) : NULL;
        for (b = 0; b < B; b++) {
            p->off[b] = st ? (int)st[b] : 0;
            if (p->off[b] + T > p->max_len)
                return nmz__fail(NMZ_ERR_SHAPE, "%d positions do not fit the context window of %d (posenc max_len)",
                                 p->off[b] + T, p->max_len);
        }
    }
    px = nmz__r(be, x);
    pP = nmz__r(be, &p->P);
    py = nmz__w(be, y);
    if (!px || !pP || !py) return nmz__op_failed();
    if ((s = be->posenc_forward(be, B, T, D, p->off, px, pP, py)) != NMZ_OK) return s;
    if (ctx->carry) {
        float *st;
        shape[0] = B;
        shape[1] = 1;
        if ((s = nmz_tensor_reshape(&p->state, 2, shape)) != NMZ_OK) return s;
        st = nmz__hw(&p->state);
        for (b = 0; b < B; b++) st[b] = (float)(p->off[b] + T);
    }
    if (NMZ_TRACING(ctx)) {
        shape[0] = T;
        shape[1] = D;
        if ((s = nmz_tensor_reshape(&p->tmp, 2, shape)) != NMZ_OK) return s;
        memcpy(nmz__hw(&p->tmp), nmz__hr(&p->P) + (size_t)p->off[0] * D, (size_t)T * D * sizeof(float));
        {
            nmz_trace_scalar sc[2] = {{"offset", p->off[0]}, {"max_len", p->max_len}};
            nmz_trace_item it[3] = {{"x", x}, {"P", &p->tmp}, {"y", y}};
            nmz_trace_emit(ctx, NMZ_TRACE_FORWARD, "posenc", nmz_posenc_name(p->kind), 2, sc, 3, it);
        }
    }
    return NMZ_OK;
}

static nmz_status pos_backward(nmz_layer *l, const nmz_tensor *x, const nmz_tensor *y, const nmz_tensor *dy,
                               nmz_tensor *dx, const nmz_ctx *ctx) {
    pos_layer *p = l->self;
    const nmz_backend *be = ctx->backend;
    const int B = x->shape[0], T = x->shape[1], D = p->D;
    nmz_status s;
    (void)y;
    if (dx && (s = copy_all(be, dy, dx)) != NMZ_OK) return s;
    if (p->kind == NMZ_POS_LEARNED) {
        const float *pdy = nmz__r(be, dy);
        float *pdP = nmz__u(be, &p->dP);
        if (!pdy || !pdP) return nmz__op_failed();
        if ((s = be->posenc_backward(be, B, T, D, p->off, pdy, pdP)) != NMZ_OK) return s;
    }
    if (NMZ_TRACING(ctx)) {
        nmz_trace_item it[3] = {{"dL/dy", dy}, {"dL/dP", &p->dP}, {"dL/dx", dx}};
        if (p->kind == NMZ_POS_LEARNED)
            nmz_trace_emit(ctx, NMZ_TRACE_BACKWARD, "posenc", nmz_posenc_name(p->kind), 0, NULL, dx ? 3 : 2, it);
        else {
            it[1] = it[2];
            nmz_trace_emit(ctx, NMZ_TRACE_BACKWARD, "posenc", nmz_posenc_name(p->kind), 0, NULL, dx ? 2 : 1, it);
        }
    }
    return NMZ_OK;
}

static int pos_param_count(const nmz_layer *l) { return ((const pos_layer *)l->self)->kind == NMZ_POS_LEARNED; }

static nmz_param pos_param(nmz_layer *l, int i) {
    pos_layer *p = l->self;
    nmz_param none = {NULL, NULL, NULL};
    return i == 0 && p->kind == NMZ_POS_LEARNED ? mkparam("P", &p->P, &p->dP) : none;
}

static nmz_status pos_save(const nmz_layer *l, nmz_writer *w) {
    const pos_layer *p = l->self;
    nmz_write_u32(w, (uint32_t)p->kind);
    return nmz_write_u32(w, (uint32_t)p->max_len);
}

static nmz_status pos_load(nmz_layer **out, nmz_reader *r) {
    uint32_t kind = 0, len = 0;
    nmz_read_u32(r, &kind);
    if (nmz_read_u32(r, &len) != NMZ_OK) return r->err;
    if (kind >= NMZ_POS_COUNT || len == 0 || len > 100000) return nmz__fail(NMZ_ERR_FORMAT, "corrupt posenc config");
    *out = nmz_layer_posenc((nmz_posenc)kind, (int)len);
    return *out ? NMZ_OK : NMZ_ERR_NOMEM;
}

static void pos_describe(const nmz_layer *l, char *buf, size_t n) {
    const pos_layer *p = l->self;
    snprintf(buf, n, "posenc %s (context %d, %d features)", nmz_posenc_name(p->kind), p->max_len, p->D);
}

static nmz_tensor *pos_state(nmz_layer *l) { return &((pos_layer *)l->self)->state; }

static void pos_destroy(nmz_layer *l) {
    pos_layer *p = l->self;
    if (!p) return;
    nmz_tensor_free(&p->P);
    nmz_tensor_free(&p->dP);
    nmz_tensor_free(&p->state);
    nmz_tensor_free(&p->tmp);
    free(p->off);
    free(p);
}

static int pos_buffers(nmz_layer *l, nmz_tensor **out, int max) {
    pos_layer *p = l->self;
    nmz_tensor *all[] = {&p->state, &p->tmp};
    return nmz__list(out, max, (int)(sizeof all / sizeof all[0]), all);
}

const nmz_layer_ops nmz__posenc_ops = {"posenc",        pos_setup, pos_init,     pos_forward, pos_backward,
                                       pos_param_count, pos_param, pos_save,     pos_load,    pos_describe,
                                       pos_destroy,     pos_state, NMZ_LAYER_NO_DECAY | NMZ_LAYER_BACKEND, NULL,
                                       pos_buffers};

nmz_layer *nmz_layer_posenc(nmz_posenc kind, int max_len) {
    pos_layer *p;
    nmz_layer *l;
    if ((int)kind < 0 || kind >= NMZ_POS_COUNT || max_len < 1 || max_len > 100000) {
        nmz__fail(NMZ_ERR_ARG, "posenc needs a kind (learned/sinusoidal) and a context length 1..100000");
        return NULL;
    }
    p = calloc(1, sizeof *p);
    if (!p) {
        nmz__fail(NMZ_ERR_NOMEM, "out of memory");
        return NULL;
    }
    p->kind = kind;
    p->max_len = max_len;
    l = nmz_layer_alloc(&nmz__posenc_ops, p);
    if (!l) free(p);
    return l;
}

/* ============================================================= layernorm */

typedef struct ln_layer {
    int D;
    float eps;
    nmz_tensor g, b, dg, db;
    nmz_tensor xhat, mean, rstd; /* of the last forward: [B,T,D], [B,T], [B,T] */
    nmz_tensor dxhat;
} ln_layer;

static nmz_status ln_setup(nmz_layer *l, int in_rank, const int *in_shape, int *out_rank, int *out_shape) {
    ln_layer *n = l->self;
    int i;
    nmz_status s;
    if (in_rank < 1) return nmz__fail(NMZ_ERR_SHAPE, "layernorm needs at least one dimension");
    n->D = in_shape[in_rank - 1];
    if ((s = nmz_tensor_init(&n->g, 1, &n->D)) != NMZ_OK || (s = nmz_tensor_init(&n->b, 1, &n->D)) != NMZ_OK ||
        (s = nmz_tensor_init(&n->dg, 1, &n->D)) != NMZ_OK || (s = nmz_tensor_init(&n->db, 1, &n->D)) != NMZ_OK)
        return s;
    *out_rank = in_rank;
    for (i = 0; i < in_rank; i++) out_shape[i] = in_shape[i];
    return NMZ_OK;
}

static nmz_status ln_init(nmz_layer *l, nmz_rng *rng, const nmz_layer *next) {
    ln_layer *n = l->self;
    size_t i;
    (void)rng;
    (void)next;
    {
        float *g = nmz__hw(&n->g);
        for (i = 0; i < n->g.size; i++) g[i] = 1.0f; /* starts as a pure normalisation */
    }
    nmz_tensor_zero(&n->b);
    return NMZ_OK;
}

/* The CPU kernels: the v2.0.1 loops (statistics in double). */
nmz_status nmz__cpu_layernorm_forward(const nmz_backend *be, size_t rows, int D, float eps, const float *x,
                                      const float *g, const float *b, float *y, float *xhat, float *mean,
                                      float *rstd) {
    size_t r;
    int k;
    (void)be;
    for (r = 0; r < rows; r++) {
        const float *xr = x + r * D;
        float *xh = xhat + r * D, *yr = y + r * D;
        double mu = 0.0, var = 0.0, rs;
        for (k = 0; k < D; k++) mu += xr[k];
        mu /= D;
        for (k = 0; k < D; k++) var += ((double)xr[k] - mu) * ((double)xr[k] - mu);
        var /= D;
        rs = 1.0 / sqrt(var + eps);
        mean[r] = (float)mu;
        rstd[r] = (float)rs;
        for (k = 0; k < D; k++) {
            xh[k] = (float)(((double)xr[k] - mu) * rs);
            yr[k] = g[k] * xh[k] + b[k];
        }
    }
    return NMZ_OK;
}

nmz_status nmz__cpu_layernorm_backward(const nmz_backend *be, size_t rows, int D, const float *dy, const float *xhat,
                                       const float *rstd, const float *gain, float *dx, float *dg, float *db,
                                       float *dxhat) {
    double stack[1024], *acc = 2 * D <= 1024 ? stack : malloc((size_t)2 * D * sizeof *acc), *ag, *ab;
    size_t r;
    int k;
    (void)be;
    if (!acc) return nmz__fail(NMZ_ERR_NOMEM, "out of memory");
    ag = acc;
    ab = acc + D;
    for (k = 0; k < 2 * D; k++) acc[k] = 0.0;
    for (r = 0; r < rows; r++) {
        const float *g = dy + r * D, *xh = xhat + r * D;
        double m1 = 0.0, m2 = 0.0, rs = rstd[r];
        for (k = 0; k < D; k++) {
            double dxh = (double)g[k] * gain[k];
            ag[k] += (double)g[k] * xh[k];
            ab[k] += g[k];
            m1 += dxh;
            m2 += dxh * xh[k];
            if (dxhat) dxhat[r * D + k] = (float)dxh;
        }
        m1 /= D;
        m2 /= D;
        if (dx) {
            float *o = dx + r * D;
            for (k = 0; k < D; k++) o[k] = (float)(rs * ((double)g[k] * gain[k] - m1 - xh[k] * m2));
        }
    }
    for (k = 0; k < D; k++) {
        dg[k] += (float)ag[k];
        db[k] += (float)ab[k];
    }
    if (acc != stack) free(acc);
    return NMZ_OK;
}

static nmz_status ln_forward(nmz_layer *l, const nmz_tensor *x, nmz_tensor *y, const nmz_ctx *ctx) {
    ln_layer *n = l->self;
    const nmz_backend *be = ctx->backend;
    const int D = n->D;
    size_t rows = x->size / (size_t)D;
    int shape[NMZ_MAX_RANK];
    const float *px, *pg, *pb;
    float *py, *pxh, *pm, *pr;
    nmz_status s;
    if ((s = shape_like(&n->xhat, x)) != NMZ_OK) return s;
    memcpy(shape, x->shape, sizeof shape);
    if ((s = nmz_tensor_reshape(&n->mean, x->rank - 1, shape)) != NMZ_OK ||
        (s = nmz_tensor_reshape(&n->rstd, x->rank - 1, shape)) != NMZ_OK)
        return s;
    px = nmz__r(be, x);
    pg = nmz__r(be, &n->g);
    pb = nmz__r(be, &n->b);
    py = nmz__w(be, y);
    pxh = nmz__w(be, &n->xhat);
    pm = nmz__w(be, &n->mean);
    pr = nmz__w(be, &n->rstd);
    if (!px || !pg || !pb || !py || !pxh || !pm || !pr) return nmz__op_failed();
    if ((s = be->layernorm_forward(be, rows, D, n->eps, px, pg, pb, py, pxh, pm, pr)) != NMZ_OK) return s;
    if (NMZ_TRACING(ctx)) {
        nmz_trace_scalar sc[1] = {{"epsilon", n->eps}};
        nmz_trace_item it[7] = {{"x", x}, {"mean", &n->mean}, {"rstd", &n->rstd}, {"x_hat", &n->xhat},
                                {"g", &n->g}, {"b", &n->b}, {"y", y}};
        nmz_trace_emit(ctx, NMZ_TRACE_FORWARD, "layernorm", NULL, 1, sc, 7, it);
    }
    return NMZ_OK;
}

static nmz_status ln_backward(nmz_layer *l, const nmz_tensor *x, const nmz_tensor *y, const nmz_tensor *dy,
                              nmz_tensor *dx, const nmz_ctx *ctx) {
    ln_layer *n = l->self;
    const nmz_backend *be = ctx->backend;
    const int D = n->D;
    size_t rows = x->size / (size_t)D;
    int tracing = NMZ_TRACING(ctx);
    const float *pdy, *pxh, *pr, *pg;
    float *pdx = NULL, *pdg, *pdb, *pdxh = NULL;
    nmz_status s;
    (void)y;
    if (n->xhat.size != x->size) return nmz__fail(NMZ_ERR_STATE, "layernorm: backward without a matching forward");
    if (tracing) {
        if ((s = shape_like(&n->dxhat, x)) != NMZ_OK) return s;
        if (!(pdxh = nmz__w(be, &n->dxhat))) return nmz__op_failed();
    }
    pdy = nmz__r(be, dy);
    pxh = nmz__r(be, &n->xhat);
    pr = nmz__r(be, &n->rstd);
    pg = nmz__r(be, &n->g);
    pdg = nmz__u(be, &n->dg);
    pdb = nmz__u(be, &n->db);
    if (dx) pdx = nmz__w(be, dx);
    if (!pdy || !pxh || !pr || !pg || !pdg || !pdb || (dx && !pdx)) return nmz__op_failed();
    if ((s = be->layernorm_backward(be, rows, D, pdy, pxh, pr, pg, pdx, pdg, pdb, pdxh)) != NMZ_OK) return s;
    if (tracing) {
        nmz_trace_item it[5] = {{"dL/dy", dy}, {"dL/dx_hat", &n->dxhat}, {"dL/dg", &n->dg}, {"dL/db", &n->db},
                                {"dL/dx", dx}};
        nmz_trace_emit(ctx, NMZ_TRACE_BACKWARD, "layernorm", NULL, 0, NULL, dx ? 5 : 4, it);
    }
    return NMZ_OK;
}

static int ln_param_count(const nmz_layer *l) {
    (void)l;
    return 2;
}

static nmz_param ln_param(nmz_layer *l, int i) {
    ln_layer *n = l->self;
    nmz_param none = {NULL, NULL, NULL};
    return i == 0 ? mkparam("g", &n->g, &n->dg) : i == 1 ? mkparam("b", &n->b, &n->db) : none;
}

static nmz_status ln_save(const nmz_layer *l, nmz_writer *w) { return nmz_write_f32(w, ((const ln_layer *)l->self)->eps); }

static nmz_status ln_load(nmz_layer **out, nmz_reader *r) {
    float eps = 0;
    if (nmz_read_f32(r, &eps) != NMZ_OK) return r->err;
    if (!(eps > 0.0f && eps < 1.0f)) return nmz__fail(NMZ_ERR_FORMAT, "corrupt layernorm config");
    *out = nmz_layer_layernorm();
    if (!*out) return NMZ_ERR_NOMEM;
    ((ln_layer *)(*out)->self)->eps = eps;
    return NMZ_OK;
}

static void ln_describe(const nmz_layer *l, char *buf, size_t n) {
    snprintf(buf, n, "layernorm %d (gain + bias)", ((const ln_layer *)l->self)->D);
}

static void ln_destroy(nmz_layer *l) {
    ln_layer *n = l->self;
    if (!n) return;
    nmz_tensor_free(&n->g);
    nmz_tensor_free(&n->b);
    nmz_tensor_free(&n->dg);
    nmz_tensor_free(&n->db);
    nmz_tensor_free(&n->xhat);
    nmz_tensor_free(&n->mean);
    nmz_tensor_free(&n->rstd);
    nmz_tensor_free(&n->dxhat);
    free(n);
}

static int ln_buffers(nmz_layer *l, nmz_tensor **out, int max) {
    ln_layer *n = l->self;
    nmz_tensor *all[] = {&n->xhat, &n->mean, &n->rstd, &n->dxhat};
    return nmz__list(out, max, (int)(sizeof all / sizeof all[0]), all);
}

const nmz_layer_ops nmz__layernorm_ops = {"layernorm",    ln_setup, ln_init, ln_forward, ln_backward,
                                          ln_param_count, ln_param, ln_save, ln_load,    ln_describe,
                                          ln_destroy,     NULL,     NMZ_LAYER_BACKEND, NULL,
                                          ln_buffers};

nmz_layer *nmz_layer_layernorm(void) {
    ln_layer *n = calloc(1, sizeof *n);
    nmz_layer *l;
    if (!n) {
        nmz__fail(NMZ_ERR_NOMEM, "out of memory");
        return NULL;
    }
    n->eps = NMZ_LAYERNORM_EPS;
    l = nmz_layer_alloc(&nmz__layernorm_ops, n);
    if (!l) free(n);
    return l;
}

/* =================================================================== ffn */

typedef struct ffn_layer {
    int D, F, F_req;
    nmz_act act;
    nmz_tensor W1, b1, W2, b2, dW1, db1, dW2, db2;
    nmz_tensor z, a, dz; /* [B, T, F] */
} ffn_layer;

static nmz_status ffn_setup(nmz_layer *l, int in_rank, const int *in_shape, int *out_rank, int *out_shape) {
    ffn_layer *f = l->self;
    int s2[2];
    nmz_status s;
    if ((s = need_rank2(in_rank, "ffn")) != NMZ_OK) return s;
    f->D = in_shape[1];
    f->F = f->F_req ? f->F_req : 4 * f->D;
    s2[0] = f->D;
    s2[1] = f->F;
    if ((s = nmz_tensor_init(&f->W1, 2, s2)) != NMZ_OK || (s = nmz_tensor_init(&f->dW1, 2, s2)) != NMZ_OK) return s;
    s2[0] = f->F;
    s2[1] = f->D;
    if ((s = nmz_tensor_init(&f->W2, 2, s2)) != NMZ_OK || (s = nmz_tensor_init(&f->dW2, 2, s2)) != NMZ_OK) return s;
    if ((s = nmz_tensor_init(&f->b1, 1, &f->F)) != NMZ_OK || (s = nmz_tensor_init(&f->db1, 1, &f->F)) != NMZ_OK ||
        (s = nmz_tensor_init(&f->b2, 1, &f->D)) != NMZ_OK || (s = nmz_tensor_init(&f->db2, 1, &f->D)) != NMZ_OK)
        return s;
    *out_rank = 2;
    out_shape[0] = in_shape[0];
    out_shape[1] = f->D;
    return NMZ_OK;
}

static nmz_status ffn_init(nmz_layer *l, nmz_rng *rng, const nmz_layer *next) {
    ffn_layer *f = l->self;
    (void)next;
    /* He before relu (decision 12), Glorot otherwise; W2 feeds the residual sum: Glorot */
    uniform(&f->W1, f->act == NMZ_ACT_RELU ? sqrt(6.0 / f->D) : sqrt(6.0 / (f->D + f->F)), rng);
    uniform(&f->W2, sqrt(6.0 / (f->D + f->F)), rng);
    nmz_tensor_zero(&f->b1);
    nmz_tensor_zero(&f->b2);
    return NMZ_OK;
}

static nmz_status ffn_forward(nmz_layer *l, const nmz_tensor *x, nmz_tensor *y, const nmz_ctx *ctx) {
    ffn_layer *f = l->self;
    const nmz_backend *be = ctx->backend;
    const int B = x->shape[0], T = x->shape[1], rows = B * T;
    int shape[3];
    const float *px, *pW1, *pb1, *pW2, *pb2;
    float *pz, *pa, *py;
    nmz_status s;
    shape[0] = B;
    shape[1] = T;
    shape[2] = f->F;
    if ((s = nmz_tensor_reshape(&f->z, 3, shape)) != NMZ_OK || (s = nmz_tensor_reshape(&f->a, 3, shape)) != NMZ_OK)
        return s;
    px = nmz__r(be, x);
    pW1 = nmz__r(be, &f->W1);
    pb1 = nmz__r(be, &f->b1);
    pW2 = nmz__r(be, &f->W2);
    pb2 = nmz__r(be, &f->b2);
    pz = nmz__w(be, &f->z);
    pa = nmz__w(be, &f->a);
    py = nmz__w(be, y);
    if (!px || !pW1 || !pb1 || !pW2 || !pb2 || !pz || !pa || !py) return nmz__op_failed();
    if ((s = be->gemm(be, 0, rows, f->F, f->D, px, f->D, pW1, f->F, pz, f->F)) != NMZ_OK ||
        (s = be->add_bias(be, (size_t)rows, f->F, pz, pb1)) != NMZ_OK ||
        (s = be->act_forward(be, f->act, 0.0f, f->z.size, pz, pa)) != NMZ_OK ||
        (s = be->gemm(be, 0, rows, f->D, f->F, pa, f->F, pW2, f->D, py, f->D)) != NMZ_OK ||
        (s = be->add_bias(be, (size_t)rows, f->D, py, pb2)) != NMZ_OK)
        return s;
    if (NMZ_TRACING(ctx)) {
        nmz_trace_item it[6] = {{"x", x}, {"W1", &f->W1}, {"z", &f->z}, {"a", &f->a}, {"W2", &f->W2}, {"y", y}};
        nmz_trace_emit(ctx, NMZ_TRACE_FORWARD, "ffn", NULL, 0, NULL, 6, it);
    }
    return NMZ_OK;
}

static nmz_status ffn_backward(nmz_layer *l, const nmz_tensor *x, const nmz_tensor *y, const nmz_tensor *dy,
                               nmz_tensor *dx, const nmz_ctx *ctx) {
    ffn_layer *f = l->self;
    const nmz_backend *be = ctx->backend;
    const int rows = x->shape[0] * x->shape[1];
    const float *px, *pdy, *pa, *pz, *pW1, *pW2;
    float *pdW1, *pdb1, *pdW2, *pdb2, *pdz, *pdx = NULL;
    nmz_status s;
    (void)y;
    if (f->z.size != (size_t)rows * f->F) return nmz__fail(NMZ_ERR_STATE, "ffn: backward without a matching forward");
    if ((s = shape_like(&f->dz, &f->z)) != NMZ_OK) return s;
    px = nmz__r(be, x);
    pdy = nmz__r(be, dy);
    pa = nmz__r(be, &f->a);
    pz = nmz__r(be, &f->z);
    pW1 = nmz__r(be, &f->W1);
    pW2 = nmz__r(be, &f->W2);
    pdW1 = nmz__u(be, &f->dW1);
    pdb1 = nmz__u(be, &f->db1);
    pdW2 = nmz__u(be, &f->dW2);
    pdb2 = nmz__u(be, &f->db2);
    pdz = nmz__w(be, &f->dz);
    if (dx) pdx = nmz__w(be, dx);
    if (!px || !pdy || !pa || !pz || !pW1 || !pW2 || !pdW1 || !pdb1 || !pdW2 || !pdb2 || !pdz || (dx && !pdx))
        return nmz__op_failed();
    if ((s = be->gemm(be, NMZ_MM_TRANS_A | NMZ_MM_ACCUMULATE, f->F, f->D, rows, pa, f->F, pdy, f->D, pdW2, f->D)) !=
            NMZ_OK ||
        (s = be->colsum_add(be, (size_t)rows, f->D, pdy, pdb2)) != NMZ_OK)
        return s;
    if ((s = be->gemm(be, NMZ_MM_TRANS_B, rows, f->F, f->D, pdy, f->D, pW2, f->D, pdz, f->F)) != NMZ_OK) return s;
    if (NMZ_TRACING(ctx)) { /* dz holds ∂L/∂a for a moment */
        nmz_trace_item it[2] = {{"dL/dy", dy}, {"dL/da", &f->dz}};
        nmz_trace_emit(ctx, NMZ_TRACE_BACKWARD, "ffn", NULL, 0, NULL, 2, it);
        if (!(pdz = nmz__u(be, &f->dz))) return nmz__op_failed(); /* the trace may have moved it to the host */
    }
    if ((s = be->act_backward(be, f->act, 0.0f, f->dz.size, pz, pdz, pdz, NULL)) != NMZ_OK) return s;
    if ((s = be->gemm(be, NMZ_MM_TRANS_A | NMZ_MM_ACCUMULATE, f->D, f->F, rows, px, f->D, pdz, f->F, pdW1, f->F)) !=
            NMZ_OK ||
        (s = be->colsum_add(be, (size_t)rows, f->F, pdz, pdb1)) != NMZ_OK)
        return s;
    if (dx && (s = be->gemm(be, NMZ_MM_TRANS_B, rows, f->D, f->F, pdz, f->F, pW1, f->F, pdx, f->D)) != NMZ_OK) return s;
    if (NMZ_TRACING(ctx)) {
        nmz_trace_item it[2] = {{"dL/dz", &f->dz}, {"dL/dx", dx}};
        nmz_trace_emit(ctx, NMZ_TRACE_BACKWARD, "ffn", NULL, 0, NULL, dx ? 2 : 1, it);
    }
    return NMZ_OK;
}

static int ffn_param_count(const nmz_layer *l) {
    (void)l;
    return 4;
}

static nmz_param ffn_param(nmz_layer *l, int i) {
    ffn_layer *f = l->self;
    nmz_param none = {NULL, NULL, NULL};
    switch (i) {
    case 0: return mkparam("W1", &f->W1, &f->dW1);
    case 1: return mkparam("b1", &f->b1, &f->db1);
    case 2: return mkparam("W2", &f->W2, &f->dW2);
    case 3: return mkparam("b2", &f->b2, &f->db2);
    default: return none;
    }
}

static nmz_status ffn_save(const nmz_layer *l, nmz_writer *w) {
    const ffn_layer *f = l->self;
    nmz_write_u32(w, (uint32_t)f->F_req);
    return nmz_write_u32(w, (uint32_t)f->act);
}

static nmz_status ffn_load(nmz_layer **out, nmz_reader *r) {
    uint32_t F = 0, act = 0;
    nmz_read_u32(r, &F);
    if (nmz_read_u32(r, &act) != NMZ_OK) return r->err;
    if (F > 1000000 || act >= NMZ_ACT_COUNT || act == NMZ_ACT_SOFTMAX) return nmz__fail(NMZ_ERR_FORMAT, "corrupt ffn config");
    *out = nmz_layer_ffn((int)F, (nmz_act)act);
    return *out ? NMZ_OK : NMZ_ERR_NOMEM;
}

static void ffn_describe(const nmz_layer *l, char *buf, size_t n) {
    const ffn_layer *f = l->self;
    snprintf(buf, n, "ffn %d->%d->%d (%s)", f->D, f->F, f->D, nmz_act_name(f->act));
}

static void ffn_destroy(nmz_layer *l) {
    ffn_layer *f = l->self;
    nmz_tensor *all[] = {&f->W1, &f->b1, &f->W2, &f->b2, &f->dW1, &f->db1, &f->dW2, &f->db2, &f->z, &f->a, &f->dz};
    size_t i;
    if (!f) return;
    for (i = 0; i < sizeof all / sizeof all[0]; i++) nmz_tensor_free(all[i]);
    free(f);
}

static int ffn_buffers(nmz_layer *l, nmz_tensor **out, int max) {
    ffn_layer *f = l->self;
    nmz_tensor *all[] = {&f->z, &f->a, &f->dz};
    return nmz__list(out, max, (int)(sizeof all / sizeof all[0]), all);
}

const nmz_layer_ops nmz__ffn_ops = {"ffn",           ffn_setup, ffn_init, ffn_forward, ffn_backward,
                                    ffn_param_count, ffn_param, ffn_save, ffn_load,    ffn_describe,
                                    ffn_destroy,     NULL,      NMZ_LAYER_BACKEND, NULL,
                                    ffn_buffers};

nmz_layer *nmz_layer_ffn(int hidden, nmz_act act) {
    ffn_layer *f;
    nmz_layer *l;
    if (hidden < 0 || hidden > 1000000 || (int)act < 0 || act >= NMZ_ACT_COUNT || act == NMZ_ACT_SOFTMAX) {
        nmz__fail(NMZ_ERR_ARG, "ffn needs a hidden size >= 0 (0 = 4 x features) and an element-wise activation");
        return NULL;
    }
    f = calloc(1, sizeof *f);
    if (!f) {
        nmz__fail(NMZ_ERR_NOMEM, "out of memory");
        return NULL;
    }
    f->F_req = f->F = hidden;
    f->act = act;
    l = nmz_layer_alloc(&nmz__ffn_ops, f);
    if (!l) free(f);
    return l;
}

/* ================================================================= block */

enum { SUB_LN1, SUB_ATTN, SUB_LN2, SUB_FFN, N_SUB };
static const char *const k_role[N_SUB] = {"ln1", "attn", "ln2", "ffn"};
static const char *const k_block_names[12] = {"ln1.g",     "ln1.b", "attn.W_qkv", "attn.b_qkv", "attn.W_o", "attn.b_o",
                                              "ln2.g",     "ln2.b", "ffn.W1",     "ffn.b1",     "ffn.W2",   "ffn.b2"};

typedef struct block_layer {
    nmz_layer *sub[N_SUB];
    float rate; /* residual dropout */
    nmz_tensor a1, o1, r1, h, a2, o2, r2, m1, m2; /* [B, T, D] each */
    nmz_tensor d_o, d_a, d_h, d_x;
} block_layer;

/* Sub-layer trace events reach the net's callback with `detail` = their
 * role in the block ("ln1", "attn", "ln2", "ffn"). */
typedef struct relay {
    nmz_trace_fn fn;
    void *user;
    const char *role;
} relay;

static void relay_fn(const nmz_trace_event *ev, void *user) {
    const relay *r = user;
    nmz_trace_event e = *ev;
    e.detail = r->role;
    r->fn(&e, r->user);
}

static nmz_ctx sub_ctx(const nmz_ctx *ctx, relay *r, int k) {
    nmz_ctx c = *ctx;
    if (ctx->trace) {
        r->fn = ctx->trace;
        r->user = ctx->trace_user;
        r->role = k_role[k];
        c.trace = relay_fn;
        c.trace_user = r;
    }
    return c;
}

static nmz_status sub_setup(nmz_layer *sub, int rank, const int *shape) {
    sub->in_rank = rank;
    memcpy(sub->in_shape, shape, sizeof(int) * (size_t)rank);
    return sub->ops->setup(sub, rank, shape, &sub->out_rank, sub->out_shape);
}

static nmz_status block_setup(nmz_layer *l, int in_rank, const int *in_shape, int *out_rank, int *out_shape) {
    block_layer *k = l->self;
    int i;
    nmz_status s;
    if ((s = need_rank2(in_rank, "block")) != NMZ_OK) return s;
    for (i = 0; i < N_SUB; i++)
        if ((s = sub_setup(k->sub[i], in_rank, in_shape)) != NMZ_OK) return s;
    *out_rank = 2;
    out_shape[0] = in_shape[0];
    out_shape[1] = in_shape[1];
    return NMZ_OK;
}

static nmz_status block_init(nmz_layer *l, nmz_rng *rng, const nmz_layer *next) {
    block_layer *k = l->self;
    int i;
    nmz_status s;
    for (i = 0; i < N_SUB; i++)
        if ((s = k->sub[i]->ops->init_params(k->sub[i], rng, i + 1 < N_SUB ? k->sub[i + 1] : next)) != NMZ_OK) return s;
    return NMZ_OK;
}

/* r = o ⊙ mask (training with dropout) or o */
static nmz_status residual_dropout(const nmz_tensor *o, nmz_tensor *mask, nmz_tensor *r, float rate,
                                   const nmz_ctx *ctx) {
    const nmz_backend *be = ctx->backend;
    nmz_status s;
    if (ctx->training && rate > 0.0f) {
        const float *po = nmz__r(be, o);
        float *pm = nmz__w(be, mask), *pr = nmz__w(be, r);
        if (!po || !pm || !pr) return nmz__op_failed();
        if ((s = be->dropout_mask(be, o->size, rate, ctx->rng, pm)) != NMZ_OK) return s;
        return be->mul(be, o->size, po, pm, pr);
    }
    if ((s = copy_all(be, o, r)) != NMZ_OK) return s;
    mask->size = 0; /* marks "no mask" for backward */
    return NMZ_OK;
}

/* out = a + b in the backend's memory */
static nmz_status add_t(const nmz_backend *be, const nmz_tensor *a, const nmz_tensor *b, nmz_tensor *out) {
    const float *pa = nmz__r(be, a), *pb = nmz__r(be, b);
    float *po = out == a || out == b ? nmz__u(be, out) : nmz__w(be, out);
    if (!pa || !pb || !po) return nmz__op_failed();
    return be->add(be, a->size, pa, pb, po);
}

/* out = a ⊙ mask, or a when there is no mask */
static nmz_status masked(const nmz_backend *be, const nmz_tensor *a, const nmz_tensor *mask, nmz_tensor *out) {
    const float *pa, *pm;
    float *po;
    if (!mask->size) return copy_all(be, a, out);
    pa = nmz__r(be, a);
    pm = nmz__r(be, mask);
    po = nmz__w(be, out);
    if (!pa || !pm || !po) return nmz__op_failed();
    return be->mul(be, a->size, pa, pm, po);
}

static nmz_status block_forward(nmz_layer *l, const nmz_tensor *x, nmz_tensor *y, const nmz_ctx *ctx) {
    block_layer *k = l->self;
    nmz_tensor *bufs[] = {&k->a1, &k->o1, &k->r1, &k->h, &k->a2, &k->o2, &k->r2, &k->m1, &k->m2};
    relay rl;
    nmz_ctx c;
    size_t i;
    nmz_status s;
    for (i = 0; i < sizeof bufs / sizeof bufs[0]; i++)
        if ((s = shape_like(bufs[i], x)) != NMZ_OK) return s;
    c = sub_ctx(ctx, &rl, SUB_LN1);
    if ((s = k->sub[SUB_LN1]->ops->forward(k->sub[SUB_LN1], x, &k->a1, &c)) != NMZ_OK) return s;
    c = sub_ctx(ctx, &rl, SUB_ATTN);
    if ((s = k->sub[SUB_ATTN]->ops->forward(k->sub[SUB_ATTN], &k->a1, &k->o1, &c)) != NMZ_OK) return s;
    if ((s = residual_dropout(&k->o1, &k->m1, &k->r1, k->rate, ctx)) != NMZ_OK) return s;
    if ((s = add_t(ctx->backend, x, &k->r1, &k->h)) != NMZ_OK) return s;
    c = sub_ctx(ctx, &rl, SUB_LN2);
    if ((s = k->sub[SUB_LN2]->ops->forward(k->sub[SUB_LN2], &k->h, &k->a2, &c)) != NMZ_OK) return s;
    c = sub_ctx(ctx, &rl, SUB_FFN);
    if ((s = k->sub[SUB_FFN]->ops->forward(k->sub[SUB_FFN], &k->a2, &k->o2, &c)) != NMZ_OK) return s;
    if ((s = residual_dropout(&k->o2, &k->m2, &k->r2, k->rate, ctx)) != NMZ_OK) return s;
    if ((s = add_t(ctx->backend, &k->h, &k->r2, y)) != NMZ_OK) return s;
    if (NMZ_TRACING(ctx)) {
        nmz_trace_scalar sc[1] = {{"dropout", k->rate}};
        nmz_trace_item it[5] = {{"x", x}, {"attn_out", &k->r1}, {"h", &k->h}, {"ffn_out", &k->r2}, {"y", y}};
        nmz_trace_emit(ctx, NMZ_TRACE_FORWARD, "block", NULL, 1, sc, 5, it);
    }
    return NMZ_OK;
}

static nmz_status block_backward(nmz_layer *l, const nmz_tensor *x, const nmz_tensor *y, const nmz_tensor *dy,
                                 nmz_tensor *dx, const nmz_ctx *ctx) {
    block_layer *k = l->self;
    const nmz_backend *be = ctx->backend;
    nmz_tensor *bufs[] = {&k->d_o, &k->d_a, &k->d_h, &k->d_x};
    relay rl;
    nmz_ctx c;
    size_t i;
    nmz_status s;
    (void)y;
    if (k->h.size != x->size) return nmz__fail(NMZ_ERR_STATE, "block: backward without a matching forward");
    for (i = 0; i < sizeof bufs / sizeof bufs[0]; i++)
        if ((s = shape_like(bufs[i], x)) != NMZ_OK) return s;
    /* y = h + drop(ffn(LN2(h))) */
    if ((s = masked(be, dy, &k->m2, &k->d_o)) != NMZ_OK) return s;
    c = sub_ctx(ctx, &rl, SUB_FFN);
    if ((s = k->sub[SUB_FFN]->ops->backward(k->sub[SUB_FFN], &k->a2, &k->o2, &k->d_o, &k->d_a, &c)) != NMZ_OK) return s;
    c = sub_ctx(ctx, &rl, SUB_LN2);
    if ((s = k->sub[SUB_LN2]->ops->backward(k->sub[SUB_LN2], &k->h, &k->a2, &k->d_a, &k->d_h, &c)) != NMZ_OK) return s;
    if ((s = add_t(be, &k->d_h, dy, &k->d_h)) != NMZ_OK) return s; /* + the residual path */
    /* h = x + drop(attention(LN1(x))) */
    if ((s = masked(be, &k->d_h, &k->m1, &k->d_o)) != NMZ_OK) return s;
    c = sub_ctx(ctx, &rl, SUB_ATTN);
    if ((s = k->sub[SUB_ATTN]->ops->backward(k->sub[SUB_ATTN], &k->a1, &k->o1, &k->d_o, &k->d_a, &c)) != NMZ_OK) return s;
    c = sub_ctx(ctx, &rl, SUB_LN1);
    if ((s = k->sub[SUB_LN1]->ops->backward(k->sub[SUB_LN1], x, &k->a1, &k->d_a, &k->d_x, &c)) != NMZ_OK) return s;
    if (dx && (s = add_t(be, &k->d_h, &k->d_x, dx)) != NMZ_OK) return s;
    if (NMZ_TRACING(ctx)) {
        nmz_trace_item it[3] = {{"dL/dy", dy}, {"dL/dh", &k->d_h}, {"dL/dx", dx}};
        nmz_trace_emit(ctx, NMZ_TRACE_BACKWARD, "block", NULL, 0, NULL, dx ? 3 : 2, it);
    }
    return NMZ_OK;
}

static int block_param_count(const nmz_layer *l) {
    (void)l;
    return 12;
}

static nmz_param block_param(nmz_layer *l, int i) {
    block_layer *k = l->self;
    nmz_param none = {NULL, NULL, NULL}, p;
    int j = i, s;
    if (i < 0 || i >= 12) return none;
    for (s = 0; s < N_SUB; s++) {
        int n = k->sub[s]->ops->param_count(k->sub[s]);
        if (j < n) {
            p = k->sub[s]->ops->param(k->sub[s], j);
            p.name = k_block_names[i];
            return p;
        }
        j -= n;
    }
    return none;
}

static nmz_status block_save(const nmz_layer *l, nmz_writer *w) {
    const block_layer *k = l->self;
    const ffn_layer *f = k->sub[SUB_FFN]->self;
    int heads = 0, hd_req = 0;
    nmz__attention_dims(k->sub[SUB_ATTN], &heads, NULL, &hd_req);
    nmz_write_u32(w, (uint32_t)heads);
    nmz_write_u32(w, (uint32_t)hd_req);
    nmz_write_u32(w, (uint32_t)f->F_req);
    nmz_write_u32(w, (uint32_t)f->act);
    return nmz_write_f32(w, k->rate);
}

static nmz_status block_load(nmz_layer **out, nmz_reader *r) {
    uint32_t heads = 0, hd = 0, F = 0, act = 0;
    float rate = 0;
    nmz_read_u32(r, &heads);
    nmz_read_u32(r, &hd);
    nmz_read_u32(r, &F);
    nmz_read_u32(r, &act);
    if (nmz_read_f32(r, &rate) != NMZ_OK) return r->err;
    if (heads == 0 || heads > 1024 || hd > 100000 || F > 1000000 || act >= NMZ_ACT_COUNT || act == NMZ_ACT_SOFTMAX ||
        !(rate >= 0.0f && rate < 1.0f))
        return nmz__fail(NMZ_ERR_FORMAT, "corrupt block config");
    *out = nmz_layer_block((int)heads, (int)hd, (int)F, (nmz_act)act, rate);
    return *out ? NMZ_OK : NMZ_ERR_NOMEM;
}

static void block_describe(const nmz_layer *l, char *buf, size_t n) {
    const block_layer *k = l->self;
    const ffn_layer *f = k->sub[SUB_FFN]->self;
    int heads = 0, hd = 0;
    nmz__attention_dims(k->sub[SUB_ATTN], &heads, &hd, NULL);
    if (k->rate > 0.0f)
        snprintf(buf, n, "block %d (%d heads x %d, ffn %d %s, dropout %g)", f->D, heads, hd, f->F, nmz_act_name(f->act),
                 (double)k->rate);
    else
        snprintf(buf, n, "block %d (%d heads x %d, ffn %d %s)", f->D, heads, hd, f->F, nmz_act_name(f->act));
}

static nmz_tensor *block_state(nmz_layer *l) {
    block_layer *k = l->self;
    return k->sub[SUB_ATTN]->ops->state(k->sub[SUB_ATTN]);
}

static void block_destroy(nmz_layer *l) {
    block_layer *k = l->self;
    nmz_tensor *bufs[] = {&k->a1, &k->o1, &k->r1,  &k->h,   &k->a2,  &k->o2, &k->r2,
                          &k->m1, &k->m2, &k->d_o, &k->d_a, &k->d_h, &k->d_x};
    size_t i;
    if (!k) return;
    for (i = 0; i < N_SUB; i++) nmz_layer_free(k->sub[i]);
    for (i = 0; i < sizeof bufs / sizeof bufs[0]; i++) nmz_tensor_free(bufs[i]);
    free(k);
}

/* its own residual buffers, then those of its four parts */
static int block_buffers(nmz_layer *l, nmz_tensor **out, int max) {
    block_layer *k = l->self;
    nmz_tensor *all[] = {&k->a1, &k->o1, &k->r1, &k->h,   &k->a2,  &k->o2, &k->r2,
                         &k->m1, &k->m2, &k->d_o, &k->d_a, &k->d_h, &k->d_x};
    int n = nmz__list(out, max, (int)(sizeof all / sizeof all[0]), all), i;
    for (i = 0; i < N_SUB; i++) {
        nmz_layer *s = k->sub[i];
        if (s->ops->buffers) n += s->ops->buffers(s, out + (n < max ? n : max), n < max ? max - n : 0);
    }
    return n;
}

const nmz_layer_ops nmz__block_ops = {"block",           block_setup, block_init, block_forward, block_backward,
                                      block_param_count, block_param, block_save, block_load,    block_describe,
                                      block_destroy,     block_state, NMZ_LAYER_BACKEND, NULL,
                                      block_buffers};

nmz_layer *nmz_layer_block(int heads, int head_dim, int ff_hidden, nmz_act act, float dropout) {
    block_layer *k;
    nmz_layer *l;
    int i;
    if (!(dropout >= 0.0f && dropout < 1.0f)) {
        nmz__fail(NMZ_ERR_ARG, "block dropout must be in [0, 1)");
        return NULL;
    }
    k = calloc(1, sizeof *k);
    if (!k) {
        nmz__fail(NMZ_ERR_NOMEM, "out of memory");
        return NULL;
    }
    k->rate = dropout;
    k->sub[SUB_LN1] = nmz_layer_layernorm();
    k->sub[SUB_ATTN] = nmz_layer_attention(heads, head_dim);
    k->sub[SUB_LN2] = nmz_layer_layernorm();
    k->sub[SUB_FFN] = nmz_layer_ffn(ff_hidden, act);
    for (i = 0; i < N_SUB; i++)
        if (!k->sub[i]) { /* the factory set the error */
            for (i = 0; i < N_SUB; i++) nmz_layer_free(k->sub[i]);
            free(k);
            return NULL;
        }
    l = nmz_layer_alloc(&nmz__block_ops, k);
    if (!l) {
        for (i = 0; i < N_SUB; i++) nmz_layer_free(k->sub[i]);
        free(k);
    }
    return l;
}

nmz_layer *nmz__block_sub(const nmz_layer *l, int i) {
    return l && l->ops == &nmz__block_ops && i >= 0 && i < N_SUB ? ((const block_layer *)l->self)->sub[i] : NULL;
}

/* ========================================================== tied output */

typedef struct tied_layer {
    int V, D;
    nmz_tensor b, db;
    nmz_tensor *E, *dE; /* the embedding's W and its gradient (not owned) */
    int emb_index;
} tied_layer;

static nmz_status tied_setup(nmz_layer *l, int in_rank, const int *in_shape, int *out_rank, int *out_shape) {
    tied_layer *t = l->self;
    nmz_status s;
    if ((s = need_rank2(in_rank, "tied output")) != NMZ_OK) return s;
    t->D = in_shape[1];
    if ((s = nmz_tensor_init(&t->b, 1, &t->V)) != NMZ_OK || (s = nmz_tensor_init(&t->db, 1, &t->V)) != NMZ_OK) return s;
    *out_rank = 2;
    out_shape[0] = in_shape[0];
    out_shape[1] = t->V;
    return NMZ_OK;
}

static nmz_status tied_bind(nmz_layer *l, nmz_layer *const *layers, int n_layers, int self_index) {
    tied_layer *t = l->self;
    int i;
    (void)n_layers;
    for (i = self_index - 1; i >= 0; i--)
        if (layers[i]->ops == &nmz__embedding_ops) {
            nmz_tensor *W = nmz__param(layers[i], 0).value;
            if (!W || W->rank != 2 || W->shape[0] != t->V || W->shape[1] != t->D)
                return nmz__fail(NMZ_ERR_SHAPE, "tied output: the embedding is %dx%d, the output needs %dx%d (vocabulary "
                                                "x features)",
                                 W ? W->shape[0] : 0, W ? W->shape[1] : 0, t->V, t->D);
            t->E = W;
            t->dE = nmz__param(layers[i], 0).grad;
            t->emb_index = i;
            return NMZ_OK;
        }
    return nmz__fail(NMZ_ERR_STATE, "tied output: no embedding layer before it to share weights with");
}

static nmz_status tied_init(nmz_layer *l, nmz_rng *rng, const nmz_layer *next) {
    (void)rng;
    (void)next;
    nmz_tensor_zero(&((tied_layer *)l->self)->b);
    return NMZ_OK;
}

static nmz_status tied_forward(nmz_layer *l, const nmz_tensor *x, nmz_tensor *y, const nmz_ctx *ctx) {
    tied_layer *t = l->self;
    const nmz_backend *be = ctx->backend;
    const int rows = x->shape[0] * x->shape[1];
    const float *px, *pE, *pb;
    float *py;
    nmz_status s;
    if (!t->E) return nmz__fail(NMZ_ERR_STATE, "tied output is not bound to an embedding (build the net first)");
    px = nmz__r(be, x);
    pE = nmz__r(be, t->E);
    pb = nmz__r(be, &t->b);
    py = nmz__w(be, y);
    if (!px || !pE || !pb || !py) return nmz__op_failed();
    if ((s = be->gemm(be, NMZ_MM_TRANS_B, rows, t->V, t->D, px, t->D, pE, t->D, py, t->V)) != NMZ_OK ||
        (s = be->add_bias(be, (size_t)rows, t->V, py, pb)) != NMZ_OK)
        return s;
    if (NMZ_TRACING(ctx)) {
        nmz_trace_scalar sc[1] = {{"embedding_layer", t->emb_index}};
        nmz_trace_item it[4] = {{"x", x}, {"W", t->E}, {"b", &t->b}, {"z", y}};
        nmz_trace_emit(ctx, NMZ_TRACE_FORWARD, "tied_output", NULL, 1, sc, 4, it);
    }
    return NMZ_OK;
}

static nmz_status tied_backward(nmz_layer *l, const nmz_tensor *x, const nmz_tensor *y, const nmz_tensor *dy,
                                nmz_tensor *dx, const nmz_ctx *ctx) {
    tied_layer *t = l->self;
    const nmz_backend *be = ctx->backend;
    const int rows = x->shape[0] * x->shape[1];
    const float *px, *pdy, *pE;
    float *pdE, *pdb, *pdx = NULL;
    nmz_status s;
    (void)y;
    if (!t->E) return nmz__fail(NMZ_ERR_STATE, "tied output is not bound to an embedding");
    px = nmz__r(be, x);
    pdy = nmz__r(be, dy);
    pE = nmz__r(be, t->E);
    pdE = nmz__u(be, t->dE);
    pdb = nmz__u(be, &t->db);
    if (dx) pdx = nmz__w(be, dx);
    if (!px || !pdy || !pE || !pdE || !pdb || (dx && !pdx)) return nmz__op_failed();
    /* ∂L/∂E += ∂L/∂zᵀ·x: ADDED to what the embedding itself contributes */
    if ((s = be->gemm(be, NMZ_MM_TRANS_A | NMZ_MM_ACCUMULATE, t->V, t->D, rows, pdy, t->V, px, t->D, pdE, t->D)) !=
            NMZ_OK ||
        (s = be->colsum_add(be, (size_t)rows, t->V, pdy, pdb)) != NMZ_OK)
        return s;
    if (dx && (s = be->gemm(be, 0, rows, t->D, t->V, pdy, t->V, pE, t->D, pdx, t->D)) != NMZ_OK) return s;
    if (NMZ_TRACING(ctx)) {
        nmz_trace_item it[3] = {{"dL/dz", dy}, {"dL/dW", t->dE}, {"dL/dx", dx}};
        nmz_trace_emit(ctx, NMZ_TRACE_BACKWARD, "tied_output", NULL, 0, NULL, dx ? 3 : 2, it);
    }
    return NMZ_OK;
}

static int tied_param_count(const nmz_layer *l) {
    (void)l;
    return 1;
}

static nmz_param tied_param(nmz_layer *l, int i) {
    tied_layer *t = l->self;
    nmz_param none = {NULL, NULL, NULL};
    return i == 0 ? mkparam("b", &t->b, &t->db) : none;
}

static nmz_status tied_save(const nmz_layer *l, nmz_writer *w) { return nmz_write_u32(w, (uint32_t)((const tied_layer *)l->self)->V); }

static nmz_status tied_load(nmz_layer **out, nmz_reader *r) {
    uint32_t V = 0;
    if (nmz_read_u32(r, &V) != NMZ_OK) return r->err;
    if (V == 0 || V > NMZ_MAX_VOCAB) return nmz__fail(NMZ_ERR_FORMAT, "corrupt tied_output config");
    *out = nmz_layer_tied_output((int)V);
    return *out ? NMZ_OK : NMZ_ERR_NOMEM;
}

static void tied_describe(const nmz_layer *l, char *buf, size_t n) {
    const tied_layer *t = l->self;
    snprintf(buf, n, "tied output %d->%d (z = x*E^T + b, E = the embedding)", t->D, t->V);
}

static void tied_destroy(nmz_layer *l) {
    tied_layer *t = l->self;
    if (!t) return;
    nmz_tensor_free(&t->b);
    nmz_tensor_free(&t->db);
    free(t);
}

const nmz_layer_ops nmz__tied_output_ops = {"tied_output",    tied_setup, tied_init, tied_forward, tied_backward,
                                            tied_param_count, tied_param, tied_save, tied_load,    tied_describe,
                                            tied_destroy,     NULL,       NMZ_LAYER_BACKEND, tied_bind, NULL};

nmz_layer *nmz_layer_tied_output(int vocab) {
    tied_layer *t;
    nmz_layer *l;
    if (vocab < 1 || vocab > NMZ_MAX_VOCAB) {
        nmz__fail(NMZ_ERR_ARG, "tied output needs the vocabulary size (1..%d)", NMZ_MAX_VOCAB);
        return NULL;
    }
    t = calloc(1, sizeof *t);
    if (!t) {
        nmz__fail(NMZ_ERR_NOMEM, "out of memory");
        return NULL;
    }
    t->V = vocab;
    t->emb_index = -1;
    l = nmz_layer_alloc(&nmz__tied_output_ops, t);
    if (!l) free(t);
    return l;
}

/* ================================================================ queries */

int nmz_layer_has_attention(const nmz_layer *l) {
    return l && (l->ops == &nmz__attention_ops || l->ops == &nmz__block_ops);
}

int nmz_net_context_length(const nmz_net *net) {
    int i, best = 0;
    if (!net) return 0;
    for (i = 0; i < net->n_layers; i++)
        if (net->layers[i]->ops == &nmz__posenc_ops) {
            int m = ((const pos_layer *)net->layers[i]->self)->max_len;
            if (best == 0 || m < best) best = m;
        }
    return best;
}

/* gradcheck: the side of 0 of every relu input inside an ffn (also inside a
 * block), so a ±ε step across a kink is recognised (decision 45) */
size_t nmz__kink_signature(const nmz_layer *l, float *out) {
    const ffn_layer *f;
    size_t e;
    if (l && l->ops == &nmz__block_ops) return nmz__kink_signature(((const block_layer *)l->self)->sub[SUB_FFN], out);
    if (!l || l->ops != &nmz__ffn_ops) return 0;
    f = l->self;
    if (f->act != NMZ_ACT_RELU && f->act != NMZ_ACT_LEAKY_RELU) return 0;
    if (out) {
        const float *z = nmz__hr(&f->z);
        for (e = 0; e < f->z.size; e++) out[e] = z[e] > 0.0f ? 1.0f : 0.0f;
    }
    return f->z.size;
}

int nmz__tied_embedding(const nmz_layer *l) {
    return l && l->ops == &nmz__tied_output_ops ? ((const tied_layer *)l->self)->emb_index : -1;
}
