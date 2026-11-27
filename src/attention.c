/* attention.c — causal multi-head self-attention. x [B, T, D] → y [B, T, D].
 *
 * H heads of d = head_dim numbers, E = H·d. The projections sit side by
 * side in one matrix (as the LSTM gates do, decision 71):
 *     [q | k | v] = x·W_qkv + b_qkv               W_qkv [D, 3E], b_qkv [3E]
 * For every head h and every query position t (only keys j ≤ t: the causal
 * mask — a position may not look at characters that come after it):
 *     s_tj = q_t·k_j / √d                         raw scores
 *     a_tj = exp(s_tj) / Σ_{i≤t} exp(s_ti)        attention weights (softmax)
 *     c_t  = Σ_{j≤t} a_tj · v_j                   context: a weighted sum of values
 * The heads' contexts are put side by side and projected back:
 *     y_t  = [c_t(head 1) | … | c_t(head H)]·W_o + b_o     W_o [E, D], b_o [D]
 *
 * Backward (per head, ⊙ element-wise, everything for j ≤ t only):
 *     ∂L/∂c = ∂L/∂y·W_oᵀ                 ∂L/∂W_o += cᵀ·∂L/∂y
 *     ∂L/∂a_tj = ∂L/∂c_t · v_j            ∂L/∂v_j += Σ_t a_tj · ∂L/∂c_t
 *     ∂L/∂s_tj = a_tj (∂L/∂a_tj − Σ_i a_ti ∂L/∂a_ti)    (softmax Jacobian)
 *     ∂L/∂q_t = Σ_j ∂L/∂s_tj · k_j / √d   ∂L/∂k_j += Σ_t ∂L/∂s_tj · q_t / √d
 *     ∂L/∂W_qkv += xᵀ·∂L/∂[q|k|v]        ∂L/∂x = ∂L/∂[q|k|v]·W_qkvᵀ
 * A masked pair (j > t) has a_tj = 0 and gets no gradient, so the output at
 * position t — and a loss on it — never depends on x_j for j > t (tested).
 *
 * KV cache (generation): with carry on, the keys and values of every
 * position seen so far are the layer's state [B, Tc, 2E] (k | v per
 * position). A forward pass over T new positions appends to it and lets the
 * new queries attend to all Tc + T positions. Each query's scores, softmax
 * and weighted sum are computed by the same code with the same operation
 * order as without the cache, so the result is bit-identical (tested;
 * decision 98). The gradient into cached keys/values is dropped, like the
 * carried state of a recurrent layer; training never carries a cache. */
#include "internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct attn_layer {
    int heads, hd, hd_req, E, D;
    nmz_tensor Wqkv, bqkv, Wo, bo, dWqkv, dbqkv, dWo, dbo;
    nmz_tensor qkv;   /* [B, T, 3E] of the last forward                          */
    nmz_tensor kv;    /* [B, Tk, 2E]: keys | values of every visible position      */
    nmz_tensor P;     /* [B, H, T, Tk]: attention weights (0 where masked)         */
    nmz_tensor S;     /* [B, H, T, Tk]: raw scores (tracing only)                  */
    nmz_tensor cv;    /* [B, T, E]: contexts, heads side by side                   */
    nmz_tensor state; /* the KV cache [B, Tc, 2E] when carrying, or empty          */
    int past;         /* cached positions the last forward attended to (Tc)        */
    nmz_tensor dcv, dqkv, dkv, dP;
    nmz_tensor tmp[6];
} attn_layer;

/* columns col0 … col0+n−1 of a [B, R, C] tensor into [B, R, n] */
static nmz_status cols3(const nmz_tensor *src, int col0, int n, nmz_tensor *dst) {
    int shape[3], b, r, B = src->shape[0], R = src->shape[1], C = src->shape[2];
    const float *sp = nmz__hr(src);
    float *dp;
    nmz_status s;
    shape[0] = B;
    shape[1] = R;
    shape[2] = n;
    if ((s = nmz_tensor_reshape(dst, 3, shape)) != NMZ_OK) return s;
    dp = nmz__hw(dst);
    for (b = 0; b < B; b++)
        for (r = 0; r < R; r++)
            memcpy(dp + ((size_t)b * R + r) * n, sp + ((size_t)b * R + r) * C + col0, (size_t)n * sizeof(float));
    return NMZ_OK;
}

static nmz_status attn_setup(nmz_layer *l, int in_rank, const int *in_shape, int *out_rank, int *out_shape) {
    attn_layer *a = l->self;
    int s2[2], n;
    nmz_status s;
    if (in_rank != 2)
        return nmz__fail(NMZ_ERR_SHAPE, "attention needs [steps, features] per sample (e.g. after an embedding), got rank %d",
                         in_rank);
    a->D = in_shape[1];
    if (a->hd_req == 0) {
        if (a->D % a->heads)
            return nmz__fail(NMZ_ERR_SHAPE, "attention: %d features do not split into %d heads (give head_dim)", a->D,
                             a->heads);
        a->hd = a->D / a->heads;
    } else {
        a->hd = a->hd_req;
    }
    a->E = a->heads * a->hd;
    if (a->E > 100000) return nmz__fail(NMZ_ERR_ARG, "attention: heads x head_dim is too large");
    s2[0] = a->D;
    s2[1] = 3 * a->E;
    if ((s = nmz_tensor_init(&a->Wqkv, 2, s2)) != NMZ_OK || (s = nmz_tensor_init(&a->dWqkv, 2, s2)) != NMZ_OK) return s;
    n = 3 * a->E;
    if ((s = nmz_tensor_init(&a->bqkv, 1, &n)) != NMZ_OK || (s = nmz_tensor_init(&a->dbqkv, 1, &n)) != NMZ_OK) return s;
    s2[0] = a->E;
    s2[1] = a->D;
    if ((s = nmz_tensor_init(&a->Wo, 2, s2)) != NMZ_OK || (s = nmz_tensor_init(&a->dWo, 2, s2)) != NMZ_OK) return s;
    if ((s = nmz_tensor_init(&a->bo, 1, &a->D)) != NMZ_OK || (s = nmz_tensor_init(&a->dbo, 1, &a->D)) != NMZ_OK) return s;
    *out_rank = 2;
    out_shape[0] = in_shape[0];
    out_shape[1] = a->D;
    return NMZ_OK;
}

static void glorot(nmz_tensor *W, int fan_in, int fan_out, nmz_rng *rng) {
    double limit = sqrt(6.0 / (fan_in + fan_out));
    float *w = nmz__hw(W);
    size_t i;
    for (i = 0; i < W->size; i++) w[i] = (float)((2.0 * nmz_rng_uniform(rng) - 1.0) * limit);
}

static nmz_status attn_init(nmz_layer *l, nmz_rng *rng, const nmz_layer *next) {
    attn_layer *a = l->self;
    (void)next;
    glorot(&a->Wqkv, a->D, a->E, rng); /* each of q, k, v is a D → E projection */
    glorot(&a->Wo, a->E, a->D, rng);
    nmz_tensor_zero(&a->bqkv);
    nmz_tensor_zero(&a->bo);
    return NMZ_OK;
}

/* ------------------------------------------------- the CPU kernels
 * The core for one (row b, head h): scores, softmax, weighted sum — the
 * v2.0.1 loops, the dot products in double. */
typedef struct core {
    int H, T, Tk, Tp, d, E;
    const float *qkv, *kv, *P, *dcv;
    float *Pw, *S, *cv, *dP, *dqkv, *dkv;
} core;

static void attend(const core *a, int b, int h) {
    const int E = a->E, d = a->d, H = a->H, T = a->T, Tk = a->Tk, Tp = a->Tp;
    const double scale = 1.0 / sqrt((double)d);
    int t, j, k;
    for (t = 0; t < T; t++) {
        const int last = Tp + t; /* keys 0 … last are visible */
        const float *q = a->qkv + ((size_t)b * T + t) * 3 * E + (size_t)h * d;
        float *p = a->Pw + (((size_t)b * H + h) * T + t) * Tk;
        float *c = a->cv + ((size_t)b * T + t) * E + (size_t)h * d;
        double mx = -HUGE_VAL, sum = 0.0;
        for (j = 0; j <= last; j++) {
            const float *kj = a->kv + ((size_t)b * Tk + j) * 2 * E + (size_t)h * d;
            p[j] = (float)(nmz__dot(q, kj, d) * scale);
            if (p[j] > mx) mx = p[j];
        }
        if (a->S) memcpy(a->S + (((size_t)b * H + h) * T + t) * Tk, p, (size_t)(last + 1) * sizeof(float));
        for (j = 0; j <= last; j++) {
            double e = exp((double)p[j] - mx);
            sum += e;
            p[j] = (float)e;
        }
        for (j = 0; j <= last; j++) p[j] = (float)(p[j] / sum);
        for (j = last + 1; j < Tk; j++) p[j] = 0.0f; /* masked: the future */
        for (k = 0; k < d; k++) c[k] = 0.0f;
        for (j = 0; j <= last; j++) {
            const float pj = p[j], *v = a->kv + ((size_t)b * Tk + j) * 2 * E + E + (size_t)h * d;
            for (k = 0; k < d; k++) c[k] += pj * v[k];
        }
    }
}

nmz_status nmz__cpu_attention_forward(const nmz_backend *be, int B, int H, int T, int Tk, int Tp, int d, int E,
                                      const float *qkv, const float *kv, float *P, float *S, float *ctx) {
    core a;
    int bh;
    (void)be;
    memset(&a, 0, sizeof a);
    a.H = H, a.T = T, a.Tk = Tk, a.Tp = Tp, a.d = d, a.E = E;
    a.qkv = qkv, a.kv = kv, a.Pw = P, a.S = S, a.cv = ctx;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (B * H > 1 && (double)T * Tk * d > 2e4)
#endif
    for (bh = 0; bh < B * H; bh++) attend(&a, bh / H, bh % H);
    return NMZ_OK;
}

/* backward of the core for one (row b, head h) */
static void attend_back(const core *a, int b, int h) {
    const int E = a->E, d = a->d, H = a->H, T = a->T, Tk = a->Tk, Tp = a->Tp;
    const double scale = 1.0 / sqrt((double)d);
    int t, j, k;
    for (t = 0; t < T; t++) {
        const int last = Tp + t;
        const float *q = a->qkv + ((size_t)b * T + t) * 3 * E + (size_t)h * d;
        const float *p = a->P + (((size_t)b * H + h) * T + t) * Tk;
        const float *dc = a->dcv + ((size_t)b * T + t) * E + (size_t)h * d;
        float *dp = a->dP + (((size_t)b * H + h) * T + t) * Tk;
        float *dq = a->dqkv + ((size_t)b * T + t) * 3 * E + (size_t)h * d;
        double dot = 0.0;
        for (j = 0; j <= last; j++) {
            const float *v = a->kv + ((size_t)b * Tk + j) * 2 * E + E + (size_t)h * d;
            float *dv = a->dkv + ((size_t)b * Tk + j) * 2 * E + E + (size_t)h * d;
            const float pj = p[j];
            double g = nmz__dot(dc, v, d); /* ∂L/∂a_tj */
            dot += pj * g;
            dp[j] = (float)g;
            for (k = 0; k < d; k++) dv[k] += pj * dc[k];
        }
        for (j = 0; j <= last; j++) {
            const float *kj = a->kv + ((size_t)b * Tk + j) * 2 * E + (size_t)h * d;
            float *dk = a->dkv + ((size_t)b * Tk + j) * 2 * E + (size_t)h * d;
            const float ds = (float)(p[j] * (dp[j] - dot)); /* ∂L/∂s_tj */
            const float g = (float)(ds * scale);
            dp[j] = ds;
            for (k = 0; k < d; k++) {
                dq[k] += g * kj[k];
                dk[k] += g * q[k];
            }
        }
        for (j = last + 1; j < Tk; j++) dp[j] = 0.0f;
    }
}

nmz_status nmz__cpu_attention_backward(const nmz_backend *be, int B, int H, int T, int Tk, int Tp, int d, int E,
                                       const float *qkv, const float *kv, const float *P, const float *dctx,
                                       float *dP, float *dqkv, float *dkv) {
    core a;
    int bh;
    (void)be;
    memset(&a, 0, sizeof a);
    a.H = H, a.T = T, a.Tk = Tk, a.Tp = Tp, a.d = d, a.E = E;
    a.qkv = qkv, a.kv = kv, a.P = P, a.dcv = dctx, a.dP = dP, a.dqkv = dqkv, a.dkv = dkv;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (B * H > 1 && (double)T * Tk * d > 2e4)
#endif
    for (bh = 0; bh < B * H; bh++) attend_back(&a, bh / H, bh % H);
    return NMZ_OK;
}

/* rows of keys | values: kv[b, Tp + t] = qkv[b, t] columns E … 3E (and back
 * for the gradient, to_kv = 0) */
static nmz_status kv_rows(const nmz_backend *be, int B, int T, int Tk, int Tp, int E, float *qkv, float *kv,
                          int to_kv) {
    int b;
    nmz_status s = NMZ_OK;
    if (Tp == 0) /* all rows line up: one strided copy */
        return to_kv ? be->copy(be, (size_t)B * T, (size_t)2 * E, qkv + E, (size_t)3 * E, kv, (size_t)2 * E)
                     : be->copy(be, (size_t)B * T, (size_t)2 * E, kv, (size_t)2 * E, qkv + E, (size_t)3 * E);
    for (b = 0; b < B && s == NMZ_OK; b++) {
        float *q = qkv + (size_t)b * T * 3 * E + E, *k = kv + ((size_t)b * Tk + Tp) * 2 * E;
        s = to_kv ? be->copy(be, (size_t)T, (size_t)2 * E, q, (size_t)3 * E, k, (size_t)2 * E)
                  : be->copy(be, (size_t)T, (size_t)2 * E, k, (size_t)2 * E, q, (size_t)3 * E);
    }
    return s;
}

static nmz_status attn_forward(nmz_layer *l, const nmz_tensor *x, nmz_tensor *y, const nmz_ctx *ctx) {
    attn_layer *a = l->self;
    const nmz_backend *be = ctx->backend;
    const int B = x->shape[0], T = x->shape[1], D = a->D, E = a->E, H = a->heads;
    int Tp = 0, Tk, shape[4], tracing = NMZ_TRACING(ctx);
    const float *px, *pWqkv, *pbqkv, *pWo, *pbo;
    float *pqkv, *pkv, *pP, *pS = NULL, *pcv, *py;
    nmz_status s;
    if (ctx->carry && ctx->training)
        return nmz__fail(NMZ_ERR_UNSUPPORTED, "attention layers do not carry state in training (the cache would hold "
                                              "keys with no gradient path); train with carry_state = 0");
    if (ctx->carry && a->state.size > 0 && a->state.rank == 3 && a->state.shape[0] == B) Tp = a->state.shape[1];
    Tk = Tp + T;
    a->past = Tp;
    shape[0] = B;
    shape[1] = T;
    shape[2] = 3 * E;
    if ((s = nmz_tensor_reshape(&a->qkv, 3, shape)) != NMZ_OK) return s;
    shape[1] = Tk;
    shape[2] = 2 * E;
    if ((s = nmz_tensor_reshape(&a->kv, 3, shape)) != NMZ_OK) return s;
    shape[0] = B;
    shape[1] = H;
    shape[2] = T;
    shape[3] = Tk;
    if ((s = nmz_tensor_reshape(&a->P, 4, shape)) != NMZ_OK) return s;
    if (tracing) {
        if ((s = nmz_tensor_reshape(&a->S, 4, shape)) != NMZ_OK || (s = nmz__zero_on(be, &a->S)) != NMZ_OK) return s;
        if (!(pS = nmz__u(be, &a->S))) return nmz__op_failed();
    }
    shape[1] = T;
    shape[2] = E;
    if ((s = nmz_tensor_reshape(&a->cv, 3, shape)) != NMZ_OK) return s;
    px = nmz__r(be, x);
    pWqkv = nmz__r(be, &a->Wqkv);
    pbqkv = nmz__r(be, &a->bqkv);
    pWo = nmz__r(be, &a->Wo);
    pbo = nmz__r(be, &a->bo);
    pqkv = nmz__w(be, &a->qkv);
    pkv = nmz__w(be, &a->kv);
    pP = nmz__w(be, &a->P);
    pcv = nmz__w(be, &a->cv);
    py = nmz__w(be, y);
    if (!px || !pWqkv || !pbqkv || !pWo || !pbo || !pqkv || !pkv || !pP || !pcv || !py) return nmz__op_failed();
    /* [q | k | v] = x·W_qkv + b_qkv */
    if ((s = be->gemm(be, 0, B * T, 3 * E, D, px, D, pWqkv, 3 * E, pqkv, 3 * E)) != NMZ_OK ||
        (s = be->add_bias(be, (size_t)B * T, 3 * E, pqkv, pbqkv)) != NMZ_OK)
        return s;
    if (Tp) { /* the cached positions, then the new keys | values */
        const float *pst = nmz__r(be, &a->state);
        if (!pst) return nmz__op_failed();
        if ((s = be->copy(be, (size_t)B, (size_t)Tp * 2 * E, pst, (size_t)Tp * 2 * E, pkv, (size_t)Tk * 2 * E)) != NMZ_OK)
            return s;
    }
    if ((s = kv_rows(be, B, T, Tk, Tp, E, pqkv, pkv, 1)) != NMZ_OK) return s;
    if ((s = be->attention_forward(be, B, H, T, Tk, Tp, a->hd, E, pqkv, pkv, pP, pS, pcv)) != NMZ_OK) return s;
    if ((s = be->gemm(be, 0, B * T, D, E, pcv, E, pWo, D, py, D)) != NMZ_OK ||
        (s = be->add_bias(be, (size_t)B * T, D, py, pbo)) != NMZ_OK)
        return s;
    if (ctx->carry && (s = nmz_tensor_copy(&a->state, &a->kv)) != NMZ_OK) return s;
    if (tracing) {
        nmz_tensor *tp = a->tmp;
        if ((s = cols3(&a->qkv, 0, E, &tp[0])) != NMZ_OK || (s = cols3(&a->kv, 0, E, &tp[1])) != NMZ_OK ||
            (s = cols3(&a->kv, E, E, &tp[2])) != NMZ_OK)
            return s;
        {
            nmz_trace_scalar sc[3] = {{"heads", H}, {"head_dim", a->hd}, {"past", Tp}};
            nmz_trace_item it[10] = {{"x", x},           {"W_qkv", &a->Wqkv}, {"q", &tp[0]},      {"k", &tp[1]},
                                     {"v", &tp[2]},      {"scores", &a->S},   {"weights", &a->P}, {"context", &a->cv},
                                     {"W_o", &a->Wo},    {"y", y}};
            nmz_trace_emit(ctx, NMZ_TRACE_FORWARD, "attention", NULL, 3, sc, 10, it);
        }
    }
    return NMZ_OK;
}

static nmz_status attn_backward(nmz_layer *l, const nmz_tensor *x, const nmz_tensor *y, const nmz_tensor *dy,
                                nmz_tensor *dx, const nmz_ctx *ctx) {
    attn_layer *a = l->self;
    const nmz_backend *be = ctx->backend;
    const int B = x->shape[0], T = x->shape[1], D = a->D, E = a->E, H = a->heads, Tp = a->past, Tk = Tp + T;
    const float *px, *pdy, *pcv, *pWo, *pqkv, *pkv, *pP, *pWqkv;
    float *pdWo, *pdbo, *pdcv, *pdkv, *pdqkv, *pdP, *pdWqkv, *pdbqkv, *pdx = NULL;
    nmz_status s;
    (void)y;
    if (a->P.rank != 4 || a->P.shape[0] != B || a->P.shape[2] != T || a->P.shape[3] != Tk)
        return nmz__fail(NMZ_ERR_STATE, "attention: backward without a matching forward");
    if ((s = nmz_tensor_reshape(&a->dcv, 3, a->cv.shape)) != NMZ_OK || (s = nmz_tensor_reshape(&a->dkv, 3, a->kv.shape)) != NMZ_OK ||
        (s = nmz_tensor_reshape(&a->dqkv, 3, a->qkv.shape)) != NMZ_OK || (s = nmz_tensor_reshape(&a->dP, 4, a->P.shape)) != NMZ_OK)
        return s;
    px = nmz__r(be, x);
    pdy = nmz__r(be, dy);
    pcv = nmz__r(be, &a->cv);
    pWo = nmz__r(be, &a->Wo);
    pqkv = nmz__r(be, &a->qkv);
    pkv = nmz__r(be, &a->kv);
    pP = nmz__r(be, &a->P);
    pWqkv = nmz__r(be, &a->Wqkv);
    pdWo = nmz__u(be, &a->dWo);
    pdbo = nmz__u(be, &a->dbo);
    pdcv = nmz__w(be, &a->dcv);
    pdkv = nmz__w(be, &a->dkv);
    pdqkv = nmz__w(be, &a->dqkv);
    pdP = nmz__w(be, &a->dP);
    pdWqkv = nmz__u(be, &a->dWqkv);
    pdbqkv = nmz__u(be, &a->dbqkv);
    if (dx) pdx = nmz__w(be, dx);
    if (!px || !pdy || !pcv || !pWo || !pqkv || !pkv || !pP || !pWqkv || !pdWo || !pdbo || !pdcv || !pdkv || !pdqkv ||
        !pdP || !pdWqkv || !pdbqkv || (dx && !pdx))
        return nmz__op_failed();
    /* output projection */
    if ((s = be->gemm(be, NMZ_MM_TRANS_A | NMZ_MM_ACCUMULATE, E, D, B * T, pcv, E, pdy, D, pdWo, D)) != NMZ_OK ||
        (s = be->colsum_add(be, (size_t)B * T, D, pdy, pdbo)) != NMZ_OK)
        return s;
    if ((s = be->gemm(be, NMZ_MM_TRANS_B, B * T, E, D, pdy, D, pWo, D, pdcv, E)) != NMZ_OK) return s;
    /* the heads */
    if ((s = be->fill(be, a->dkv.size, 0.0f, pdkv)) != NMZ_OK || (s = be->fill(be, a->dqkv.size, 0.0f, pdqkv)) != NMZ_OK)
        return s;
    if ((s = be->attention_backward(be, B, H, T, Tk, Tp, a->hd, E, pqkv, pkv, pP, pdcv, pdP, pdqkv, pdkv)) != NMZ_OK)
        return s;
    /* the new positions' keys | values (cached ones get no gradient) */
    if ((s = kv_rows(be, B, T, Tk, Tp, E, pdqkv, pdkv, 0)) != NMZ_OK) return s;
    /* input projection */
    if ((s = be->gemm(be, NMZ_MM_TRANS_A | NMZ_MM_ACCUMULATE, D, 3 * E, B * T, px, D, pdqkv, 3 * E, pdWqkv, 3 * E)) !=
            NMZ_OK ||
        (s = be->colsum_add(be, (size_t)B * T, 3 * E, pdqkv, pdbqkv)) != NMZ_OK)
        return s;
    if (dx && (s = be->gemm(be, NMZ_MM_TRANS_B, B * T, D, 3 * E, pdqkv, 3 * E, pWqkv, 3 * E, pdx, D)) != NMZ_OK) return s;
    if (NMZ_TRACING(ctx)) {
        nmz_tensor *tp = a->tmp;
        if ((s = cols3(&a->dqkv, 0, E, &tp[3])) != NMZ_OK || (s = cols3(&a->dkv, 0, E, &tp[4])) != NMZ_OK ||
            (s = cols3(&a->dkv, E, E, &tp[5])) != NMZ_OK)
            return s;
        {
            nmz_trace_item it[7] = {{"dL/dy", dy},       {"dL/dcontext", &a->dcv}, {"dL/dscores", &a->dP},
                                    {"dL/dq", &tp[3]},   {"dL/dk", &tp[4]},        {"dL/dv", &tp[5]},
                                    {"dL/dx", dx}};
            nmz_trace_emit(ctx, NMZ_TRACE_BACKWARD, "attention", NULL, 0, NULL, dx ? 7 : 6, it);
        }
    }
    return NMZ_OK;
}

static int attn_param_count(const nmz_layer *l) {
    (void)l;
    return 4;
}

static nmz_param attn_param(nmz_layer *l, int i) {
    attn_layer *a = l->self;
    nmz_param p = {NULL, NULL, NULL};
    switch (i) {
    case 0: p.name = "W_qkv", p.value = &a->Wqkv, p.grad = &a->dWqkv; break;
    case 1: p.name = "b_qkv", p.value = &a->bqkv, p.grad = &a->dbqkv; break;
    case 2: p.name = "W_o", p.value = &a->Wo, p.grad = &a->dWo; break;
    case 3: p.name = "b_o", p.value = &a->bo, p.grad = &a->dbo; break;
    default: break;
    }
    return p;
}

static nmz_status attn_save(const nmz_layer *l, nmz_writer *w) {
    const attn_layer *a = l->self;
    nmz_write_u32(w, (uint32_t)a->heads);
    return nmz_write_u32(w, (uint32_t)a->hd_req);
}

static nmz_status attn_load(nmz_layer **out, nmz_reader *r) {
    uint32_t heads = 0, hd = 0;
    nmz_read_u32(r, &heads);
    if (nmz_read_u32(r, &hd) != NMZ_OK) return r->err;
    if (heads == 0 || heads > 1024 || hd > 100000) return nmz__fail(NMZ_ERR_FORMAT, "corrupt attention config");
    *out = nmz_layer_attention((int)heads, (int)hd);
    return *out ? NMZ_OK : NMZ_ERR_NOMEM;
}

static void attn_describe(const nmz_layer *l, char *buf, size_t n) {
    const attn_layer *a = l->self;
    snprintf(buf, n, "attention %d->%d (%d heads x %d, causal)", a->D, a->D, a->heads, a->hd);
}

static nmz_tensor *attn_state(nmz_layer *l) { return &((attn_layer *)l->self)->state; }

static void attn_destroy(nmz_layer *l) {
    attn_layer *a = l->self;
    nmz_tensor *all[] = {&a->Wqkv, &a->bqkv, &a->Wo, &a->bo, &a->dWqkv, &a->dbqkv, &a->dWo, &a->dbo, &a->qkv,
                         &a->kv,   &a->P,    &a->S,  &a->cv, &a->state, &a->dcv,   &a->dqkv,  &a->dkv, &a->dP};
    size_t i;
    if (!a) return;
    for (i = 0; i < sizeof all / sizeof all[0]; i++) nmz_tensor_free(all[i]);
    for (i = 0; i < 6; i++) nmz_tensor_free(&a->tmp[i]);
    free(a);
}

static int attn_buffers(nmz_layer *l, nmz_tensor **out, int max) {
    attn_layer *a = l->self;
    nmz_tensor *all[] = {&a->qkv, &a->kv, &a->P, &a->S, &a->cv, &a->state, &a->dcv, &a->dqkv, &a->dkv, &a->dP, &a->tmp[0], &a->tmp[1], &a->tmp[2], &a->tmp[3], &a->tmp[4], &a->tmp[5]};
    return nmz__list(out, max, (int)(sizeof all / sizeof all[0]), all);
}

const nmz_layer_ops nmz__attention_ops = {"attention",  attn_setup, attn_init,     attn_forward, attn_backward,
                                          attn_param_count, attn_param, attn_save, attn_load,    attn_describe,
                                          attn_destroy, attn_state, NMZ_LAYER_BACKEND, NULL, attn_buffers};

nmz_layer *nmz_layer_attention(int heads, int head_dim) {
    attn_layer *a;
    nmz_layer *l;
    if (heads < 1 || heads > 1024 || head_dim < 0 || head_dim > 100000) {
        nmz__fail(NMZ_ERR_ARG, "attention needs 1..1024 heads and a head size >= 0 (0 = features / heads)");
        return NULL;
    }
    a = calloc(1, sizeof *a);
    if (!a) {
        nmz__fail(NMZ_ERR_NOMEM, "out of memory");
        return NULL;
    }
    a->heads = heads;
    a->hd_req = a->hd = head_dim;
    l = nmz_layer_alloc(&nmz__attention_ops, a);
    if (!l) free(a);
    return l;
}

void nmz__attention_dims(const nmz_layer *l, int *heads, int *head_dim, int *head_dim_req) {
    const attn_layer *a = l->self;
    if (heads) *heads = a->heads;
    if (head_dim) *head_dim = a->hd;
    if (head_dim_req) *head_dim_req = a->hd_req;
}
