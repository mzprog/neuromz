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
    nmz_tensor rec[5];  /* trace: per-step values [B, T, ·]            */
    nmz_tensor gda, grh, gprev, gdir, gdrh; /* generic path            */
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

/* The CPU kernels: the v2.0.1 loops (each row through all its steps, the
 * gate inputs summed in double). With rec, the trace's per-step values are
 * recorded instead of being emitted from inside the loop. */
nmz_status nmz__cpu_gru_forward(const nmz_backend *be, int B, int T, int D, int H, const float *x, const float *h0,
                                const float *Wx, const float *Wh, const float *b, float *y, float *gates) {
    const int G = 3 * H;
    double stack[1024], *acc = G <= 1024 ? stack : malloc((size_t)G * sizeof *acc);
    float *rh = malloc((size_t)H * sizeof *rh);
    int bi, t, i, j;
    (void)be;
    if (!acc || !rh) {
        if (acc != stack) free(acc);
        free(rh);
        return nmz__fail(NMZ_ERR_NOMEM, "out of memory");
    }
    for (bi = 0; bi < B; bi++)
        for (t = 0; t < T; t++) {
            size_t row = (size_t)bi * T + (size_t)t;
            const float *xt = x + row * D;
            const float *hp = t ? y + (row - 1) * H : h0 + (size_t)bi * H;
            float *gt = gates + row * G, *ht = y + row * H;
            for (j = 0; j < G; j++) acc[j] = b[j];
            for (i = 0; i < D; i++) {
                const float *w = Wx + (size_t)i * G;
                double xi = xt[i];
                if (xi != 0.0)
                    for (j = 0; j < G; j++) acc[j] += xi * w[j];
            }
            for (i = 0; i < H; i++) { /* z and r see h_{t−1} */
                const float *w = Wh + (size_t)i * G;
                double hi = hp[i];
                if (hi != 0.0)
                    for (j = 0; j < 2 * H; j++) acc[j] += hi * w[j];
            }
            for (j = 0; j < H; j++) {
                gt[j] = sigmoid(acc[j]);
                gt[H + j] = sigmoid(acc[H + j]);
                rh[j] = gt[H + j] * hp[j];
            }
            for (i = 0; i < H; i++) { /* the candidate sees r ⊙ h_{t−1} */
                const float *w = Wh + (size_t)i * G + 2 * H;
                double v = rh[i];
                if (v != 0.0)
                    for (j = 0; j < H; j++) acc[2 * H + j] += v * w[j];
            }
            for (j = 0; j < H; j++) {
                float z = gt[j], n = (float)tanh(acc[2 * H + j]);
                gt[2 * H + j] = n;
                ht[j] = z * hp[j] + (1.0f - z) * n;
            }
        }
    if (acc != stack) free(acc);
    free(rh);
    return NMZ_OK;
}

nmz_status nmz__cpu_gru_backward(const nmz_backend *be, int B, int T, int D, int H, const float *x, const float *y,
                                 const float *h0, const float *gates, const float *Wx, const float *Wh,
                                 const float *dy, float *dWx, float *dWh, float *db, float *dx, float *dh_next,
                                 const nmz_seq_record *rec) {
    const int G = 3 * H;
    float *da = malloc((size_t)G * sizeof *da), *rh = malloc((size_t)H * sizeof *rh);
    float *drh = malloc((size_t)H * sizeof *drh), *direct = malloc((size_t)H * sizeof *direct);
    int bi, t, i, j;
    (void)be;
    if (!da || !rh || !drh || !direct) {
        free(da);
        free(rh);
        free(drh);
        free(direct);
        return nmz__fail(NMZ_ERR_NOMEM, "out of memory");
    }
    memset(dh_next, 0, (size_t)B * H * sizeof(float));
    for (t = T - 1; t >= 0; t--) {
        for (bi = 0; bi < B; bi++) {
            size_t row = (size_t)bi * T + (size_t)t, rb = (size_t)bi * H;
            const float *xt = x + row * D;
            const float *hp = t ? y + (row - 1) * H : h0 + rb;
            const float *gt = gates + row * G;
            const float *gout = dy + row * H;
            float *dhn = dh_next + rb;
            for (j = 0; j < H; j++) {
                float z = gt[j], n = gt[2 * H + j];
                float dh = gout[j] + dhn[j];
                da[j] = dh * (hp[j] - n) * z * (1.0f - z);     /* ∂L/∂a_z */
                da[2 * H + j] = dh * (1.0f - z) * (1.0f - n * n); /* ∂L/∂a_ĥ */
                direct[j] = dh * z;
                rh[j] = gt[H + j] * hp[j];
                if (rec) {
                    rec->dh_next[row * H + j] = dhn[j];
                    rec->dh[row * H + j] = dh;
                    rec->direct[row * H + j] = direct[j];
                }
            }
            for (i = 0; i < H; i++) /* ∂L/∂(r⊙h_{t−1}) = ∂L/∂a_ĥ·W_hĥᵀ */
                drh[i] = (float)nmz__dot(Wh + (size_t)i * G + 2 * H, da + 2 * H, H);
            for (j = 0; j < H; j++) {
                float r = gt[H + j];
                da[H + j] = drh[j] * hp[j] * r * (1.0f - r); /* ∂L/∂a_r */
            }
            if (rec) memcpy(rec->da + row * G, da, (size_t)G * sizeof(float));
            for (j = 0; j < G; j++) db[j] += da[j];
            for (i = 0; i < D; i++) {
                float *dw = dWx + (size_t)i * G, xi = xt[i];
                if (xi != 0.0f)
                    for (j = 0; j < G; j++) dw[j] += xi * da[j];
            }
            for (i = 0; i < H; i++) {
                float *dw = dWh + (size_t)i * G, hi = hp[i], ri = rh[i];
                if (hi != 0.0f)
                    for (j = 0; j < 2 * H; j++) dw[j] += hi * da[j];
                if (ri != 0.0f)
                    for (j = 0; j < H; j++) dw[2 * H + j] += ri * da[2 * H + j];
            }
            if (dx) {
                float *dxt = dx + row * D;
                for (i = 0; i < D; i++) dxt[i] = (float)nmz__dot(Wx + (size_t)i * G, da, G);
            }
            for (i = 0; i < H; i++) { /* ∂L/∂h_{t−1}: direct + through r⊙h + through the z, r products */
                double acc = (double)direct[i] + (double)drh[i] * gt[H + i];
                acc += nmz__dot(Wh + (size_t)i * G, da, 2 * H);
                dhn[i] = (float)acc;
                if (rec) rec->dh_prev[row * H + i] = dhn[i];
            }
        }
    }
    free(da);
    free(rh);
    free(drh);
    free(direct);
    return NMZ_OK;
}

static nmz_rows rows_of(const float *p, int ld) {
    nmz_rows r;
    r.p = (float *)p;
    r.ld = ld;
    return r;
}

static nmz_status gru_forward(nmz_layer *l, const nmz_tensor *x, nmz_tensor *y, const nmz_ctx *ctx) {
    gru_layer *g = l->self;
    const nmz_backend *be = ctx->backend;
    int B = x->shape[0], T = x->shape[1], H = g->units, D = g->D, G = 3 * H, t;
    int shape[3];
    const float *px, *pWx, *pWh, *pb, *ph0;
    float *py, *pg;
    nmz_status s;
    shape[0] = B;
    shape[1] = H;
    if ((s = nmz_tensor_reshape(&g->h0, 2, shape)) != NMZ_OK) return s;
    if (ctx->carry && g->state.size == (size_t)B * H) { /* continue the stream */
        if ((s = nmz_tensor_copy(&g->h0, &g->state)) != NMZ_OK) return s;
    } else if ((s = nmz__zero_on(be, &g->h0)) != NMZ_OK) {
        return s;
    }
    shape[1] = T;
    shape[2] = G;
    if ((s = nmz_tensor_reshape(&g->gates, 3, shape)) != NMZ_OK) return s;
    px = nmz__r(be, x);
    pWx = nmz__r(be, &g->Wx);
    pWh = nmz__r(be, &g->Wh);
    pb = nmz__r(be, &g->b);
    ph0 = nmz__r(be, &g->h0);
    py = nmz__w(be, y);
    pg = nmz__w(be, &g->gates);
    if (!px || !pWx || !pWh || !pb || !ph0 || !py || !pg) return nmz__op_failed();
    /* a fused kernel may decline a shape it does not handle (NMZ_ERR_UNSUPPORTED) */
    s = NMZ_FUSED(be, gru_forward) ? be->gru_forward(be, B, T, D, H, px, ph0, pWx, pWh, pb, py, pg)
                                   : NMZ_ERR_UNSUPPORTED;
    if (s == NMZ_ERR_UNSUPPORTED) {
        float *prh;
        shape[0] = B;
        shape[1] = H;
        if ((s = nmz_tensor_reshape(&g->gdrh, 2, shape)) != NMZ_OK) return s;
        if (!(prh = nmz__w(be, &g->gdrh))) return nmz__op_failed();
        s = be->gemm(be, 0, B * T, G, D, px, D, pWx, G, pg, G); /* a = x·W_x + b for all steps */
        if (s == NMZ_OK) s = be->add_bias(be, (size_t)B * T, G, pg, pb);
        for (t = 0; t < T && s == NMZ_OK; t++) {
            nmz_rows a = rows_of(pg + (size_t)t * G, T * G), hp = t ? rows_of(py + (size_t)(t - 1) * H, T * H)
                                                                    : rows_of(ph0, H);
            s = nmz__rec_product(be, B, T, t, H, G, 0, 2 * H, ph0, py, pWh, pg); /* z, r see h_{t−1} */
            if (s == NMZ_OK) s = be->gru_cell_forward1(be, B, H, a, hp, rows_of(prh, H));
            if (s == NMZ_OK) /* the candidate sees r ⊙ h_{t−1} */
                s = be->gemm(be, NMZ_MM_ACCUMULATE, B, H, H, prh, H, pWh + 2 * H, G, a.p + 2 * H, T * G);
            if (s == NMZ_OK) s = be->gru_cell_forward2(be, B, H, a, hp, rows_of(py + (size_t)t * H, T * H));
        }
    }
    if (s != NMZ_OK) return s;
    if (ctx->carry) { /* remember h_{T−1} for the next batch of the same streams */
        float *pst;
        shape[0] = B;
        shape[1] = H;
        if ((s = nmz_tensor_reshape(&g->state, 2, shape)) != NMZ_OK) return s;
        if (!(pst = nmz__w(be, &g->state))) return nmz__op_failed();
        if ((s = be->copy(be, (size_t)B, (size_t)H, py + (size_t)(T - 1) * H, (size_t)T * H, pst, (size_t)H)) != NMZ_OK)
            return s;
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
    const nmz_backend *be = ctx->backend;
    int B = x->shape[0], T = x->shape[1], H = g->units, D = g->D, G = 3 * H, t;
    int tracing = NMZ_TRACING(ctx), shape[3];
    nmz_tensor *tp = g->tmp;
    nmz_seq_record rec = {0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    const float *px, *py, *ph0, *pg, *pWx, *pWh, *pdy;
    float *pdWx, *pdWh, *pdb, *pdx = NULL, *pdhn;
    nmz_status s;
    shape[0] = B;
    shape[1] = H;
    if ((s = nmz_tensor_reshape(&g->dh_next, 2, shape)) != NMZ_OK) return s;
    if (tracing) { /* dh_next dh da direct dh_prev */
        float **slots[5];
        int widths[5] = {H, H, 0, H, H};
        slots[0] = &rec.dh_next;
        slots[1] = &rec.dh;
        slots[2] = &rec.da;
        slots[3] = &rec.direct;
        slots[4] = &rec.dh_prev;
        if ((s = nmz__rec_alloc(be, g->rec, 5, B, T, G, &rec, slots, widths)) != NMZ_OK) return s;
    }
    px = nmz__r(be, x);
    py = nmz__r(be, y);
    ph0 = nmz__r(be, &g->h0);
    pg = nmz__r(be, &g->gates);
    pWx = nmz__r(be, &g->Wx);
    pWh = nmz__r(be, &g->Wh);
    pdy = nmz__r(be, dy);
    pdWx = nmz__u(be, &g->dWx);
    pdWh = nmz__u(be, &g->dWh);
    pdb = nmz__u(be, &g->db);
    pdhn = nmz__w(be, &g->dh_next);
    if (dx) pdx = nmz__w(be, dx);
    if (!px || !py || !ph0 || !pg || !pWx || !pWh || !pdy || !pdWx || !pdWh || !pdb || !pdhn || (dx && !pdx))
        return nmz__op_failed();
    s = NMZ_FUSED(be, gru_backward) ? be->gru_backward(be, B, T, D, H, px, py, ph0, pg, pWx, pWh, pdy, pdWx, pdWh, pdb,
                                                       pdx, pdhn, tracing ? &rec : NULL)
                                    : NMZ_ERR_UNSUPPORTED;
    if (s == NMZ_ERR_UNSUPPORTED) {
        float *pda, *prh, *pgp, *pdir, *pdrh;
        shape[1] = T;
        shape[2] = G;
        if ((s = nmz_tensor_reshape(&g->gda, 3, shape)) != NMZ_OK) return s;
        shape[2] = H;
        if ((s = nmz_tensor_reshape(&g->grh, 3, shape)) != NMZ_OK || (s = nmz_tensor_reshape(&g->gprev, 3, shape)) != NMZ_OK)
            return s;
        shape[1] = H;
        if ((s = nmz_tensor_reshape(&g->gdir, 2, shape)) != NMZ_OK || (s = nmz_tensor_reshape(&g->gdrh, 2, shape)) != NMZ_OK)
            return s;
        pda = nmz__w(be, &g->gda);
        prh = nmz__w(be, &g->grh);
        pgp = nmz__w(be, &g->gprev);
        pdir = nmz__w(be, &g->gdir);
        pdrh = nmz__w(be, &g->gdrh);
        if (!pda || !prh || !pgp || !pdir || !pdrh) return nmz__op_failed();
        s = be->fill(be, (size_t)B * H, 0.0f, pdhn);
        for (t = T - 1; t >= 0 && s == NMZ_OK; t--) {
            nmz_rows gt = rows_of(pg + (size_t)t * G, T * G), a = rows_of(pda + (size_t)t * G, T * G);
            nmz_rows hp = t ? rows_of(py + (size_t)(t - 1) * H, T * H) : rows_of(ph0, H);
            s = be->gru_cell_backward1(be, B, H, gt, hp, rows_of(pdy + (size_t)t * H, T * H), rows_of(pdhn, H), a,
                                       rows_of(pdir, H), rows_of(prh + (size_t)t * H, T * H), tracing ? &rec : NULL, t);
            if (s == NMZ_OK) /* ∂L/∂(r⊙h_{t−1}) = ∂L/∂a_ĥ·W_hĥᵀ */
                s = be->gemm(be, NMZ_MM_TRANS_B, B, H, H, a.p + 2 * H, T * G, pWh + 2 * H, G, pdrh, H);
            if (s == NMZ_OK) s = be->gru_cell_backward2(be, B, H, gt, hp, rows_of(pdrh, H), rows_of(pdir, H), a,
                                                        rows_of(pdhn, H));
            if (s == NMZ_OK) /* + [∂L/∂a_z ∂L/∂a_r]·[W_hz W_hr]ᵀ */
                s = be->gemm(be, NMZ_MM_TRANS_B | NMZ_MM_ACCUMULATE, B, H, 2 * H, a.p, T * G, pWh, G, pdhn, H);
            if (s == NMZ_OK && tracing)
                s = be->copy(be, (size_t)B, (size_t)H, pdhn, (size_t)H, rec.dh_prev + (size_t)t * H, (size_t)T * H);
        }
        if (s == NMZ_OK) s = be->gemm(be, NMZ_MM_TRANS_A | NMZ_MM_ACCUMULATE, D, G, B * T, px, D, pda, G, pdWx, G);
        if (s == NMZ_OK) s = nmz__rec_prev(be, B, T, H, ph0, py, pgp);
        if (s == NMZ_OK)
            s = be->gemm(be, NMZ_MM_TRANS_A | NMZ_MM_ACCUMULATE, H, 2 * H, B * T, pgp, H, pda, G, pdWh, G);
        if (s == NMZ_OK)
            s = be->gemm(be, NMZ_MM_TRANS_A | NMZ_MM_ACCUMULATE, H, H, B * T, prh, H, pda + 2 * H, G, pdWh + 2 * H, G);
        if (s == NMZ_OK) s = be->colsum_add(be, (size_t)B * T, G, pda, pdb);
        if (s == NMZ_OK && dx) s = be->gemm(be, NMZ_MM_TRANS_B, B * T, D, G, pda, G, pWx, G, pdx, D);
        if (s == NMZ_OK && tracing) s = be->copy(be, 1, (size_t)B * T * G, pda, (size_t)B * T * G, rec.da, (size_t)B * T * G);
    }
    if (s != NMZ_OK) return s;
    if (tracing) {
        for (t = T - 1; t >= 0; t--) {
            int k;
            if ((s = nmz__slice_t(dy, t, &tp[0])) != NMZ_OK) return s;
            for (k = 0; k < 5; k++)
                if ((s = nmz__slice_t(&g->rec[k], t, &tp[1 + k])) != NMZ_OK) return s;
            {
                nmz_trace_item it[6] = {{"dL/dh_out", &tp[0]}, {"dL/dh_next", &tp[1]},   {"dL/dh", &tp[2]},
                                        {"dL/da", &tp[3]},     {"dL/dh_direct", &tp[4]}, {"dL/dh_prev", &tp[5]}};
                nmz_trace_emit_step(ctx, NMZ_TRACE_BACKWARD, "gru", "z r h~", t, 6, it);
            }
        }
        {
            nmz_trace_scalar sc[1] = {{"steps", T}};
            nmz_trace_item all[5] = {{"dL/dW_x", &g->dWx}, {"dL/dW_h", &g->dWh}, {"dL/db", &g->db},
                                     {"W_h", &g->Wh},      {"dL/dx", dx}};
            nmz_trace_emit(ctx, NMZ_TRACE_BACKWARD, "gru", "z r h~", 1, sc, dx ? 5 : 4, all);
        }
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
    for (i = 0; i < 5; i++) nmz_tensor_free(&g->rec[i]);
    nmz_tensor_free(&g->gda);
    nmz_tensor_free(&g->grh);
    nmz_tensor_free(&g->gprev);
    nmz_tensor_free(&g->gdir);
    nmz_tensor_free(&g->gdrh);
    free(g);
}

static int gru_buffers(nmz_layer *l, nmz_tensor **out, int max) {
    gru_layer *g = l->self;
    nmz_tensor *all[] = {&g->h0, &g->state, &g->gates, &g->dh_next, &g->tmp[0], &g->tmp[1], &g->tmp[2], &g->tmp[3], &g->tmp[4], &g->tmp[5], &g->tmp[6], &g->tmp[7], &g->rec[0], &g->rec[1], &g->rec[2], &g->rec[3], &g->rec[4], &g->gda, &g->grh, &g->gprev, &g->gdir, &g->gdrh};
    return nmz__list(out, max, (int)(sizeof all / sizeof all[0]), all);
}

const nmz_layer_ops nmz__gru_ops = {"gru",          gru_setup, gru_init, gru_forward, gru_backward, gru_param_count,
                                    gru_param,      gru_save,  gru_load, gru_describe, gru_destroy, gru_state,
                                    NMZ_LAYER_BACKEND, NULL, gru_buffers};

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
