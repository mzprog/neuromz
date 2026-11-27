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
    nmz_tensor rec[7];           /* trace: per-step values [B, T, ·]          */
    nmz_tensor gda, gprev;       /* generic path: ∂L/∂a [B, T, 4H], h_{t−1}    */
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
    {
        float *b = nmz__hu(&m->b);
        for (j = 0; j < m->units; j++) b[m->units + j] = m->forget_bias; /* f starts ≈ σ(1) = 0.73 */
    }
    return nmz__recurrent_init(m->init, &m->resolved, &m->Wx, &m->Wh, m->D, m->units, 4, rng);
}

/* The CPU kernels: the v2.0.1 loops (each row through all its steps, the
 * gate inputs summed in double). With rec, the trace's per-step values are
 * recorded instead of being emitted from inside the loop. */
nmz_status nmz__cpu_lstm_forward(const nmz_backend *be, int B, int T, int D, int H, const float *x, const float *h0,
                                 const float *c0, const float *Wx, const float *Wh, const float *b, float *y,
                                 float *gates, float *c) {
    const int G = 4 * H;
    double stack[1024], *acc = G <= 1024 ? stack : malloc((size_t)G * sizeof *acc);
    int bi, t, i, j;
    (void)be;
    if (!acc) return nmz__fail(NMZ_ERR_NOMEM, "out of memory");
    for (bi = 0; bi < B; bi++)
        for (t = 0; t < T; t++) {
            size_t row = (size_t)bi * T + (size_t)t;
            const float *xt = x + row * D;
            const float *hp = t ? y + (row - 1) * H : h0 + (size_t)bi * H;
            const float *cp = t ? c + (row - 1) * H : c0 + (size_t)bi * H;
            float *gt = gates + row * G, *ct = c + row * H, *ht = y + row * H;
            for (j = 0; j < G; j++) acc[j] = b[j];
            for (i = 0; i < D; i++) {
                const float *w = Wx + (size_t)i * G;
                double xi = xt[i];
                if (xi != 0.0)
                    for (j = 0; j < G; j++) acc[j] += xi * w[j];
            }
            for (i = 0; i < H; i++) {
                const float *w = Wh + (size_t)i * G;
                double hi = hp[i];
                if (hi != 0.0)
                    for (j = 0; j < G; j++) acc[j] += hi * w[j];
            }
            for (j = 0; j < H; j++) {
                float ig = sigmoid(acc[j]), fg = sigmoid(acc[H + j]);
                float gg = (float)tanh(acc[2 * H + j]), og = sigmoid(acc[3 * H + j]);
                float cv = fg * cp[j] + ig * gg;
                gt[j] = ig;
                gt[H + j] = fg;
                gt[2 * H + j] = gg;
                gt[3 * H + j] = og;
                ct[j] = cv;
                ht[j] = og * tanhf(cv);
            }
        }
    if (acc != stack) free(acc);
    return NMZ_OK;
}

nmz_status nmz__cpu_lstm_backward(const nmz_backend *be, int B, int T, int D, int H, const float *x, const float *y,
                                  const float *h0, const float *c0, const float *gates, const float *c,
                                  const float *Wx, const float *Wh, const float *dy, float *dWx, float *dWh,
                                  float *db, float *dx, float *dh_next, float *dc_next, const nmz_seq_record *rec) {
    const int G = 4 * H;
    float *da = malloc((size_t)G * sizeof *da);
    int bi, t, i, j;
    (void)be;
    if (!da) return nmz__fail(NMZ_ERR_NOMEM, "out of memory");
    memset(dh_next, 0, (size_t)B * H * sizeof(float)); /* nothing comes from beyond the last step */
    memset(dc_next, 0, (size_t)B * H * sizeof(float));
    for (t = T - 1; t >= 0; t--) {
        for (bi = 0; bi < B; bi++) {
            size_t row = (size_t)bi * T + (size_t)t, rb = (size_t)bi * H;
            const float *xt = x + row * D;
            const float *hp = t ? y + (row - 1) * H : h0 + rb;
            const float *cp = t ? c + (row - 1) * H : c0 + rb;
            const float *gt = gates + row * G, *ct = c + row * H;
            const float *gout = dy + row * H;
            float *dhn = dh_next + rb, *dcn = dc_next + rb;
            for (j = 0; j < H; j++) {
                float ig = gt[j], fg = gt[H + j], gg = gt[2 * H + j], og = gt[3 * H + j];
                float tc = tanhf(ct[j]);
                float dh = gout[j] + dhn[j];               /* from above at t + from step t+1 */
                float dc = dcn[j] + dh * og * (1.0f - tc * tc); /* along c + through h_t = o·tanh(c_t) */
                da[j] = dc * gg * ig * (1.0f - ig);
                da[H + j] = dc * cp[j] * fg * (1.0f - fg);
                da[2 * H + j] = dc * ig * (1.0f - gg * gg);
                da[3 * H + j] = dh * tc * og * (1.0f - og);
                if (rec) {
                    rec->dh_next[row * H + j] = dhn[j];
                    rec->dh[row * H + j] = dh;
                    rec->dc_next[row * H + j] = dcn[j];
                    rec->dc[row * H + j] = dc;
                    rec->dc_prev[row * H + j] = dc * fg;
                }
                dcn[j] = dc * fg; /* ∂L/∂c_{t−1}: only the forget gate in between */
            }
            if (rec) memcpy(rec->da + row * G, da, (size_t)G * sizeof(float));
            for (j = 0; j < G; j++) db[j] += da[j];
            for (i = 0; i < D; i++) {
                float *dw = dWx + (size_t)i * G, xi = xt[i];
                if (xi != 0.0f)
                    for (j = 0; j < G; j++) dw[j] += xi * da[j];
            }
            for (i = 0; i < H; i++) {
                float *dw = dWh + (size_t)i * G, hi = hp[i];
                if (hi != 0.0f)
                    for (j = 0; j < G; j++) dw[j] += hi * da[j];
            }
            if (dx) {
                float *dxt = dx + row * D;
                for (i = 0; i < D; i++) dxt[i] = (float)nmz__dot(Wx + (size_t)i * G, da, G);
            }
            for (i = 0; i < H; i++) { /* ∂L/∂h_{t−1} = ∂L/∂a·W_hᵀ, overwrites dh_next */
                dhn[i] = (float)nmz__dot(Wh + (size_t)i * G, da, G);
                if (rec) rec->dh_prev[row * H + i] = dhn[i];
            }
        }
    }
    free(da);
    return NMZ_OK;
}

static nmz_status lstm_initial(const nmz_backend *be, const nmz_ctx *ctx, lstm_layer *m, int B) {
    const int H = m->units;
    int shape[2];
    nmz_status s;
    shape[0] = B;
    shape[1] = H;
    if ((s = nmz_tensor_reshape(&m->h0, 2, shape)) != NMZ_OK || (s = nmz_tensor_reshape(&m->c0, 2, shape)) != NMZ_OK)
        return s;
    if (ctx->carry && m->state.size == (size_t)B * 2 * H) { /* continue the streams: h | c */
        const float *ps = nmz__r(be, &m->state);
        float *ph = nmz__w(be, &m->h0), *pc = nmz__w(be, &m->c0);
        if (!ps || !ph || !pc) return nmz__op_failed();
        if ((s = be->copy(be, (size_t)B, (size_t)H, ps, (size_t)2 * H, ph, (size_t)H)) != NMZ_OK) return s;
        return be->copy(be, (size_t)B, (size_t)H, ps + H, (size_t)2 * H, pc, (size_t)H);
    }
    if ((s = nmz__zero_on(be, &m->h0)) != NMZ_OK) return s;
    return nmz__zero_on(be, &m->c0);
}

static nmz_status lstm_forward(nmz_layer *l, const nmz_tensor *x, nmz_tensor *y, const nmz_ctx *ctx) {
    lstm_layer *m = l->self;
    const nmz_backend *be = ctx->backend;
    int B = x->shape[0], T = x->shape[1], H = m->units, D = m->D, G = 4 * H, t;
    int shape[3];
    const float *px, *pWx, *pWh, *pb, *ph0, *pc0;
    float *py, *pg, *pc;
    nmz_status s;
    if ((s = lstm_initial(be, ctx, m, B)) != NMZ_OK) return s;
    shape[0] = B;
    shape[1] = T;
    shape[2] = G;
    if ((s = nmz_tensor_reshape(&m->gates, 3, shape)) != NMZ_OK) return s;
    shape[2] = H;
    if ((s = nmz_tensor_reshape(&m->c, 3, shape)) != NMZ_OK) return s;
    px = nmz__r(be, x);
    pWx = nmz__r(be, &m->Wx);
    pWh = nmz__r(be, &m->Wh);
    pb = nmz__r(be, &m->b);
    ph0 = nmz__r(be, &m->h0);
    pc0 = nmz__r(be, &m->c0);
    py = nmz__w(be, y);
    pg = nmz__w(be, &m->gates);
    pc = nmz__w(be, &m->c);
    if (!px || !pWx || !pWh || !pb || !ph0 || !pc0 || !py || !pg || !pc) return nmz__op_failed();
    if (NMZ_FUSED(be, lstm_forward)) {
        s = be->lstm_forward(be, B, T, D, H, px, ph0, pc0, pWx, pWh, pb, py, pg, pc);
    } else { /* a = x·W_x + b for all steps; per step a_t += h_{t−1}·W_h, then the gates */
        s = be->gemm(be, 0, B * T, G, D, px, D, pWx, G, pg, G);
        if (s == NMZ_OK) s = be->add_bias(be, (size_t)B * T, G, pg, pb);
        for (t = 0; t < T && s == NMZ_OK; t++) {
            nmz_rows a, cp, ct, ht;
            a.p = pg + (size_t)t * G;
            a.ld = T * G;
            cp.p = t ? pc + (size_t)(t - 1) * H : (float *)pc0;
            cp.ld = t ? T * H : H;
            ct.p = pc + (size_t)t * H;
            ct.ld = T * H;
            ht.p = py + (size_t)t * H;
            ht.ld = T * H;
            if ((s = nmz__rec_product(be, B, T, t, H, G, 0, G, ph0, py, pWh, pg)) == NMZ_OK)
                s = be->lstm_cell_forward(be, B, H, a, cp, ct, ht);
        }
    }
    if (s != NMZ_OK) return s;
    if (ctx->carry) { /* remember h_{T−1} | c_{T−1} for the next batch of the same streams */
        float *pst;
        shape[0] = B;
        shape[1] = 2 * H;
        if ((s = nmz_tensor_reshape(&m->state, 2, shape)) != NMZ_OK) return s;
        if (!(pst = nmz__w(be, &m->state))) return nmz__op_failed();
        if ((s = be->copy(be, (size_t)B, (size_t)H, py + (size_t)(T - 1) * H, (size_t)T * H, pst, (size_t)2 * H)) !=
                NMZ_OK ||
            (s = be->copy(be, (size_t)B, (size_t)H, pc + (size_t)(T - 1) * H, (size_t)T * H, pst + H, (size_t)2 * H)) !=
                NMZ_OK)
            return s;
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
    const nmz_backend *be = ctx->backend;
    int B = x->shape[0], T = x->shape[1], H = m->units, D = m->D, G = 4 * H, t;
    int tracing = NMZ_TRACING(ctx), shape[2];
    nmz_tensor *tp = m->tmp;
    nmz_seq_record rec = {0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    const float *px, *py, *ph0, *pc0, *pg, *pc, *pWx, *pWh, *pdy;
    float *pdWx, *pdWh, *pdb, *pdx = NULL, *pdhn, *pdcn;
    nmz_status s;
    shape[0] = B;
    shape[1] = H;
    if ((s = nmz_tensor_reshape(&m->dh_next, 2, shape)) != NMZ_OK || (s = nmz_tensor_reshape(&m->dc_next, 2, shape)) != NMZ_OK)
        return s;
    if (tracing) { /* dh_next dh dc_next dc da dh_prev dc_prev */
        float **slots[7];
        int widths[7] = {H, H, H, H, 0, H, H};
        slots[0] = &rec.dh_next;
        slots[1] = &rec.dh;
        slots[2] = &rec.dc_next;
        slots[3] = &rec.dc;
        slots[4] = &rec.da;
        slots[5] = &rec.dh_prev;
        slots[6] = &rec.dc_prev;
        if ((s = nmz__rec_alloc(be, m->rec, 7, B, T, G, &rec, slots, widths)) != NMZ_OK) return s;
    }
    px = nmz__r(be, x);
    py = nmz__r(be, y);
    ph0 = nmz__r(be, &m->h0);
    pc0 = nmz__r(be, &m->c0);
    pg = nmz__r(be, &m->gates);
    pc = nmz__r(be, &m->c);
    pWx = nmz__r(be, &m->Wx);
    pWh = nmz__r(be, &m->Wh);
    pdy = nmz__r(be, dy);
    pdWx = nmz__u(be, &m->dWx);
    pdWh = nmz__u(be, &m->dWh);
    pdb = nmz__u(be, &m->db);
    pdhn = nmz__w(be, &m->dh_next);
    pdcn = nmz__w(be, &m->dc_next);
    if (dx) pdx = nmz__w(be, dx);
    if (!px || !py || !ph0 || !pc0 || !pg || !pc || !pWx || !pWh || !pdy || !pdWx || !pdWh || !pdb || !pdhn || !pdcn ||
        (dx && !pdx))
        return nmz__op_failed();
    if (NMZ_FUSED(be, lstm_backward)) {
        s = be->lstm_backward(be, B, T, D, H, px, py, ph0, pc0, pg, pc, pWx, pWh, pdy, pdWx, pdWh, pdb, pdx, pdhn, pdcn,
                              tracing ? &rec : NULL);
    } else {
        int s3[3];
        float *pda, *pgp;
        s3[0] = B;
        s3[1] = T;
        s3[2] = G;
        if ((s = nmz_tensor_reshape(&m->gda, 3, s3)) != NMZ_OK) return s;
        s3[2] = H;
        if ((s = nmz_tensor_reshape(&m->gprev, 3, s3)) != NMZ_OK) return s;
        pda = nmz__w(be, &m->gda);
        pgp = nmz__w(be, &m->gprev);
        if (!pda || !pgp) return nmz__op_failed();
        s = be->fill(be, (size_t)B * H, 0.0f, pdhn);
        if (s == NMZ_OK) s = be->fill(be, (size_t)B * H, 0.0f, pdcn);
        for (t = T - 1; t >= 0 && s == NMZ_OK; t--) {
            nmz_rows g, c, cp, d, hn, cn, a;
            g.p = (float *)pg + (size_t)t * G;
            g.ld = T * G;
            c.p = (float *)pc + (size_t)t * H;
            c.ld = T * H;
            cp.p = t ? (float *)pc + (size_t)(t - 1) * H : (float *)pc0;
            cp.ld = t ? T * H : H;
            d.p = (float *)pdy + (size_t)t * H;
            d.ld = T * H;
            hn.p = pdhn;
            hn.ld = H;
            cn.p = pdcn;
            cn.ld = H;
            a.p = pda + (size_t)t * G;
            a.ld = T * G;
            s = be->lstm_cell_backward(be, B, H, g, c, cp, d, hn, cn, a, tracing ? &rec : NULL, t);
            if (s == NMZ_OK) s = be->gemm(be, NMZ_MM_TRANS_B, B, H, G, a.p, T * G, pWh, G, pdhn, H); /* ∂L/∂h_{t−1} */
            if (s == NMZ_OK && tracing)
                s = be->copy(be, (size_t)B, (size_t)H, pdhn, (size_t)H, rec.dh_prev + (size_t)t * H, (size_t)T * H);
        }
        if (s == NMZ_OK) s = be->gemm(be, NMZ_MM_TRANS_A | NMZ_MM_ACCUMULATE, D, G, B * T, px, D, pda, G, pdWx, G);
        if (s == NMZ_OK) s = nmz__rec_prev(be, B, T, H, ph0, py, pgp);
        if (s == NMZ_OK) s = be->gemm(be, NMZ_MM_TRANS_A | NMZ_MM_ACCUMULATE, H, G, B * T, pgp, H, pda, G, pdWh, G);
        if (s == NMZ_OK) s = be->colsum_add(be, (size_t)B * T, G, pda, pdb);
        if (s == NMZ_OK && dx) s = be->gemm(be, NMZ_MM_TRANS_B, B * T, D, G, pda, G, pWx, G, pdx, D);
        if (s == NMZ_OK && tracing) s = be->copy(be, 1, (size_t)B * T * G, pda, (size_t)B * T * G, rec.da, (size_t)B * T * G);
    }
    if (s != NMZ_OK) return s;
    if (tracing) {
        for (t = T - 1; t >= 0; t--) {
            static const int order[7] = {1, 2, 3, 4, 5, 6, 7}; /* tmp slot of rec[k] */
            int k;
            if ((s = nmz__slice_t(dy, t, &tp[0])) != NMZ_OK) return s;
            for (k = 0; k < 7; k++)
                if ((s = nmz__slice_t(&m->rec[k], t, &tp[order[k]])) != NMZ_OK) return s;
            {
                nmz_trace_item it[8] = {{"dL/dh_out", &tp[0]},  {"dL/dh_next", &tp[1]}, {"dL/dh", &tp[2]},
                                        {"dL/dc_next", &tp[3]}, {"dL/dc", &tp[4]},      {"dL/da", &tp[5]},
                                        {"dL/dh_prev", &tp[6]}, {"dL/dc_prev", &tp[7]}};
                nmz_trace_emit_step(ctx, NMZ_TRACE_BACKWARD, "lstm", "i f g o", t, 8, it);
            }
        }
        {
            nmz_trace_scalar sc[1] = {{"steps", T}};
            nmz_trace_item all[5] = {{"dL/dW_x", &m->dWx}, {"dL/dW_h", &m->dWh}, {"dL/db", &m->db},
                                     {"W_h", &m->Wh},      {"dL/dx", dx}};
            nmz_trace_emit(ctx, NMZ_TRACE_BACKWARD, "lstm", "i f g o", 1, sc, dx ? 5 : 4, all);
        }
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
    for (i = 0; i < 7; i++) nmz_tensor_free(&m->rec[i]);
    nmz_tensor_free(&m->gda);
    nmz_tensor_free(&m->gprev);
    free(m);
}

static int lstm_buffers(nmz_layer *l, nmz_tensor **out, int max) {
    lstm_layer *m = l->self;
    nmz_tensor *all[] = {&m->h0, &m->c0, &m->state, &m->gates, &m->c, &m->dh_next, &m->dc_next, &m->tmp[0], &m->tmp[1], &m->tmp[2], &m->tmp[3], &m->tmp[4], &m->tmp[5], &m->tmp[6], &m->tmp[7], &m->tmp[8], &m->tmp[9], &m->tmp[10], &m->tmp[11], &m->rec[0], &m->rec[1], &m->rec[2], &m->rec[3], &m->rec[4], &m->rec[5], &m->rec[6], &m->gda, &m->gprev};
    return nmz__list(out, max, (int)(sizeof all / sizeof all[0]), all);
}

const nmz_layer_ops nmz__lstm_ops = {"lstm",          lstm_setup,  lstm_init,  lstm_forward,  lstm_backward,
                                     lstm_param_count, lstm_param, lstm_save,  lstm_load,     lstm_describe,
                                     lstm_destroy,    lstm_state,  NMZ_LAYER_BACKEND, NULL,
                                     lstm_buffers};

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
