/* test_gated.c — LSTM and GRU layers (phase 4): forward values by hand,
 * numerical gradient checks (multi-step, carried state, stacked, dropout in
 * evaluation mode), the gradient through c_{t−1}, orthogonal
 * initialisation, the forget-gate bias, save/load. */
#include "test.h"
#include "neuromz_layer.h"

#include <stdlib.h>
#include <string.h>

static void fill(nmz_tensor *t, nmz_rng *r, double lo, double hi) {
    size_t i;
    for (i = 0; i < t->size; i++) t->data[i] = (float)(lo + (hi - lo) * nmz_rng_uniform(r));
}

static void fill_ids(nmz_tensor *t, nmz_rng *r, int V) {
    size_t i;
    for (i = 0; i < t->size; i++) t->data[i] = (float)nmz_rng_below(r, (uint32_t)V);
}

static void onehot_targets(nmz_tensor *t, nmz_rng *r, int from) {
    int B = t->shape[0], T = t->shape[1], V = t->shape[2], b, s;
    nmz_tensor_zero(t);
    for (b = 0; b < B; b++)
        for (s = from; s < T; s++) t->data[((size_t)b * T + s) * V + nmz_rng_below(r, (uint32_t)V)] = 1.0f;
}

static void report(const char *label, const nmz_gradcheck_result *res) {
    CHECK_MSG(res->failures == 0 && res->checked > 0,
              "%s: %zu/%zu mismatches; worst layer %d %s[%zu]: analytic %.6g numeric %.6g", label, res->failures,
              res->checked, res->worst_layer, res->worst_param, res->worst_index, res->worst_analytic,
              res->worst_numeric);
    test_note("%-44s %5zu grads, max |err| %.2e%s", label, res->checked, res->max_abs_err,
              res->kinks ? " (kinks skipped)" : "");
}

static double sig(double a) { return 1.0 / (1.0 + exp(-a)); }

/* ---------------------------------------------------------------- by hand */

/* 1 unit, 1 input, 3 steps: every gate and c_t, h_t in double precision */
static void test_lstm_by_hand(void) {
    nmz_net *net = NULL;
    nmz_tensor x = NMZ_TENSOR_EMPTY;
    const nmz_tensor *y;
    int in[2] = {NMZ_ANY_LENGTH, 1}, xs[3] = {1, 3, 1}, s;
    /* gates i f g o */
    const double wx[4] = {0.5, -0.3, 0.8, 0.2}, wh[4] = {0.1, 0.4, -0.6, 0.7}, b[4] = {0.0, 1.0, 0.1, -0.2};
    const double xv[3] = {1.0, -2.0, 0.5};
    double h = 0, c = 0;
    nmz_layer *l;
    CHECK_OK(nmz_net_create(&net, 2, in, 1));
    CHECK_OK(nmz_net_add(net, nmz_layer_lstm(1, NMZ_INIT_AUTO, NMZ_LSTM_FORGET_BIAS)));
    CHECK_OK(nmz_net_build(net));
    l = nmz_net_layer(net, 0);
    CHECK_NEAR(nmz_layer_param_value(l, 2)->data[1], 1.0, 0); /* forget bias starts at 1 */
    for (s = 0; s < 4; s++) {
        nmz_layer_param_value(l, 0)->data[s] = (float)wx[s];
        nmz_layer_param_value(l, 1)->data[s] = (float)wh[s];
        nmz_layer_param_value(l, 2)->data[s] = (float)b[s];
    }
    CHECK_OK(nmz_tensor_init(&x, 3, xs));
    for (s = 0; s < 3; s++) x.data[s] = (float)xv[s];
    CHECK_OK(nmz_net_forward(net, &x, &y));
    for (s = 0; s < 3; s++) {
        double i = sig(wx[0] * xv[s] + wh[0] * h + b[0]), f = sig(wx[1] * xv[s] + wh[1] * h + b[1]);
        double g = tanh(wx[2] * xv[s] + wh[2] * h + b[2]), o = sig(wx[3] * xv[s] + wh[3] * h + b[3]);
        c = f * c + i * g;
        h = o * tanh(c);
        CHECK_NEAR(y->data[s], h, 1e-6);
    }
    nmz_tensor_free(&x);
    nmz_net_free(net);
}

static void test_gru_by_hand(void) {
    nmz_net *net = NULL;
    nmz_tensor x = NMZ_TENSOR_EMPTY;
    const nmz_tensor *y;
    int in[2] = {NMZ_ANY_LENGTH, 1}, xs[3] = {1, 3, 1}, s;
    /* gates z r ĥ */
    const double wx[3] = {0.5, -0.3, 0.8}, wh[3] = {0.4, 0.9, -0.6}, b[3] = {0.2, -0.1, 0.1};
    const double xv[3] = {1.0, -2.0, 0.5};
    double h = 0;
    nmz_layer *l;
    CHECK_OK(nmz_net_create(&net, 2, in, 1));
    CHECK_OK(nmz_net_add(net, nmz_layer_gru(1, NMZ_INIT_AUTO)));
    CHECK_OK(nmz_net_build(net));
    l = nmz_net_layer(net, 0);
    for (s = 0; s < 3; s++) {
        nmz_layer_param_value(l, 0)->data[s] = (float)wx[s];
        nmz_layer_param_value(l, 1)->data[s] = (float)wh[s];
        nmz_layer_param_value(l, 2)->data[s] = (float)b[s];
    }
    CHECK_OK(nmz_tensor_init(&x, 3, xs));
    for (s = 0; s < 3; s++) x.data[s] = (float)xv[s];
    CHECK_OK(nmz_net_forward(net, &x, &y));
    for (s = 0; s < 3; s++) {
        double z = sig(wx[0] * xv[s] + wh[0] * h + b[0]), r = sig(wx[1] * xv[s] + wh[1] * h + b[1]);
        double n = tanh(wx[2] * xv[s] + wh[2] * (r * h) + b[2]); /* Cho et al.: r before the product */
        h = z * h + (1 - z) * n;
        CHECK_NEAR(y->data[s], h, 1e-6);
    }
    nmz_tensor_free(&x);
    nmz_net_free(net);
}

/* ------------------------------------------------------------- gradchecks */

static nmz_layer *gated(int kind, int H) {
    return kind == 0 ? nmz_layer_lstm(H, NMZ_INIT_AUTO, NMZ_LSTM_FORGET_BIAS) : nmz_layer_gru(H, NMZ_INIT_AUTO);
}

static const char *kind_name(int kind) { return kind == 0 ? "lstm" : "gru"; }

/* continuous input [B, T, D] → lstm/gru → dense (per step) → MSE */
static void test_gradcheck_multistep(void) {
    int kind;
    for (kind = 0; kind < 2; kind++) {
        nmz_net *net = NULL;
        nmz_tensor x = NMZ_TENSOR_EMPTY, t = NMZ_TENSOR_EMPTY;
        nmz_gradcheck_result res;
        int in[2] = {NMZ_ANY_LENGTH, 3}, xs[3] = {2, 6, 3}, ts[3] = {2, 6, 2};
        char label[64];
        nmz_rng r;
        nmz_rng_seed(&r, 11, (uint64_t)kind);
        CHECK_OK(nmz_net_create(&net, 2, in, 3));
        CHECK_OK(nmz_net_add(net, gated(kind, 4)));
        CHECK_OK(nmz_net_add(net, nmz_layer_dense(2, NMZ_INIT_AUTO)));
        CHECK_OK(nmz_net_build(net));
        fill(nmz_layer_param_value(nmz_net_layer(net, 0), 2), &r, -0.5, 0.5);
        CHECK_OK(nmz_tensor_init(&x, 3, xs));
        CHECK_OK(nmz_tensor_init(&t, 3, ts));
        fill(&x, &r, -1, 1);
        fill(&t, &r, -1, 1);
        CHECK_OK(nmz_net_gradcheck(net, &x, &t, 1e-3, 0, 0, &res));
        snprintf(label, sizeof label, "%s 3->4, 6 steps, per-step dense, mse", kind_name(kind));
        report(label, &res);
        nmz_tensor_free(&x);
        nmz_tensor_free(&t);
        nmz_net_free(net);
    }
}

/* embedding → layers... → dense(V) → softmax, cross-entropy */
static nmz_net *char_net2(int V, int D, const int *kinds, int n, int H, float drop, uint64_t seed) {
    nmz_net *net = NULL;
    int in = NMZ_ANY_LENGTH, k;
    if (nmz_net_create(&net, 1, &in, seed) != NMZ_OK) return NULL;
    nmz_net_add(net, nmz_layer_embedding(V, D, NMZ_INIT_AUTO));
    for (k = 0; k < n; k++) {
        if (k > 0 && drop > 0) nmz_net_add(net, nmz_layer_dropout(drop));
        nmz_net_add(net, kinds[k] == 2 ? nmz_layer_rnn(H, NMZ_INIT_AUTO) : gated(kinds[k], H));
    }
    nmz_net_add_dense(net, V, NMZ_ACT_SOFTMAX, NMZ_INIT_AUTO);
    nmz_net_set_loss(net, NMZ_LOSS_CROSS_ENTROPY);
    if (nmz_net_build(net) != NMZ_OK) {
        nmz_net_free(net);
        return NULL;
    }
    return net;
}

static void char_case(const char *label, const int *kinds, int n, float drop, int carry, int last_only, int training) {
    const int V = 6, B = 2, T = 6;
    nmz_net *net = char_net2(V, 3, kinds, n, 4, drop, 21);
    nmz_tensor x = NMZ_TENSOR_EMPTY, t = NMZ_TENSOR_EMPTY;
    nmz_gradcheck_result res;
    int xs[2], ts[3], k;
    nmz_rng r;
    CHECK(net != NULL);
    if (!net) return;
    nmz_rng_seed(&r, 22, (uint64_t)n);
    xs[0] = ts[0] = B;
    xs[1] = ts[1] = T;
    ts[2] = V;
    CHECK_OK(nmz_tensor_init(&x, 2, xs));
    CHECK_OK(nmz_tensor_init(&t, 3, ts));
    for (k = 0; k < nmz_net_layer_count(net); k++) { /* non-zero biases everywhere */
        nmz_layer *l = nmz_net_layer(net, k);
        if (nmz_layer_is_recurrent(l)) fill(nmz_layer_param_value(l, 2), &r, -0.5, 0.5);
    }
    fill_ids(&x, &r, V);
    onehot_targets(&t, &r, last_only ? T - 1 : 0);
    if (carry) { /* a first batch leaves h (and c) behind; the checked pass starts from it */
        nmz_net_set_carry(net, 1);
        CHECK_OK(nmz_net_forward(net, &x, NULL));
        fill_ids(&x, &r, V);
    }
    CHECK_OK(nmz_net_gradcheck_mode(net, &x, &t, 1e-3, 0, 0, training, &res));
    report(label, &res);
    if (carry) {
        nmz_layer *l = nmz_net_layer(net, 1);
        CHECK(l->ops->state(l)->size > 0); /* gradcheck left the carried state alone */
        nmz_net_set_carry(net, 0);
    }
    nmz_tensor_free(&x);
    nmz_tensor_free(&t);
    nmz_net_free(net);
}

static void test_gradcheck_char(void) {
    static const int lstm1[1] = {0}, gru1[1] = {1};
    char_case("embed->lstm->dense->softmax/CE", lstm1, 1, 0, 0, 0, 0);
    char_case("lstm, loss at the last step only", lstm1, 1, 0, 0, 1, 0);
    char_case("embed->gru->dense->softmax/CE", gru1, 1, 0, 0, 0, 0);
    char_case("gru, loss at the last step only", gru1, 1, 0, 0, 1, 0);
}

static void test_gradcheck_carry(void) {
    static const int lstm1[1] = {0}, gru1[1] = {1};
    char_case("lstm with carried h and c", lstm1, 1, 0, 1, 0, 0);
    char_case("gru with carried h", gru1, 1, 0, 1, 0, 0);
}

static void test_gradcheck_stacked(void) {
    static const int ll[2] = {0, 0}, gg[2] = {1, 1}, mix[3] = {0, 1, 2};
    char_case("lstm->lstm", ll, 2, 0, 0, 0, 0);
    char_case("gru->gru", gg, 2, 0, 0, 0, 0);
    char_case("lstm->dropout->gru->dropout->rnn, eval mode", mix, 3, 0.3f, 0, 0, 0);
    char_case("lstm->dropout->lstm, training mode", ll, 2, 0.3f, 0, 0, 1);
    char_case("lstm->lstm with carried state", ll, 2, 0, 1, 1, 0);
}

/* ∂L/∂c_{t−1} and ∂L/∂h_{t−1} as reported by the trace at t = 0 (the
 * gradient that would flow into the carried state) against central
 * differences of the loss in that carried state — the gradient along the
 * cell-state path, checked numerically. */
static void state_grad_case(int kind) {
    nmz_net *net = NULL;
    nmz_tensor x = NMZ_TENSOR_EMPTY, t = NMZ_TENSOR_EMPTY, keep = NMZ_TENSOR_EMPTY;
    nmz_trace_recorder *rec = NULL;
    const nmz_tensor *dh = NULL, *dc = NULL;
    int in[2] = {NMZ_ANY_LENGTH, 2}, xs[3] = {2, 5, 2}, ts[3] = {2, 5, 3}, H = 3, B = 2, bi, j, part;
    nmz_layer *l;
    nmz_tensor *st;
    double worst = 0;
    size_t k;
    nmz_rng r;
    nmz_rng_seed(&r, 31, (uint64_t)kind);
    CHECK_OK(nmz_net_create(&net, 2, in, 5));
    CHECK_OK(nmz_net_add(net, gated(kind, H)));
    CHECK_OK(nmz_net_add(net, nmz_layer_dense(3, NMZ_INIT_AUTO)));
    CHECK_OK(nmz_net_build(net));
    l = nmz_net_layer(net, 0);
    CHECK_OK(nmz_tensor_init(&x, 3, xs));
    CHECK_OK(nmz_tensor_init(&t, 3, ts));
    fill(&x, &r, -1, 1);
    fill(&t, &r, -1, 1);
    nmz_net_set_carry(net, 1);
    CHECK_OK(nmz_net_forward(net, &x, NULL)); /* leaves a non-zero state */
    st = l->ops->state(l);
    CHECK_OK(nmz_tensor_copy(&keep, st));
    fill(&x, &r, -1, 1);
    CHECK_OK(nmz_trace_recorder_create(&rec));
    nmz_net_set_trace(net, nmz_trace_recorder_fn, rec);
    CHECK_OK(nmz_net_forward(net, &x, NULL));
    CHECK_OK(nmz_net_backward(net, &t, NULL));
    nmz_net_set_trace(net, NULL, NULL);
    for (k = 0; k < nmz_trace_recorder_count(rec); k++) {
        const nmz_trace_event *e = nmz_trace_recorder_get(rec, k);
        if (e->phase == NMZ_TRACE_BACKWARD && e->time_step == 0) {
            dh = nmz_trace_find(e, "dL/dh_prev");
            dc = nmz_trace_find(e, "dL/dc_prev");
        }
    }
    CHECK(dh != NULL && (kind != 0 || dc != NULL));
    for (part = 0; part < (kind == 0 ? 2 : 1) && dh; part++)
        for (bi = 0; bi < B; bi++)
            for (j = 0; j < H; j++) {
                size_t e = kind == 0 ? (size_t)bi * 2 * H + (size_t)part * H + j : (size_t)bi * H + j;
                double lp, lm, num, ana = (part ? dc : dh)->data[(size_t)bi * H + j], eps = 1e-3;
                nmz_tensor_copy(st, &keep);
                st->data[e] += (float)eps;
                CHECK_OK(nmz_net_forward(net, &x, NULL));
                CHECK_OK(nmz_net_compute_loss(net, &t, &lp));
                nmz_tensor_copy(st, &keep);
                st->data[e] -= (float)eps;
                CHECK_OK(nmz_net_forward(net, &x, NULL));
                CHECK_OK(nmz_net_compute_loss(net, &t, &lm));
                num = (lp - lm) / (2 * eps);
                CHECK_MSG(fabs(num - ana) < 2e-4 + 2e-3 * fabs(num), "%s %s_prev[%d,%d]: analytic %.6g numeric %.6g",
                          kind_name(kind), part ? "c" : "h", bi, j, ana, num);
                if (fabs(num - ana) > worst) worst = fabs(num - ana);
            }
    test_note("%-44s %5d grads, max |err| %.2e", kind == 0 ? "lstm dL/dh_{-1}, dL/dc_{-1} (carried state)" :
                                                              "gru dL/dh_{-1} (carried state)",
              B * H * (kind == 0 ? 2 : 1), worst);
    nmz_net_set_carry(net, 0);
    nmz_trace_recorder_free(rec);
    nmz_tensor_free(&keep);
    nmz_tensor_free(&x);
    nmz_tensor_free(&t);
    nmz_net_free(net);
}

static void test_state_gradient(void) {
    state_grad_case(0);
    state_grad_case(1);
}

/* ---------------------------------------------------------- initialisation */

/* max |(WᵀW − I)_ij| of the n×n block at column col0 of a row-major [n, ld] matrix */
static double orth_error(const float *W, int n, int ld, int col0) {
    double worst = 0;
    int a, b, k;
    for (a = 0; a < n; a++)
        for (b = 0; b < n; b++) {
            double d = 0;
            for (k = 0; k < n; k++) d += (double)W[(size_t)k * ld + col0 + a] * W[(size_t)k * ld + col0 + b];
            d -= a == b;
            if (fabs(d) > worst) worst = fabs(d);
        }
    return worst;
}

static void test_orthogonal_init(void) {
    const int H = 24, D = 5;
    int kind, g;
    for (kind = 0; kind < 3; kind++) {
        nmz_net *net = NULL;
        int in[2] = {NMZ_ANY_LENGTH, D}, G = kind == 0 ? 4 : kind == 1 ? 3 : 1;
        nmz_layer *l = kind == 0 ? nmz_layer_lstm(H, NMZ_INIT_AUTO, 1.0f)
                                 : kind == 1 ? nmz_layer_gru(H, NMZ_INIT_AUTO) : nmz_layer_rnn(H, NMZ_INIT_AUTO);
        const nmz_tensor *Wh;
        double worst = 0;
        CHECK_OK(nmz_net_create(&net, 2, in, 77));
        CHECK_OK(nmz_net_add(net, l));
        CHECK_OK(nmz_net_build(net));
        l = nmz_net_layer(net, 0);
        CHECK(nmz_layer_init_used(l) == NMZ_INIT_ORTHOGONAL); /* AUTO resolves to orthogonal */
        Wh = nmz_layer_param_value(l, 1);
        CHECK(Wh->shape[0] == H && Wh->shape[1] == G * H);
        for (g = 0; g < G; g++) { /* each gate's H×H block: WᵀW = I */
            double e = orth_error(Wh->data, H, G * H, g * H);
            if (e > worst) worst = e;
        }
        CHECK_MSG(worst < 1e-5, "%s W_h: max |W^T W - I| = %g", kind == 0 ? "lstm" : kind == 1 ? "gru" : "rnn", worst);
        test_note("%-6s W_h %dx%d, %d orthogonal block(s): max |WtW - I| = %.1e", kind == 0 ? "lstm" : kind == 1 ? "gru" : "rnn",
                  H, G * H, G, worst);
        nmz_net_free(net);
    }
    {
        /* the old option is still there and is NOT orthogonal */
        nmz_net *net = NULL;
        int in[2] = {NMZ_ANY_LENGTH, D};
        CHECK_OK(nmz_net_create(&net, 2, in, 77));
        CHECK_OK(nmz_net_add(net, nmz_layer_rnn(H, NMZ_INIT_XAVIER)));
        CHECK_OK(nmz_net_build(net));
        CHECK(nmz_layer_init_used(nmz_net_layer(net, 0)) == NMZ_INIT_XAVIER);
        CHECK(orth_error(nmz_layer_param_value(nmz_net_layer(net, 0), 1)->data, H, H, 0) > 0.1);
        nmz_net_free(net);
    }
    {
        nmz_init i;
        CHECK_OK(nmz_init_parse("orthogonal", &i));
        CHECK(i == NMZ_INIT_ORTHOGONAL && !strcmp(nmz_init_name(i), "orthogonal"));
        CHECK(nmz_layer_dense(3, NMZ_INIT_ORTHOGONAL) == NULL); /* recurrent layers only */
        CHECK(nmz_layer_embedding(3, 2, NMZ_INIT_ORTHOGONAL) == NULL);
        CHECK(nmz_layer_conv2d(2, 3, 3, 1, 0, NMZ_INIT_ORTHOGONAL) == NULL);
    }
}

static void test_forget_bias_and_saveload(void) {
    const int V = 5, H = 6;
    nmz_net *net = NULL, *back = NULL;
    nmz_tensor x = NMZ_TENSOR_EMPTY;
    const nmz_tensor *y;
    int in = NMZ_ANY_LENGTH, xs[2] = {2, 7}, j;
    float *before;
    nmz_rng r;
    char d[160];
    nmz_rng_seed(&r, 41, 41);
    CHECK_OK(nmz_net_create(&net, 1, &in, 9));
    CHECK_OK(nmz_net_add(net, nmz_layer_embedding(V, 3, NMZ_INIT_AUTO)));
    CHECK_OK(nmz_net_add(net, nmz_layer_lstm(H, NMZ_INIT_AUTO, 2.5f)));
    CHECK_OK(nmz_net_add(net, nmz_layer_dropout(0.2f)));
    CHECK_OK(nmz_net_add(net, nmz_layer_gru(H, NMZ_INIT_XAVIER)));
    CHECK_OK(nmz_net_add_dense(net, V, NMZ_ACT_SOFTMAX, NMZ_INIT_AUTO));
    CHECK_OK(nmz_net_set_loss(net, NMZ_LOSS_CROSS_ENTROPY));
    CHECK_OK(nmz_net_build(net));
    for (j = 0; j < 4 * H; j++) /* b = [i f g o]: only the forget block starts at the bias */
        CHECK(nmz_layer_param_value(nmz_net_layer(net, 1), 2)->data[j] == (j >= H && j < 2 * H ? 2.5f : 0.0f));
    nmz_layer_describe(nmz_net_layer(net, 1), d, sizeof d);
    CHECK_MSG(strstr(d, "forget bias 2.5") && strstr(d, "orthogonal"), "%s", d);
    CHECK(nmz_layer_lstm(4, NMZ_INIT_AUTO, NAN) == NULL);
    CHECK_OK(nmz_tensor_init(&x, 2, xs));
    fill_ids(&x, &r, V);
    CHECK_OK(nmz_net_forward(net, &x, &y));
    before = malloc(y->size * sizeof *before);
    if (before) memcpy(before, y->data, y->size * sizeof *before);
    CHECK_OK(nmz_net_save(net, test_path("t_gated.nmz")));
    CHECK_OK(nmz_net_load(&back, test_path("t_gated.nmz")));
    if (back && before) {
        size_t k;
        double worst = 0;
        CHECK(nmz_net_layer_count(back) == nmz_net_layer_count(net));
        CHECK(!strcmp(nmz_layer_type(nmz_net_layer(back, 1)), "lstm") && !strcmp(nmz_layer_type(nmz_net_layer(back, 3)), "gru"));
        CHECK(nmz_layer_init_used(nmz_net_layer(back, 3)) == NMZ_INIT_XAVIER);
        nmz_layer_describe(nmz_net_layer(back, 1), d, sizeof d);
        CHECK_MSG(strstr(d, "forget bias 2.5") != NULL, "%s", d);
        CHECK_OK(nmz_net_forward(back, &x, &y));
        for (k = 0; k < y->size; k++)
            if (fabs(y->data[k] - before[k]) > worst) worst = fabs(y->data[k] - before[k]);
        CHECK(worst == 0.0);
    }
    free(before);
    nmz_tensor_free(&x);
    nmz_net_free(back);
    nmz_net_free(net);
}

/* carried state shapes, reset, and that carry really continues a stream:
 * two halves with carry = the whole sequence at once */
static void test_carry_continues_stream(void) {
    int kind;
    for (kind = 0; kind < 2; kind++) {
        const int V = 5, H = 4, T = 8;
        int kinds[1];
        nmz_net *net;
        nmz_tensor x = NMZ_TENSOR_EMPTY, a = NMZ_TENSOR_EMPTY, b = NMZ_TENSOR_EMPTY;
        const nmz_tensor *y;
        nmz_layer *l;
        int xs[2] = {1, T}, hs[2] = {1, T / 2}, s;
        nmz_rng r;
        double worst = 0;
        kinds[0] = kind;
        net = char_net2(V, 3, kinds, 1, H, 0, 51);
        CHECK(net != NULL);
        if (!net) continue;
        l = nmz_net_layer(net, 1);
        nmz_rng_seed(&r, 52, 52);
        CHECK_OK(nmz_tensor_init(&x, 2, xs));
        CHECK_OK(nmz_tensor_init(&a, 2, hs));
        CHECK_OK(nmz_tensor_init(&b, 2, hs));
        fill_ids(&x, &r, V);
        for (s = 0; s < T / 2; s++) {
            a.data[s] = x.data[s];
            b.data[s] = x.data[T / 2 + s];
        }
        CHECK_OK(nmz_net_forward(net, &x, &y));
        {
            float last[8];
            memcpy(last, y->data + (size_t)(T - 1) * V, (size_t)V * sizeof(float));
            nmz_net_set_carry(net, 1);
            CHECK_OK(nmz_net_forward(net, &a, NULL));
            CHECK(l->ops->state(l)->size == (size_t)(kind == 0 ? 2 * H : H)); /* lstm: h | c */
            CHECK_OK(nmz_net_forward(net, &b, &y));
            for (s = 0; s < V; s++)
                if (fabs(y->data[(size_t)(T / 2 - 1) * V + s] - last[s]) > worst)
                    worst = fabs(y->data[(size_t)(T / 2 - 1) * V + s] - last[s]);
            CHECK_MSG(worst < 1e-6, "%s: carried halves differ from the whole by %g", kind_name(kind), worst);
            nmz_net_reset_state(net);
            CHECK(l->ops->state(l)->size == 0);
            nmz_net_set_carry(net, 0);
        }
        nmz_tensor_free(&x);
        nmz_tensor_free(&a);
        nmz_tensor_free(&b);
        nmz_net_free(net);
    }
}

const test_case gated_tests[] = {
    {"gated/lstm_forward_by_hand", test_lstm_by_hand},
    {"gated/gru_forward_by_hand", test_gru_by_hand},
    {"gated/gradcheck_multistep", test_gradcheck_multistep},
    {"gated/gradcheck_char_head", test_gradcheck_char},
    {"gated/gradcheck_carried_state", test_gradcheck_carry},
    {"gated/gradcheck_stacked_dropout", test_gradcheck_stacked},
    {"gated/gradient_through_c_prev", test_state_gradient},
    {"gated/orthogonal_init", test_orthogonal_init},
    {"gated/forget_bias_saveload", test_forget_bias_and_saveload},
    {"gated/carry_continues_stream", test_carry_continues_stream},
    {NULL, NULL},
};
