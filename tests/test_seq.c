/* test_seq.c — Embedding, RNN (BPTT, carried state), the per-step output
 * head, gradient clipping, validation splits and variable sequence lengths. */
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

/* one-hot targets [B, T, V]; rows with t < from are left all-zero (no loss there) */
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
    test_note("%-40s %5zu grads, max |err| %.2e%s", label, res->checked, res->max_abs_err,
              res->kinks ? " (kinks skipped)" : "");
}

/* continuous input [B, T, D] → rnn → dense (per step) → MSE */
static void test_gradcheck_rnn(void) {
    nmz_net *net = NULL;
    nmz_tensor x = NMZ_TENSOR_EMPTY, t = NMZ_TENSOR_EMPTY;
    nmz_gradcheck_result res;
    int in[2] = {NMZ_ANY_LENGTH, 3}, xs[3] = {2, 5, 3}, ts[3] = {2, 5, 2};
    nmz_rng r;
    nmz_rng_seed(&r, 5, 5);
    CHECK_OK(nmz_net_create(&net, 2, in, 3));
    CHECK_OK(nmz_net_add(net, nmz_layer_rnn(4, NMZ_INIT_AUTO)));
    CHECK_OK(nmz_net_add(net, nmz_layer_dense(2, NMZ_INIT_AUTO))); /* acts on every time step (last dim) */
    CHECK_OK(nmz_net_build(net));
    fill(nmz_layer_param_value(nmz_net_layer(net, 0), 2), &r, -0.3, 0.3);
    CHECK_OK(nmz_tensor_init(&x, 3, xs));
    CHECK_OK(nmz_tensor_init(&t, 3, ts));
    fill(&x, &r, -1, 1);
    fill(&t, &r, -1, 1);
    CHECK_OK(nmz_net_gradcheck(net, &x, &t, 1e-3, 0, 0, &res));
    report("rnn 3->4, 5 steps, per-step dense, mse", &res);
    nmz_tensor_free(&x);
    nmz_tensor_free(&t);
    nmz_net_free(net);
}

/* The phase-3 character net. Its RNN keeps the phase-3 init (Glorot, now
 * the explicit `xavier` option; AUTO became orthogonal in phase 4), so the
 * exact generated strings below remain a regression test of phase 3. */
static nmz_net *char_net(int V, int D, int H, uint64_t seed) {
    nmz_net *net = NULL;
    int in = NMZ_ANY_LENGTH;
    if (nmz_net_create(&net, 1, &in, seed) != NMZ_OK) return NULL;
    nmz_net_add(net, nmz_layer_embedding(V, D, NMZ_INIT_AUTO));
    nmz_net_add(net, nmz_layer_rnn(H, NMZ_INIT_XAVIER));
    nmz_net_add_dense(net, V, NMZ_ACT_SOFTMAX, NMZ_INIT_AUTO);
    nmz_net_set_loss(net, NMZ_LOSS_CROSS_ENTROPY);
    if (nmz_net_build(net) != NMZ_OK) {
        nmz_net_free(net);
        return NULL;
    }
    return net;
}

static void test_gradcheck_char_rnn(void) {
    const int V = 6, B = 2, T = 6;
    nmz_net *net = char_net(V, 3, 5, 7);
    nmz_tensor x = NMZ_TENSOR_EMPTY, t = NMZ_TENSOR_EMPTY;
    nmz_gradcheck_result res;
    int xs[2], ts[3];
    nmz_rng r;
    CHECK(net != NULL);
    if (!net) return;
    nmz_rng_seed(&r, 8, 8);
    xs[0] = ts[0] = B;
    xs[1] = ts[1] = T;
    ts[2] = V;
    CHECK_OK(nmz_tensor_init(&x, 2, xs));
    CHECK_OK(nmz_tensor_init(&t, 3, ts));
    fill_ids(&x, &r, V);
    fill(nmz_layer_param_value(nmz_net_layer(net, 1), 2), &r, -0.3, 0.3);
    onehot_targets(&t, &r, 0);
    CHECK_OK(nmz_net_gradcheck(net, &x, &t, 1e-3, 0, 0, &res));
    report("embed->rnn->dense->softmax/CE, 6 steps", &res);
    /* loss ONLY at the last step: gradients for earlier inputs can only
     * arrive through h_{t-1} → h_t (backpropagation through time) */
    onehot_targets(&t, &r, T - 1);
    CHECK_OK(nmz_net_gradcheck(net, &x, &t, 1e-3, 0, 0, &res));
    report("same, loss at the last step only", &res);
    {
        /* and they do arrive: the embedding row of the FIRST token gets a gradient */
        double L;
        const nmz_tensor *dW;
        size_t k, nz = 0;
        int first = (int)x.data[0];
        CHECK_OK(nmz_net_forward(net, &x, NULL));
        CHECK_OK(nmz_net_backward(net, &t, &L));
        dW = nmz_layer_param_grad(nmz_net_layer(net, 0), 0);
        for (k = 0; k < 3; k++) nz += dW->data[(size_t)first * 3 + k] != 0.0f;
        CHECK_MSG(nz > 0, "no gradient reached step 0 through time");
    }
    nmz_tensor_free(&x);
    nmz_tensor_free(&t);
    nmz_net_free(net);
}

static void test_gradcheck_carry(void) {
    const int V = 5, B = 3, T = 4;
    nmz_net *net = char_net(V, 3, 4, 9);
    nmz_tensor x = NMZ_TENSOR_EMPTY, t = NMZ_TENSOR_EMPTY;
    nmz_gradcheck_result res;
    nmz_layer *rnn;
    int xs[2], ts[3];
    nmz_rng r;
    if (!net) return;
    nmz_rng_seed(&r, 10, 10);
    xs[0] = ts[0] = B;
    xs[1] = ts[1] = T;
    ts[2] = V;
    CHECK_OK(nmz_tensor_init(&x, 2, xs));
    CHECK_OK(nmz_tensor_init(&t, 3, ts));
    fill_ids(&x, &r, V);
    onehot_targets(&t, &r, 0);
    rnn = nmz_net_layer(net, 1);
    nmz_net_set_carry(net, 1);
    CHECK_OK(nmz_net_forward(net, &x, NULL)); /* first batch: leaves h_{T-1} as carried state */
    CHECK(rnn->ops->state(rnn)->size == (size_t)B * 4);
    {
        /* the next forward starts from it: outputs differ from a fresh start */
        const nmz_tensor *y;
        float a, b;
        CHECK_OK(nmz_net_forward(net, &x, &y));
        a = y->data[0];
        nmz_net_reset_state(net);
        CHECK(rnn->ops->state(rnn)->size == 0);
        CHECK_OK(nmz_net_forward(net, &x, &y));
        b = y->data[0];
        CHECK(a != b);
        CHECK_OK(nmz_net_forward(net, &x, NULL)); /* state again */
    }
    fill_ids(&x, &r, V); /* the next batch of the streams */
    CHECK_OK(nmz_net_gradcheck(net, &x, &t, 1e-3, 0, 0, &res));
    report("char rnn with carried state (h0 != 0)", &res);
    CHECK(rnn->ops->state(rnn)->size == (size_t)B * 4); /* gradcheck left the state alone */
    nmz_net_set_carry(net, 0);
    nmz_tensor_free(&x);
    nmz_tensor_free(&t);
    nmz_net_free(net);
}

/* a 1-unit RNN over 3 steps, by hand (double precision) */
static void test_rnn_by_hand(void) {
    nmz_net *net = NULL;
    nmz_tensor x = NMZ_TENSOR_EMPTY, t = NMZ_TENSOR_EMPTY;
    const nmz_tensor *y;
    int in[2] = {NMZ_ANY_LENGTH, 1}, xs[3] = {1, 3, 1};
    const double wx = 0.5, wh = 0.8, b = 0.1, xv[3] = {1.0, -2.0, 0.5}, tv[3] = {0.2, -0.4, 0.6};
    double h[4] = {0}, dh_next = 0, gwx = 0, gwh = 0, gb = 0;
    int s;
    nmz_layer *l;
    CHECK_OK(nmz_net_create(&net, 2, in, 1));
    CHECK_OK(nmz_net_add(net, nmz_layer_rnn(1, NMZ_INIT_AUTO)));
    CHECK_OK(nmz_net_build(net));
    l = nmz_net_layer(net, 0);
    nmz_layer_param_value(l, 0)->data[0] = (float)wx;
    nmz_layer_param_value(l, 1)->data[0] = (float)wh;
    nmz_layer_param_value(l, 2)->data[0] = (float)b;
    CHECK_OK(nmz_tensor_init(&x, 3, xs));
    CHECK_OK(nmz_tensor_init(&t, 3, xs));
    for (s = 0; s < 3; s++) {
        x.data[s] = (float)xv[s];
        t.data[s] = (float)tv[s];
        h[s + 1] = tanh(wx * xv[s] + wh * h[s] + b); /* h[0] = h_{-1} = 0 */
    }
    CHECK_OK(nmz_net_forward(net, &x, &y));
    for (s = 0; s < 3; s++) CHECK_NEAR(y->data[s], h[s + 1], 1e-6);
    CHECK_OK(nmz_net_backward(net, &t, NULL));
    for (s = 2; s >= 0; s--) { /* MSE over 3 rows: ∂L/∂h_t = (h_t − t_t)/3 */
        double dh = (h[s + 1] - tv[s]) / 3.0 + dh_next, dz = dh * (1 - h[s + 1] * h[s + 1]);
        gwx += dz * xv[s];
        gwh += dz * h[s];
        gb += dz;
        dh_next = dz * wh;
    }
    CHECK_NEAR(nmz_layer_param_grad(l, 0)->data[0], gwx, 1e-6);
    CHECK_NEAR(nmz_layer_param_grad(l, 1)->data[0], gwh, 1e-6);
    CHECK_NEAR(nmz_layer_param_grad(l, 2)->data[0], gb, 1e-6);
    nmz_tensor_free(&x);
    nmz_tensor_free(&t);
    nmz_net_free(net);
}

/* the embedding gradient is sparse, but Adam still updates every row the
 * way dense Adam would (momentum keeps moving rows not in this batch) */
static void test_embedding_sparse_adam(void) {
    nmz_net *net = NULL;
    nmz_tensor x = NMZ_TENSOR_EMPTY, t = NMZ_TENSOR_EMPTY;
    nmz_train_opts o = nmz_train_defaults();
    int in = NMZ_ANY_LENGTH, xs[2] = {1, 1}, ts[3] = {1, 1, 2}, k;
    nmz_layer *e;
    float w3_after1, w3_after2, g1;
    const nmz_tensor *g;
    CHECK_OK(nmz_net_create(&net, 1, &in, 4));
    CHECK_OK(nmz_net_add(net, nmz_layer_embedding(5, 2, NMZ_INIT_AUTO)));
    CHECK_OK(nmz_net_build(net));
    e = nmz_net_layer(net, 0);
    CHECK(nmz_layer_init_used(e) == NMZ_INIT_XAVIER);
    o.optimizer = NMZ_OPT_ADAM;
    CHECK_OK(nmz_tensor_init(&x, 2, xs));
    CHECK_OK(nmz_tensor_init(&t, 3, ts));
    /* step 1: token 3 */
    x.data[0] = 3;
    t.data[0] = 1;
    CHECK_OK(nmz_net_forward(net, &x, NULL));
    CHECK_OK(nmz_net_backward(net, &t, NULL));
    g = nmz_layer_param_grad(e, 0);
    for (k = 0; k < 10; k++) CHECK((k / 2 == 3) == (g->data[k] != 0.0f)); /* only row 3 */
    g1 = g->data[6];
    CHECK_OK(nmz_net_optimizer_step(net, &o, 0.1f));
    w3_after1 = nmz_layer_param_value(e, 0)->data[6];
    /* step 2: token 1 only — row 3 has zero gradient now, but its Adam moment m ≠ 0 */
    x.data[0] = 1;
    CHECK_OK(nmz_net_forward(net, &x, NULL));
    CHECK_OK(nmz_net_backward(net, &t, NULL));
    CHECK(nmz_layer_param_grad(e, 0)->data[6] == 0.0f);
    CHECK_OK(nmz_net_optimizer_step(net, &o, 0.1f));
    w3_after2 = nmz_layer_param_value(e, 0)->data[6];
    {
        /* dense Adam for that row at t = 2 with g = 0: m = β1·m1, v = β2·v1 */
        double b1 = 0.9, b2 = 0.999, m = b1 * (1 - b1) * g1, v = b2 * (1 - b2) * (double)g1 * g1;
        double delta = -0.1 * (m / (1 - b1 * b1)) / (sqrt(v / (1 - b2 * b2)) + 1e-8);
        CHECK_MSG(w3_after2 != w3_after1, "row 3 did not move at step 2: lazy (sparse) Adam instead of Adam");
        CHECK_NEAR(w3_after2 - w3_after1, delta, 1e-6);
    }
    /* invalid ids are rejected clearly */
    x.data[0] = 7;
    CHECK_STATUS(nmz_net_forward(net, &x, NULL), NMZ_ERR_ARG);
    x.data[0] = 1.5f;
    CHECK_STATUS(nmz_net_forward(net, &x, NULL), NMZ_ERR_ARG);
    nmz_tensor_free(&x);
    nmz_tensor_free(&t);
    nmz_net_free(net);
}

static void test_clipping(void) {
    nmz_net *net = char_net(4, 2, 3, 2);
    nmz_tensor x = NMZ_TENSOR_EMPTY, t = NMZ_TENSOR_EMPTY;
    nmz_trace_recorder *rec = NULL;
    int xs[2] = {2, 4}, ts[3] = {2, 4, 4}, i, j;
    double n0, n1, sc, sq = 0;
    nmz_rng r;
    float before[64];
    size_t k = 0;
    if (!net) return;
    nmz_rng_seed(&r, 1, 1);
    CHECK_OK(nmz_tensor_init(&x, 2, xs));
    CHECK_OK(nmz_tensor_init(&t, 3, ts));
    fill_ids(&x, &r, 4);
    onehot_targets(&t, &r, 0);
    CHECK_OK(nmz_net_forward(net, &x, NULL));
    CHECK_OK(nmz_net_backward(net, &t, NULL));
    for (i = 0; i < nmz_net_layer_count(net); i++)
        for (j = 0; j < nmz_layer_param_count(nmz_net_layer(net, i)); j++) {
            const nmz_tensor *g = nmz_layer_param_grad(nmz_net_layer(net, i), j);
            size_t e;
            for (e = 0; e < g->size; e++) {
                sq += (double)g->data[e] * g->data[e];
                if (k < 64) before[k++] = g->data[e];
            }
        }
    /* above the norm: no-op */
    CHECK_OK(nmz_net_clip_gradients(net, sqrt(sq) * 2, &n0, &sc));
    CHECK_NEAR(n0, sqrt(sq), 1e-6);
    CHECK(sc == 1.0);
    /* below: every gradient scaled by the same factor, new norm = max */
    CHECK_OK(nmz_trace_recorder_create(&rec));
    nmz_net_set_trace(net, nmz_trace_recorder_fn, rec);
    CHECK_OK(nmz_net_clip_gradients(net, sqrt(sq) / 4, &n0, &sc));
    nmz_net_set_trace(net, NULL, NULL);
    CHECK_NEAR(sc, 0.25, 1e-9);
    CHECK_OK(nmz_net_clip_gradients(net, 1e9, &n1, NULL));
    CHECK_NEAR(n1, sqrt(sq) / 4, 1e-6);
    k = 0;
    for (i = 0; i < nmz_net_layer_count(net) && k < 64; i++)
        for (j = 0; j < nmz_layer_param_count(nmz_net_layer(net, i)); j++) {
            const nmz_tensor *g = nmz_layer_param_grad(nmz_net_layer(net, i), j);
            size_t e;
            for (e = 0; e < g->size && k < 64; e++, k++) CHECK_NEAR(g->data[e], before[k] * 0.25, 1e-7);
        }
    {
        const nmz_trace_event *ev = nmz_trace_recorder_get(rec, 0);
        double v = 0;
        CHECK(ev && ev->phase == NMZ_TRACE_UPDATE && !strcmp(ev->layer_type, "clip") && !strcmp(ev->detail, "clipped"));
        CHECK(ev && nmz_trace_scalar_get(ev, "scale", &v) && fabs(v - 0.25) < 1e-9);
    }
    CHECK_STATUS(nmz_net_clip_gradients(net, 0, NULL, NULL), NMZ_ERR_ARG);
    nmz_trace_recorder_free(rec);
    nmz_tensor_free(&x);
    nmz_tensor_free(&t);
    nmz_net_free(net);
}

static void write_text(const char *p, const char *s) {
    FILE *f = fopen(p, "wb");
    if (!f) return;
    fputs(s, f);
    fclose(f);
}

typedef struct seen_epoch {
    int n, clipped;
    double val;
} seen_epoch;

static int on_epoch(const nmz_epoch_info *info, void *user) {
    seen_epoch *s = user;
    s->n++;
    s->val = info->val_loss;
    s->clipped += info->clipped_batches;
    return 0;
}

/* a tiny repeating text is learned; carry, validation and variable lengths work */
static void test_char_training(void) {
    const char *p = test_path("t_abc.txt");
    char text[1200];
    nmz_text_opts to = nmz_text_defaults();
    nmz_dataset *ds = NULL;
    int i, mode;
    for (i = 0; i < 1199; i++) text[i] = "abcde\n"[i % 6];
    text[1199] = '\0';
    write_text(p, text);
    to.seq_len = 8;
    CHECK_OK(nmz_dataset_load_text(&ds, p, &to));
    if (!ds) return;
    for (mode = 0; mode < 2; mode++) {
        nmz_net *net = char_net(ds->vocab->size, 4, 12, 3);
        nmz_train_opts o = nmz_train_defaults();
        nmz_train_result res;
        seen_epoch se = {0, 0, 0};
        if (!net) break;
        o.optimizer = NMZ_OPT_ADAM;
        o.learning_rate = 0.02f;
        o.batch_size = 8;
        o.max_epochs = 15;
        o.target_loss = 0;
        o.val_fraction = 0.1f;
        o.clip_norm = 1.0f;
        o.carry_state = mode;
        o.on_epoch = on_epoch;
        o.user = &se;
        CHECK_OK(nmz_net_train(net, ds, &o, &res));
        test_note("abcde pattern, %s: train loss %.4f, val loss %.4f, val acc %.0f%%, clipped %d batches",
                  mode ? "carry on " : "carry off", res.loss, res.val_loss, 100 * res.val_accuracy, se.clipped);
        CHECK(se.n == 15 && isfinite(se.val) && se.val == res.val_loss);
        CHECK_MSG(res.val_loss < 0.1 && res.val_accuracy > 0.95, "val loss %g acc %g", res.val_loss, res.val_accuracy);
        CHECK(nmz_net_vocab(net) && nmz_net_vocab(net)->size == 6);
        {
            /* the same net runs on 1 step and on 13 steps */
            nmz_tensor x = NMZ_TENSOR_EMPTY;
            const nmz_tensor *y;
            int xs[2] = {1, 1};
            CHECK_OK(nmz_tensor_init(&x, 2, xs));
            CHECK_OK(nmz_net_forward(net, &x, &y));
            CHECK(y->rank == 3 && y->shape[1] == 1 && y->shape[2] == 6);
            xs[1] = 13;
            CHECK_OK(nmz_tensor_reshape(&x, 2, xs));
            nmz_tensor_zero(&x);
            CHECK_OK(nmz_net_forward(net, &x, &y));
            CHECK(y->shape[1] == 13);
            nmz_tensor_free(&x);
        }
        nmz_net_free(net);
    }
    {
        /* tabular data can use the validation split too */
        static const float X[16] = {0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1};
        static const float Y[8] = {0, 1, 1, 0, 0, 1, 1, 0};
        nmz_dataset *xd = NULL;
        nmz_net *xn = NULL;
        int sizes[3] = {2, 4, 1};
        nmz_train_opts o;
        nmz_train_result res;
        CHECK_OK(nmz_dataset_create(&xd, 8, 2, 1, X, Y));
        CHECK_OK(nmz_net_create_quick(&xn, 3, sizes, 1, NULL));
        o = *nmz_net_train_config(xn);
        o.val_fraction = 0.5f;
        CHECK_OK(nmz_net_train(xn, xd, &o, &res));
        CHECK(res.converged && isfinite(res.val_loss) && res.val_accuracy == 1.0);
        o.val_fraction = 0.05f; /* 0.4 of a sample: no validation split possible */
        CHECK_STATUS(nmz_net_train(xn, xd, &o, &res), NMZ_ERR_ARG);
        nmz_net_free(xn);
        nmz_dataset_free(xd);
    }
    nmz_dataset_free(ds);
}

/* train a small model on the "abcde\n" pattern (shared by the next tests) */
static nmz_net *trained_pattern_net(void) {
    const char *p = test_path("t_abc2.txt");
    char text[1201];
    nmz_text_opts to = nmz_text_defaults();
    nmz_dataset *ds = NULL;
    nmz_net *net;
    nmz_train_opts o = nmz_train_defaults();
    int i;
    for (i = 0; i < 1200; i++) text[i] = "abcde\n"[i % 6];
    text[1200] = '\0';
    write_text(p, text);
    to.seq_len = 8;
    if (nmz_dataset_load_text(&ds, p, &to) != NMZ_OK) return NULL;
    net = char_net(ds->vocab->size, 4, 12, 3);
    o.optimizer = NMZ_OPT_ADAM;
    o.learning_rate = 0.02f;
    o.batch_size = 8;
    o.max_epochs = 15;
    o.target_loss = 0;
    o.seq_len = 8;
    o.clip_norm = 5.0f;
    if (net && nmz_net_train(net, ds, &o, NULL) != NMZ_OK) {
        nmz_net_free(net);
        net = NULL;
    }
    if (net) *nmz_net_train_config(net) = o;
    nmz_dataset_free(ds);
    return net;
}

static void test_generate(void) {
    nmz_net *net = trained_pattern_net(), *back = NULL;
    nmz_gen_opts g = nmz_gen_defaults();
    char *a = NULL, *b = NULL, *c = NULL;
    size_t n = 0;
    CHECK(net != NULL);
    if (!net) return;
    g.prime = "abc";
    g.length = 15;
    g.seed = 7;
    CHECK_OK(nmz_net_generate(net, &g, &a, &n));
    CHECK_OK(nmz_net_generate(net, &g, &b, NULL));
    CHECK(a && b && n == 15 && !strcmp(a, b)); /* same seed, same text */
    CHECK_MSG(a && !strcmp(a, "de\nabcde\nabcde\n"), "learned pattern not reproduced: '%s'", a ? a : "");
    free(b);
    b = NULL;
    /* greedy (top-k 1) ignores the seed; a high temperature with different seeds differs */
    g.top_k = 1;
    g.seed = 99;
    CHECK_OK(nmz_net_generate(net, &g, &b, NULL));
    CHECK(b && !strcmp(a, b));
    free(b);
    b = NULL;
    g.top_k = 0;
    g.temperature = 5.0f;
    g.seed = 1;
    CHECK_OK(nmz_net_generate(net, &g, &b, NULL));
    g.seed = 2;
    CHECK_OK(nmz_net_generate(net, &g, &c, NULL));
    CHECK(b && c && strcmp(b, c) != 0);
    free(b);
    free(c);
    b = c = NULL;
    /* one item: stops right after the end-of-item token */
    g.temperature = 1.0f;
    g.prime = NULL; /* default prime: '\n' */
    g.stop_at_end = 1;
    g.length = 100;
    CHECK_OK(nmz_net_generate(net, &g, &b, &n));
    CHECK_MSG(b && !strcmp(b, "abcde\n") && n == 6, "item: '%s'", b ? b : "");
    free(b);
    b = NULL;
    g.prime = "abz";
    CHECK_STATUS(nmz_net_generate(net, &g, &b, NULL), NMZ_ERR_ARG);
    CHECK(strstr(nmz_last_error(), "U+007A") != NULL && b == NULL);
    g.prime = "abc";
    g.temperature = 0.0f;
    CHECK_STATUS(nmz_net_generate(net, &g, &b, NULL), NMZ_ERR_ARG);
    /* vocabulary and text settings survive save/load; generation is identical */
    CHECK_OK(nmz_net_save(net, test_path("t_charnet.nmz")));
    CHECK_OK(nmz_net_load(&back, test_path("t_charnet.nmz")));
    if (back) {
        const nmz_vocab *v1 = nmz_net_vocab(net), *v2 = nmz_net_vocab(back);
        CHECK(v2 && v2->size == v1->size && !memcmp(v1->codepoints, v2->codepoints, sizeof(uint32_t) * (size_t)v1->size));
        CHECK(v2 && !memcmp(v1->counts, v2->counts, sizeof(uint64_t) * (size_t)v1->size));
        CHECK(nmz_net_is_sequence(back));
        CHECK(nmz_net_train_config(back)->seq_len == 8 && nmz_net_train_config(back)->clip_norm == 5.0f);
        g.temperature = 1.0f;
        g.stop_at_end = 0;
        g.length = 15;
        g.seed = 7;
        CHECK_OK(nmz_net_generate(back, &g, &c, NULL));
        CHECK(c && !strcmp(a, c));
        free(c);
        nmz_net_free(back);
    }
    free(a);
    nmz_net_free(net);
}

static void test_generate_arabic(void) {
    nmz_text_opts to = nmz_text_defaults();
    nmz_dataset *ds = NULL;
    nmz_net *net;
    nmz_train_opts o = nmz_train_defaults();
    nmz_gen_opts g = nmz_gen_defaults();
    char *txt = NULL;
    uint32_t *cp = NULL;
    size_t n = 0, bad = 1, i, nl = 0;
    to.items = 1;
    to.seq_len = 12;
    CHECK_OK(nmz_dataset_load_text(&ds, "examples/names_ar.txt", &to));
    if (!ds) return;
    CHECK(ds->invalid_utf8 == 0 && ds->vocab->size == 37); /* 36 Arabic letters + '\n' */
    net = char_net(ds->vocab->size, 8, 32, 5);
    o.optimizer = NMZ_OPT_ADAM;
    o.learning_rate = 0.01f;
    o.max_epochs = 3;
    o.target_loss = 0;
    CHECK_OK(nmz_net_train(net, ds, &o, NULL));
    g.length = 40;
    g.seed = 3;
    CHECK_OK(nmz_net_generate(net, &g, &txt, NULL));
    if (txt) {
        CHECK_OK(nmz_utf8_decode(txt, strlen(txt), &cp, &n, &bad));
        CHECK(n == 40 && bad == 0); /* 40 code points, valid UTF-8 (each Arabic letter is 2 bytes) */
        for (i = 0; i < n; i++) {
            CHECK(nmz_vocab_index(ds->vocab, cp[i]) >= 0);
            nl += cp[i] == '\n';
        }
        CHECK(strlen(txt) > n); /* multi-byte */
        free(cp);
        free(txt);
    }
    nmz_net_free(net);
    nmz_dataset_free(ds);
}

static int track_min(const nmz_epoch_info *info, void *user) {
    double *best = user;
    if (info->val_loss < *best) *best = info->val_loss;
    return 0;
}

/* early stopping: the small Arabic list overfits after ~10 epochs */
static void test_patience(void) {
    nmz_text_opts to = nmz_text_defaults();
    nmz_dataset *ds = NULL;
    nmz_net *net;
    nmz_train_opts o = nmz_train_defaults();
    nmz_train_result r;
    double best = HUGE_VAL;
    to.items = 1;
    to.seq_len = 12;
    CHECK_OK(nmz_dataset_load_text(&ds, "examples/names_ar.txt", &to));
    if (!ds) return;
    net = char_net(ds->vocab->size, 16, 64, 42);
    o.optimizer = NMZ_OPT_ADAM;
    o.learning_rate = 0.005f;
    o.batch_size = 16;
    o.max_epochs = 80;
    o.target_loss = 0;
    o.val_fraction = 0.1f;
    o.patience = 4;
    o.on_epoch = track_min;
    o.user = &best;
    CHECK_OK(nmz_net_train(net, ds, &o, &r));
    test_note("Arabic names, patience 4: stopped after %d epochs, best epoch %d, val loss %.3f", r.epochs, r.best_epoch,
              r.val_loss);
    CHECK(r.epochs < 80 && r.best_epoch > 0 && r.epochs == r.best_epoch + 4);
    CHECK_NEAR(r.val_loss, best, 1e-9); /* the best weights were restored */
    nmz_net_free(net);
    nmz_dataset_free(ds);
}

static void test_phase2_file_loads(void) {
    /* format 3 written by the phase-2 code: no vocab / train_ext sections */
    static const float want[3][3] = {{0x1.f78916p-1f, 0x1.dab4dp-7f, 0x1.0c16f6p-9f},
                                     {0x1.a7cb68p-3f, 0x1.c26f5ap-2f, 0x1.69aaf2p-2f},
                                     {0x1.76e92ep-12f, 0x1.5c02cap-5f, 0x1.ea10f6p-1f}};
    nmz_net *net = NULL;
    nmz_dataset *ds = NULL;
    nmz_csv_opts co = nmz_csv_defaults();
    int i, k;
    CHECK_OK(nmz_net_load(&net, "tests/data/v3_phase2_iris.nmz"));
    CHECK_OK(nmz_dataset_load_csv(&ds, "examples/iris_train.csv", &co));
    if (net && ds) {
        for (i = 0; i < 3; i++) {
            float y[3];
            CHECK_OK(nmz_net_predict(net, ds->x + 4 * 40 * i, y));
            for (k = 0; k < 3; k++) CHECK_MSG(y[k] == want[i][k], "%a vs %a", (double)y[k], (double)want[i][k]);
        }
        CHECK(nmz_net_vocab(net) == NULL && nmz_net_train_config(net)->clip_norm == 0.0f);
        CHECK(nmz_net_optimizer_state(net, NULL) == NMZ_OPT_ADAM);
        CHECK(nmz_net_normalization(net, NULL, NULL, NULL, NULL) == NMZ_NORM_ZSCORE);
    }
    nmz_net_free(net);
    nmz_dataset_free(ds);
}

const test_case seq_tests[] = {
    {"gen/deterministic_topk_items_saveload", test_generate},
    {"gen/arabic_model_valid_utf8", test_generate_arabic},
    {"io/phase2_format3_file_loads", test_phase2_file_loads},
    {"train/early_stopping_restores_best", test_patience},
    {"seq/gradcheck_rnn_multistep", test_gradcheck_rnn},
    {"seq/gradcheck_char_rnn_head", test_gradcheck_char_rnn},
    {"seq/gradcheck_carried_state", test_gradcheck_carry},
    {"seq/rnn_bptt_by_hand", test_rnn_by_hand},
    {"seq/embedding_sparse_with_adam", test_embedding_sparse_adam},
    {"seq/gradient_clipping", test_clipping},
    {"seq/char_training_val_carry", test_char_training},
    {NULL, NULL}};
