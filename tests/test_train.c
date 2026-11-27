/* test_train.c — convergence, multi-output (the v1 bug), determinism, trace. */
#include "test.h"

#include <stdlib.h>
#include <string.h>

static const float XOR_X[8] = {0, 0, 0, 1, 1, 0, 1, 1};
static const float XOR_Y[4] = {0, 1, 1, 0};
static const float XOR_AND_Y[8] = {0, 0, 1, 0, 1, 0, 0, 1}; /* columns: XOR, AND */

static int all_close(nmz_net *net, const nmz_dataset *ds, float tol) {
    int i, k, n_out = ds->n_outputs;
    float y[8];
    for (i = 0; i < ds->n_samples; i++) {
        if (nmz_net_predict(net, ds->x + i * ds->n_inputs, y) != NMZ_OK) return 0;
        for (k = 0; k < n_out; k++)
            if (fabsf(y[k] - ds->y[i * n_out + k]) > tol) return 0;
    }
    return 1;
}

static void test_xor_quick_seeds(void) {
    nmz_dataset *ds = NULL;
    int sizes[3] = {2, 4, 1}, seed, ok = 0, total_epochs = 0;
    const int n_seeds = 20;
    CHECK_OK(nmz_dataset_create(&ds, 4, 2, 1, XOR_X, XOR_Y));
    for (seed = 1; seed <= n_seeds; seed++) {
        nmz_net *net = NULL;
        nmz_train_result res;
        CHECK_OK(nmz_net_create_quick(&net, 3, sizes, (uint64_t)seed, NULL));
        CHECK_OK(nmz_net_train(net, ds, NULL, &res));
        if (res.converged && all_close(net, ds, 0.1f)) {
            ok++;
            total_epochs += res.epochs;
        }
        nmz_net_free(net);
    }
    test_note("XOR 2-4-1 quick defaults: %d/%d seeds converged, mean %d epochs", ok, n_seeds,
              ok ? total_epochs / ok : 0);
    CHECK_MSG(ok == n_seeds, "only %d/%d seeds converged", ok, n_seeds);
    nmz_dataset_free(ds);
}

static void test_xor_relu_he(void) {
    /* advanced configuration: relu hidden + He init, sigmoid out, full batch */
    nmz_dataset *ds = NULL;
    int in = 2, seed, ok = 0;
    const int n_seeds = 20;
    CHECK_OK(nmz_dataset_create(&ds, 4, 2, 1, XOR_X, XOR_Y));
    for (seed = 1; seed <= n_seeds; seed++) {
        nmz_net *net = NULL;
        nmz_train_result res;
        nmz_train_opts o = nmz_train_defaults();
        o.max_epochs = 5000;
        o.batch_size = 4;
        o.learning_rate = 0.5f;
        o.momentum = 0.9f;
        CHECK_OK(nmz_net_create(&net, 1, &in, (uint64_t)seed));
        CHECK_OK(nmz_net_add_dense(net, 8, NMZ_ACT_RELU, NMZ_INIT_HE));
        CHECK_OK(nmz_net_add_dense(net, 1, NMZ_ACT_SIGMOID, NMZ_INIT_XAVIER));
        CHECK_OK(nmz_net_train(net, ds, &o, &res));
        ok += res.converged && all_close(net, ds, 0.1f);
        nmz_net_free(net);
    }
    test_note("XOR 2-8(relu,he)-1: %d/%d seeds converged", ok, n_seeds);
    CHECK_MSG(ok >= n_seeds * 8 / 10, "only %d/%d seeds converged", ok, n_seeds);
    nmz_dataset_free(ds);
}

/* XOR and AND at once from one network with two sigmoid outputs. In v1 the
 * second output never learned (all deltas were applied to column 0). */
static void test_xor_and_multi_output(void) {
    nmz_dataset *ds = NULL;
    int in = 2, seed, ok = 0;
    const int n_seeds = 20;
    CHECK_OK(nmz_dataset_create(&ds, 4, 2, 2, XOR_X, XOR_AND_Y));
    for (seed = 1; seed <= n_seeds; seed++) {
        nmz_net *net = NULL;
        nmz_train_result res;
        nmz_train_opts o = nmz_train_defaults();
        nmz_tensor W0 = NMZ_TENSOR_EMPTY;
        nmz_tensor *W;
        o.max_epochs = 5000;
        o.learning_rate = 0.5f;
        CHECK_OK(nmz_net_create(&net, 1, &in, (uint64_t)seed));
        CHECK_OK(nmz_net_add_dense(net, 4, NMZ_ACT_TANH, NMZ_INIT_AUTO));
        CHECK_OK(nmz_net_add_dense(net, 2, NMZ_ACT_SIGMOID, NMZ_INIT_AUTO));
        CHECK_OK(nmz_net_build(net));
        W = nmz_layer_param_value(nmz_net_layer(net, 2), 0); /* hidden → output weights [4, 2] */
        CHECK_OK(nmz_tensor_copy(&W0, W));
        CHECK_OK(nmz_net_train(net, ds, &o, &res));
        if (res.converged && all_close(net, ds, 0.1f)) ok++;
        if (seed == 1) {
            int i, col1_changed = 0;
            for (i = 0; i < 4; i++) col1_changed += W->data[i * 2 + 1] != W0.data[i * 2 + 1];
            CHECK_MSG(col1_changed == 4, "weights into output 1 did not all change (%d/4)", col1_changed);
        }
        nmz_tensor_free(&W0);
        nmz_net_free(net);
    }
    test_note("XOR+AND 2-4-2 (sigmoid outputs, mse): %d/%d seeds converged", ok, n_seeds);
    CHECK_MSG(ok == n_seeds, "only %d/%d seeds converged", ok, n_seeds);
    nmz_dataset_free(ds);
}

static void test_determinism(void) {
    nmz_dataset *ds = NULL;
    int sizes[3] = {2, 3, 1}, run;
    double loss[2];
    float y[2];
    nmz_train_opts o;
    CHECK_OK(nmz_dataset_create(&ds, 4, 2, 1, XOR_X, XOR_Y));
    for (run = 0; run < 2; run++) {
        nmz_net *net = NULL;
        nmz_train_result res;
        CHECK_OK(nmz_net_create_quick(&net, 3, sizes, 99, NULL));
        o = *nmz_net_train_config(net);
        o.max_epochs = 50;
        o.batch_size = 1; /* shuffling matters with batch 1 */
        CHECK_OK(nmz_net_train(net, ds, &o, &res));
        CHECK(res.epochs == 50 && !res.converged);
        CHECK_OK(nmz_net_predict(net, XOR_X + 2, &y[run]));
        loss[run] = res.loss;
        nmz_net_free(net);
    }
    CHECK(loss[0] == loss[1]);
    CHECK(y[0] == y[1]);
    nmz_dataset_free(ds);
}

typedef struct epoch_log {
    int calls;
    double first, last;
} epoch_log;

static int on_epoch(const nmz_epoch_info *info, void *user) {
    epoch_log *log = user;
    if (log->calls++ == 0) log->first = info->loss;
    log->last = info->loss;
    return info->epoch >= 7; /* ask to stop after 7 epochs */
}

static void test_callback_and_stop(void) {
    nmz_dataset *ds = NULL;
    nmz_net *net = NULL;
    nmz_train_result res;
    nmz_train_opts o;
    epoch_log log = {0, 0, 0};
    int sizes[3] = {2, 4, 1};
    CHECK_OK(nmz_dataset_create(&ds, 4, 2, 1, XOR_X, XOR_Y));
    CHECK_OK(nmz_net_create_quick(&net, 3, sizes, 3, NULL));
    o = *nmz_net_train_config(net);
    o.on_epoch = on_epoch;
    o.user = &log;
    CHECK_OK(nmz_net_train(net, ds, &o, &res));
    CHECK(res.stopped && res.epochs == 7 && log.calls == 7);
    CHECK(res.loss == log.last);
    /* a huge learning rate on a linear output must end in a clean NUMERIC
     * error, not garbage (sigmoid outputs saturate instead of exploding) */
    {
        nmz_net *lin = NULL;
        int in = 2;
        o = nmz_train_defaults();
        o.learning_rate = 1e3f;
        o.max_epochs = 1000;
        CHECK_OK(nmz_net_create(&lin, 1, &in, 1));
        CHECK_OK(nmz_net_add_dense(lin, 1, NMZ_ACT_LINEAR, NMZ_INIT_AUTO));
        CHECK_STATUS(nmz_net_train(lin, ds, &o, &res), NMZ_ERR_NUMERIC);
        CHECK_MSG(strstr(nmz_last_error(), "learning rate") != NULL, "%s", nmz_last_error());
        nmz_net_free(lin);
    }
    /* dataset that does not fit */
    {
        nmz_dataset *bad = NULL;
        CHECK_OK(nmz_dataset_create(&bad, 4, 1, 1, XOR_X, XOR_Y));
        CHECK_STATUS(nmz_net_train(net, bad, NULL, &res), NMZ_ERR_SHAPE);
        nmz_dataset_free(bad);
    }
    nmz_net_free(net);
    nmz_dataset_free(ds);
}

/* ------------------------------------------------------------------ trace */

typedef struct counter {
    int n;
    char seq[64];
} counter;

static void count_events(const nmz_trace_event *ev, void *user) {
    counter *c = user;
    if (c->n < 63) {
        c->seq[c->n] = "FLBU"[ev->phase];
        c->seq[c->n + 1] = '\0';
    }
    c->n++;
}

static void test_trace(void) {
    nmz_net *net = NULL;
    nmz_trace_recorder *rec = NULL;
    nmz_tensor x = NMZ_TENSOR_EMPTY, t = NMZ_TENSOR_EMPTY;
    const nmz_tensor *y;
    counter c;
    int sizes[3] = {2, 3, 1}, shape[2] = {1, 2}, tshape[2] = {1, 1};
    size_t i;
    const nmz_trace_event *ev;
    double L = 0, Ls = 0;
    FILE *f;
    memset(&c, 0, sizeof c);
    CHECK_OK(nmz_net_create_quick(&net, 3, sizes, 1, NULL));
    CHECK_OK(nmz_tensor_init(&x, 2, shape));
    CHECK_OK(nmz_tensor_init(&t, 2, tshape));
    x.data[0] = 1;
    t.data[0] = 1;
    /* off by default: no events */
    CHECK_OK(nmz_net_forward(net, &x, &y));
    CHECK_OK(nmz_net_backward(net, &t, NULL));
    nmz_net_set_trace(net, count_events, &c);
    CHECK_OK(nmz_net_forward(net, &x, &y));
    CHECK_OK(nmz_net_backward(net, &t, &L));
    CHECK_OK(nmz_net_update(net, 0.5f, 0.9f));
    /* layers: dense, tanh, dense, sigmoid → 4 F, 1 L, 1 loss-B + 4 B, 4 params → 4 U */
    CHECK_MSG(strcmp(c.seq, "FFFFLBBBBBUUUU") == 0, "sequence %s", c.seq);

    CHECK_OK(nmz_trace_recorder_create(&rec));
    nmz_net_set_trace(net, nmz_trace_recorder_fn, rec);
    CHECK_OK(nmz_net_forward(net, &x, &y));
    CHECK_OK(nmz_net_backward(net, &t, &L));
    CHECK_OK(nmz_net_update(net, 0.5f, 0.9f));
    nmz_net_set_trace(net, NULL, NULL);
    CHECK(nmz_trace_recorder_status(rec) == NMZ_OK);
    CHECK(nmz_trace_recorder_count(rec) == 14);
    /* the recorder holds deep copies: values survive later passes */
    ev = nmz_trace_recorder_get(rec, 0);
    CHECK(ev->phase == NMZ_TRACE_FORWARD && strcmp(ev->layer_type, "dense") == 0 && ev->layer == 0);
    CHECK(nmz_trace_find(ev, "x")->data[0] == 1.0f && nmz_trace_find(ev, "W") && nmz_trace_find(ev, "z"));
    ev = nmz_trace_recorder_get(rec, 4);
    CHECK(ev->phase == NMZ_TRACE_LOSS && nmz_trace_scalar_get(ev, "L", &Ls) && Ls == L);
    ev = nmz_trace_recorder_get(rec, 6); /* backward through sigmoid */
    CHECK(strcmp(ev->detail, "sigmoid") == 0 && nmz_trace_find(ev, "f'(z)") != NULL);
    {
        /* δ = dL/da · f'(z) holds in the recorded numbers */
        const nmz_tensor *g = nmz_trace_find(ev, "dL/da"), *fp = nmz_trace_find(ev, "f'(z)"),
                         *d = nmz_trace_find(ev, "dL/dz");
        CHECK_NEAR(d->data[0], g->data[0] * fp->data[0], 1e-7);
    }
    ev = nmz_trace_recorder_get(rec, 10);
    CHECK(ev->phase == NMZ_TRACE_UPDATE && strcmp(ev->detail, "W") == 0);
    {
        double lr = 0, mu = 0;
        const nmz_tensor *v0 = nmz_trace_find(ev, "velocity_prev"), *g = nmz_trace_find(ev, "grad"),
                         *dl = nmz_trace_find(ev, "delta");
        CHECK(nmz_trace_scalar_get(ev, "lr", &lr) && nmz_trace_scalar_get(ev, "momentum", &mu));
        for (i = 0; i < dl->size; i++) CHECK_NEAR(dl->data[i], mu * v0->data[i] - lr * g->data[i], 1e-6);
    }
    /* JSON lines */
    f = fopen(test_path("t_trace.jsonl"), "w+");
    CHECK(f != NULL);
    if (f) {
        char line[4096];
        for (i = 0; i < nmz_trace_recorder_count(rec); i++)
            CHECK_OK(nmz_trace_event_write_json(nmz_trace_recorder_get(rec, i), f));
        rewind(f);
        CHECK(fgets(line, sizeof line, f) != NULL);
        CHECK(strncmp(line, "{\"step\":", 8) == 0 && strstr(line, "\"phase\":\"forward\"") &&
              strstr(line, "\"W\":{\"shape\":[2,3]"));
        fclose(f);
    }
    nmz_trace_recorder_free(rec);
    nmz_tensor_free(&x);
    nmz_tensor_free(&t);
    nmz_net_free(net);
}

const test_case train_tests[] = {
    {"train/xor_quick_20_seeds", test_xor_quick_seeds},
    {"train/xor_relu_he_20_seeds", test_xor_relu_he},
    {"train/xor_and_multi_output", test_xor_and_multi_output},
    {"train/deterministic_with_seed", test_determinism},
    {"train/callback_stop_and_errors", test_callback_and_stop},
    {"trace/events_recorder_json", test_trace},
    {NULL, NULL}};
