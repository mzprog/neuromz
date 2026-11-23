/* test_optim.c — Adam, learning-rate schedules, input normalization, and
 * the format-3 file sections (optimizer state, normalization, compatibility). */
#include "test.h"
#include "neuromz_layer.h"

#include <stdlib.h>
#include <string.h>

/* 1 → 1 linear net with chosen w, b */
static nmz_net *tiny(float w, float b) {
    nmz_net *net = NULL;
    int in = 1;
    if (nmz_net_create(&net, 1, &in, 1) != NMZ_OK) return NULL;
    nmz_net_add(net, nmz_layer_dense(1, NMZ_INIT_XAVIER));
    nmz_net_build(net);
    nmz_layer_param_value(nmz_net_layer(net, 0), 0)->data[0] = w;
    nmz_layer_param_value(nmz_net_layer(net, 0), 1)->data[0] = b;
    return net;
}

static void one_pass(nmz_net *net, float xv, float tv) {
    nmz_tensor x = NMZ_TENSOR_EMPTY, t = NMZ_TENSOR_EMPTY;
    int s[2] = {1, 1};
    nmz_tensor_init(&x, 2, s);
    nmz_tensor_init(&t, 2, s);
    x.data[0] = xv;
    t.data[0] = tv;
    CHECK_OK(nmz_net_forward(net, &x, NULL));
    CHECK_OK(nmz_net_backward(net, &t, NULL));
    nmz_tensor_free(&x);
    nmz_tensor_free(&t);
}

/* Adam against a by-hand computation in double precision, 3 steps */
static void test_adam_by_hand(void) {
    nmz_net *net = tiny(0.5f, 0.0f);
    nmz_train_opts o = nmz_train_defaults();
    double w = 0.5, b = 0.0, mw = 0, vw = 0, mb = 0, vb = 0;
    const double x = 2.0, t = 3.0, lr = 0.1, b1 = 0.9, b2 = 0.999, eps = 1e-8;
    int step;
    long tt = 0;
    o.optimizer = NMZ_OPT_ADAM;
    if (!net) return;
    for (step = 1; step <= 3; step++) {
        double y = w * x + b, gy = y - t; /* MSE: ∂L/∂y = y − t (one row) */
        double gw = gy * x, gb = gy;
        mw = b1 * mw + (1 - b1) * gw;
        vw = b2 * vw + (1 - b2) * gw * gw;
        mb = b1 * mb + (1 - b1) * gb;
        vb = b2 * vb + (1 - b2) * gb * gb;
        w -= lr * (mw / (1 - pow(b1, step))) / (sqrt(vw / (1 - pow(b2, step))) + eps);
        b -= lr * (mb / (1 - pow(b1, step))) / (sqrt(vb / (1 - pow(b2, step))) + eps);
        one_pass(net, (float)x, (float)t);
        CHECK_OK(nmz_net_optimizer_step(net, &o, (float)lr));
        CHECK_NEAR(nmz_layer_param_value(nmz_net_layer(net, 0), 0)->data[0], w, 1e-6);
        CHECK_NEAR(nmz_layer_param_value(nmz_net_layer(net, 0), 1)->data[0], b, 1e-6);
    }
    /* every Adam step moves a weight by ≈ lr whatever the gradient size (y < t, so w grows) */
    CHECK_NEAR(0.5 + 0.1 * 3, w, 0.02);
    CHECK(nmz_net_optimizer_state(net, &tt) == NMZ_OPT_ADAM && tt == 3);
    /* switching to SGD starts a fresh state */
    o.optimizer = NMZ_OPT_SGD;
    o.momentum = 0.5f;
    one_pass(net, 1, 0);
    CHECK_OK(nmz_net_optimizer_step(net, &o, 0.01f));
    CHECK(nmz_net_optimizer_state(net, &tt) == NMZ_OPT_SGD && tt == 0);
    nmz_net_free(net);
}

static void test_adam_trace_and_preview(void) {
    nmz_net *net = tiny(0.5f, 0.25f);
    nmz_train_opts o = nmz_train_defaults();
    nmz_trace_recorder *rec = NULL;
    const nmz_trace_event *ev;
    double lr = 0, t = 0, b1 = 0, b2 = 0, eps = 0;
    float w0;
    long tt = -1;
    if (!net) return;
    o.optimizer = NMZ_OPT_ADAM;
    CHECK_OK(nmz_trace_recorder_create(&rec));
    one_pass(net, 2, 3);
    nmz_net_set_trace(net, nmz_trace_recorder_fn, rec);
    w0 = nmz_layer_param_value(nmz_net_layer(net, 0), 0)->data[0];
    CHECK_OK(nmz_net_optimizer_preview(net, &o, 0.1f));
    CHECK(nmz_layer_param_value(nmz_net_layer(net, 0), 0)->data[0] == w0); /* nothing changed */
    CHECK(nmz_net_optimizer_state(net, &tt) == -1 && tt == 0);
    CHECK(nmz_trace_recorder_count(rec) == 2);
    ev = nmz_trace_recorder_get(rec, 0);
    CHECK(ev->phase == NMZ_TRACE_UPDATE && !strcmp(ev->detail, "W"));
    CHECK(nmz_trace_scalar_get(ev, "lr", &lr) && nmz_trace_scalar_get(ev, "t", &t) &&
          nmz_trace_scalar_get(ev, "beta1", &b1) && nmz_trace_scalar_get(ev, "beta2", &b2) &&
          nmz_trace_scalar_get(ev, "epsilon", &eps));
    CHECK(t == 1.0);
    {
        const nmz_tensor *g = nmz_trace_find(ev, "grad"), *m = nmz_trace_find(ev, "m"), *v = nmz_trace_find(ev, "v"),
                         *mh = nmz_trace_find(ev, "m_hat"), *vh = nmz_trace_find(ev, "v_hat"),
                         *d = nmz_trace_find(ev, "delta"), *val = nmz_trace_find(ev, "value");
        CHECK(g && m && v && mh && vh && d && val);
        if (g && m && v && mh && vh && d && val) {
            CHECK_NEAR(m->data[0], (1 - b1) * g->data[0], 1e-6);
            CHECK_NEAR(v->data[0], (1 - b2) * g->data[0] * g->data[0], 1e-6);
            CHECK_NEAR(mh->data[0], g->data[0], 1e-5); /* bias correction undoes (1−β) at t = 1 */
            CHECK_NEAR(d->data[0], -lr * mh->data[0] / (sqrt(vh->data[0]) + eps), 1e-6);
            CHECK(val->data[0] == w0);
        }
    }
    /* applying gives exactly the previewed delta */
    nmz_trace_recorder_clear(rec);
    CHECK_OK(nmz_net_optimizer_step(net, &o, 0.1f));
    ev = nmz_trace_recorder_get(rec, 0);
    CHECK(ev && nmz_trace_find(ev, "value")->data[0] == w0); /* value = before the step */
    CHECK_NEAR(nmz_layer_param_value(nmz_net_layer(net, 0), 0)->data[0], w0 + nmz_trace_find(ev, "delta")->data[0], 1e-7);
    nmz_net_set_trace(net, NULL, NULL);
    nmz_trace_recorder_free(rec);
    nmz_net_free(net);
}

static void test_schedules(void) {
    nmz_train_opts o = nmz_train_defaults();
    o.learning_rate = 0.1f;
    CHECK_NEAR(nmz_schedule_lr(&o, 7), 0.1, 1e-7);
    o.schedule = NMZ_SCHED_STEP;
    o.step_size = 2;
    o.gamma = 0.5f;
    CHECK_NEAR(nmz_schedule_lr(&o, 1), 0.1, 1e-7);
    CHECK_NEAR(nmz_schedule_lr(&o, 2), 0.1, 1e-7);
    CHECK_NEAR(nmz_schedule_lr(&o, 3), 0.05, 1e-7);
    CHECK_NEAR(nmz_schedule_lr(&o, 5), 0.025, 1e-7);
    o.schedule = NMZ_SCHED_COSINE;
    o.max_epochs = 10;
    o.min_lr = 0.01f;
    CHECK_NEAR(nmz_schedule_lr(&o, 1), 0.1, 1e-7);
    CHECK_NEAR(nmz_schedule_lr(&o, 6), 0.055, 1e-6); /* halfway: (η + η_min)/2 */
    CHECK(nmz_schedule_lr(&o, 10) > 0.01f && nmz_schedule_lr(&o, 10) < 0.015f);
    o.min_lr = 1.0f; /* > lr */
    CHECK_STATUS(nmz_net_train(NULL, NULL, &o, NULL), NMZ_ERR_ARG);
}

static void test_adam_xor(void) {
    static const float X[8] = {0, 0, 0, 1, 1, 0, 1, 1}, Y[4] = {0, 1, 1, 0};
    nmz_dataset *ds = NULL;
    int sizes[3] = {2, 4, 1}, seed, ok = 0, epochs[6] = {0};
    CHECK_OK(nmz_dataset_create(&ds, 4, 2, 1, X, Y));
    for (seed = 1; seed <= 20; seed++) {
        nmz_net *net = NULL;
        nmz_train_result r;
        nmz_train_opts o;
        CHECK_OK(nmz_net_create_quick(&net, 3, sizes, (uint64_t)seed, NULL));
        o = *nmz_net_train_config(net);
        o.optimizer = NMZ_OPT_ADAM;
        o.learning_rate = 0.05f;
        o.schedule = seed % 2 ? NMZ_SCHED_COSINE : NMZ_SCHED_STEP;
        o.step_size = 500;
        o.gamma = 0.7f;
        CHECK_OK(nmz_net_train(net, ds, &o, &r));
        ok += r.converged;
        if (seed <= 6) epochs[seed - 1] = r.epochs;
        nmz_net_free(net);
    }
    test_note("XOR 2-4-1 with Adam (cosine/step schedules): %d/20 converged, e.g. %d %d %d epochs", ok, epochs[0],
              epochs[1], epochs[2]);
    CHECK_MSG(ok >= 18, "%d/20", ok);
    nmz_dataset_free(ds);
}

/* ------------------------------------------------------------ normalization */

static void test_normalization(void) {
    nmz_dataset *ds = NULL;
    nmz_net *net = NULL;
    const float *off, *sc;
    int fitted = 0, groups = 0, in = 2, i;
    float x[8] = {10, -1, 20, -1, 30, 1, 40, 1}, y[4] = {0, 1, 0, 1}, out1, out2, xn[2];
    CHECK_OK(nmz_dataset_create(&ds, 4, 2, 1, x, y));
    CHECK_OK(nmz_net_create(&net, 1, &in, 3));
    CHECK_OK(nmz_net_add_dense(net, 1, NMZ_ACT_SIGMOID, NMZ_INIT_AUTO));
    CHECK_OK(nmz_net_set_normalization(net, NMZ_NORM_ZSCORE));
    CHECK_STATUS(nmz_net_predict(net, x, &out1), NMZ_ERR_STATE); /* not fitted yet */
    CHECK_OK(nmz_net_fit_normalization(net, ds));
    CHECK(nmz_net_normalization(net, &fitted, &groups, &off, &sc) == NMZ_NORM_ZSCORE && fitted && groups == 2);
    CHECK_NEAR(off[0], 25, 1e-5);
    CHECK_NEAR(1.0 / sc[0], sqrt(125.0), 1e-4); /* population std of 10,20,30,40 */
    CHECK_NEAR(off[1], 0, 1e-6);
    CHECK_NEAR(sc[1], 1, 1e-6);
    /* predict(raw) == predict-without-normalization(normalized by hand) */
    CHECK_OK(nmz_net_predict(net, x + 2, &out1));
    for (i = 0; i < 2; i++) xn[i] = (x[2 + i] - off[i]) * sc[i];
    CHECK_OK(nmz_net_set_normalization(net, NMZ_NORM_NONE));
    CHECK_OK(nmz_net_predict(net, xn, &out2));
    CHECK_NEAR(out1, out2, 1e-6);
    /* min-max */
    CHECK_OK(nmz_net_set_normalization(net, NMZ_NORM_MINMAX));
    CHECK_OK(nmz_net_fit_normalization(net, ds));
    nmz_net_normalization(net, NULL, NULL, &off, &sc);
    CHECK_NEAR(off[0], 10, 1e-6);
    CHECK_NEAR(sc[0], 1.0 / 30, 1e-7);
    CHECK_NEAR(off[1], -1, 1e-6);
    CHECK_NEAR(sc[1], 0.5, 1e-7);
    nmz_net_free(net);
    nmz_dataset_free(ds);
}

static void test_normalization_channels_and_grad(void) {
    nmz_net *net = NULL;
    nmz_dataset *ds = NULL;
    int shape[3] = {2, 3, 3}, groups = 0, i;
    const float *off, *sc;
    float xs[2 * 18], ys[2 * 2] = {1, 0, 0, 1};
    nmz_rng r;
    nmz_rng_seed(&r, 4, 4);
    for (i = 0; i < 36; i++) xs[i] = (float)((i % 18) < 9 ? 100 + 50 * nmz_rng_uniform(&r) : -5 + nmz_rng_uniform(&r));
    CHECK_OK(nmz_dataset_create(&ds, 2, 18, 2, xs, ys));
    CHECK_OK(nmz_net_create(&net, 3, shape, 2));
    CHECK_OK(nmz_net_add(net, nmz_layer_conv2d(2, 2, 2, 1, 0, NMZ_INIT_AUTO)));
    CHECK_OK(nmz_net_add(net, nmz_layer_flatten()));
    CHECK_OK(nmz_net_add_dense(net, 2, NMZ_ACT_SOFTMAX, NMZ_INIT_AUTO));
    CHECK_OK(nmz_net_set_loss(net, NMZ_LOSS_CROSS_ENTROPY));
    CHECK_OK(nmz_net_set_normalization(net, NMZ_NORM_ZSCORE));
    CHECK_OK(nmz_net_fit_normalization(net, ds));
    nmz_net_normalization(net, NULL, &groups, &off, &sc);
    CHECK(groups == 2); /* one per channel */
    CHECK(off[0] > 100 && off[0] < 150 && off[1] > -5 && off[1] < -4);
    {
        /* the input gradient is w.r.t. the RAW input: gradcheck perturbs raw x */
        nmz_tensor x = NMZ_TENSOR_EMPTY, t = NMZ_TENSOR_EMPTY;
        int bx[4] = {2, 2, 3, 3}, bt[2] = {2, 2};
        nmz_gradcheck_result res;
        CHECK_OK(nmz_tensor_init(&x, 4, bx));
        CHECK_OK(nmz_tensor_init(&t, 2, bt));
        memcpy(x.data, xs, sizeof xs);
        memcpy(t.data, ys, sizeof ys);
        CHECK_OK(nmz_net_gradcheck(net, &x, &t, 2e-2, 0, 0, &res));
        CHECK_MSG(res.failures == 0, "%zu failures, worst %s %.6g vs %.6g", res.failures, res.worst_param,
                  res.worst_analytic, res.worst_numeric);
        nmz_tensor_free(&x);
        nmz_tensor_free(&t);
    }
    nmz_net_free(net);
    nmz_dataset_free(ds);
}

/* ------------------------------------------------------------- file format */

static uint32_t crc32_ref(const unsigned char *p, size_t n) {
    uint32_t c = 0xFFFFFFFFu;
    size_t i;
    int k;
    for (i = 0; i < n; i++) {
        c ^= p[i];
        for (k = 0; k < 8; k++) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
    }
    return ~c;
}

static void test_v2_fixture(void) {
    /* written by the format-2 writer (phase 1), see tests/data */
    static const float X[8] = {0, 0, 0, 1, 1, 0, 1, 1};
    static const float want[4] = {0x1.6d90a8p-6f, 0x1.e86732p-1f, 0x1.e6d36cp-1f, 0x1.bcec1p-5f};
    nmz_net *net = NULL;
    const char *p = getenv("NMZ_FIXTURES") ? getenv("NMZ_FIXTURES") : "tests/data/v2_xor.nmz";
    int i;
    CHECK_OK(nmz_net_load(&net, p));
    if (!net) return;
    for (i = 0; i < 4; i++) {
        float y;
        CHECK_OK(nmz_net_predict(net, X + 2 * i, &y));
        CHECK_MSG(y == want[i], "sample %d: %a vs %a", i, (double)y, (double)want[i]);
    }
    CHECK(nmz_net_train_config(net)->optimizer == NMZ_OPT_SGD);
    CHECK(nmz_net_train_config(net)->schedule == NMZ_SCHED_CONSTANT);
    CHECK(nmz_net_normalization(net, NULL, NULL, NULL, NULL) == NMZ_NORM_NONE);
    CHECK(nmz_net_class_count(net) == 1 && !strcmp(nmz_net_class_name(net, 0), "xor"));
    CHECK(nmz_net_optimizer_state(net, NULL) == -1);
    nmz_net_free(net);
}

static void test_optimizer_state_resume(void) {
    static const float X[8] = {0, 0, 0, 1, 1, 0, 1, 1}, Y[4] = {0, 1, 1, 0};
    nmz_dataset *ds = NULL;
    nmz_net *a = NULL, *b = NULL, *c = NULL;
    nmz_train_result r;
    nmz_train_opts o;
    int sizes[3] = {2, 4, 1};
    long ta = 0, tb = 0;
    float ya, yb;
    CHECK_OK(nmz_dataset_create(&ds, 4, 2, 1, X, Y));
    CHECK_OK(nmz_net_create_quick(&a, 3, sizes, 11, NULL));
    CHECK_OK(nmz_net_create_quick(&b, 3, sizes, 11, NULL));
    o = *nmz_net_train_config(a);
    o.optimizer = NMZ_OPT_ADAM;
    o.learning_rate = 0.02f;
    o.batch_size = 2;
    o.target_loss = 0;
    o.max_epochs = 10;
    CHECK_OK(nmz_net_set_normalization(a, NMZ_NORM_MINMAX));
    CHECK_OK(nmz_net_set_normalization(b, NMZ_NORM_MINMAX));
    CHECK_OK(nmz_net_train(a, ds, &o, &r)); /* 10 epochs straight */
    o.max_epochs = 5;
    CHECK_OK(nmz_net_train(b, ds, &o, &r)); /* 5 epochs, save, load, 5 more */
    *nmz_net_train_config(b) = o;           /* stored with the model */
    CHECK_OK(nmz_net_save(b, test_path("t_adam.nmz")));
    nmz_net_free(b);
    b = NULL;
    CHECK_OK(nmz_net_load(&b, test_path("t_adam.nmz")));
    if (!b) goto out;
    CHECK(nmz_net_optimizer_state(b, &tb) == NMZ_OPT_ADAM && tb == 10);
    CHECK(nmz_net_normalization(b, NULL, NULL, NULL, NULL) == NMZ_NORM_MINMAX);
    CHECK(nmz_net_train_config(b)->optimizer == NMZ_OPT_ADAM && nmz_net_train_config(b)->batch_size == 2);
    CHECK_OK(nmz_net_train(b, ds, &o, &r));
    nmz_net_optimizer_state(a, &ta);
    nmz_net_optimizer_state(b, &tb);
    CHECK(ta == 20 && tb == 20);
    CHECK_OK(nmz_net_predict(a, X + 2, &ya));
    CHECK_OK(nmz_net_predict(b, X + 2, &yb));
    CHECK_MSG(ya == yb, "resumed Adam differs: %a vs %a", (double)ya, (double)yb);
    /* without optimizer state: smaller file, state starts fresh */
    CHECK_OK(nmz_net_save_ex(b, test_path("t_noopt.nmz"), NMZ_SAVE_NO_OPTIMIZER));
    CHECK_OK(nmz_net_load(&c, test_path("t_noopt.nmz")));
    CHECK(c && nmz_net_optimizer_state(c, NULL) == -1);
out:
    nmz_net_free(a);
    nmz_net_free(b);
    nmz_net_free(c);
    nmz_dataset_free(ds);
}

static void test_unknown_section_skipped(void) {
    nmz_net *net = NULL, *back = NULL;
    int sizes[3] = {2, 3, 1};
    FILE *f;
    unsigned char *buf, extra[4 + 6 + 4 + 3] = {6, 0, 0, 0, 'f', 'u', 't', 'u', 'r', 'e', 3, 0, 0, 0, 'a', 'b', 'c'};
    long n;
    uint32_t crc;
    CHECK_OK(nmz_net_create_quick(&net, 3, sizes, 1, NULL)); /* untrained, no normalization: only train_ext */
    CHECK_OK(nmz_net_save(net, test_path("t_sec.nmz")));
    f = fopen(test_path("t_sec.nmz"), "rb");
    if (!f) return;
    fseek(f, 0, SEEK_END);
    n = ftell(f);
    rewind(f);
    buf = malloc((size_t)n + sizeof extra);
    if (buf && fread(buf, 1, (size_t)n, f) == (size_t)n) {
        fclose(f);
        /* layout of the tail: u32 count = 1, "train_ext" section (4+9 + 4 + 20 bytes), CRC */
        {
            long cnt = n - 4 - 37 - 4;
            CHECK(buf[cnt] == 1 && buf[cnt + 1] == 0 && buf[cnt + 4] == 9 && !memcmp(buf + cnt + 8, "train_ext", 9));
            buf[cnt] = 2; /* append one more section after train_ext */
        }
        memcpy(buf + n - 4, extra, sizeof extra);
        crc = crc32_ref(buf, (size_t)n - 4 + sizeof extra);
        buf[n - 4 + sizeof extra] = (unsigned char)crc;
        buf[n - 3 + sizeof extra] = (unsigned char)(crc >> 8);
        buf[n - 2 + sizeof extra] = (unsigned char)(crc >> 16);
        buf[n - 1 + sizeof extra] = (unsigned char)(crc >> 24);
        f = fopen(test_path("t_sec2.nmz"), "wb");
        if (f) {
            fwrite(buf, 1, (size_t)n + sizeof extra, f);
            fclose(f);
        }
        CHECK_OK(nmz_net_load(&back, test_path("t_sec2.nmz")));
        nmz_net_free(back);
    } else {
        fclose(f);
    }
    free(buf);
    nmz_net_free(net);
}

const test_case optim_tests[] = {
    {"optim/adam_by_hand", test_adam_by_hand},
    {"optim/adam_trace_and_preview", test_adam_trace_and_preview},
    {"optim/schedules", test_schedules},
    {"optim/adam_xor_20_seeds", test_adam_xor},
    {"norm/zscore_minmax_predict", test_normalization},
    {"norm/per_channel_and_raw_input_grad", test_normalization_channels_and_grad},
    {"io/v2_file_still_loads", test_v2_fixture},
    {"io/optimizer_state_resume", test_optimizer_state_resume},
    {"io/unknown_section_skipped", test_unknown_section_skipped},
    {NULL, NULL}};
