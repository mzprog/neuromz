/* test_backend.c — the backend layer (DESIGN §11), without a GPU.
 *
 *   - selection and information (auto falls back to the CPU here);
 *   - every layer type on "cpu" (the reference kernels) against
 *     "cpu-generic" (the compositions a GPU runs): outputs, every
 *     parameter gradient and the input gradient agree to float rounding;
 *   - a fake device registered through the public API: the CPU's ops, but
 *     data in separate "device" memory with explicit copies. It must give
 *     exactly the bits of cpu-generic (same arithmetic, so any stale copy
 *     or missed transfer shows), pass the gradient checks of every layer
 *     type, train (data resident on the device), generate (KV cache = no
 *     cache), trace, move between backends, run a host-only third-party
 *     layer, and fail cleanly when its memory runs out. */
#include "test.h"
#include "neuromz_backend.h"
#include "neuromz_layer.h"

#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

/* ----------------------------------------------------------- fake device */

static nmz_backend g_fake;
static size_t g_fake_used, g_fake_cap, g_fake_moves; /* bytes, bytes (0 = no limit), transfers */

static void *fake_alloc(const nmz_backend *be, size_t bytes) {
    size_t *p;
    (void)be;
    if (g_fake_cap && g_fake_used + bytes > g_fake_cap) {
        nmz_backend_error(NMZ_ERR_NOMEM, "fake device: out of memory (%zu bytes in use, %zu more needed): try a smaller "
                                         "batch size", g_fake_used, bytes);
        return NULL;
    }
    p = malloc(bytes + sizeof(size_t) * 2);
    if (!p) return NULL;
    p[0] = bytes;
    g_fake_used += bytes;
    memset(p + 2, 0xff, bytes); /* NaN: reading memory nobody wrote shows up */
    return p + 2;
}

static void fake_release(const nmz_backend *be, void *q) {
    size_t *p = (size_t *)q - 2;
    (void)be;
    g_fake_used -= p[0];
    free(p);
}

static nmz_status fake_move(const nmz_backend *be, void *dst, const void *src, size_t bytes) {
    (void)be;
    memcpy(dst, src, bytes);
    g_fake_moves++;
    return NMZ_OK;
}

static void fake_info(const nmz_backend *be, nmz_device_info *out) {
    (void)be;
    snprintf(out->device, sizeof out->device, "fake device");
    out->mem_used = g_fake_used;
}

static int use_fake(void) {
    static int registered = 0;
    if (!registered) {
        if (nmz_backend_select("cpu-generic") != NMZ_OK) return 0;
        g_fake = *nmz_backend_current();
        g_fake.name = "fake-device";
        g_fake.summary = "test";
        g_fake.host_memory = 0;
        g_fake.alloc = fake_alloc;
        g_fake.release = fake_release;
        g_fake.upload = fake_move;
        g_fake.download = fake_move;
        g_fake.info = fake_info;
        if (nmz_backend_register(&g_fake) != NMZ_OK) return 0;
        registered = 1;
    }
    return nmz_backend_select("fake-device") == NMZ_OK;
}

/* -------------------------------------------------------------- the zoo */

enum { Z_DENSE, Z_ACTS, Z_CONV, Z_RNN, Z_LSTM, Z_GRU, Z_GPT, Z_PARTS, Z_COUNT };
static const char *const k_zoo[Z_COUNT] = {"dense+softmax/ce", "activations/mse", "conv+pool+dropout", "embed+rnn (carry)",
                                           "lstm+dropout+lstm", "gru (carry)", "transformer (tied)",
                                           "sinusoid+ln+attention+ffn"};

typedef struct model {
    nmz_net *net;
    nmz_tensor x, t;
    int carry; /* warm the carried state up before the measured pass */
} model;

static void fill_r(nmz_tensor *t, nmz_rng *r, double lo, double hi) {
    size_t i;
    for (i = 0; i < t->size; i++) t->data[i] = (float)(lo + (hi - lo) * nmz_rng_uniform(r));
}

/* the same net, weights and batch on whatever backend is selected */
static int make(int k, model *m) {
    nmz_rng r;
    int i, j, V = 11, shape[4];
    memset(m, 0, sizeof *m);
    nmz_rng_seed(&r, 77 + (uint64_t)k, 3);
    switch (k) {
    case Z_DENSE:
        shape[0] = 5;
        if (nmz_net_create(&m->net, 1, shape, 5) != NMZ_OK) return 0;
        nmz_net_add_dense(m->net, 7, NMZ_ACT_TANH, NMZ_INIT_AUTO);
        nmz_net_add_dense(m->net, 3, NMZ_ACT_SOFTMAX, NMZ_INIT_AUTO);
        nmz_net_set_loss(m->net, NMZ_LOSS_CROSS_ENTROPY);
        nmz_net_set_normalization(m->net, NMZ_NORM_ZSCORE);
        break;
    case Z_ACTS:
        shape[0] = 4;
        if (nmz_net_create(&m->net, 1, shape, 6) != NMZ_OK) return 0;
        nmz_net_add_dense(m->net, 6, NMZ_ACT_SIGMOID, NMZ_INIT_AUTO);
        nmz_net_add_dense(m->net, 6, NMZ_ACT_LEAKY_RELU, NMZ_INIT_AUTO);
        nmz_net_add_dense(m->net, 6, NMZ_ACT_GELU, NMZ_INIT_AUTO);
        nmz_net_add_dense(m->net, 5, NMZ_ACT_RELU, NMZ_INIT_AUTO);
        nmz_net_add_dense(m->net, 3, NMZ_ACT_LINEAR, NMZ_INIT_AUTO);
        nmz_net_add(m->net, nmz_layer_activation(NMZ_ACT_SOFTMAX, 0)); /* softmax + MSE: the Jacobian path */
        break;
    case Z_CONV:
        shape[0] = 2, shape[1] = 7, shape[2] = 6;
        if (nmz_net_create(&m->net, 3, shape, 7) != NMZ_OK) return 0;
        nmz_net_add(m->net, nmz_layer_conv2d(3, 3, 3, 1, NMZ_PAD_SAME, NMZ_INIT_AUTO));
        nmz_net_add(m->net, nmz_layer_activation(NMZ_ACT_TANH, 0));
        nmz_net_add(m->net, nmz_layer_maxpool2d(2, 1));
        nmz_net_add(m->net, nmz_layer_conv2d(4, 2, 3, 2, 1, NMZ_INIT_XAVIER));
        nmz_net_add(m->net, nmz_layer_activation(NMZ_ACT_SIGMOID, 0));
        nmz_net_add(m->net, nmz_layer_flatten());
        nmz_net_add(m->net, nmz_layer_dropout(0.3f));
        nmz_net_add_dense(m->net, 3, NMZ_ACT_SOFTMAX, NMZ_INIT_AUTO);
        nmz_net_set_loss(m->net, NMZ_LOSS_CROSS_ENTROPY);
        break;
    default: /* text models: ids [B, T] → per-step distributions */
        shape[0] = NMZ_ANY_LENGTH;
        if (nmz_net_create(&m->net, 1, shape, 8 + (uint64_t)k) != NMZ_OK) return 0;
        nmz_net_add(m->net, nmz_layer_embedding(V, 6, NMZ_INIT_AUTO));
        if (k == Z_RNN) {
            nmz_net_add(m->net, nmz_layer_rnn(7, NMZ_INIT_AUTO));
            m->carry = 1;
        } else if (k == Z_LSTM) {
            nmz_net_add(m->net, nmz_layer_lstm(5, NMZ_INIT_AUTO, 1.0f));
            nmz_net_add(m->net, nmz_layer_dropout(0.25f));
            nmz_net_add(m->net, nmz_layer_lstm(4, NMZ_INIT_XAVIER, 0.5f));
        } else if (k == Z_GRU) {
            nmz_net_add(m->net, nmz_layer_gru(6, NMZ_INIT_AUTO));
            m->carry = 1;
        } else if (k == Z_GPT) {
            nmz_net_add(m->net, nmz_layer_posenc(NMZ_POS_LEARNED, 16));
            nmz_net_add(m->net, nmz_layer_block(2, 0, 12, NMZ_ACT_GELU, 0.2f));
            nmz_net_add(m->net, nmz_layer_layernorm());
            nmz_net_add(m->net, nmz_layer_tied_output(V));
        } else {
            nmz_net_add(m->net, nmz_layer_posenc(NMZ_POS_SINUSOIDAL, 16));
            nmz_net_add(m->net, nmz_layer_layernorm());
            nmz_net_add(m->net, nmz_layer_attention(2, 3));
            nmz_net_add(m->net, nmz_layer_ffn(10, NMZ_ACT_RELU));
            nmz_net_add(m->net, nmz_layer_dense(V, NMZ_INIT_AUTO));
        }
        if (k != Z_GPT && k != Z_PARTS) nmz_net_add(m->net, nmz_layer_dense(V, NMZ_INIT_AUTO));
        nmz_net_add(m->net, nmz_layer_activation(NMZ_ACT_SOFTMAX, 0));
        nmz_net_set_loss(m->net, NMZ_LOSS_CROSS_ENTROPY);
        break;
    }
    if (nmz_net_build(m->net) != NMZ_OK) return 0;
    for (i = 0; i < nmz_net_layer_count(m->net); i++) { /* no trivially 0 or 1 parameter */
        nmz_layer *l = nmz_net_layer(m->net, i);
        for (j = 0; j < nmz_layer_param_count(l); j++) {
            nmz_tensor *v = nmz_layer_param_value(l, j);
            size_t e;
            for (e = 0; e < v->size; e++) v->data[e] += (float)(0.3 * (2.0 * nmz_rng_uniform(&r) - 1.0));
        }
    }
    if (k >= Z_RNN) {
        int s3[3] = {3, 6, V};
        size_t e;
        shape[0] = 3, shape[1] = 6;
        nmz_tensor_init(&m->x, 2, shape);
        nmz_tensor_init(&m->t, 3, s3);
        for (e = 0; e < m->x.size; e++) m->x.data[e] = (float)nmz_rng_below(&r, (uint32_t)V);
        for (e = 0; e < 18; e++) m->t.data[e * V + nmz_rng_below(&r, (uint32_t)V)] = 1.0f;
    } else {
        const nmz_layer *l0 = nmz_net_layer(m->net, 0);
        const int out = nmz_net_output_size(m->net);
        int s2[2];
        size_t rr;
        shape[0] = 4;
        for (i = 0; i < l0->in_rank; i++) shape[i + 1] = l0->in_shape[i];
        nmz_tensor_init(&m->x, l0->in_rank + 1, shape);
        fill_r(&m->x, &r, -1.0, 2.0);
        s2[0] = 4;
        s2[1] = out;
        nmz_tensor_init(&m->t, 2, s2);
        for (rr = 0; rr < 4; rr++) m->t.data[rr * out + nmz_rng_below(&r, (uint32_t)out)] = 1.0f;
        if (k == Z_DENSE) { /* a fitted normalization */
            nmz_dataset *ds = NULL;
            nmz_dataset_create(&ds, 4, 5, 3, m->x.data, m->t.data);
            nmz_net_fit_normalization(m->net, ds);
            nmz_dataset_free(ds);
        }
    }
    return 1;
}

static void unmake(model *m) {
    nmz_net_free(m->net);
    nmz_tensor_free(&m->x);
    nmz_tensor_free(&m->t);
}

/* one training-mode pass: loss, output, every gradient, the input gradient */
typedef struct result {
    double loss;
    float *v;
    size_t n;
} result;

static void push(result *r, const nmz_tensor *t) {
    if (!t || !t->size) return;
    r->v = realloc(r->v, (r->n + t->size) * sizeof *r->v);
    memcpy(r->v + r->n, t->data, t->size * sizeof(float));
    r->n += t->size;
}

static int run_pass(int k, const char *backend, result *out) {
    model m;
    const nmz_tensor *y;
    int i, j;
    memset(out, 0, sizeof *out);
    if (nmz_backend_select(backend) != NMZ_OK || !make(k, &m)) return 0;
    if (m.carry) { /* a carried state h0 ≠ 0 */
        nmz_net_set_carry(m.net, 1);
        if (nmz_net_forward(m.net, &m.x, NULL) != NMZ_OK) return 0;
    }
    if (nmz_net_forward_mode(m.net, &m.x, 1, &y) != NMZ_OK) return 0;
    push(out, y);
    if (nmz_net_backward(m.net, &m.t, &out->loss) != NMZ_OK) return 0;
    for (i = 0; i < nmz_net_layer_count(m.net); i++)
        for (j = 0; j < nmz_layer_param_count(nmz_net_layer(m.net, i)); j++)
            push(out, nmz_layer_param_grad(nmz_net_layer(m.net, i), j));
    push(out, nmz_net_input_grad(m.net));
    unmake(&m);
    nmz_backend_select("cpu");
    return 1;
}

/* largest difference relative to the largest value of the pass */
static double rel_diff(const result *a, const result *b) {
    double worst = 0, scale = 1e-6;
    size_t i;
    if (a->n != b->n) return 1e9;
    for (i = 0; i < a->n; i++) {
        double d = (double)a->v[i] - b->v[i];
        if (fabs(a->v[i]) > scale) scale = fabs(a->v[i]);
        if (!(fabs(d) <= worst)) worst = fabs(d);
    }
    return worst / scale;
}

/* ----------------------------------------------------------------- tests */

static void test_select(void) {
    nmz_device_info di;
    int i, have_cuda = 0;
    CHECK(nmz_backend_count() >= 2);
    CHECK(strcmp(nmz_backend_name(0), "cpu") == 0);
    for (i = 0; i < nmz_backend_count(); i++) have_cuda |= strcmp(nmz_backend_name(i), "cuda") == 0;
    CHECK(nmz_backend_name(nmz_backend_count()) == NULL);
    CHECK_OK(nmz_backend_info(&di));
    CHECK(strcmp(di.backend, "cpu") == 0 && di.threads >= 1 && di.mem_total == 0);
    CHECK_STATUS(nmz_backend_select("tpu"), NMZ_ERR_ARG);
    CHECK_STATUS(nmz_backend_select(NULL), NMZ_ERR_ARG);
    CHECK_OK(nmz_backend_select("auto"));
    CHECK_OK(nmz_backend_info(&di));
    CHECK(strncmp(di.note, "auto:", 5) == 0);
    test_note("auto -> %s: %s", di.backend, di.note);
    if (!have_cuda) {
        CHECK(strcmp(di.backend, "cpu") == 0);
        CHECK_STATUS(nmz_backend_select("cuda"), NMZ_ERR_UNSUPPORTED);
        CHECK(strstr(nmz_last_error(), "NMZ_CUDA") != NULL);
        CHECK_STATUS(nmz_backend_query("cuda", &di), NMZ_ERR_UNSUPPORTED);
    } else if (strcmp(di.backend, "cpu") == 0) { /* a CUDA build without a usable GPU */
        CHECK_STATUS(nmz_backend_select("cuda"), NMZ_ERR_UNSUPPORTED);
        CHECK(nmz_last_error()[0] != '\0');
    }
    CHECK_OK(nmz_backend_query("cpu-generic", &di));
    CHECK_OK(nmz_backend_select("cpu"));
    CHECK_OK(nmz_backend_info(&di));
    CHECK(strcmp(di.note, "requested") == 0);
}

/* the fused CPU kernels and the generic compositions compute the same math */
static void test_generic_matches_cpu(void) {
    int k;
    for (k = 0; k < Z_COUNT; k++) {
        result a, b;
        double d;
        CHECK_MSG(run_pass(k, "cpu", &a), "%s on cpu", k_zoo[k]);
        CHECK_MSG(run_pass(k, "cpu-generic", &b), "%s on cpu-generic", k_zoo[k]);
        d = rel_diff(&a, &b);
        CHECK_MSG(d < 2e-5 && fabs(a.loss - b.loss) < 1e-5, "%s: max relative difference %.3g, loss %.9g vs %.9g",
                  k_zoo[k], d, a.loss, b.loss);
        test_note("%-28s %6zu numbers, max rel diff %.1e", k_zoo[k], a.n, d);
        free(a.v);
        free(b.v);
    }
    nmz_backend_select("cpu");
}

/* the device path moves data correctly: exactly cpu-generic's bits */
static void test_fake_device_bits(void) {
    int k;
    if (!use_fake()) {
        CHECK_MSG(0, "fake device: %s", nmz_last_error());
        return;
    }
    for (k = 0; k < Z_COUNT; k++) {
        result a, b;
        CHECK_MSG(run_pass(k, "cpu-generic", &a), "%s on cpu-generic", k_zoo[k]);
        CHECK_MSG(run_pass(k, "fake-device", &b), "%s on the fake device: %s", k_zoo[k], nmz_last_error());
        CHECK_MSG(a.n == b.n && memcmp(a.v, b.v, a.n * sizeof(float)) == 0 && a.loss == b.loss,
                  "%s: the device run differs from cpu-generic (rel %.3g)", k_zoo[k], rel_diff(&a, &b));
        free(a.v);
        free(b.v);
    }
    nmz_backend_select("cpu");
}

static void test_fake_device_gradcheck(void) {
    int k;
    if (!use_fake()) return;
    for (k = 0; k < Z_COUNT; k++) {
        model m;
        nmz_gradcheck_result res;
        memset(&res, 0, sizeof res);
        if (!make(k, &m)) {
            CHECK_MSG(0, "%s: %s", k_zoo[k], nmz_last_error());
            continue;
        }
        if (m.carry) {
            nmz_net_set_carry(m.net, 1);
            CHECK_OK(nmz_net_forward(m.net, &m.x, NULL));
        }
        CHECK_OK(nmz_net_gradcheck_mode(m.net, &m.x, &m.t, 1e-3, 2e-4, 3e-2, 1, &res));
        CHECK_MSG(res.failures == 0 && res.checked > 0, "%s: %zu/%zu mismatches; worst layer %d %s[%zu]: %.6g vs %.6g",
                  k_zoo[k], res.failures, res.checked, res.worst_layer, res.worst_param, res.worst_index,
                  res.worst_analytic, res.worst_numeric);
        test_note("%-28s %5zu grads, max |err| %.2e", k_zoo[k], res.checked, res.max_abs_err);
        unmake(&m);
    }
    nmz_backend_select("cpu");
}

/* training with the data on the device: the same bits as cpu-generic,
 * the same file; then the net moves to cpu-generic and continues */
static unsigned long long file_hash(const char *path) {
    FILE *f = fopen(path, "rb");
    unsigned long long h = 14695981039346656037ull;
    int c;
    if (!f) return 0;
    while ((c = fgetc(f)) != EOF) h = (h ^ (unsigned char)c) * 1099511628211ull;
    fclose(f);
    return h;
}

static unsigned long long train_text(const char *b1, const char *b2, int kind, double *loss) {
    nmz_dataset *ds = NULL;
    nmz_text_opts to = nmz_text_defaults();
    nmz_train_opts o = nmz_train_defaults();
    nmz_train_result r;
    nmz_net *net = NULL;
    int shape[1] = {NMZ_ANY_LENGTH}, V;
    unsigned long long h = 0;
    to.items = 1;
    to.seq_len = 12;
    if (nmz_dataset_load_text(&ds, "examples/names_en.txt", &to) != NMZ_OK) return 0;
    nmz_dataset_truncate(ds, 96);
    V = ds->vocab->size;
    nmz_backend_select(b1);
    nmz_net_create(&net, 1, shape, 21);
    nmz_net_set_vocab(net, ds->vocab);
    nmz_net_add(net, nmz_layer_embedding(V, 10, NMZ_INIT_AUTO));
    if (kind == 0) {
        nmz_net_add(net, nmz_layer_lstm(12, NMZ_INIT_AUTO, 1.0f));
        nmz_net_add(net, nmz_layer_dropout(0.1f));
        nmz_net_add(net, nmz_layer_gru(10, NMZ_INIT_AUTO));
        nmz_net_add_dense(net, V, NMZ_ACT_SOFTMAX, NMZ_INIT_AUTO);
        o.carry_state = 1;
        o.shuffle = 0;
        o.clip_norm = 5;
    } else {
        nmz_net_add(net, nmz_layer_posenc(NMZ_POS_LEARNED, 12));
        nmz_net_add(net, nmz_layer_block(2, 0, 0, NMZ_ACT_GELU, 0.1f));
        nmz_net_add(net, nmz_layer_layernorm());
        nmz_net_add(net, nmz_layer_tied_output(V));
        nmz_net_add(net, nmz_layer_activation(NMZ_ACT_SOFTMAX, 0));
        o.optimizer = NMZ_OPT_ADAMW;
        o.weight_decay = 0.1f;
        o.schedule = NMZ_SCHED_WARMUP_COSINE;
        o.warmup_steps = 4;
        o.clip_norm = 1;
        o.val_fraction = 0.25f;
        o.patience = 2;
    }
    nmz_net_set_loss(net, NMZ_LOSS_CROSS_ENTROPY);
    o.optimizer = kind ? o.optimizer : NMZ_OPT_ADAM;
    o.learning_rate = 0.01f;
    o.batch_size = 8;
    o.max_epochs = 2;
    o.target_loss = 0;
    if (nmz_net_train(net, ds, &o, &r) == NMZ_OK) {
        nmz_backend_select(b2); /* the same net continues elsewhere */
        if (nmz_net_train(net, ds, &o, &r) == NMZ_OK && nmz_net_save(net, test_path("backend_train.nmz")) == NMZ_OK) {
            h = file_hash(test_path("backend_train.nmz"));
            *loss = r.loss;
        }
    }
    if (!h) test_note("train on %s/%s: %s", b1, b2, nmz_last_error());
    nmz_net_free(net);
    nmz_dataset_free(ds);
    nmz_backend_select("cpu");
    return h;
}

static void test_fake_device_training(void) {
    int kind;
    if (!use_fake()) return;
    for (kind = 0; kind < 2; kind++) {
        double l1 = 0, l2 = 0, l3 = 0;
        unsigned long long a = train_text("cpu-generic", "cpu-generic", kind, &l1);
        unsigned long long b = train_text("fake-device", "fake-device", kind, &l2);
        unsigned long long c = train_text("fake-device", "cpu-generic", kind, &l3);
        CHECK_MSG(a && a == b, "%s: device training differs from cpu-generic (loss %.9g vs %.9g)",
                  kind ? "transformer" : "lstm+gru", l1, l2);
        CHECK_MSG(a && a == c, "%s: moving from the device to the host mid-way changed the result",
                  kind ? "transformer" : "lstm+gru");
    }
    {
        nmz_device_info di;
        use_fake();
        CHECK_OK(nmz_backend_info(&di));
        CHECK(strcmp(di.backend, "fake-device") == 0);
        test_note("fake device: %zu bytes still held, %zu transfers", di.mem_used, g_fake_moves);
        nmz_backend_select("cpu");
    }
}

/* generation: KV cache = recomputation, and the same text as cpu-generic */
static void test_fake_device_generate(void) {
    char *a[3] = {0}, *b[3] = {0}, *c[3] = {0};
    nmz_net *net = NULL;
    nmz_gen_opts g = nmz_gen_defaults();
    int i;
    if (!use_fake()) return;
    nmz_backend_select("cpu");
    CHECK_OK(nmz_net_load(&net, "tests/data/v3_phase4_names_en.nmz"));
    if (!net) return;
    g.length = 30;
    g.seed = 3;
    g.temperature = 0.9f;
    CHECK_OK(nmz_backend_select("cpu-generic"));
    CHECK_OK(nmz_net_generate_batch(net, &g, 3, a, NULL));
    use_fake();
    CHECK_OK(nmz_net_generate_batch(net, &g, 3, b, NULL));
    for (i = 0; i < 3; i++) CHECK_MSG(a[i] && b[i] && strcmp(a[i], b[i]) == 0, "row %d: '%s' vs '%s'", i, a[i], b[i]);
    nmz_net_free(net);
    net = NULL;
    for (i = 0; i < 3; i++) free(a[i]), free(b[i]), a[i] = b[i] = NULL;
    { /* a small Transformer: the cache, a sliding window, rows that finish */
        int shape[1] = {NMZ_ANY_LENGTH}, V = 12, j;
        uint32_t cps[12];
        uint64_t counts[12];
        nmz_vocab *v = NULL;
        for (j = 0; j < V; j++) cps[j] = j ? (uint32_t)('a' + j) : '\n', counts[j] = (uint64_t)(j + 1);
        nmz_backend_select("cpu");
        nmz_vocab_create(&v, V, cps, counts, 1);
        nmz_net_create(&net, 1, shape, 4);
        nmz_net_set_vocab(net, v);
        nmz_net_add(net, nmz_layer_embedding(V, 8, NMZ_INIT_AUTO));
        nmz_net_add(net, nmz_layer_posenc(NMZ_POS_LEARNED, 9));
        nmz_net_add(net, nmz_layer_block(2, 0, 0, NMZ_ACT_GELU, 0));
        nmz_net_add(net, nmz_layer_tied_output(V));
        nmz_net_add(net, nmz_layer_activation(NMZ_ACT_SOFTMAX, 0));
        nmz_vocab_free(v);
        g.length = 25;
        g.stop_at_end = 1;
        g.prime = "bcd";
        use_fake();
        CHECK_OK(nmz_net_generate_batch(net, &g, 3, a, NULL));
        g.no_cache = 1;
        CHECK_OK(nmz_net_generate_batch(net, &g, 3, b, NULL));
        g.no_cache = 0;
        CHECK_OK(nmz_backend_select("cpu-generic"));
        CHECK_OK(nmz_net_generate_batch(net, &g, 3, c, NULL));
        for (i = 0; i < 3; i++) {
            CHECK_MSG(a[i] && b[i] && strcmp(a[i], b[i]) == 0, "cache vs no cache, row %d: '%s' vs '%s'", i, a[i], b[i]);
            CHECK_MSG(a[i] && c[i] && strcmp(a[i], c[i]) == 0, "device vs host, row %d: '%s' vs '%s'", i, a[i], c[i]);
            free(a[i]), free(b[i]), free(c[i]);
        }
        nmz_net_free(net);
    }
    nmz_backend_select("cpu");
}

/* traced numbers on the device = traced numbers on the host */
static void trace_hash(const nmz_trace_event *ev, void *user) {
    unsigned long long *h = user;
    int i;
    for (i = 0; i < ev->n_items; i++) {
        const nmz_tensor *t = ev->items[i].t;
        size_t e;
        for (e = 0; t && e < t->size; e++) {
            unsigned u;
            memcpy(&u, &t->data[e], sizeof u);
            *h = (*h ^ u) * 1099511628211ull;
        }
    }
}

static void test_fake_device_trace(void) {
    int k;
    if (!use_fake()) return;
    for (k = 0; k < Z_COUNT; k++) {
        unsigned long long h[2] = {14695981039346656037ull, 14695981039346656037ull};
        int pass;
        for (pass = 0; pass < 2; pass++) {
            model m;
            nmz_train_opts o = nmz_train_defaults();
            nmz_backend_select(pass ? "fake-device" : "cpu-generic");
            if (!make(k, &m)) break;
            o.optimizer = NMZ_OPT_ADAM;
            nmz_net_set_trace(m.net, trace_hash, &h[pass]);
            CHECK_OK(nmz_net_forward_mode(m.net, &m.x, 1, NULL));
            CHECK_OK(nmz_net_backward(m.net, &m.t, NULL));
            CHECK_OK(nmz_net_clip_gradients(m.net, 0.5, NULL, NULL));
            CHECK_OK(nmz_net_optimizer_preview(m.net, &o, 0.01f));
            CHECK_OK(nmz_net_optimizer_step(m.net, &o, 0.01f));
            CHECK_OK(nmz_net_forward(m.net, &m.x, NULL));
            unmake(&m);
        }
        CHECK_MSG(h[0] == h[1], "%s: traced values differ between the host and the device", k_zoo[k]);
    }
    nmz_backend_select("cpu");
}

/* a layer written for v2.0 (it reads ->data, no NMZ_LAYER_BACKEND):
 * y = x · s, one parameter s per feature */
typedef struct scale_layer {
    int n;
    nmz_tensor s, ds;
} scale_layer;

static nmz_status sc_setup(nmz_layer *l, int in_rank, const int *in_shape, int *out_rank, int *out_shape) {
    scale_layer *c = l->self;
    int i;
    c->n = in_shape[in_rank - 1];
    *out_rank = in_rank;
    for (i = 0; i < in_rank; i++) out_shape[i] = in_shape[i];
    if (nmz_tensor_init(&c->s, 1, &c->n) != NMZ_OK) return NMZ_ERR_NOMEM;
    return nmz_tensor_init(&c->ds, 1, &c->n);
}
static nmz_status sc_init(nmz_layer *l, nmz_rng *rng, const nmz_layer *next) {
    scale_layer *c = l->self;
    int i;
    (void)next;
    for (i = 0; i < c->n; i++) c->s.data[i] = (float)(0.5 + nmz_rng_uniform(rng));
    return NMZ_OK;
}
static nmz_status sc_forward(nmz_layer *l, const nmz_tensor *x, nmz_tensor *y, const nmz_ctx *ctx) {
    scale_layer *c = l->self;
    size_t e;
    (void)ctx;
    for (e = 0; e < x->size; e++) y->data[e] = x->data[e] * c->s.data[e % (size_t)c->n];
    return NMZ_OK;
}
static nmz_status sc_backward(nmz_layer *l, const nmz_tensor *x, const nmz_tensor *y, const nmz_tensor *dy,
                              nmz_tensor *dx, const nmz_ctx *ctx) {
    scale_layer *c = l->self;
    size_t e;
    (void)y;
    (void)ctx;
    for (e = 0; e < x->size; e++) {
        c->ds.data[e % (size_t)c->n] += dy->data[e] * x->data[e];
        if (dx) dx->data[e] = dy->data[e] * c->s.data[e % (size_t)c->n];
    }
    return NMZ_OK;
}
static int sc_count(const nmz_layer *l) {
    (void)l;
    return 1;
}
static nmz_param sc_param(nmz_layer *l, int i) {
    scale_layer *c = l->self;
    nmz_param p = {"s", &c->s, &c->ds};
    (void)i;
    return p;
}
static nmz_status sc_save(const nmz_layer *l, nmz_writer *w) {
    (void)l;
    (void)w;
    return NMZ_OK;
}
static void sc_describe(const nmz_layer *l, char *buf, size_t n) {
    (void)l;
    snprintf(buf, n, "scale");
}
static void sc_destroy(nmz_layer *l) {
    scale_layer *c = l->self;
    nmz_tensor_free(&c->s);
    nmz_tensor_free(&c->ds);
    free(c);
}
static const nmz_layer_ops k_scale_ops = {"scale", sc_setup, sc_init, sc_forward, sc_backward, sc_count, sc_param,
                                          sc_save, NULL, sc_describe, sc_destroy, NULL, 0, NULL, NULL};

static void test_host_layer_on_device(void) {
    int pass;
    double loss[2] = {0, 0};
    float w[2] = {0, 0};
    if (!use_fake()) return;
    for (pass = 0; pass < 2; pass++) {
        nmz_net *net = NULL;
        nmz_dataset *ds = NULL;
        nmz_train_opts o = nmz_train_defaults();
        nmz_train_result r;
        int shape[1] = {2};
        static const float x[8] = {0, 0, 0, 1, 1, 0, 1, 1}, y[4] = {0, 1, 1, 0};
        nmz_layer *l = nmz_layer_alloc(&k_scale_ops, calloc(1, sizeof(scale_layer)));
        nmz_backend_select(pass ? "fake-device" : "cpu-generic");
        CHECK_OK(nmz_net_create(&net, 1, shape, 3));
        CHECK_OK(nmz_net_add_dense(net, 4, NMZ_ACT_TANH, NMZ_INIT_AUTO));
        CHECK_OK(nmz_net_add(net, l));
        CHECK_OK(nmz_net_add_dense(net, 1, NMZ_ACT_SIGMOID, NMZ_INIT_AUTO));
        CHECK_OK(nmz_dataset_create(&ds, 4, 2, 1, x, y));
        o.max_epochs = 50;
        o.learning_rate = 0.5f;
        o.batch_size = 2;
        CHECK_OK(nmz_net_train(net, ds, &o, &r));
        loss[pass] = r.loss;
        w[pass] = nmz_layer_param_value(nmz_net_layer(net, 2), 0)->data[3];
        nmz_net_free(net);
        nmz_dataset_free(ds);
    }
    CHECK_MSG(loss[0] == loss[1] && w[0] == w[1], "host layer: %.9g vs %.9g", loss[0], loss[1]);
    nmz_backend_select("cpu");
}

/* out of device memory: a clean error that says what to do, and the
 * device still works afterwards */
static void test_fake_device_oom(void) {
    nmz_net *net = NULL;
    nmz_dataset *ds = NULL;
    nmz_train_opts o = nmz_train_defaults();
    nmz_train_result r;
    int shape[1] = {64}, i;
    float *x = malloc(sizeof(float) * 64 * 512), *y = calloc(512 * 4, sizeof(float));
    nmz_status s;
    if (!use_fake() || !x || !y) {
        free(x), free(y);
        return;
    }
    for (i = 0; i < 64 * 512; i++) x[i] = (float)(i % 7) - 3.0f;
    for (i = 0; i < 512; i++) y[i * 4 + i % 4] = 1.0f;
    CHECK_OK(nmz_dataset_create(&ds, 512, 64, 4, x, y));
    CHECK_OK(nmz_net_create(&net, 1, shape, 2));
    CHECK_OK(nmz_net_add_dense(net, 512, NMZ_ACT_RELU, NMZ_INIT_AUTO));
    CHECK_OK(nmz_net_add_dense(net, 4, NMZ_ACT_SOFTMAX, NMZ_INIT_AUTO));
    CHECK_OK(nmz_net_set_loss(net, NMZ_LOSS_CROSS_ENTROPY));
    o.max_epochs = 1;
    o.batch_size = 512;
    o.optimizer = NMZ_OPT_ADAM;
    o.learning_rate = 0.001f;
    g_fake_cap = g_fake_used + 3000000; /* small batches fit, a batch of 512 does not */
    s = nmz_net_train(net, ds, &o, &r);
    CHECK_STATUS(s, NMZ_ERR_NOMEM);
    CHECK(strstr(nmz_last_error(), "batch") != NULL);
    test_note("out of memory: %s", nmz_last_error());
    o.batch_size = 16; /* smaller batches fit */
    CHECK_OK(nmz_net_train(net, ds, &o, &r));
    g_fake_cap = 0;
    nmz_net_free(net);
    nmz_dataset_free(ds);
    free(x);
    free(y);
    nmz_backend_select("cpu");
}

const test_case backend_tests[] = {{"backend/select_and_info", test_select},
                                   {"backend/generic_matches_cpu", test_generic_matches_cpu},
                                   {"backend/fake_device_bits", test_fake_device_bits},
                                   {"backend/fake_device_gradcheck", test_fake_device_gradcheck},
                                   {"backend/fake_device_training", test_fake_device_training},
                                   {"backend/fake_device_generate", test_fake_device_generate},
                                   {"backend/fake_device_trace", test_fake_device_trace},
                                   {"backend/host_layer_on_device", test_host_layer_on_device},
                                   {"backend/fake_device_oom", test_fake_device_oom},
                                   {NULL, NULL}};
