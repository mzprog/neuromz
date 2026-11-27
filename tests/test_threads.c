/* test_threads.c — several nets used from several threads at once
 * (DESIGN §11.8, decision 158):
 *   - four model types trained concurrently give the same files, byte for
 *     byte, as when they are trained one after another;
 *   - a thread's own backend selection does not leak into other threads;
 *   - the last-error message is per thread;
 *   - registry calls from many threads at once;
 *   - an on_batch callback stops training in the middle of an epoch, after
 *     exactly the batches it allowed (the server's cancel). */
#include "test.h"

#include <stdlib.h>
#include <string.h>

enum { M_DENSE, M_CNN, M_LSTM, M_GPT, M_COUNT };
static const char *const k_names[M_COUNT] = {"dense", "cnn", "lstm+gru", "transformer"};

typedef struct job {
    int kind;
    const char *backend; /* this thread's own selection, or NULL */
    char path[600];
    nmz_status status;
    char err[300];
    char seen_backend[16]; /* nmz_backend_info inside the thread */
    char seen_note[256];
} job;

/* synthetic tabular data: 4 inputs, 3 classes; or 1x8x8 "images" */
static nmz_dataset *tabular(int images) {
    int n = images ? 96 : 240, n_in = images ? 64 : 4, i, j;
    float *x = malloc((size_t)n * n_in * sizeof *x), *y = calloc((size_t)n * 3, sizeof *y);
    nmz_dataset *ds = NULL;
    nmz_rng r;
    nmz_rng_seed(&r, images ? 11u : 5u, 1u);
    if (!x || !y) {
        free(x);
        free(y);
        return NULL;
    }
    for (i = 0; i < n; i++) {
        double score[3] = {0, 0, 0};
        int best = 0;
        for (j = 0; j < n_in; j++) {
            float v = (float)(2.0 * nmz_rng_uniform(&r) - 1.0);
            x[(size_t)i * n_in + j] = v;
            score[images ? (j % 8 < 3 ? 0 : j % 8 < 6 ? 1 : 2) : j % 3] += v;
        }
        for (j = 1; j < 3; j++)
            if (score[j] > score[best]) best = j;
        y[(size_t)i * 3 + best] = 1.0f;
    }
    nmz_dataset_create(&ds, n, n_in, 3, x, y);
    free(x);
    free(y);
    return ds;
}

static nmz_dataset *names_data(void) {
    nmz_dataset *ds = NULL;
    nmz_text_opts to = nmz_text_defaults();
    to.items = 1;
    to.seq_len = 16;
    if (nmz_dataset_load_text(&ds, "examples/names_en.txt", &to) != NMZ_OK) return NULL;
    nmz_dataset_truncate(ds, 300);
    return ds;
}

static nmz_dataset *data_for(int kind) {
    return kind == M_DENSE ? tabular(0) : kind == M_CNN ? tabular(1) : names_data();
}

static nmz_net *net_for(int kind, const nmz_dataset *ds) {
    nmz_net *net = NULL;
    if (kind == M_DENSE) {
        int shape[1] = {4};
        nmz_net_create(&net, 1, shape, 31);
        nmz_net_add_dense(net, 8, NMZ_ACT_TANH, NMZ_INIT_AUTO);
        nmz_net_add_dense(net, 3, NMZ_ACT_SOFTMAX, NMZ_INIT_AUTO);
    } else if (kind == M_CNN) {
        int shape[3] = {1, 8, 8};
        nmz_net_create(&net, 3, shape, 32);
        nmz_net_add(net, nmz_layer_conv2d(4, 3, 3, 1, 0, NMZ_INIT_AUTO));
        nmz_net_add(net, nmz_layer_activation(NMZ_ACT_RELU, 0));
        nmz_net_add(net, nmz_layer_maxpool2d(2, 0));
        nmz_net_add(net, nmz_layer_flatten());
        nmz_net_add(net, nmz_layer_dropout(0.2f));
        nmz_net_add_dense(net, 3, NMZ_ACT_SOFTMAX, NMZ_INIT_AUTO);
    } else {
        int shape[1] = {NMZ_ANY_LENGTH}, V = ds->vocab->size;
        nmz_net_create(&net, 1, shape, 33);
        nmz_net_set_vocab(net, ds->vocab);
        nmz_net_add(net, nmz_layer_embedding(V, 12, NMZ_INIT_AUTO));
        if (kind == M_LSTM) {
            nmz_net_add(net, nmz_layer_lstm(16, NMZ_INIT_AUTO, 1.0f));
            nmz_net_add(net, nmz_layer_dropout(0.1f));
            nmz_net_add(net, nmz_layer_gru(12, NMZ_INIT_AUTO));
            nmz_net_add_dense(net, V, NMZ_ACT_SOFTMAX, NMZ_INIT_AUTO);
        } else {
            nmz_net_add(net, nmz_layer_posenc(NMZ_POS_LEARNED, 16));
            nmz_net_add(net, nmz_layer_block(3, 0, 0, NMZ_ACT_GELU, 0.1f));
            nmz_net_add(net, nmz_layer_layernorm());
            nmz_net_add(net, nmz_layer_tied_output(V));
            nmz_net_add(net, nmz_layer_activation(NMZ_ACT_SOFTMAX, 0));
        }
    }
    if (net) nmz_net_set_loss(net, NMZ_LOSS_CROSS_ENTROPY);
    return net;
}

static nmz_train_opts opts_for(int kind) {
    nmz_train_opts o = nmz_train_defaults();
    o.max_epochs = kind >= M_LSTM ? 2 : 4;
    o.target_loss = 0;
    o.batch_size = 16;
    o.optimizer = kind == M_CNN ? NMZ_OPT_SGD : kind == M_GPT ? NMZ_OPT_ADAMW : NMZ_OPT_ADAM;
    o.learning_rate = kind == M_CNN ? 0.05f : 0.01f;
    o.momentum = 0.9f;
    if (kind >= M_LSTM) o.clip_norm = 1.0f, o.val_fraction = 0.2f;
    if (kind == M_GPT) o.schedule = NMZ_SCHED_WARMUP_COSINE, o.warmup_steps = 5;
    return o;
}

static void train_one(void *arg) {
    job *j = (job *)arg;
    nmz_dataset *ds;
    nmz_net *net;
    nmz_train_opts o = opts_for(j->kind);
    nmz_train_result r;
    nmz_device_info di;
    if (j->backend && (j->status = nmz_backend_select_thread(j->backend)) != NMZ_OK) {
        snprintf(j->err, sizeof j->err, "%s", nmz_last_error());
        return;
    }
    memset(&di, 0, sizeof di);
    nmz_backend_info(&di);
    snprintf(j->seen_backend, sizeof j->seen_backend, "%s", di.backend);
    snprintf(j->seen_note, sizeof j->seen_note, "%s", di.note);
    ds = data_for(j->kind);
    net = ds ? net_for(j->kind, ds) : NULL;
    j->status = net ? nmz_net_train(net, ds, &o, &r) : NMZ_ERR_NOMEM;
    if (j->status == NMZ_OK) j->status = nmz_net_save(net, j->path);
    if (j->status != NMZ_OK) snprintf(j->err, sizeof j->err, "%s", nmz_last_error());
    nmz_net_free(net);
    nmz_dataset_free(ds);
    if (j->backend) nmz_backend_select_thread(NULL);
}

static int same_file(const char *a, const char *b) {
    FILE *fa = fopen(a, "rb"), *fb = fopen(b, "rb");
    int ca, cb, same = fa && fb;
    while (same) {
        ca = fgetc(fa);
        cb = fgetc(fb);
        if (ca != cb) same = 0;
        if (ca == EOF) break;
    }
    if (fa) fclose(fa);
    if (fb) fclose(fb);
    return same;
}

static void test_concurrent_training(void) {
    enum { N = M_COUNT + 2 }; /* every kind, and two of them twice */
    job seq[N], par[N];
    void *args[N];
    int i;
    memset(seq, 0, sizeof seq);
    memset(par, 0, sizeof par);
    for (i = 0; i < N; i++) {
        char name[64];
        seq[i].kind = par[i].kind = i < M_COUNT ? i : i == M_COUNT ? M_DENSE : M_GPT;
        snprintf(name, sizeof name, "threads_seq%d.nmz", i);
        snprintf(seq[i].path, sizeof seq[i].path, "%s", test_path(name));
        snprintf(name, sizeof name, "threads_par%d.nmz", i);
        snprintf(par[i].path, sizeof par[i].path, "%s", test_path(name));
        args[i] = &par[i];
    }
    for (i = 0; i < N; i++) train_one(&seq[i]);
    CHECK(test_run_threads(N, train_one, args));
    for (i = 0; i < N; i++) {
        CHECK_MSG(seq[i].status == NMZ_OK, "%s, alone: %s", k_names[seq[i].kind], seq[i].err);
        CHECK_MSG(par[i].status == NMZ_OK, "%s, in a thread: %s", k_names[par[i].kind], par[i].err);
        CHECK_MSG(same_file(seq[i].path, par[i].path), "%s: trained in a thread, the model file differs",
                  k_names[par[i].kind]);
    }
    test_note("%d nets trained in %d threads at once: every file identical to the sequential run", N, N);
}

static void test_thread_selection(void) {
    job alone, other;
    void *args[1];
    nmz_device_info di;
    memset(&alone, 0, sizeof alone);
    memset(&other, 0, sizeof other);
    CHECK_OK(nmz_backend_select("cpu"));
    other.kind = alone.kind = M_LSTM;
    other.backend = "cpu-generic";
    snprintf(other.path, sizeof other.path, "%s", test_path("threads_generic_a.nmz"));
    snprintf(alone.path, sizeof alone.path, "%s", test_path("threads_generic_b.nmz"));
    args[0] = &other;
    CHECK(test_run_threads(1, train_one, args));
    CHECK_MSG(other.status == NMZ_OK, "%s", other.err);
    CHECK(strcmp(other.seen_backend, "cpu-generic") == 0);
    CHECK(strstr(other.seen_note, "this thread") != NULL);
    memset(&di, 0, sizeof di);
    CHECK_OK(nmz_backend_info(&di));
    CHECK_MSG(strcmp(di.backend, "cpu") == 0, "the main thread sees %s", di.backend);
    /* the same choice made process-wide gives the same model */
    CHECK_OK(nmz_backend_select("cpu-generic"));
    train_one(&alone);
    CHECK_OK(nmz_backend_select("cpu"));
    CHECK_MSG(alone.status == NMZ_OK, "%s", alone.err);
    CHECK(same_file(alone.path, other.path));
    /* a thread selection wins over the process-wide one, NULL gives it back */
    CHECK_OK(nmz_backend_select_thread("cpu-generic"));
    CHECK_OK(nmz_backend_info(&di));
    CHECK(strcmp(di.backend, "cpu-generic") == 0);
    CHECK_OK(nmz_backend_select_thread(NULL));
    CHECK_OK(nmz_backend_info(&di));
    CHECK(strcmp(di.backend, "cpu") == 0);
    CHECK_STATUS(nmz_backend_select_thread("tpu"), NMZ_ERR_ARG);
}

typedef struct err_job {
    char msg[300];
} err_job;

static void make_error(void *arg) {
    err_job *e = (err_job *)arg;
    nmz_net *net = NULL;
    int shape[1] = {-5};
    nmz_net_create(&net, 1, shape, 1); /* a bad shape */
    snprintf(e->msg, sizeof e->msg, "%s", nmz_last_error());
}

static void test_errors_per_thread(void) {
    err_job e;
    void *args[1];
    char mine[300];
    memset(&e, 0, sizeof e);
    CHECK_STATUS(nmz_backend_select("no-such-backend"), NMZ_ERR_ARG);
    snprintf(mine, sizeof mine, "%s", nmz_last_error());
    args[0] = &e;
    CHECK(test_run_threads(1, make_error, args));
    CHECK(e.msg[0] != '\0' && strcmp(e.msg, mine) != 0);
    CHECK_MSG(strcmp(nmz_last_error(), mine) == 0, "the thread's error replaced this one: %s", nmz_last_error());
}

static void registry_storm(void *arg) {
    int *fails = (int *)arg, i;
    for (i = 0; i < 200; i++) {
        nmz_device_info di;
        if (nmz_backend_select_thread(i % 2 ? "cpu-generic" : "auto") != NMZ_OK) (*fails)++;
        if (nmz_backend_info(&di) != NMZ_OK) (*fails)++;
        if (nmz_backend_query("cpu", &di) != NMZ_OK) (*fails)++;
        if (nmz_backend_count() < 2 || !nmz_backend_name(0)) (*fails)++;
        if (nmz_backend_select_thread(NULL) != NMZ_OK) (*fails)++;
    }
}

static void test_registry_from_threads(void) {
    enum { N = 8 };
    int fails[N], i, total = 0;
    void *args[N];
    for (i = 0; i < N; i++) {
        fails[i] = 0;
        args[i] = &fails[i];
    }
    CHECK(test_run_threads(N, registry_storm, args));
    for (i = 0; i < N; i++) total += fails[i];
    CHECK_MSG(total == 0, "%d registry calls failed", total);
}

typedef struct stopper {
    int calls, stop_at;
} stopper;

static int stop_batch(const nmz_batch_info *bi, void *user) {
    stopper *s = (stopper *)user;
    s->calls++;
    return bi->batch >= s->stop_at;
}

static void test_stop_mid_epoch(void) {
    nmz_dataset *ds = tabular(0), *first = tabular(0);
    nmz_net *a, *b;
    nmz_train_opts o = opts_for(M_DENSE);
    nmz_train_result ra, rb;
    stopper st = {0, 3};
    int i, j, same = 1;
    if (!ds || !first) {
        CHECK(0);
        return;
    }
    a = net_for(M_DENSE, ds);
    b = net_for(M_DENSE, ds);
    o.shuffle = 0;
    o.on_batch = stop_batch;
    o.user = &st;
    CHECK_OK(nmz_net_train(a, ds, &o, &ra));
    CHECK(ra.stopped == 1 && st.calls == 3);
    /* the same three batches as a whole (one-epoch) run */
    nmz_dataset_truncate(first, 3 * o.batch_size);
    o.on_batch = NULL;
    o.max_epochs = 1;
    CHECK_OK(nmz_net_train(b, first, &o, &rb));
    for (i = 0; i < nmz_net_layer_count(a); i++)
        for (j = 0; j < nmz_layer_param_count(nmz_net_layer(a, i)); j++) {
            const nmz_tensor *pa = nmz_layer_param_value(nmz_net_layer(a, i), j);
            const nmz_tensor *pb = nmz_layer_param_value(nmz_net_layer(b, i), j);
            if (pa->size != pb->size || memcmp(pa->data, pb->data, pa->size * sizeof(float))) same = 0;
        }
    CHECK_MSG(same, "stopped after 3 batches, the weights differ from 3 batches run as an epoch");
    nmz_net_free(a);
    nmz_net_free(b);
    nmz_dataset_free(ds);
    nmz_dataset_free(first);
}

const test_case thread_tests[] = {{"threads/concurrent_training", test_concurrent_training},
                                  {"threads/per_thread_selection", test_thread_selection},
                                  {"threads/errors_per_thread", test_errors_per_thread},
                                  {"threads/registry_from_threads", test_registry_from_threads},
                                  {"threads/stop_mid_epoch", test_stop_mid_epoch},
                                  {NULL, NULL}};
