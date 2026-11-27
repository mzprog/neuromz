/* bench_device.c â€” training time per epoch on a chosen backend, for the
 * models of the README's device table (README "Performance", DESIGN آ§11):
 *
 *   mnist-cnn      the CLI preset: conv 8 â†’ pool â†’ conv 16 â†’ pool â†’ 64 â†’ 10
 *                  (27.6 k weights), Adam, batch 32, 60 000 images
 *   alice-lstm     the char-lstm preset: embed 32 â†’ lstm 64 â†’ softmax
 *                  (32 k), 16 streams of 64 characters, carried state
 *   alice-gpt      the char-gpt preset: 2 blocks of 32, context 128 (32 k)
 *   alice-gpt-213k 4 blocks of 64, 4 heads, dropout 0.2, context 128 (213 k)
 *   alice-gpt-big  8 blocks of 512, 8 heads, context 256, batch 16 (25 M):
 *                  sized for a 6 GB GPU (about half of it)
 *
 * Usage: bench_device MODEL [--device cpu|cuda|auto] [--epochs N]
 *                           [--batches N] [--batch N] [--data DIR]
 * Two times are reported: a whole epoch as the CLI shows it (the updates
 * and the evaluation of the whole training and validation sets), and the
 * updates alone: the time from the first to the last update of epoch 1,
 * scaled to its number of batches. --batches N stops after N updates (for
 * runs that would take hours on one CPU core): then only the second is
 * measured. The last line is machine-readable:
 *   RESULT model device threads params epoch_s updates_s measured_batches
 *          batches_per_epoch train_loss val_loss vram_peak_MiB vram_total_MiB
 * Exit 77 = the data is missing (cmake --build build --target data data-text). */
#include "neuromz.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}

#define TRY(expr)                                                                                                    \
    do {                                                                                                             \
        if ((expr) != NMZ_OK) {                                                                                      \
            fprintf(stderr, "%s failed: %s\n", #expr, nmz_last_error());                                             \
            exit(1);                                                                                                 \
        }                                                                                                            \
    } while (0)

typedef struct timing {
    double t0, t_epoch, t_first, t_last;
    int batches, max_batches, n_batches, epochs;
    double loss, val_loss;
} timing;

static int on_batch(const nmz_batch_info *b, void *u) {
    timing *t = u;
    t->n_batches = b->n_batches;
    if (b->epoch == 1) { /* the first epoch's updates; the first one warms up (allocations, the device) */
        if (b->batch == 1) t->t_first = now();
        t->t_last = now();
        t->batches = b->batch;
    }
    return t->max_batches > 0 && b->batch >= t->max_batches;
}

static int on_epoch(const nmz_epoch_info *e, void *u) {
    timing *t = u;
    double dt = now() - t->t0;
    t->t_epoch = dt;
    t->epochs = e->epoch;
    t->loss = e->loss;
    t->val_loss = e->val_loss;
    printf("  epoch %d: %.2f s, train loss %.4f, val loss %.4f\n", e->epoch, dt, e->loss, e->val_loss);
    fflush(stdout);
    t->t0 = now();
    return 0;
}

static int exists(const char *p) {
    FILE *f = fopen(p, "rb");
    if (f) fclose(f);
    return f != NULL;
}

int main(int argc, char **argv) {
    const char *model = argc > 1 ? argv[1] : "", *device = "cpu", *dir = "data";
    char p1[512], p2[512];
    nmz_net *net = NULL;
    nmz_dataset *ds = NULL;
    nmz_train_opts o = nmz_train_defaults();
    nmz_train_result res;
    timing tm;
    int epochs = 1, i, threads = nmz_threads(), seq = 0, batch = 0;
    double t_total;
    memset(&tm, 0, sizeof tm);
    for (i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--device") && i + 1 < argc) device = argv[++i];
        else if (!strcmp(argv[i], "--epochs") && i + 1 < argc) epochs = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--batches") && i + 1 < argc) tm.max_batches = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--batch") && i + 1 < argc) batch = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--data") && i + 1 < argc) dir = argv[++i];
    }
#ifdef NMZ_HAVE_BACKENDS
    {
        nmz_device_info di;
        TRY(nmz_backend_select(device));
        nmz_backend_info(&di);
        printf("device: %s â€” %s (%s)\n", di.backend, di.device, di.note);
        if (di.mem_total) printf("        compute %d.%d, %.0f MiB\n", di.cc_major, di.cc_minor, di.mem_total / 1048576.0);
    }
#else
    if (strcmp(device, "cpu")) {
        fprintf(stderr, "this library has no device backends\n");
        return 2;
    }
#endif
    if (!strcmp(model, "mnist-cnn")) {
        int shape[3] = {1, 28, 28};
        snprintf(p1, sizeof p1, "%s/train-images-idx3-ubyte", dir);
        snprintf(p2, sizeof p2, "%s/train-labels-idx1-ubyte", dir);
        if (!exists(p1) || !exists(p2)) return printf("MNIST not found in %s/: skipped\n", dir), 77;
        TRY(nmz_dataset_load_idx(&ds, p1, p2, 0));
        TRY(nmz_net_create(&net, 3, shape, 42));
        TRY(nmz_net_add(net, nmz_layer_conv2d(8, 3, 3, 1, 0, NMZ_INIT_AUTO)));
        TRY(nmz_net_add(net, nmz_layer_activation(NMZ_ACT_RELU, 0)));
        TRY(nmz_net_add(net, nmz_layer_maxpool2d(2, 0)));
        TRY(nmz_net_add(net, nmz_layer_conv2d(16, 3, 3, 1, 0, NMZ_INIT_AUTO)));
        TRY(nmz_net_add(net, nmz_layer_activation(NMZ_ACT_RELU, 0)));
        TRY(nmz_net_add(net, nmz_layer_maxpool2d(2, 0)));
        TRY(nmz_net_add(net, nmz_layer_flatten()));
        TRY(nmz_net_add_dense(net, 64, NMZ_ACT_RELU, NMZ_INIT_AUTO));
        TRY(nmz_net_add_dense(net, 10, NMZ_ACT_SOFTMAX, NMZ_INIT_AUTO));
        TRY(nmz_net_set_loss(net, NMZ_LOSS_CROSS_ENTROPY));
        TRY(nmz_net_set_normalization(net, NMZ_NORM_ZSCORE));
        o.optimizer = NMZ_OPT_ADAM;
        o.learning_rate = 0.001f;
        o.batch_size = 32;
        o.schedule = NMZ_SCHED_COSINE;
    } else if (!strncmp(model, "alice-", 6)) {
        nmz_text_opts to = nmz_text_defaults();
        int shape[1] = {NMZ_ANY_LENGTH}, V, D = 32, blocks = 2, heads = 4, ctx = 128, ff = 0;
        float drop = 0.0f;
        snprintf(p1, sizeof p1, "%s/alice.txt", dir);
        if (!exists(p1)) return printf("%s not found: skipped (--target data-text)\n", p1), 77;
        seq = 1;
        if (!strcmp(model, "alice-gpt-213k")) D = 64, blocks = 4, drop = 0.2f;
        if (!strcmp(model, "alice-gpt-big")) D = 512, blocks = 8, heads = 8, ctx = 256, drop = 0.1f;
        to.seq_len = !strcmp(model, "alice-lstm") ? 64 : ctx;
        TRY(nmz_dataset_load_text(&ds, p1, &to));
        V = ds->vocab->size;
        TRY(nmz_net_create(&net, 1, shape, 42));
        TRY(nmz_net_set_vocab(net, ds->vocab));
        TRY(nmz_net_add(net, nmz_layer_embedding(V, D, NMZ_INIT_AUTO)));
        if (!strcmp(model, "alice-lstm")) {
            TRY(nmz_net_add(net, nmz_layer_lstm(64, NMZ_INIT_AUTO, NMZ_LSTM_FORGET_BIAS)));
            TRY(nmz_net_add_dense(net, V, NMZ_ACT_SOFTMAX, NMZ_INIT_AUTO));
            o.optimizer = NMZ_OPT_ADAM;
            o.learning_rate = 0.01f;
            o.batch_size = 16;
            o.carry_state = 1;
            o.shuffle = 0;
            o.clip_norm = 5.0f;
        } else if (!strncmp(model, "alice-gpt", 9)) {
            int b;
            TRY(nmz_net_add(net, nmz_layer_posenc(NMZ_POS_LEARNED, ctx)));
            for (b = 0; b < blocks; b++) TRY(nmz_net_add(net, nmz_layer_block(heads, 0, ff, NMZ_ACT_GELU, drop)));
            TRY(nmz_net_add(net, nmz_layer_layernorm()));
            TRY(nmz_net_add(net, nmz_layer_tied_output(V)));
            TRY(nmz_net_add(net, nmz_layer_activation(NMZ_ACT_SOFTMAX, 0)));
            o.optimizer = NMZ_OPT_ADAMW;
            o.weight_decay = 0.1f;
            o.learning_rate = D >= 512 ? 0.0005f : D >= 64 ? 0.005f : 0.01f;
            o.batch_size = D >= 512 ? 16 : 8;
            o.schedule = NMZ_SCHED_WARMUP_COSINE;
            o.min_lr = o.learning_rate / 10;
            o.clip_norm = 1.0f;
        } else {
            fprintf(stderr, "unknown model '%s'\n", model);
            return 2;
        }
        TRY(nmz_net_set_loss(net, NMZ_LOSS_CROSS_ENTROPY));
        o.val_fraction = 0.1f;
    } else {
        fprintf(stderr, "usage: bench_device mnist-cnn|alice-lstm|alice-gpt|alice-gpt-213k|alice-gpt-big "
                        "[--device cpu|cuda|auto] [--epochs N] [--batches N] [--data DIR]\n");
        return 2;
    }
    TRY(nmz_net_build(net));
    if (batch > 0) o.batch_size = batch;
    if (o.schedule == NMZ_SCHED_WARMUP_COSINE) {
        long per = (long)((ds->n_samples - (long)(ds->n_samples * 0.1)) + o.batch_size - 1) / o.batch_size;
        o.warmup_steps = (int)(0.05 * per * epochs) > 10 ? (int)(0.05 * per * epochs) : 10;
    }
    o.max_epochs = epochs;
    o.target_loss = 0;
    o.on_epoch = on_epoch;
    o.on_batch = on_batch;
    o.user = &tm;
    printf("%s: %zu weights, %d samples%s, batch %d, %d thread(s)\n", model, nmz_net_param_count(net), ds->n_samples,
           seq ? " (sequences)" : "", o.batch_size, threads);
    t_total = now();
    tm.t0 = t_total;
    TRY(nmz_net_train(net, ds, &o, &res));
    t_total = now() - t_total;
    {
        double per_epoch = tm.epochs ? tm.t_epoch : -1, peak = 0, total = 0, updates = -1;
        int measured = tm.batches;
        if (measured > 1) updates = (tm.t_last - tm.t_first) / (measured - 1) * tm.n_batches;
        printf("updates: %.2f s per epoch (%d of %d batches timed%s), %.1f s in all\n", updates, measured, tm.n_batches,
               measured < tm.n_batches ? ", scaled to the whole epoch" : "", t_total);
#ifdef NMZ_HAVE_BACKENDS
        {
            nmz_device_info di;
            nmz_backend_info(&di);
            peak = di.mem_peak / 1048576.0;
            total = di.mem_total / 1048576.0;
            if (di.mem_total)
                printf("device memory: %.1f MiB held now, %.1f MiB at most (of %.0f MiB; %.0f MiB free)\n",
                       di.mem_used / 1048576.0, peak, total, di.mem_free / 1048576.0);
        }
#endif
        printf("RESULT %s %s %d %zu %.3f %.3f %d %d %.4f %.4f %.1f %.0f\n", model, device, threads,
               nmz_net_param_count(net), per_epoch, updates, measured, tm.n_batches, res.loss, res.val_loss, peak, total);
    }
    nmz_net_free(net);
    nmz_dataset_free(ds);
    return 0;
}
