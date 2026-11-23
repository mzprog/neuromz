/* train.c — mini-batch training loop and evaluation.
 *
 * One epoch = shuffle (optional) → for each mini-batch: forward, backward
 * (all gradients), update → loss over the WHOLE training set. Training stops
 * when that loss <= target_loss (converged) or after max_epochs. */
#include "internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

nmz_status nmz__train_opts_validate(const nmz_train_opts *o) {
    if (o->max_epochs < 1) return nmz__fail(NMZ_ERR_ARG, "epochs must be >= 1");
    if (o->batch_size < 1) return nmz__fail(NMZ_ERR_ARG, "batch size must be >= 1");
    if (!(o->learning_rate > 0.0f) || !isfinite(o->learning_rate))
        return nmz__fail(NMZ_ERR_ARG, "learning rate must be > 0");
    if (!(o->momentum >= 0.0f && o->momentum < 1.0f))
        return nmz__fail(NMZ_ERR_ARG, "momentum must be in [0, 1)");
    if (!(o->target_loss >= 0.0) || !isfinite(o->target_loss))
        return nmz__fail(NMZ_ERR_ARG, "target loss must be >= 0");
    if ((int)o->optimizer < 0 || o->optimizer >= NMZ_OPT_COUNT) return nmz__fail(NMZ_ERR_ARG, "invalid optimizer");
    if (!(o->beta1 >= 0.0f && o->beta1 < 1.0f && o->beta2 >= 0.0f && o->beta2 < 1.0f))
        return nmz__fail(NMZ_ERR_ARG, "Adam betas must be in [0, 1)");
    if (!(o->epsilon > 0.0f) || !isfinite(o->epsilon)) return nmz__fail(NMZ_ERR_ARG, "epsilon must be > 0");
    if ((int)o->schedule < 0 || o->schedule >= NMZ_SCHED_COUNT) return nmz__fail(NMZ_ERR_ARG, "invalid schedule");
    if (o->schedule == NMZ_SCHED_STEP && (o->step_size < 1 || !(o->gamma > 0.0f && o->gamma <= 1.0f)))
        return nmz__fail(NMZ_ERR_ARG, "step schedule needs step size >= 1 and 0 < gamma <= 1");
    if (o->schedule == NMZ_SCHED_COSINE && !(o->min_lr >= 0.0f && o->min_lr <= o->learning_rate))
        return nmz__fail(NMZ_ERR_ARG, "cosine schedule needs 0 <= min lr <= lr");
    return NMZ_OK;
}

nmz_status nmz__net_check_dataset(const nmz_net *net, const nmz_dataset *ds) {
    if (!ds || !ds->x || !ds->y || ds->n_samples < 1)
        return nmz__fail(NMZ_ERR_ARG, "dataset is empty");
    if (ds->n_inputs != nmz_net_input_size(net))
        return nmz__fail(NMZ_ERR_SHAPE, "dataset has %d input column(s) but the network expects %d",
                         ds->n_inputs, nmz_net_input_size(net));
    if (ds->n_outputs != nmz_net_output_size(net))
        return nmz__fail(NMZ_ERR_SHAPE, "dataset has %d target value(s) per row but the network has %d output(s)",
                         ds->n_outputs, nmz_net_output_size(net));
    return NMZ_OK;
}

static nmz_status batch_shapes(const nmz_net *net, int b, nmz_tensor *xb, nmz_tensor *tb) {
    int shape[NMZ_MAX_RANK], i;
    nmz_status s;
    const nmz_layer *last = net->layers[net->n_layers - 1];
    shape[0] = b;
    for (i = 0; i < net->in_rank; i++) shape[i + 1] = net->in_shape[i];
    if ((s = nmz_tensor_reshape(xb, net->in_rank + 1, shape)) != NMZ_OK) return s;
    for (i = 0; i < last->out_rank; i++) shape[i + 1] = last->out_shape[i];
    return nmz_tensor_reshape(tb, last->out_rank + 1, shape);
}

static void gather(const nmz_dataset *ds, const int *idx, int start, int b, nmz_tensor *xb,
                   nmz_tensor *tb) {
    int r;
    for (r = 0; r < b; r++) {
        int k = idx ? idx[start + r] : start + r;
        memcpy(xb->data + (size_t)r * ds->n_inputs, ds->x + (size_t)k * ds->n_inputs,
               (size_t)ds->n_inputs * sizeof(float));
        memcpy(tb->data + (size_t)r * ds->n_outputs, ds->y + (size_t)k * ds->n_outputs,
               (size_t)ds->n_outputs * sizeof(float));
    }
}

static int argmax(const float *v, int n) {
    int i, best = 0;
    for (i = 1; i < n; i++)
        if (v[i] > v[best]) best = i;
    return best;
}

/* 1 = argmax (softmax), 2 = threshold 0.5 (binary targets), 0 = n/a */
static int accuracy_mode(const nmz_net *net, const nmz_dataset *ds) {
    size_t i, n;
    if (nmz__net_fused_ce_possible(net) && ds->n_outputs > 1) return 1;
    n = (size_t)ds->n_samples * ds->n_outputs;
    for (i = 0; i < n; i++)
        if (ds->y[i] != 0.0f && ds->y[i] != 1.0f) return 0;
    return 2;
}

nmz_status nmz_net_evaluate(nmz_net *net, const nmz_dataset *ds, double *loss, double *accuracy) {
    nmz_tensor xb = NMZ_TENSOR_EMPTY, tb = NMZ_TENSOR_EMPTY;
    nmz_trace_fn saved_fn;
    nmz_status s;
    double total = 0.0;
    long correct = 0;
    int start, mode;
    const int chunk = 256;
    if (!net) return nmz__fail(NMZ_ERR_ARG, "net is NULL");
    if ((s = nmz_net_build(net)) != NMZ_OK) return s;
    if ((s = nmz__net_check_dataset(net, ds)) != NMZ_OK) return s;
    mode = accuracy_mode(net, ds);
    saved_fn = net->trace;
    net->trace = NULL; /* evaluation is not part of the traced math */
    for (start = 0; start < ds->n_samples; start += chunk) {
        int b = ds->n_samples - start < chunk ? ds->n_samples - start : chunk, r, k;
        const nmz_tensor *y;
        double L;
        if ((s = batch_shapes(net, b, &xb, &tb)) != NMZ_OK) break;
        gather(ds, NULL, start, b, &xb, &tb);
        if ((s = nmz__net_forward(net, &xb, 0, &y)) != NMZ_OK) break;
        if ((s = nmz_net_compute_loss(net, &tb, &L)) != NMZ_OK) break;
        total += L * b;
        for (r = 0; r < b; r++) {
            const float *yr = y->data + (size_t)r * ds->n_outputs, *tr = tb.data + (size_t)r * ds->n_outputs;
            if (mode == 1) {
                correct += argmax(yr, ds->n_outputs) == argmax(tr, ds->n_outputs);
            } else if (mode == 2) {
                int ok = 1;
                for (k = 0; k < ds->n_outputs; k++) ok &= (yr[k] >= 0.5f) == (tr[k] >= 0.5f);
                correct += ok;
            }
        }
    }
    net->trace = saved_fn;
    nmz_tensor_free(&xb);
    nmz_tensor_free(&tb);
    if (s != NMZ_OK) return s;
    if (loss) *loss = total / ds->n_samples;
    if (accuracy) *accuracy = mode ? (double)correct / ds->n_samples : NAN;
    return NMZ_OK;
}

nmz_status nmz_net_train(nmz_net *net, const nmz_dataset *ds, const nmz_train_opts *opts,
                         nmz_train_result *result) {
    nmz_train_opts o;
    nmz_train_result res;
    nmz_tensor xb = NMZ_TENSOR_EMPTY, tb = NMZ_TENSOR_EMPTY;
    int *perm = NULL, epoch, i, bs, n;
    double bl = 0.0;
    nmz_status s;
    if (!net) return nmz__fail(NMZ_ERR_ARG, "net is NULL");
    o = opts ? *opts : net->train;
    if ((s = nmz__train_opts_validate(&o)) != NMZ_OK) return s;
    if ((s = nmz_net_build(net)) != NMZ_OK) return s;
    if ((s = nmz__net_check_dataset(net, ds)) != NMZ_OK) return s;
    if (net->norm != NMZ_NORM_NONE && !net->norm_fitted) /* fitted once, on the first training data */
        if ((s = nmz_net_fit_normalization(net, ds)) != NMZ_OK) return s;
    if (ds->n_classes == ds->n_outputs && ds->n_classes > 0 && net->n_classes == 0)
        if ((s = nmz_net_set_class_names(net, ds->n_classes, (const char *const *)ds->class_names)) != NMZ_OK)
            return s;

    memset(&res, 0, sizeof res);
    res.loss = res.accuracy = NAN;
    n = ds->n_samples;
    bs = o.batch_size < n ? o.batch_size : n;
    perm = malloc((size_t)n * sizeof *perm);
    if (!perm) return nmz__fail(NMZ_ERR_NOMEM, "out of memory");
    for (i = 0; i < n; i++) perm[i] = i;

    for (epoch = 1; epoch <= o.max_epochs; epoch++) {
        int start, batch = 0, n_batches = (n + bs - 1) / bs;
        float lr = nmz_schedule_lr(&o, epoch);
        nmz_epoch_info info;
        /* each epoch's order depends only on the PRNG state, so training for
         * 5 + 5 epochs (even across save/load) equals training for 10 */
        for (i = 0; i < n; i++) perm[i] = i;
        if (o.shuffle)
            for (i = n - 1; i > 0; i--) { /* Fisher–Yates */
                int j = (int)nmz_rng_below(&net->rng, (uint32_t)i + 1u), t = perm[i];
                perm[i] = perm[j];
                perm[j] = t;
            }
        net->training = 1;
        for (start = 0; start < n; start += bs) {
            int b = n - start < bs ? n - start : bs;
            if ((s = batch_shapes(net, b, &xb, &tb)) != NMZ_OK) goto done;
            gather(ds, perm, start, b, &xb, &tb);
            if ((s = nmz__net_forward(net, &xb, 1, NULL)) != NMZ_OK) goto done;
            if ((s = nmz_net_backward(net, &tb, &bl)) != NMZ_OK) {
                if (s == NMZ_ERR_NUMERIC)
                    nmz__fail(s, "loss became NaN/Inf in epoch %d — the learning rate (%g) is probably too high",
                              epoch, (double)o.learning_rate);
                goto done;
            }
            if ((s = nmz_net_optimizer_step(net, &o, lr)) != NMZ_OK) goto done;
            if (o.on_batch) {
                nmz_batch_info bi;
                bi.epoch = epoch;
                bi.batch = ++batch;
                bi.n_batches = n_batches;
                bi.loss = bl;
                if (o.on_batch(&bi, o.user)) {
                    res.stopped = 1;
                    s = NMZ_OK;
                    goto done;
                }
            }
        }
        net->training = 0;
        if ((s = nmz_net_evaluate(net, ds, &res.loss, &res.accuracy)) != NMZ_OK) goto done;
        res.epochs = epoch;
        if (!isfinite(res.loss)) {
            s = nmz__fail(NMZ_ERR_NUMERIC, "loss became NaN/Inf in epoch %d — the learning rate (%g) is probably too high",
                          epoch, (double)o.learning_rate);
            goto done;
        }
        info.epoch = epoch;
        info.learning_rate = lr;
        info.loss = res.loss;
        info.accuracy = res.accuracy;
        if (res.loss <= o.target_loss) res.converged = 1;
        if (o.on_epoch && o.on_epoch(&info, o.user)) res.stopped = 1;
        if (res.converged || res.stopped) break;
    }
    s = NMZ_OK;
done:
    net->training = 0;
    free(perm);
    nmz_tensor_free(&xb);
    nmz_tensor_free(&tb);
    if (result) *result = res;
    return s;
}
