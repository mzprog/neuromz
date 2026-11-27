/* train.c — mini-batch training loop and evaluation.
 *
 * One epoch = shuffle (optional) → for each mini-batch: forward, backward
 * (all gradients), clip (optional), update → loss over the WHOLE training
 * split (and the validation split, if any). Training stops when the training
 * loss <= target_loss (converged) or after max_epochs.
 *
 * Sequence datasets (time_steps > 0) store token ids; targets are expanded
 * to one-hot rows only for the batch being processed. With carry_state the
 * batches are cut into B parallel streams so that row r of batch k+1 is the
 * text right after row r of batch k, and recurrent layers keep their hidden
 * state between batches (truncated BPTT).
 *
 * On a device backend the data set is copied to the device once per run
 * (`resident`) and every batch is gathered there, so nothing but the
 * shuffled order crosses to the device during an epoch, and only the
 * batch loss and the epoch's metrics come back. If it does not fit, the
 * batches are gathered on the host and copied one by one. */
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
    if ((o->schedule == NMZ_SCHED_COSINE || o->schedule == NMZ_SCHED_WARMUP_COSINE) &&
        !(o->min_lr >= 0.0f && o->min_lr <= o->learning_rate))
        return nmz__fail(NMZ_ERR_ARG, "cosine schedule needs 0 <= min lr <= lr");
    if (o->warmup_steps < 0 || o->warmup_steps > 100000000) return nmz__fail(NMZ_ERR_ARG, "warm-up steps must be >= 0");
    if (!(o->weight_decay >= 0.0f && o->weight_decay < 10.0f))
        return nmz__fail(NMZ_ERR_ARG, "weight decay must be in [0, 10)");
    if (!(o->val_fraction >= 0.0f && o->val_fraction < 1.0f))
        return nmz__fail(NMZ_ERR_ARG, "validation fraction must be in [0, 1)");
    if (!(o->clip_norm >= 0.0f) || !isfinite(o->clip_norm)) return nmz__fail(NMZ_ERR_ARG, "clip norm must be >= 0");
    if (o->seq_len < 0 || o->seq_len > 100000) return nmz__fail(NMZ_ERR_ARG, "sequence length must be 0..100000");
    if (o->patience < 0) return nmz__fail(NMZ_ERR_ARG, "patience must be >= 0");
    return NMZ_OK;
}

static int steps_of(const nmz_dataset *ds) { return ds->time_steps > 0 ? ds->time_steps : 1; }

nmz_status nmz__net_check_dataset(const nmz_net *net, const nmz_dataset *ds) {
    int T;
    if (!ds || !ds->x || !ds->y || ds->n_samples < 1) return nmz__fail(NMZ_ERR_ARG, "dataset is empty");
    T = steps_of(ds);
    if (nmz_net_is_sequence(net) != (ds->time_steps > 0))
        return nmz__fail(NMZ_ERR_SHAPE, nmz_net_is_sequence(net)
                                            ? "this is a sequence network: train it on text (sequence) data"
                                            : "sequence (text) data needs a sequence network (input shape any-length)");
    if (ds->n_inputs != nmz_net_input_size(net) * T)
        return nmz__fail(NMZ_ERR_SHAPE, "dataset has %d input column(s) but the network expects %d",
                         ds->n_inputs / T, nmz_net_input_size(net));
    if (ds->n_outputs != nmz_net_output_size(net))
        return nmz__fail(NMZ_ERR_SHAPE, "dataset has %d target value(s) per %s but the network has %d output(s)",
                         ds->n_outputs, ds->time_steps ? "step" : "row", nmz_net_output_size(net));
    return NMZ_OK;
}

static nmz_status batch_shapes(const nmz_net *net, const nmz_dataset *ds, int b, nmz_tensor *xb, nmz_tensor *tb) {
    int shape[NMZ_MAX_RANK], i, T = steps_of(ds);
    nmz_status s;
    const nmz_layer *last = net->layers[net->n_layers - 1];
    shape[0] = b;
    for (i = 0; i < net->in_rank; i++) shape[i + 1] = net->in_shape[i] == NMZ_ANY_LENGTH ? T : net->in_shape[i];
    if ((s = nmz_tensor_reshape(xb, net->in_rank + 1, shape)) != NMZ_OK) return s;
    for (i = 0; i < last->out_rank; i++) shape[i + 1] = last->out_shape[i] == NMZ_ANY_LENGTH ? T : last->out_shape[i];
    return nmz_tensor_reshape(tb, last->out_rank + 1, shape);
}

static nmz_status gather(const nmz_dataset *ds, const int *idx, int start, int b, nmz_tensor *xb, nmz_tensor *tb) {
    int r, t, T = steps_of(ds);
    float *xd = nmz__hw(xb), *td;
    if (ds->y_index) nmz_tensor_zero(tb);
    td = nmz__hu(tb);
    for (r = 0; r < b; r++) {
        int k = idx ? idx[start + r] : start + r;
        memcpy(xd + (size_t)r * ds->n_inputs, ds->x + (size_t)k * ds->n_inputs, (size_t)ds->n_inputs * sizeof(float));
        if (!ds->y_index) {
            memcpy(td + (size_t)r * ds->n_outputs, ds->y + (size_t)k * ds->n_outputs,
                   (size_t)ds->n_outputs * sizeof(float));
            continue;
        }
        for (t = 0; t < T; t++) { /* class id → one-hot row */
            float id = ds->y[(size_t)k * T + t];
            if (!(id >= 0.0f && id < (float)ds->n_outputs))
                return nmz__fail(NMZ_ERR_FORMAT, "target id %g out of range 0..%d", (double)id, ds->n_outputs - 1);
            td[((size_t)r * T + t) * ds->n_outputs + (size_t)id] = 1.0f;
        }
    }
    return NMZ_OK;
}

/* ------------------------------------------------- data on the device */

typedef struct resident {
    const nmz_backend *be;
    float *x, *y;   /* every sample's inputs [n, n_inputs], targets [n, n_outputs] (or ids [n, T]) */
    float *idx;     /* the epoch's order (floats), n                                         */
    nmz_tensor ids; /* target ids of a batch [b, T]                                          */
    int n;
} resident;

static size_t y_width(const nmz_dataset *ds) { return ds->y_index ? (size_t)steps_of(ds) : (size_t)ds->n_outputs; }

static void resident_free(resident *r) {
    if (r->be) {
        if (r->x) r->be->release(r->be, r->x);
        if (r->y) r->be->release(r->be, r->y);
        if (r->idx) r->be->release(r->be, r->idx);
    }
    nmz_tensor_free(&r->ids);
    memset(r, 0, sizeof *r);
}

/* Copy a data set to the device: 1 = done, 0 = keep it on the host (a host
 * backend, or it does not fit: then batches are copied one at a time). */
static int resident_load(nmz_net *net, const nmz_dataset *ds, resident *r, nmz_status *s) {
    const nmz_backend *be = net->be;
    size_t nx = (size_t)ds->n_samples * ds->n_inputs, ny = (size_t)ds->n_samples * y_width(ds);
    memset(r, 0, sizeof *r);
    *s = NMZ_OK;
    if (be->host_memory) return 0;
    /* ids are checked once here: a device kernel cannot report a bad one */
    if ((net->layers[0]->ops->flags & NMZ_LAYER_DISCRETE_INPUT) &&
        (*s = nmz__check_ids(net->layers[0], ds->x, nx)) != NMZ_OK)
        return 0;
    r->be = be;
    r->n = ds->n_samples;
    r->x = be->alloc(be, nx * sizeof(float));
    r->y = r->x ? be->alloc(be, ny * sizeof(float)) : NULL;
    r->idx = r->y ? be->alloc(be, (size_t)ds->n_samples * sizeof(float)) : NULL;
    if (!r->idx || be->upload(be, r->x, ds->x, nx * sizeof(float)) != NMZ_OK ||
        be->upload(be, r->y, ds->y, ny * sizeof(float)) != NMZ_OK) {
        resident_free(r); /* too big for the device: gather on the host */
        return 0;
    }
    return 1;
}

/* the epoch's order to the device (floats are exact below 2^24 samples) */
static nmz_status resident_order(resident *r, const int *perm, int n) {
    float *f = malloc((size_t)n * sizeof *f);
    nmz_status s;
    int i;
    if (!f) return nmz__fail(NMZ_ERR_NOMEM, "out of memory");
    for (i = 0; i < n; i++) f[i] = (float)perm[i];
    s = r->be->upload(r->be, r->idx, f, (size_t)n * sizeof *f);
    if (s == NMZ_OK) s = r->be->sync(r->be); /* f is freed below */
    free(f);
    return s;
}

/* samples idx[start …] (in order) or start … (perm = 0) into xb, tb */
static nmz_status resident_gather(const nmz_dataset *ds, resident *r, int ordered, int start, int b, nmz_tensor *xb,
                                  nmz_tensor *tb) {
    const nmz_backend *be = r->be;
    float *px = nmz__w(be, xb), *pt = nmz__w(be, tb), *py;
    size_t yw = y_width(ds);
    nmz_status s;
    if (!px || !pt) return nmz__op_failed();
    if (ds->y_index) {
        int shape[2];
        shape[0] = b;
        shape[1] = (int)yw;
        if ((s = nmz_tensor_reshape(&r->ids, 2, shape)) != NMZ_OK) return s;
        if (!(py = nmz__w(be, &r->ids))) return nmz__op_failed();
    } else {
        py = pt;
    }
    if (ordered) {
        if ((s = be->gather_rows(be, b, (size_t)ds->n_inputs, r->idx + start, r->x, px)) != NMZ_OK ||
            (s = be->gather_rows(be, b, yw, r->idx + start, r->y, py)) != NMZ_OK)
            return s;
    } else {
        if ((s = be->copy(be, 1, (size_t)b * ds->n_inputs, r->x + (size_t)start * ds->n_inputs, (size_t)b * ds->n_inputs,
                          px, (size_t)b * ds->n_inputs)) != NMZ_OK ||
            (s = be->copy(be, 1, (size_t)b * yw, r->y + (size_t)start * yw, (size_t)b * yw, py, (size_t)b * yw)) != NMZ_OK)
            return s;
    }
    return ds->y_index ? be->onehot(be, (size_t)b * yw, ds->n_outputs, py, pt) : NMZ_OK;
}

static int argmax(const float *v, int n) {
    int i, best = 0;
    for (i = 1; i < n; i++)
        if (v[i] > v[best]) best = i;
    return best;
}

/* 1 = argmax (softmax / class ids), 2 = threshold 0.5 (binary targets), 0 = n/a */
static int accuracy_mode(const nmz_net *net, const nmz_dataset *ds) {
    size_t i, n;
    if (ds->y_index) return ds->n_outputs > 1 ? 1 : 0;
    if (nmz__net_fused_ce_possible(net) && ds->n_outputs > 1) return 1;
    n = (size_t)ds->n_samples * ds->n_outputs;
    for (i = 0; i < n; i++)
        if (ds->y[i] != 0.0f && ds->y[i] != 1.0f) return 0;
    return 2;
}

/* loss/accuracy over samples [first, first+count). Stateless (every sample
 * from a zero state) unless `carry`: then the samples are read in order as
 * one stream, one at a time, with the recurrent state carried — the way a
 * model trained with carry_state is meant to run. `rd` (may be NULL): the
 * data set on the device. */
static nmz_status evaluate_range(nmz_net *net, const nmz_dataset *ds, resident *rd, int first, int count, int carry,
                                 double *loss, double *accuracy) {
    nmz_tensor xb = NMZ_TENSOR_EMPTY, tb = NMZ_TENSOR_EMPTY;
    nmz_trace_fn saved_fn = net->trace;
    int saved_carry = net->carry;
    nmz_status s = NMZ_OK;
    double total = 0.0, *q = NULL; /* device: each chunk's loss and count, read at the end */
    long correct = 0;
    int start, mode = accuracy_mode(net, ds), T = steps_of(ds), c = 0;
    int chunk = carry ? 1 : ds->time_steps > 0 ? 64 : 256;
    /* on a device no larger than the training batch, whose activations fit (the
     * CPU keeps v2.0.1's chunks: they decide the order the losses are summed in) */
    if (rd && rd->x) {
        int bsz = net->train_batch > 0 ? net->train_batch : net->train.batch_size; /* this run's, or the stored one */
        if (bsz > 0 && chunk > bsz) chunk = bsz;
    }
    if (rd && rd->x && !(q = calloc((size_t)(count / chunk + 2) * 2, sizeof *q)))
        return nmz__fail(NMZ_ERR_NOMEM, "out of memory");
    net->trace = NULL; /* evaluation is not part of the traced math */
    net->carry = carry != 0;
    nmz_net_reset_state(net);
    for (start = first; start < first + count; start += chunk) {
        int b = first + count - start < chunk ? first + count - start : chunk, r, k;
        const nmz_tensor *y;
        double L, right = 0.0;
        if ((s = batch_shapes(net, ds, b, &xb, &tb)) != NMZ_OK) break;
        if (rd && rd->x) { /* nothing waits for the device until the end (or a full result queue) */
            const nmz_backend *be = rd->be;
            const nmz_tensor *z = nmz__net_fused_ce(net) ? &net->acts[net->n_layers - 1] : NULL;
            const float *py, *pt;
            (void)right;
            if ((s = resident_gather(ds, rd, 0, start, b, &xb, &tb)) != NMZ_OK) break;
            if ((s = nmz__net_forward(net, &xb, 0, &y)) != NMZ_OK) break;
            py = nmz__r(be, y);
            pt = nmz__r(be, &tb);
            if (!py || !pt) {
                s = nmz__op_failed();
                break;
            }
            if (mode && (s = be->count_correct(be, mode, (size_t)b * T, ds->n_outputs, py, pt, &q[2 * c + 1])) != NMZ_OK)
                break;
            if ((s = nmz__loss_value_queued(be, net->loss, y, &tb, z, &q[2 * c])) != NMZ_OK) break;
            c++;
            continue;
        }
        if ((s = gather(ds, NULL, start, b, &xb, &tb)) != NMZ_OK) break;
        if ((s = nmz__net_forward(net, &xb, 0, &y)) != NMZ_OK) break;
        if ((s = nmz_net_compute_loss(net, &tb, &L)) != NMZ_OK) break;
        total += L * b; /* the loss is a mean over rows; every sample has T rows */
        {
            const float *yd = nmz__hr(y), *td = nmz__hr(&tb);
            for (r = 0; r < b * T; r++) {
                const float *yr = yd + (size_t)r * ds->n_outputs, *tr = td + (size_t)r * ds->n_outputs;
                if (mode == 1) {
                    correct += argmax(yr, ds->n_outputs) == argmax(tr, ds->n_outputs);
                } else if (mode == 2) {
                    int ok = 1;
                    for (k = 0; k < ds->n_outputs; k++) ok &= (yr[k] >= 0.5f) == (tr[k] >= 0.5f);
                    correct += ok;
                }
            }
        }
    }
    if (q) { /* the queued results: the mean over rows, × the chunk's samples */
        int k;
        if (s == NMZ_OK) s = rd->be->sync(rd->be);
        for (k = 0; k < c && s == NMZ_OK; k++) {
            int b = first + count - (first + k * chunk) < chunk ? first + count - (first + k * chunk) : chunk;
            total += q[2 * k] * b;
            correct += (long)q[2 * k + 1];
        }
        free(q);
    }
    net->trace = saved_fn;
    net->carry = saved_carry;
    nmz_net_reset_state(net);
    nmz_tensor_free(&xb);
    nmz_tensor_free(&tb);
    if (s != NMZ_OK) return s;
    if (loss) *loss = total / count;
    if (accuracy) *accuracy = mode ? (double)correct / ((double)count * T) : NAN;
    return NMZ_OK;
}

nmz_status nmz_net_evaluate(nmz_net *net, const nmz_dataset *ds, double *loss, double *accuracy) {
    resident rd;
    nmz_status s;
    if (!net) return nmz__fail(NMZ_ERR_ARG, "net is NULL");
    if ((s = nmz_net_build(net)) != NMZ_OK) return s;
    if ((s = nmz__net_check_dataset(net, ds)) != NMZ_OK) return s;
    nmz__net_backend(net);
    resident_load(net, ds, &rd, &s);
    if (s != NMZ_OK) return s;
    s = evaluate_range(net, ds, &rd, 0, ds->n_samples, net->train.carry_state && ds->time_steps > 0, loss, accuracy);
    resident_free(&rd);
    if (s == NMZ_ERR_NOMEM) nmz__net_drop_buffers(net);
    return s;
}

/* copy all parameters to/from one flat buffer in be's memory (early stopping) */
static nmz_status params_copy(nmz_net *net, nmz_tensor *buf, int to_buf) {
    const nmz_backend *be = net->be;
    float *b = to_buf ? nmz__u(be, buf) : (float *)nmz__r(be, buf);
    nmz_status s = NMZ_OK;
    int k;
    size_t off = 0;
    if (!b) return nmz__op_failed();
    for (k = 0; k < net->n_params && s == NMZ_OK; k++) {
        nmz_tensor *w = nmz__param(net->layers[net->param_refs[k].layer], net->param_refs[k].index).value;
        float *pw = to_buf ? (float *)nmz__r(be, w) : nmz__w(be, w);
        if (!pw) return nmz__op_failed();
        if (to_buf)
            s = be->copy(be, 1, w->size, pw, w->size, b + off, w->size);
        else
            s = be->copy(be, 1, w->size, b + off, w->size, pw, w->size);
        off += w->size;
    }
    return s;
}

nmz_status nmz_net_train(nmz_net *net, const nmz_dataset *ds, const nmz_train_opts *opts,
                         nmz_train_result *result) {
    return nmz_net_train_val(net, ds, NULL, opts, result);
}

nmz_status nmz_net_train_val(nmz_net *net, const nmz_dataset *ds, const nmz_dataset *val, const nmz_train_opts *opts,
                             nmz_train_result *result) {
    nmz_train_opts o;
    nmz_train_result res;
    nmz_tensor xb = NMZ_TENSOR_EMPTY, tb = NMZ_TENSOR_EMPTY, best = NMZ_TENSOR_EMPTY;
    int *perm = NULL, epoch, i, bs, n, n_val = 0, vfirst = 0, used, best_epoch = 0, keep_best = 0;
    double bl = 0.0, best_val = HUGE_VAL;
    const nmz_dataset *vds = val ? val : ds; /* validation samples [vfirst, vfirst + n_val) of vds */
    resident rd, rv, *rvp;
    nmz_status s;
    memset(&rd, 0, sizeof rd);
    memset(&rv, 0, sizeof rv);
    if (!net) return nmz__fail(NMZ_ERR_ARG, "net is NULL");
    o = opts ? *opts : net->train;
    if ((s = nmz__train_opts_validate(&o)) != NMZ_OK) return s;
    if ((s = nmz_net_build(net)) != NMZ_OK) return s;
    if ((s = nmz__net_check_dataset(net, ds)) != NMZ_OK) return s;
    nmz__net_backend(net);
    if (val) { /* a separate validation set: all of ds trains, val_fraction is ignored */
        if ((s = nmz__net_check_dataset(net, val)) != NMZ_OK) return s;
        if (val->n_samples < 1) return nmz__fail(NMZ_ERR_ARG, "the validation data set is empty");
        n_val = val->n_samples;
    } else if (o.val_fraction > 0.0f) {
        n_val = (int)((double)ds->n_samples * o.val_fraction);
        if (n_val < 1 || n_val >= ds->n_samples)
            return nmz__fail(NMZ_ERR_ARG, "a validation fraction of %g leaves no training or no validation samples "
                                          "(%d samples)", (double)o.val_fraction, ds->n_samples);
    }
    /* without a separate set, the LAST n_val samples are the validation split */
    n = val ? ds->n_samples : ds->n_samples - n_val;
    vfirst = val ? 0 : n;
    if (net->norm != NMZ_NORM_NONE && !net->norm_fitted) { /* fitted once, on the first training data */
        nmz_dataset view = *ds;
        view.n_samples = n; /* statistics of the training split only */
        if ((s = nmz_net_fit_normalization(net, &view)) != NMZ_OK) return s;
    }
    if (ds->n_classes == ds->n_outputs && ds->n_classes > 0 && net->n_classes == 0)
        if ((s = nmz_net_set_class_names(net, ds->n_classes, (const char *const *)ds->class_names)) != NMZ_OK)
            return s;
    if (ds->vocab && !net->vocab && (s = nmz_net_set_vocab(net, ds->vocab)) != NMZ_OK) return s;

    memset(&res, 0, sizeof res);
    res.loss = res.accuracy = res.val_loss = res.val_accuracy = NAN;
    bs = o.batch_size < n ? o.batch_size : n;
    /* carry: batch k, row r = sample r·nb + k (B parallel streams); a
     * remainder that does not fill a batch is left out of the epoch */
    used = o.carry_state ? (n / bs) * bs : n;
    perm = malloc((size_t)n * sizeof *perm);
    if (!perm) return nmz__fail(NMZ_ERR_NOMEM, "out of memory");
    if (o.patience > 0 && n_val > 0) {
        int total = (int)nmz_net_param_count(net) + 1;
        keep_best = 1;
        if ((s = nmz_tensor_reshape(&best, 1, &total)) != NMZ_OK) {
            free(perm);
            return s;
        }
    }
    /* on a device: the data sets live there for the whole run */
    resident_load(net, ds, &rd, &s);
    if (s == NMZ_OK && val) resident_load(net, val, &rv, &s);
    if (s != NMZ_OK) goto done;
    rvp = val ? &rv : &rd;

    for (epoch = 1; epoch <= o.max_epochs; epoch++) {
        int start, batch = 0, n_batches = (used + bs - 1) / bs, clipped = 0;
        float lr = nmz_schedule_lr(&o, epoch);
        long total_steps = (long)n_batches * o.max_epochs;
        nmz_epoch_info info;
        if (o.carry_state) {
            int nb = used / bs, k, r;
            for (k = 0; k < nb; k++)
                for (r = 0; r < bs; r++) perm[k * bs + r] = r * nb + k;
            nmz_net_reset_state(net);
        } else {
            /* each epoch's order depends only on the PRNG state, so training for
             * 5 + 5 epochs (even across save/load) equals training for 10 */
            for (i = 0; i < n; i++) perm[i] = i;
            if (o.shuffle)
                for (i = n - 1; i > 0; i--) { /* Fisher–Yates */
                    int j = (int)nmz_rng_below(&net->rng, (uint32_t)i + 1u), t = perm[i];
                    perm[i] = perm[j];
                    perm[j] = t;
                }
        }
        if (rd.x && (s = resident_order(&rd, perm, used)) != NMZ_OK) goto done;
        net->training = 1;
        net->carry = o.carry_state != 0;
        net->train_batch = bs;
        for (start = 0; start < used; start += bs) {
            int b = used - start < bs ? used - start : bs;
            double gnorm = NAN, scale = 1.0;
            if (o.schedule == NMZ_SCHED_WARMUP_COSINE) /* per batch */
                lr = nmz_schedule_lr_step(&o, (long)(epoch - 1) * n_batches + start / bs + 1, total_steps, n_batches);
            if ((s = batch_shapes(net, ds, b, &xb, &tb)) != NMZ_OK) goto done;
            if (rd.x)
                s = resident_gather(ds, &rd, 1, start, b, &xb, &tb);
            else
                s = gather(ds, perm, start, b, &xb, &tb);
            if (s != NMZ_OK) goto done;
            if ((s = nmz__net_forward(net, &xb, 1, NULL)) != NMZ_OK) goto done;
            if ((s = nmz_net_backward(net, &tb, &bl)) != NMZ_OK) {
                if (s == NMZ_ERR_NUMERIC)
                    nmz__fail(s, "loss became NaN/Inf in epoch %d — the learning rate (%g) is probably too high",
                              epoch, (double)o.learning_rate);
                goto done;
            }
            if (o.clip_norm > 0.0f) {
                if ((s = nmz_net_clip_gradients(net, o.clip_norm, &gnorm, &scale)) != NMZ_OK) goto done;
                clipped += scale < 1.0;
            }
            if ((s = nmz_net_optimizer_step(net, &o, lr)) != NMZ_OK) goto done;
            if (o.on_batch) {
                nmz_batch_info bi;
                bi.epoch = epoch;
                bi.batch = ++batch;
                bi.n_batches = n_batches;
                bi.loss = bl;
                bi.grad_norm = gnorm;
                bi.clipped = scale < 1.0;
                bi.learning_rate = lr;
                if (o.on_batch(&bi, o.user)) {
                    res.stopped = 1;
                    s = NMZ_OK;
                    goto done;
                }
            }
        }
        net->training = 0;
        net->carry = 0;
        nmz_net_reset_state(net);
        if ((s = evaluate_range(net, ds, &rd, 0, n, o.carry_state, &res.loss, &res.accuracy)) != NMZ_OK) goto done;
        if (n_val && (s = evaluate_range(net, vds, rvp, vfirst, n_val, o.carry_state, &res.val_loss, &res.val_accuracy)) !=
                         NMZ_OK)
            goto done;
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
        info.val_loss = res.val_loss;
        info.val_accuracy = res.val_accuracy;
        info.clipped_batches = clipped;
        info.improved = n_val > 0 && res.val_loss < best_val;
        if (info.improved) {
            best_val = res.val_loss;
            best_epoch = epoch;
            if (keep_best && (s = params_copy(net, &best, 1)) != NMZ_OK) goto done;
        }
        if (res.loss <= o.target_loss) res.converged = 1;
        if (o.on_epoch && o.on_epoch(&info, o.user)) res.stopped = 1;
        if (res.converged || res.stopped) break;
        if (keep_best && epoch - best_epoch >= o.patience) break; /* early stopping */
    }
    if (keep_best && best_epoch > 0 && best_epoch != res.epochs) { /* go back to the best weights */
        if ((s = params_copy(net, &best, 0)) != NMZ_OK) goto done;
        res.best_epoch = best_epoch;
        if ((s = evaluate_range(net, ds, &rd, 0, n, o.carry_state, &res.loss, &res.accuracy)) != NMZ_OK) goto done;
        if ((s = evaluate_range(net, vds, rvp, vfirst, n_val, o.carry_state, &res.val_loss, &res.val_accuracy)) != NMZ_OK)
            goto done;
    } else if (keep_best) {
        res.best_epoch = best_epoch;
    }
    s = NMZ_OK;
done:
    net->training = 0;
    net->carry = 0;
    nmz_net_reset_state(net);
    resident_free(&rd);
    resident_free(&rv);
    if (s == NMZ_ERR_NOMEM) nmz__net_drop_buffers(net); /* a retry with a smaller batch starts from nothing */
    free(perm);
    nmz_tensor_free(&best);
    nmz_tensor_free(&xb);
    nmz_tensor_free(&tb);
    if (result) *result = res;
    return s;
}
