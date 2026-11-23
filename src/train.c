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
 * state between batches (truncated BPTT). */
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
    if (ds->y_index) nmz_tensor_zero(tb);
    for (r = 0; r < b; r++) {
        int k = idx ? idx[start + r] : start + r;
        memcpy(xb->data + (size_t)r * ds->n_inputs, ds->x + (size_t)k * ds->n_inputs,
               (size_t)ds->n_inputs * sizeof(float));
        if (!ds->y_index) {
            memcpy(tb->data + (size_t)r * ds->n_outputs, ds->y + (size_t)k * ds->n_outputs,
                   (size_t)ds->n_outputs * sizeof(float));
            continue;
        }
        for (t = 0; t < T; t++) { /* class id → one-hot row */
            float id = ds->y[(size_t)k * T + t];
            if (!(id >= 0.0f && id < (float)ds->n_outputs))
                return nmz__fail(NMZ_ERR_FORMAT, "target id %g out of range 0..%d", (double)id, ds->n_outputs - 1);
            tb->data[((size_t)r * T + t) * ds->n_outputs + (size_t)id] = 1.0f;
        }
    }
    return NMZ_OK;
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
 * model trained with carry_state is meant to run. */
static nmz_status evaluate_range(nmz_net *net, const nmz_dataset *ds, int first, int count, int carry, double *loss,
                                 double *accuracy) {
    nmz_tensor xb = NMZ_TENSOR_EMPTY, tb = NMZ_TENSOR_EMPTY;
    nmz_trace_fn saved_fn = net->trace;
    int saved_carry = net->carry;
    nmz_status s = NMZ_OK;
    double total = 0.0;
    long correct = 0;
    int start, mode = accuracy_mode(net, ds), T = steps_of(ds);
    const int chunk = carry ? 1 : ds->time_steps > 0 ? 64 : 256;
    net->trace = NULL; /* evaluation is not part of the traced math */
    net->carry = carry != 0;
    nmz_net_reset_state(net);
    for (start = first; start < first + count; start += chunk) {
        int b = first + count - start < chunk ? first + count - start : chunk, r, k;
        const nmz_tensor *y;
        double L;
        if ((s = batch_shapes(net, ds, b, &xb, &tb)) != NMZ_OK) break;
        if ((s = gather(ds, NULL, start, b, &xb, &tb)) != NMZ_OK) break;
        if ((s = nmz__net_forward(net, &xb, 0, &y)) != NMZ_OK) break;
        if ((s = nmz_net_compute_loss(net, &tb, &L)) != NMZ_OK) break;
        total += L * b; /* the loss is a mean over rows; every sample has T rows */
        for (r = 0; r < b * T; r++) {
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
    nmz_status s;
    if (!net) return nmz__fail(NMZ_ERR_ARG, "net is NULL");
    if ((s = nmz_net_build(net)) != NMZ_OK) return s;
    if ((s = nmz__net_check_dataset(net, ds)) != NMZ_OK) return s;
    return evaluate_range(net, ds, 0, ds->n_samples, net->train.carry_state && ds->time_steps > 0, loss, accuracy);
}

/* copy all parameters to/from one flat buffer (early stopping) */
static void params_copy(nmz_net *net, float *buf, int to_buf) {
    int k;
    size_t off = 0;
    for (k = 0; k < net->n_params; k++) {
        nmz_tensor *w = nmz_layer_param_value(net->layers[net->param_refs[k].layer], net->param_refs[k].index);
        if (to_buf)
            memcpy(buf + off, w->data, w->size * sizeof(float));
        else
            memcpy(w->data, buf + off, w->size * sizeof(float));
        off += w->size;
    }
}

nmz_status nmz_net_train(nmz_net *net, const nmz_dataset *ds, const nmz_train_opts *opts,
                         nmz_train_result *result) {
    return nmz_net_train_val(net, ds, NULL, opts, result);
}

nmz_status nmz_net_train_val(nmz_net *net, const nmz_dataset *ds, const nmz_dataset *val, const nmz_train_opts *opts,
                             nmz_train_result *result) {
    nmz_train_opts o;
    nmz_train_result res;
    nmz_tensor xb = NMZ_TENSOR_EMPTY, tb = NMZ_TENSOR_EMPTY;
    int *perm = NULL, epoch, i, bs, n, n_val = 0, vfirst = 0, used, best_epoch = 0;
    double bl = 0.0, best_val = HUGE_VAL;
    float *best = NULL;
    const nmz_dataset *vds = val ? val : ds; /* validation samples [vfirst, vfirst + n_val) of vds */
    nmz_status s;
    if (!net) return nmz__fail(NMZ_ERR_ARG, "net is NULL");
    o = opts ? *opts : net->train;
    if ((s = nmz__train_opts_validate(&o)) != NMZ_OK) return s;
    if ((s = nmz_net_build(net)) != NMZ_OK) return s;
    if ((s = nmz__net_check_dataset(net, ds)) != NMZ_OK) return s;
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
        best = malloc(nmz_net_param_count(net) * sizeof *best + 1);
        if (!best) {
            free(perm);
            return nmz__fail(NMZ_ERR_NOMEM, "out of memory");
        }
    }

    for (epoch = 1; epoch <= o.max_epochs; epoch++) {
        int start, batch = 0, n_batches = (used + bs - 1) / bs, clipped = 0;
        float lr = nmz_schedule_lr(&o, epoch);
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
        net->training = 1;
        net->carry = o.carry_state != 0;
        for (start = 0; start < used; start += bs) {
            int b = used - start < bs ? used - start : bs;
            double gnorm = NAN, scale = 1.0;
            if ((s = batch_shapes(net, ds, b, &xb, &tb)) != NMZ_OK) goto done;
            if ((s = gather(ds, perm, start, b, &xb, &tb)) != NMZ_OK) goto done;
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
        if ((s = evaluate_range(net, ds, 0, n, o.carry_state, &res.loss, &res.accuracy)) != NMZ_OK) goto done;
        if (n_val && (s = evaluate_range(net, vds, vfirst, n_val, o.carry_state, &res.val_loss, &res.val_accuracy)) != NMZ_OK) goto done;
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
            if (best) params_copy(net, best, 1);
        }
        if (res.loss <= o.target_loss) res.converged = 1;
        if (o.on_epoch && o.on_epoch(&info, o.user)) res.stopped = 1;
        if (res.converged || res.stopped) break;
        if (best && epoch - best_epoch >= o.patience) break; /* early stopping */
    }
    if (best && best_epoch > 0 && best_epoch != res.epochs) { /* go back to the best weights */
        params_copy(net, best, 0);
        res.best_epoch = best_epoch;
        if ((s = evaluate_range(net, ds, 0, n, o.carry_state, &res.loss, &res.accuracy)) != NMZ_OK) goto done;
        if ((s = evaluate_range(net, vds, vfirst, n_val, o.carry_state, &res.val_loss, &res.val_accuracy)) != NMZ_OK) goto done;
    } else if (best) {
        res.best_epoch = best_epoch;
    }
    s = NMZ_OK;
done:
    net->training = 0;
    net->carry = 0;
    nmz_net_reset_state(net);
    free(perm);
    free(best);
    nmz_tensor_free(&xb);
    nmz_tensor_free(&tb);
    if (result) *result = res;
    return s;
}
