/* normalize.c — input normalization fitted on training data, learning-rate
 * schedules, and the names of optimizers/schedules/normalizations.
 *
 * Normalization is part of the net (not a layer): it has no trainable
 * parameters, it is fitted once from data, saved with the model, and applied
 * automatically by forward/predict, so raw inputs can always be fed in.
 * Groups: one per channel for [C,H,W] samples, one per feature otherwise. */
#include "internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static const char *const k_norm_names[NMZ_NORM_COUNT] = {"none", "minmax", "zscore"};
static const char *const k_opt_names[NMZ_OPT_COUNT] = {"sgd", "adam", "adamw"};
static const char *const k_sched_names[NMZ_SCHED_COUNT] = {"constant", "step", "cosine", "warmup-cosine"};

static nmz_status parse_name(const char *s, const char *const *names, int n, int *out, const char *what) {
    int i;
    if (!s || !out) return nmz__fail(NMZ_ERR_ARG, "NULL argument");
    for (i = 0; i < n; i++)
        if (!strcmp(s, names[i])) {
            *out = i;
            return NMZ_OK;
        }
    return nmz__fail(NMZ_ERR_ARG, "unknown %s '%s'", what, s);
}

const char *nmz_norm_name(nmz_norm n) { return (int)n >= 0 && n < NMZ_NORM_COUNT ? k_norm_names[n] : "?"; }
const char *nmz_optimizer_name(nmz_optimizer o) { return (int)o >= 0 && o < NMZ_OPT_COUNT ? k_opt_names[o] : "?"; }
const char *nmz_schedule_name(nmz_schedule s) { return (int)s >= 0 && s < NMZ_SCHED_COUNT ? k_sched_names[s] : "?"; }

nmz_status nmz_norm_parse(const char *s, nmz_norm *out) {
    int v = 0;
    nmz_status st;
    if (s && (!strcmp(s, "z-score") || !strcmp(s, "standard"))) s = "zscore";
    if (s && !strcmp(s, "min-max")) s = "minmax";
    st = parse_name(s, k_norm_names, NMZ_NORM_COUNT, &v, "normalization (expected none, minmax or zscore)");
    if (st == NMZ_OK) *out = (nmz_norm)v;
    return st;
}

nmz_status nmz_optimizer_parse(const char *s, nmz_optimizer *out) {
    int v = 0;
    nmz_status st = parse_name(s, k_opt_names, NMZ_OPT_COUNT, &v, "optimizer (expected sgd, adam or adamw)");
    if (st == NMZ_OK) *out = (nmz_optimizer)v;
    return st;
}

nmz_status nmz_schedule_parse(const char *s, nmz_schedule *out) {
    int v = 0;
    nmz_status st = parse_name(s, k_sched_names, NMZ_SCHED_COUNT, &v, "schedule (expected constant, step, cosine or warmup-cosine)");
    if (st == NMZ_OK) *out = (nmz_schedule)v;
    return st;
}

float nmz_schedule_lr(const nmz_train_opts *o, int epoch) {
    double lr;
    if (!o) return 0.0f;
    lr = o->learning_rate;
    if (epoch < 1) epoch = 1;
    switch (o->schedule) {
    case NMZ_SCHED_STEP:
        lr *= pow((double)o->gamma, (double)((epoch - 1) / (o->step_size > 0 ? o->step_size : 1)));
        break;
    case NMZ_SCHED_COSINE: {
        const double pi = 3.14159265358979323846;
        double frac = o->max_epochs > 0 ? (double)(epoch - 1) / o->max_epochs : 0.0;
        if (frac > 1.0) frac = 1.0;
        lr = o->min_lr + 0.5 * (lr - o->min_lr) * (1.0 + cos(pi * frac));
        break;
    }
    case NMZ_SCHED_WARMUP_COSINE: /* per batch: see nmz_schedule_lr_step */
    case NMZ_SCHED_CONSTANT:
    case NMZ_SCHED_COUNT: break;
    }
    return (float)lr;
}

/* Warm-up + cosine, per batch (DESIGN decision 95): the first W updates ramp
 * linearly up to η (Adam's second-moment estimates are still noisy, and a
 * Transformer's early updates through LayerNorm and attention can be large),
 * then a cosine from η towards η_min: batch W+1 uses η, batch S uses a rate
 * one step above η_min (with η_min = 0 the last update is not wasted). */
float nmz_schedule_lr_step(const nmz_train_opts *o, long step, long total, long per_epoch) {
    const double pi = 3.14159265358979323846;
    double lr, frac;
    long W;
    if (!o) return 0.0f;
    if (o->schedule != NMZ_SCHED_WARMUP_COSINE)
        return nmz_schedule_lr(o, per_epoch > 0 ? (int)((step - 1) / per_epoch) + 1 : 1);
    lr = o->learning_rate;
    W = o->warmup_steps > 0 ? o->warmup_steps : 0;
    if (step < 1) step = 1;
    if (W > 0 && step <= W) return (float)(lr * (double)step / (double)W);
    frac = total > W ? (double)(step - W - 1) / (double)(total - W) : 1.0;
    if (frac > 1.0) frac = 1.0;
    return (float)(o->min_lr + 0.5 * (lr - o->min_lr) * (1.0 + cos(pi * frac)));
}

/* ------------------------------------------------------------ normalization */

int nmz__norm_group_count(const nmz_net *net) {
    if (net->in_rank == 3) return net->in_shape[0];
    return net->in_shape[net->in_rank - 1];
}

int nmz__norm_group_of(const nmz_net *net, size_t e) {
    if (net->in_rank == 3) return (int)(e / ((size_t)net->in_shape[1] * net->in_shape[2]));
    return (int)(e % (size_t)net->in_shape[net->in_rank - 1]);
}

void nmz__norm_free(nmz_net *net) {
    free(net->norm_offset);
    free(net->norm_scale);
    net->norm_offset = net->norm_scale = NULL;
    net->norm_groups = 0;
    net->norm_fitted = 0;
}

nmz_status nmz__norm_alloc(nmz_net *net, int groups) {
    nmz__norm_free(net);
    net->norm_offset = calloc((size_t)groups, sizeof(float));
    net->norm_scale = malloc((size_t)groups * sizeof(float));
    if (!net->norm_offset || !net->norm_scale) {
        nmz__norm_free(net);
        return nmz__fail(NMZ_ERR_NOMEM, "out of memory");
    }
    net->norm_groups = groups;
    return NMZ_OK;
}

static size_t sample_size(const nmz_net *net) {
    size_t n = 1;
    int i;
    for (i = 0; i < net->in_rank; i++) n *= (size_t)net->in_shape[i];
    return n;
}

void nmz__norm_apply(const nmz_net *net, nmz_tensor *t) {
    size_t per = sample_size(net), e;
    if (!net->norm_fitted || net->norm == NMZ_NORM_NONE) return;
    for (e = 0; e < t->size; e++) {
        int g = nmz__norm_group_of(net, e % per);
        t->data[e] = (t->data[e] - net->norm_offset[g]) * net->norm_scale[g];
    }
}

void nmz__norm_scale_grad(const nmz_net *net, nmz_tensor *gr) {
    size_t per = sample_size(net), e;
    if (!net->norm_fitted || net->norm == NMZ_NORM_NONE) return;
    for (e = 0; e < gr->size; e++) gr->data[e] *= net->norm_scale[nmz__norm_group_of(net, e % per)];
}

nmz_status nmz_net_set_normalization(nmz_net *net, nmz_norm kind) {
    if (!net) return nmz__fail(NMZ_ERR_ARG, "net is NULL");
    if ((int)kind < 0 || kind >= NMZ_NORM_COUNT) return nmz__fail(NMZ_ERR_ARG, "invalid normalization");
    if (kind != net->norm) nmz__norm_free(net);
    net->norm = kind;
    net->have_forward = net->have_grads = 0;
    return NMZ_OK;
}

nmz_status nmz_net_fit_normalization(nmz_net *net, const nmz_dataset *ds) {
    int groups, g;
    size_t per, i, e;
    double *sum = NULL, *sq = NULL, *mn = NULL, *mx = NULL, *cnt = NULL;
    nmz_status s;
    if (!net || !ds || !ds->x || ds->n_samples < 1) return nmz__fail(NMZ_ERR_ARG, "net or dataset is NULL/empty");
    if (ds->n_inputs != nmz_net_input_size(net))
        return nmz__fail(NMZ_ERR_SHAPE, "dataset has %d inputs, the network expects %d", ds->n_inputs,
                         nmz_net_input_size(net));
    if (net->norm == NMZ_NORM_NONE) return NMZ_OK;
    groups = nmz__norm_group_count(net);
    per = sample_size(net);
    sum = calloc((size_t)groups, sizeof *sum);
    sq = calloc((size_t)groups, sizeof *sq);
    mn = malloc((size_t)groups * sizeof *mn);
    mx = malloc((size_t)groups * sizeof *mx);
    cnt = calloc((size_t)groups, sizeof *cnt);
    if (!sum || !sq || !mn || !mx || !cnt) {
        s = nmz__fail(NMZ_ERR_NOMEM, "out of memory");
        goto out;
    }
    for (g = 0; g < groups; g++) {
        mn[g] = HUGE_VAL;
        mx[g] = -HUGE_VAL;
    }
    for (i = 0; i < (size_t)ds->n_samples; i++)
        for (e = 0; e < per; e++) {
            double v = ds->x[i * per + e];
            g = nmz__norm_group_of(net, e);
            sum[g] += v;
            sq[g] += v * v;
            cnt[g] += 1.0;
            if (v < mn[g]) mn[g] = v;
            if (v > mx[g]) mx[g] = v;
        }
    if ((s = nmz__norm_alloc(net, groups)) != NMZ_OK) goto out;
    for (g = 0; g < groups; g++) {
        if (net->norm == NMZ_NORM_MINMAX) {
            double range = mx[g] - mn[g];
            net->norm_offset[g] = (float)mn[g];
            net->norm_scale[g] = range > 1e-12 ? (float)(1.0 / range) : 1.0f;
        } else {
            double mean = sum[g] / cnt[g], var = sq[g] / cnt[g] - mean * mean;
            net->norm_offset[g] = (float)mean;
            net->norm_scale[g] = var > 1e-24 ? (float)(1.0 / sqrt(var)) : 1.0f;
        }
    }
    net->norm_fitted = 1;
    net->have_forward = net->have_grads = 0;
    s = NMZ_OK;
out:
    free(sum);
    free(sq);
    free(mn);
    free(mx);
    free(cnt);
    return s;
}

nmz_norm nmz_net_normalization(const nmz_net *net, int *fitted, int *groups, const float **offset,
                               const float **scale) {
    if (fitted) *fitted = net ? net->norm_fitted : 0;
    if (groups) *groups = net ? net->norm_groups : 0;
    if (offset) *offset = net && net->norm_fitted ? net->norm_offset : NULL;
    if (scale) *scale = net && net->norm_fitted ? net->norm_scale : NULL;
    return net ? net->norm : NMZ_NORM_NONE;
}
