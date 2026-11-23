/* gradcheck.c — numerical gradient check.
 *
 * For every parameter w (and every input x) compare the backprop gradient
 * with the central difference
 *
 *     ∂L/∂w ≈ (L(w + ε) − L(w − ε)) / 2ε
 *
 * which has O(ε²) truncation error. If backprop is right the two agree to
 * float precision. This is how the engine's own derivatives are tested, and
 * it is public so students can check layers they write themselves. */
#include "internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* The "switch pattern" of a forward pass: which side of 0 every relu /
 * leaky-relu input is on, and every max-pool winner. If a perturbation
 * changes it, the loss is not differentiable between the two points. */
static size_t signature(const nmz_net *net, float *out) {
    size_t n = 0, e;
    int i;
    for (i = 0; i < net->n_layers; i++) {
        const nmz_layer *l = net->layers[i];
        int k = nmz_layer_activation_kind(l);
        if (k == NMZ_ACT_RELU || k == NMZ_ACT_LEAKY_RELU) {
            for (e = 0; e < net->acts[i].size; e++, n++)
                if (out) out[n] = net->acts[i].data[e] > 0.0f ? 1.0f : 0.0f;
        } else if (l->ops == &nmz__maxpool2d_ops) {
            const nmz_tensor *am = nmz__maxpool_argmax(l);
            for (e = 0; e < am->size; e++, n++)
                if (out) out[n] = am->data[e];
        }
    }
    return n;
}

static int same_signature(const nmz_net *net, const float *base, float *tmp, size_t n) {
    if (n == 0) return 1;
    signature(net, tmp);
    return memcmp(base, tmp, n * sizeof(float)) == 0;
}

/* Every forward pass starts from the same PRNG state, so stochastic layers
 * (dropout) make identical choices and L is a deterministic function of w. */
typedef struct sigbuf {
    float *base, *tmp;
    size_t n;
    int crossed; /* set when a forward's switch pattern differs from base */
} sigbuf;

static nmz_status loss_at(nmz_net *net, const nmz_tensor *x, const nmz_tensor *t, int training,
                          const nmz_rng *rng0, sigbuf *sb, double *L) {
    nmz_status s;
    net->rng = *rng0;
    s = nmz__net_forward(net, x, training, NULL);
    if (s != NMZ_OK) return s;
    if (!same_signature(net, sb->base, sb->tmp, sb->n)) sb->crossed = 1;
    return nmz_net_compute_loss(net, t, L);
}

/* Track errors; "worst" = largest error relative to its tolerance. */
static void compare(nmz_gradcheck_result *r, double *worst_ratio, double a, double n, int layer,
                    const char *name, size_t index, double atol, double rtol) {
    double abs_err = fabs(a - n), denom = fabs(a) + fabs(n);
    double rel = abs_err / (denom > 1e-8 ? denom : 1e-8);
    double ratio = abs_err / (atol + rtol * fmax(fabs(a), fabs(n)));
    r->checked++;
    if (ratio > 1.0) r->failures++;
    if (abs_err > r->max_abs_err) r->max_abs_err = abs_err;
    if (rel > r->max_rel_err) r->max_rel_err = rel;
    if (r->checked == 1 || ratio > *worst_ratio) {
        *worst_ratio = ratio;
        r->worst_layer = layer;
        strncpy(r->worst_param, name, sizeof r->worst_param - 1);
        r->worst_param[sizeof r->worst_param - 1] = '\0';
        r->worst_index = index;
        r->worst_analytic = a;
        r->worst_numeric = n;
    }
}

nmz_status nmz_net_gradcheck(nmz_net *net, const nmz_tensor *x, const nmz_tensor *t, double eps,
                             double atol, double rtol, nmz_gradcheck_result *result) {
    return nmz_net_gradcheck_mode(net, x, t, eps, atol, rtol, 1, result);
}

nmz_status nmz_net_gradcheck_mode(nmz_net *net, const nmz_tensor *x, const nmz_tensor *t, double eps,
                                  double atol, double rtol, int training, nmz_gradcheck_result *result) {
    nmz_gradcheck_result r;
    nmz_trace_fn saved_fn;
    nmz_tensor xp = NMZ_TENSOR_EMPTY, grads = NMZ_TENSOR_EMPTY;
    nmz_status s;
    size_t total = 0, off, e;
    long saved_step;
    double worst = 0.0;
    nmz_rng rng0;
    sigbuf sb = {NULL, NULL, 0, 0};
    int k;
    if (!net || !x || !t || !result) return nmz__fail(NMZ_ERR_ARG, "NULL argument");
    if (eps <= 0) eps = 1e-3;
    if (atol <= 0) atol = 1e-4;
    if (rtol <= 0) rtol = 2e-2;
    if ((s = nmz_net_build(net)) != NMZ_OK) return s;
    memset(&r, 0, sizeof r);
    saved_fn = net->trace;
    saved_step = net->step;
    rng0 = net->rng;
    net->trace = NULL;

    /* analytic gradients, copied out (they are overwritten by later passes) */
    net->rng = rng0;
    if ((s = nmz__net_forward(net, x, training, NULL)) != NMZ_OK) goto out;
    sb.n = signature(net, NULL); /* switch pattern at the unperturbed point */
    if (sb.n) {
        sb.base = malloc(sb.n * sizeof(float));
        sb.tmp = malloc(sb.n * sizeof(float));
        if (!sb.base || !sb.tmp) {
            s = nmz__fail(NMZ_ERR_NOMEM, "out of memory");
            goto out;
        }
        signature(net, sb.base);
    }
    net->training = training;
    s = nmz_net_backward(net, t, NULL);
    net->training = 0;
    if (s != NMZ_OK) goto out;
    for (k = 0; k < net->n_params; k++)
        total += nmz_layer_param_grad(net->layers[net->param_refs[k].layer], net->param_refs[k].index)->size;
    total += net->dacts[0].size;
    {
        int shape = (int)total;
        if ((s = nmz_tensor_init(&grads, 1, &shape)) != NMZ_OK) goto out;
    }
    for (k = 0, off = 0; k < net->n_params; k++) {
        const nmz_tensor *g = nmz_layer_param_grad(net->layers[net->param_refs[k].layer], net->param_refs[k].index);
        memcpy(grads.data + off, g->data, g->size * sizeof(float));
        off += g->size;
    }
    memcpy(grads.data + off, net->dacts[0].data, net->dacts[0].size * sizeof(float));

    /* parameters */
    for (k = 0, off = 0; k < net->n_params; k++) {
        nmz_layer *l = net->layers[net->param_refs[k].layer];
        int pi = net->param_refs[k].index;
        nmz_tensor *w = nmz_layer_param_value(l, pi);
        for (e = 0; e < w->size; e++) {
            float orig = w->data[e];
            double lp, lm;
            w->data[e] = (float)(orig + eps);
            sb.crossed = 0;
            s = loss_at(net, x, t, training, &rng0, &sb, &lp);
            w->data[e] = (float)(orig - eps);
            if (s == NMZ_OK) s = loss_at(net, x, t, training, &rng0, &sb, &lm);
            w->data[e] = orig;
            if (s != NMZ_OK) goto out;
            if (sb.crossed) {
                r.kinks++;
                continue;
            }
            compare(&r, &worst, grads.data[off + e], (lp - lm) / (2.0 * eps), net->param_refs[k].layer,
                    nmz_layer_param_name(l, pi), e, atol, rtol);
        }
        off += w->size;
    }
    /* input */
    if ((s = nmz_tensor_copy(&xp, x)) != NMZ_OK) goto out;
    for (e = 0; e < xp.size; e++) {
        float orig = xp.data[e];
        double lp, lm;
        xp.data[e] = (float)(orig + eps);
        sb.crossed = 0;
        s = loss_at(net, &xp, t, training, &rng0, &sb, &lp);
        xp.data[e] = (float)(orig - eps);
        if (s == NMZ_OK) s = loss_at(net, &xp, t, training, &rng0, &sb, &lm);
        xp.data[e] = orig;
        if (s != NMZ_OK) goto out;
        if (sb.crossed) {
            r.kinks++;
            continue;
        }
        compare(&r, &worst, grads.data[off + e], (lp - lm) / (2.0 * eps), -1, "x", e, atol, rtol);
    }
    s = NMZ_OK;
out:
    net->trace = saved_fn;
    net->step = saved_step;
    net->rng = rng0; /* the check consumes no randomness */
    net->have_forward = net->have_grads = 0;
    nmz_tensor_free(&xp);
    nmz_tensor_free(&grads);
    free(sb.base);
    free(sb.tmp);
    *result = r;
    return s;
}
