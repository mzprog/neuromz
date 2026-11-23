/* generate.c — sample text from a character-level model.
 *
 * The prime is fed as one sequence (from a zero state), then the model runs
 * one step at a time with its hidden state carried over: each step turns the
 * logits z of the last time step into probabilities
 *     p_k = exp(z_k / τ) / Σ_j exp(z_j / τ)          (τ = temperature)
 * keeps only the top-k (if set), renormalises, and samples the next
 * character with its own PCG32 (seeded by the caller, so a seed gives the
 * same text every time and generation never disturbs the training PRNG). */
#include "internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

nmz_gen_opts nmz_gen_defaults(void) {
    nmz_gen_opts o;
    o.prime = NULL;
    o.length = 200;
    o.temperature = 1.0f;
    o.top_k = 0;
    o.seed = 1;
    o.stop_at_end = 0;
    return o;
}

/* logits of the last time step of the last forward (softmax input if the
 * net ends in softmax, else the outputs themselves) */
static const float *last_logits(const nmz_net *net, int *V) {
    const nmz_tensor *z = nmz__net_fused_ce_possible(net) ? &net->acts[net->n_layers - 1] : &net->acts[net->n_layers];
    int T = z->shape[1];
    *V = z->shape[2];
    return z->data + (size_t)(T - 1) * (size_t)*V;
}

int nmz__sample_next(const float *z, int V, double temperature, int top_k, nmz_rng *rng, double *probs) {
    double mx = -HUGE_VAL, sum = 0.0, u, acc = 0.0;
    int k, kept = V;
    for (k = 0; k < V; k++)
        if (z[k] > mx) mx = z[k];
    for (k = 0; k < V; k++) sum += probs[k] = exp(((double)z[k] - mx) / temperature);
    if (top_k > 0 && top_k < V) {
        /* threshold = the top_k-th largest probability (ties at it are kept) */
        double *sorted = malloc((size_t)V * sizeof *sorted), thr;
        int i, j;
        if (sorted) {
            memcpy(sorted, probs, (size_t)V * sizeof *sorted);
            for (i = 0; i < top_k; i++) { /* partial selection sort, V is small */
                int best = i;
                for (j = i + 1; j < V; j++)
                    if (sorted[j] > sorted[best]) best = j;
                thr = sorted[best];
                sorted[best] = sorted[i];
                sorted[i] = thr;
            }
            thr = sorted[top_k - 1];
            free(sorted);
            sum = 0.0;
            kept = 0;
            for (k = 0; k < V; k++) {
                if (probs[k] < thr) probs[k] = 0.0;
                else kept++;
                sum += probs[k];
            }
        }
    }
    (void)kept;
    for (k = 0; k < V; k++) probs[k] /= sum;
    u = nmz_rng_uniform(rng);
    for (k = 0; k < V; k++) {
        acc += probs[k];
        if (u < acc) return k;
    }
    for (k = V - 1; k > 0 && probs[k] == 0.0; k--) {
    }
    return k; /* rounding: the last character with non-zero probability */
}

static nmz_status forward_ids(nmz_net *net, const int *ids, int n, nmz_tensor *x) {
    int shape[2], i;
    nmz_status s;
    shape[0] = 1;
    shape[1] = n;
    if ((s = nmz_tensor_reshape(x, 2, shape)) != NMZ_OK) return s;
    for (i = 0; i < n; i++) x->data[i] = (float)ids[i];
    return nmz__net_forward(net, x, 0, NULL);
}

nmz_status nmz_net_generate(nmz_net *net, const nmz_gen_opts *opts, char **out, size_t *n_chars) {
    nmz_gen_opts o = opts ? *opts : nmz_gen_defaults();
    const nmz_vocab *v;
    nmz_tensor x = NMZ_TENSOR_EMPTY;
    nmz_trace_fn saved = NULL;
    nmz_rng rng;
    uint32_t *pcp = NULL;
    size_t np = 0, bad = 0, cap = 64, len = 0, made = 0, i;
    int *ids = NULL, V, id, end_id;
    double *probs = NULL;
    char *buf = NULL;
    nmz_status s;
    if (!net || !out) return nmz__fail(NMZ_ERR_ARG, "NULL argument");
    *out = NULL;
    if (n_chars) *n_chars = 0;
    v = net->vocab;
    if (!v || !nmz_net_is_sequence(net) || net->n_layers < 1 || !(net->layers[0]->ops->flags & NMZ_LAYER_DISCRETE_INPUT))
        return nmz__fail(NMZ_ERR_STATE, "generation needs a text model (embedding input and a vocabulary)");
    if (o.length < 0 || o.length > 10000000) return nmz__fail(NMZ_ERR_ARG, "length must be 0..10000000");
    if (!(o.temperature > 0.0f) || !isfinite(o.temperature)) return nmz__fail(NMZ_ERR_ARG, "temperature must be > 0");
    if (o.top_k < 0) return nmz__fail(NMZ_ERR_ARG, "top-k must be >= 0");
    if ((s = nmz_net_build(net)) != NMZ_OK) return s;
    end_id = nmz_vocab_index(v, '\n');
    /* prime → ids */
    if (o.prime && *o.prime) {
        if ((s = nmz_utf8_decode(o.prime, strlen(o.prime), &pcp, &np, &bad)) != NMZ_OK) return s;
    } else {
        /* default: the end-of-item / newline token, else the most frequent character */
        int best = 0;
        for (id = 1; id < v->size; id++)
            if (v->counts[id] > v->counts[best]) best = id;
        pcp = malloc(sizeof *pcp);
        if (!pcp) return nmz__fail(NMZ_ERR_NOMEM, "out of memory");
        pcp[0] = end_id >= 0 ? '\n' : v->codepoints[best];
        np = 1;
    }
    ids = malloc((np ? np : 1) * sizeof *ids);
    probs = malloc((size_t)v->size * sizeof *probs);
    buf = malloc(cap);
    if (!ids || !probs || !buf) {
        s = nmz__fail(NMZ_ERR_NOMEM, "out of memory");
        goto done;
    }
    for (i = 0; i < np; i++) {
        ids[i] = nmz_vocab_index(v, pcp[i]);
        if (ids[i] < 0) {
            char u[5];
            nmz_utf8_encode(pcp[i], u);
            s = nmz__fail(NMZ_ERR_ARG, "the prime contains '%s' (U+%04X), which is not in the model's vocabulary", u,
                          (unsigned)pcp[i]);
            goto done;
        }
    }
    nmz_rng_seed(&rng, o.seed, 0x67656eULL /* "gen" */);
    saved = net->trace;
    net->trace = NULL;
    nmz_net_reset_state(net);
    net->carry = 1;
    if ((s = forward_ids(net, ids, (int)np, &x)) != NMZ_OK) goto done;
    buf[0] = '\0';
    while (made < (size_t)o.length) {
        const float *z = last_logits(net, &V);
        char u[5];
        int nb;
        if (V != v->size) {
            s = nmz__fail(NMZ_ERR_STATE, "the model has %d outputs but the vocabulary has %d characters", V, v->size);
            goto done;
        }
        id = nmz__sample_next(z, V, o.temperature, o.top_k, &rng, probs);
        nb = nmz_utf8_encode(v->codepoints[id], u);
        if (len + (size_t)nb + 1 > cap) {
            char *p = realloc(buf, cap *= 2);
            if (!p) {
                s = nmz__fail(NMZ_ERR_NOMEM, "out of memory");
                goto done;
            }
            buf = p;
        }
        memcpy(buf + len, u, (size_t)nb + 1);
        len += (size_t)nb;
        made++;
        if (o.stop_at_end && id == end_id) break;
        if ((s = forward_ids(net, &id, 1, &x)) != NMZ_OK) goto done;
    }
    s = NMZ_OK;
done:
    net->carry = 0;
    nmz_net_reset_state(net);
    if (saved) net->trace = saved;
    net->have_forward = net->have_grads = 0;
    nmz_tensor_free(&x);
    free(pcp);
    free(ids);
    free(probs);
    if (s == NMZ_OK) {
        *out = buf;
        if (n_chars) *n_chars = made;
    } else {
        free(buf);
    }
    return s;
}
