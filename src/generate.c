/* generate.c — sample text from a character-level model.
 *
 * The prime is fed as one sequence (from a zero state), then the model runs
 * one step at a time with its hidden state carried over: each step turns the
 * logits z of the last time step into probabilities
 *     p_k = exp(z_k / τ) / Σ_j exp(z_j / τ)          (τ = temperature)
 * keeps only the top-k (if set), renormalises, and samples the next
 * character with its own PCG32 (seeded by the caller, so a seed gives the
 * same text every time and generation never disturbs the training PRNG).
 *
 * Batches: nmz_net_generate_batch runs `count` independent rows at once
 * (one forward pass per step for all of them). Row r draws from its own
 * PCG32 seeded with seed + r, and every layer computes each row on its own,
 * so row r is exactly what nmz_net_generate with seed + r produces — the
 * output does not depend on the batch size. A row that has finished (end
 * of item, or length reached) leaves the batch: its row of every layer's
 * carried state is removed, so no work is spent on padding. */
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

/* logits of the last time step of row r of the last forward (softmax
 * input if the net ends in softmax, else the outputs themselves) */
static const float *last_logits(const nmz_net *net, int r, int *V) {
    const nmz_tensor *z = nmz__net_fused_ce_possible(net) ? &net->acts[net->n_layers - 1] : &net->acts[net->n_layers];
    int T = z->shape[1];
    *V = z->shape[2];
    return z->data + ((size_t)r * T + (size_t)(T - 1)) * (size_t)*V;
}

int nmz__sample_next(const float *z, int V, double temperature, int top_k, nmz_rng *rng, double *probs) {
    double mx = -HUGE_VAL, sum = 0.0, u, acc = 0.0;
    int k;
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
            for (k = 0; k < V; k++) {
                if (probs[k] < thr) probs[k] = 0.0;
                sum += probs[k];
            }
        }
    }
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

/* Keep only the streams keep[0..n) (old row numbers, ascending) in every
 * layer's carried state. A state tensor holds one row per stream ([B, ...],
 * DESIGN §3), so this works for any recurrent layer without knowing it. */
static void compact_states(nmz_net *net, int B, const int *keep, int n) {
    int i, k;
    for (i = 0; i < net->n_layers; i++) {
        nmz_layer *l = net->layers[i];
        nmz_tensor *st = l->ops->state ? l->ops->state(l) : NULL;
        size_t per;
        int shape[NMZ_MAX_RANK];
        if (!st || st->size == 0 || st->rank < 1 || st->shape[0] != B) continue;
        per = st->size / (size_t)B;
        for (k = 0; k < n; k++)
            if (keep[k] != k) memmove(st->data + (size_t)k * per, st->data + (size_t)keep[k] * per, per * sizeof(float));
        memcpy(shape, st->shape, sizeof shape);
        shape[0] = n;
        nmz_tensor_reshape(st, st->rank, shape); /* smaller: keeps the data */
    }
}

/* one growing UTF-8 output per row */
typedef struct gen_row {
    char *buf;
    size_t len, cap, made;
    int done;
    nmz_rng rng;
} gen_row;

static nmz_status row_append(gen_row *g, uint32_t cp) {
    char u[5];
    int nb = nmz_utf8_encode(cp, u);
    if (g->len + (size_t)nb + 1 > g->cap) {
        size_t cap = g->cap ? g->cap * 2 : 64;
        char *p = realloc(g->buf, cap);
        if (!p) return nmz__fail(NMZ_ERR_NOMEM, "out of memory");
        g->buf = p;
        g->cap = cap;
    }
    memcpy(g->buf + g->len, u, (size_t)nb + 1);
    g->len += (size_t)nb;
    g->made++;
    return NMZ_OK;
}

nmz_status nmz_net_generate_batch(nmz_net *net, const nmz_gen_opts *opts, int count, char **outs, size_t *n_chars) {
    nmz_gen_opts o = opts ? *opts : nmz_gen_defaults();
    const nmz_vocab *v;
    nmz_tensor x = NMZ_TENSOR_EMPTY;
    nmz_trace_fn saved = NULL;
    uint32_t *pcp = NULL;
    size_t np = 0, bad = 0, i;
    int *ids = NULL, *act = NULL, *keep = NULL, V, id, end_id, r, n_act, k;
    double *probs = NULL;
    gen_row *rows = NULL;
    nmz_status s;
    if (!net || !outs) return nmz__fail(NMZ_ERR_ARG, "NULL argument");
    if (count < 1 || count > 100000) return nmz__fail(NMZ_ERR_ARG, "count must be 1..100000");
    for (r = 0; r < count; r++) outs[r] = NULL;
    if (n_chars)
        for (r = 0; r < count; r++) n_chars[r] = 0;
    v = net->vocab;
    if (!v || !nmz_net_is_sequence(net) || net->n_layers < 1 || !(net->layers[0]->ops->flags & NMZ_LAYER_DISCRETE_INPUT))
        return nmz__fail(NMZ_ERR_STATE, "generation needs a text model (embedding input and a vocabulary)");
    if (o.length < 0 || o.length > 10000000) return nmz__fail(NMZ_ERR_ARG, "length must be 0..10000000");
    if (!(o.temperature > 0.0f) || !isfinite(o.temperature)) return nmz__fail(NMZ_ERR_ARG, "temperature must be > 0");
    if (o.top_k < 0) return nmz__fail(NMZ_ERR_ARG, "top-k must be >= 0");
    if ((s = nmz_net_build(net)) != NMZ_OK) return s;
    end_id = nmz_vocab_index(v, '\n');
    /* prime → ids (normalised like the training text, e.g. tashkeel stripped) */
    if (o.prime && *o.prime) {
        if ((s = nmz_utf8_decode(o.prime, strlen(o.prime), &pcp, &np, &bad)) != NMZ_OK) return s;
        np = nmz_text_normalize(pcp, np, v->normalize);
    }
    if (np == 0) {
        /* default: the end-of-item / newline token, else the most frequent character */
        int best = 0;
        for (id = 1; id < v->size; id++)
            if (v->counts[id] > v->counts[best]) best = id;
        free(pcp);
        pcp = malloc(sizeof *pcp);
        if (!pcp) return nmz__fail(NMZ_ERR_NOMEM, "out of memory");
        pcp[0] = end_id >= 0 ? '\n' : v->codepoints[best];
        np = 1;
    }
    ids = malloc(np * (size_t)count * sizeof *ids);
    probs = malloc((size_t)v->size * sizeof *probs);
    rows = calloc((size_t)count, sizeof *rows);
    act = malloc((size_t)count * sizeof *act);
    keep = malloc((size_t)count * sizeof *keep);
    if (!ids || !probs || !rows || !act || !keep) {
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
    for (r = 1; r < count; r++) memcpy(ids + (size_t)r * np, ids, np * sizeof *ids); /* every row: same prime */
    for (r = 0; r < count; r++) { /* row r = what a single run with seed + r would give */
        nmz_rng_seed(&rows[r].rng, o.seed + (uint64_t)r, 0x67656eULL /* "gen" */);
        if ((s = row_append(&rows[r], 0)) != NMZ_OK) goto done; /* allocate, "" */
        rows[r].len = rows[r].made = 0;
        rows[r].buf[0] = '\0';
    }
    saved = net->trace;
    net->trace = NULL;
    nmz_net_reset_state(net);
    net->carry = 1;
    {
        int shape[2];
        shape[0] = count;
        shape[1] = (int)np;
        if ((s = nmz_tensor_reshape(&x, 2, shape)) != NMZ_OK) goto done;
        for (i = 0; i < np * (size_t)count; i++) x.data[i] = (float)ids[i];
        if ((s = nmz__net_forward(net, &x, 0, NULL)) != NMZ_OK) goto done;
    }
    /* act[k] = the output row of batch row k; finished rows leave the batch */
    n_act = o.length > 0 ? count : 0;
    for (r = 0; r < n_act; r++) act[r] = r;
    while (n_act > 0) {
        int n_keep = 0;
        for (k = 0; k < n_act; k++) {
            gen_row *g = &rows[act[k]];
            const float *z = last_logits(net, k, &V);
            if (V != v->size) {
                s = nmz__fail(NMZ_ERR_STATE, "the model has %d outputs but the vocabulary has %d characters", V, v->size);
                goto done;
            }
            id = nmz__sample_next(z, V, o.temperature, o.top_k, &g->rng, probs);
            if ((s = row_append(g, v->codepoints[id])) != NMZ_OK) goto done;
            if (g->made >= (size_t)o.length || (o.stop_at_end && id == end_id)) {
                g->done = 1;
                continue;
            }
            keep[n_keep] = k;
            x.data[n_keep++] = (float)id; /* the next input of this row (n_keep <= k: safe in place) */
        }
        if (n_keep == 0) break;
        if (n_keep < n_act) { /* drop finished rows: no wasted work on them */
            compact_states(net, n_act, keep, n_keep);
            for (k = 0; k < n_keep; k++) act[k] = act[keep[k]];
        }
        n_act = n_keep;
        {
            int shape[2];
            shape[0] = n_act;
            shape[1] = 1;
            if ((s = nmz_tensor_reshape(&x, 2, shape)) != NMZ_OK) goto done;
            if ((s = nmz__net_forward(net, &x, 0, NULL)) != NMZ_OK) goto done;
        }
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
    free(act);
    free(keep);
    free(probs);
    if (rows) {
        for (r = 0; r < count; r++) {
            if (s == NMZ_OK) {
                outs[r] = rows[r].buf;
                if (n_chars) n_chars[r] = rows[r].made;
            } else {
                free(rows[r].buf);
            }
        }
        free(rows);
    }
    return s;
}

nmz_status nmz_net_generate(nmz_net *net, const nmz_gen_opts *opts, char **out, size_t *n_chars) {
    if (!out) return nmz__fail(NMZ_ERR_ARG, "NULL argument");
    return nmz_net_generate_batch(net, opts, 1, out, n_chars);
}
