/* print_seq.c — the trace of a character RNN, unrolled through time.
 *
 *   1. the sequence, and each character's embedding
 *   2. the unrolled RNN: for every time step the input character,
 *      h_t[j] = tanh(x_t·W_xh[:,j] + h_{t−1}·W_hh[:,j] + b[j]) with real numbers
 *      for one hidden unit j, and the top-5 next-character probabilities
 *   3. a heatmap of all hidden units over time
 *   4. backpropagation through time: how ∂L/∂h_t is assembled at every step
 *      and how much of it is passed on to h_{t−1}
 *   5. "vanishing gradients": the gradient of the LAST step's loss alone,
 *      followed back through time (from a separate backward pass)
 *   6. clipping and the weight update */
#include "cli.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static const char *nm(double v) {
    static char pool[48][32];
    static int k = 0;
    char *b = pool[k++ % 48];
    return fmt_num(b, sizeof pool[0], v);
}

static const char *ch(const nmz_vocab *v, int id) {
    static char pool[16][24];
    static int k = 0;
    char *b = pool[k++ % 16];
    if (!v || id < 0 || id >= v->size) {
        snprintf(b, sizeof pool[0], "#%d", id);
        return b;
    }
    return cli_char_label(v->codepoints[id], b, sizeof pool[0]);
}

/* label padded to `w` characters (not bytes) */
static const char *chp(const nmz_vocab *v, int id, int w) {
    static char pool[16][48];
    static int k = 0;
    char *b = pool[k++ % 16];
    const char *c = ch(v, id);
    int n = 0, i;
    for (i = 0; c[i]; i++) n += ((unsigned char)c[i] & 0xC0) != 0x80;
    snprintf(b, sizeof pool[0], "%s", c);
    for (i = (int)strlen(b); n < w && i < (int)sizeof pool[0] - 1; n++) b[i++] = ' ', b[i] = '\0';
    return b;
}

static double norm_row(const float *v, int n) {
    double s = 0;
    int i;
    for (i = 0; i < n; i++) s += (double)v[i] * v[i];
    return sqrt(s);
}

static double norm_t(const nmz_tensor *t) { return t ? norm_row(t->data, (int)t->size) : 0.0; }

static const nmz_trace_event *find(const nmz_trace_recorder *rec, nmz_trace_phase ph, const char *type, int time_step,
                                   int nth) {
    size_t i;
    for (i = 0; i < nmz_trace_recorder_count(rec); i++) {
        const nmz_trace_event *e = nmz_trace_recorder_get(rec, i);
        if (e->phase == ph && !strcmp(e->layer_type, type) && e->time_step == time_step && nth-- == 0) return e;
    }
    return NULL;
}

static void bar(double frac, int width) {
    int n = (int)(frac * width + 0.5), i;
    for (i = 0; i < n && i < width; i++) fputs(print_is_ascii() ? "#" : "█", stdout);
}

static void section(const char *title) {
    printf("\n");
    print_rule();
    printf("%s\n", title);
    print_rule();
}

/* top-5 of row t of probs [1, T, V] */
static void top5(const float *p, int V, int *idx, int k) {
    int i, j;
    for (i = 0; i < k; i++) idx[i] = -1;
    for (j = 0; j < V; j++)
        for (i = 0; i < k; i++)
            if (idx[i] < 0 || p[j] > p[idx[i]]) {
                int m;
                for (m = k - 1; m > i; m--) idx[m] = idx[m - 1];
                idx[i] = j;
                break;
            }
}

/* the unit whose activation changes most over the sequence */
static int busiest_unit(const nmz_tensor *h) {
    int T = h->shape[1], H = h->shape[2], j, t, best = 0;
    double best_range = -1;
    for (j = 0; j < H; j++) {
        double lo = 2, hi = -2;
        for (t = 0; t < T; t++) {
            double v = h->data[(size_t)t * H + j];
            if (v < lo) lo = v;
            if (v > hi) hi = v;
        }
        if (hi - lo > best_range) {
            best_range = hi - lo;
            best = j;
        }
    }
    return best;
}

void print_seq_trace(const nmz_trace_recorder *rec, const nmz_vocab *v, const print_opts *o, int applied, int unit,
                     const double *last_norms, int n_last) {
    const nmz_trace_event *emb = find(rec, NMZ_TRACE_FORWARD, "embedding", -1, 0);
    const nmz_trace_event *rnn = find(rec, NMZ_TRACE_FORWARD, "rnn", -1, 0);
    const nmz_trace_event *soft = find(rec, NMZ_TRACE_FORWARD, "activation", -1, 0);
    const nmz_trace_event *loss = find(rec, NMZ_TRACE_LOSS, "loss", -1, 0);
    const nmz_tensor *ids, *e, *h, *Wxh, *Whh, *b, *p = NULL, *tg = NULL;
    int T, D, H, V = v ? v->size : 0, t, j, k, rows;
    double L = NAN, total_nll = 0;
    if (!emb || !rnn) {
        printf("(this network has no embedding + rnn pair; showing nothing unrolled)\n");
        return;
    }
    ids = nmz_trace_find(emb, "ids");
    e = nmz_trace_find(emb, "e");
    h = nmz_trace_find(rnn, "h");
    Wxh = nmz_trace_find(rnn, "W_xh");
    Whh = nmz_trace_find(rnn, "W_hh");
    b = nmz_trace_find(rnn, "b");
    /* the last softmax forward is the output head */
    for (k = 0; (soft = find(rec, NMZ_TRACE_FORWARD, "activation", -1, k)) != NULL; k++)
        if (!strcmp(soft->detail, "softmax")) p = nmz_trace_find(soft, "a");
    if (loss) {
        tg = nmz_trace_find(loss, "t");
        nmz_trace_scalar_get(loss, "L", &L);
    }
    T = h->shape[1];
    H = h->shape[2];
    D = e->shape[2];
    if (unit < 0 || unit >= H) unit = busiest_unit(h);
    rows = o->max_rows > 0 && o->max_rows < T ? o->max_rows : T;

    /* ------------------------------------------------------------ sequence */
    printf("\ninput :");
    for (t = 0; t < T; t++) printf(" %s", ch(v, (int)ids->data[t]));
    if (tg) {
        printf("\ntarget:");
        for (t = 0; t < T; t++) {
            const float *r = tg->data + (size_t)t * V;
            int best = 0;
            for (k = 1; k < V; k++)
                if (r[k] > r[best]) best = k;
            printf(" %s", ch(v, best));
        }
    }
    printf("\n(the target is the input shifted by one character: at every step the net predicts the next one)\n");

    section("EMBEDDING   each character id selects one row of W (a learned vector)");
    printf("  e_t = W[id_t]        W is %d characters x %d numbers\n", V, D);
    for (t = 0; t < rows; t++) {
        printf("  t=%-3d %s id %-4d e = [", t, chp(v, (int)ids->data[t], 4), (int)ids->data[t]);
        for (k = 0; k < D && k < 6; k++) printf("%s%s", k ? ", " : "", nm(e->data[(size_t)t * D + k]));
        printf(D > 6 ? ", ...]\n" : "]\n");
    }
    if (rows < T) printf("  %s %d more steps (use --full)\n", sym("…", "..."), T - rows);

    /* ------------------------------------------------------------ unrolled */
    section("RNN UNROLLED THROUGH TIME   the same weights at every step; h carries the past forward");
    printf("  h_t = tanh(x_t%sW_xh + h_{t-1}%sW_hh + b)      shown for hidden unit j = %d of %d%s\n", sym("·", "*"),
           sym("·", "*"), unit + 1, H, o->at_set ? "" : " (the unit that changes most here; choose with --unit J)");
    printf("  x_t = e_t (from the embedding), h_{-1} = 0\n");
    for (t = 0; t < rows; t++) {
        const nmz_trace_event *st = find(rec, NMZ_TRACE_FORWARD, "rnn", t, 0);
        const nmz_tensor *xt = st ? nmz_trace_find(st, "x_t") : NULL, *hp = st ? nmz_trace_find(st, "h_prev") : NULL;
        double ax = 0, ah = 0, zj;
        int i;
        if (!xt || !hp) continue;
        for (i = 0; i < D; i++) ax += (double)xt->data[i] * Wxh->data[(size_t)i * H + unit];
        for (i = 0; i < H; i++) ah += (double)hp->data[i] * Whh->data[(size_t)i * H + unit];
        zj = ax + ah + b->data[unit];
        printf("\n  t=%-3d input %s h_t[%d] = tanh(%s + %s + %s) = tanh(%s) = %s\n", t, chp(v, (int)ids->data[t], 4),
               unit + 1, nm(ax), nm(ah), nm(b->data[unit]), nm(zj), nm(h->data[(size_t)t * H + unit]));
        printf("        %*s x_t%sW_xh[:,j]   h_{t-1}%sW_hh[:,j]   b[j]\n", 0, "", sym("·", "*"), sym("·", "*"));
        if (p && V > 0) {
            const float *pr = p->data + (size_t)t * V;
            int top[5], target = -1;
            if (tg) {
                const float *r = tg->data + (size_t)t * V;
                target = 0;
                for (k = 1; k < V; k++)
                    if (r[k] > r[target]) target = k;
            }
            top5(pr, V, top, V < 5 ? V : 5);
            for (k = 0; k < 5 && k < V; k++) {
                printf("        %s %s %5.1f%% ", k == 0 ? "next:" : "     ", chp(v, top[k], 4), 100.0 * pr[top[k]]);
                bar(pr[top[k]], 30);
                printf("%s\n", top[k] == target ? "   <- target" : "");
            }
            if (target >= 0) {
                double nll = -log(pr[target] > 1e-30 ? pr[target] : 1e-30);
                total_nll += nll;
                printf("        target %s: p = %s, loss -log p = %s\n", ch(v, target), nm(pr[target]), nm(nll));
            }
        }
    }
    if (rows < T) printf("\n  %s %d more steps (use --full)\n", sym("…", "..."), T - rows);
    if (!isnan(L))
        printf("\n  L = mean over the %d steps of -log p(target) = %s   (perplexity e^L = %s: as unsure as a choice "
               "among %.1f characters)\n",
               T, nm(L), nm(exp(L)), exp(L));

    /* ------------------------------------------------------------- heatmap */
    section("HIDDEN STATE OVER TIME   rows = hidden units, columns = time steps");
    {
        int units = o->full || H <= 24 ? H : 24;
        printf("  '%c' = -1 ... '%c' = 0 ... '%c' = +1   (tanh range)\n", heat_char(-1, -1, 1), heat_char(0, -1, 1),
               heat_char(1, -1, 1));
        printf("  %6s ", "");
        for (t = 0; t < T; t++) {
            char lbl[24];
            const char *c = ch(v, (int)ids->data[t]);
            snprintf(lbl, sizeof lbl, "%s", c);
            /* one column per step: show the character if it is a single glyph, else '.' */
            {
                int w = 0, q;
                for (q = 0; lbl[q]; q++) w += ((unsigned char)lbl[q] & 0xC0) != 0x80;
                fputs(w == 1 ? lbl : ".", stdout);
            }
        }
        printf("\n");
        for (j = 0; j < units; j++) {
            printf("  %s%4d ", j == unit ? ">" : " ", j + 1);
            for (t = 0; t < T; t++) putchar(heat_char(h->data[(size_t)t * H + j], -1, 1));
            printf("\n");
        }
        if (units < H) printf("  %s %d more units (use --full)\n", sym("…", "..."), H - units);
    }

    /* --------------------------------------------------------------- BPTT */
    section("BACKPROPAGATION THROUGH TIME   from the last step back to the first");
    printf("  at every step:  dL/dh_t = (from the output at t) + (from step t+1)\n");
    printf("                  %s_t = dL/dh_t %s (1 - h_t%s)            (through tanh)\n", sym("δ", "delta"),
           sym("⊙", "*"), sym("²", "^2"));
    printf("                  dL/dh_{t-1} = %s_t %s W_hh%s           (passed back to the previous step)\n",
           sym("δ", "delta"), sym("·", "*"), sym("ᵀ", "^T"));
    printf("  and every step adds to the SAME weight gradients: dL/dW_hh += h_{t-1}%s %s %s_t\n\n", sym("ᵀ", "^T"),
           sym("·", "*"), sym("δ", "delta"));
    printf("  %4s %-5s %12s %12s %12s %12s %12s\n", "t", "char", "|from out|", "|from t+1|", "|dL/dh_t|",
           print_is_ascii() ? "|delta_t|" : "|δ_t|", "|to t-1|");
    for (t = T - 1; t >= 0; t--) {
        const nmz_trace_event *st = find(rec, NMZ_TRACE_BACKWARD, "rnn", t, 0);
        if (!st) continue;
        if (o->max_rows > 0 && T - 1 - t >= o->max_rows) {
            printf("  %s %d more steps (use --full)\n", sym("…", "..."), t + 1);
            break;
        }
        printf("  %4d %s %12s %12s %12s %12s %12s\n", t, chp(v, (int)ids->data[t], 5),
               nm(norm_t(nmz_trace_find(st, "dL/dh_out"))), nm(norm_t(nmz_trace_find(st, "dL/dh_next"))),
               nm(norm_t(nmz_trace_find(st, "dL/dh"))), nm(norm_t(nmz_trace_find(st, "dL/dz"))),
               nm(norm_t(nmz_trace_find(st, "dL/dh_prev"))));
    }
    printf("  (|v| = L2 norm; at t = 0 the gradient to h_{-1} is dropped: that is where truncation cuts)\n");

    if (last_norms && n_last > 1) {
        double prod = 1.0, mx = 0;
        int steps = 0;
        section("VANISHING / EXPLODING GRADIENTS   the loss of the LAST character only, followed back in time");
        printf("  |dL_last/dh_t|: how strongly the last prediction can still teach step t\n\n");
        for (t = n_last - 1; t >= 0; t--)
            if (last_norms[t] > mx) mx = last_norms[t];
        for (t = n_last - 1; t >= 0; t--) {
            double r = t < n_last - 1 && last_norms[t + 1] > 0 ? last_norms[t] / last_norms[t + 1] : NAN;
            double lg = mx > 0 && last_norms[t] > 0 ? (log10(last_norms[t]) - log10(mx) + 6) / 6 : 0; /* 6 decades */
            if (o->max_rows > 0 && n_last - 1 - t >= 2 * o->max_rows) {
                printf("  %s\n", sym("…", "..."));
                break;
            }
            printf("  t=%-4d %s %11s  ", t, chp(v, (int)ids->data[t], 4), nm(last_norms[t]));
            if (isnan(r))
                printf("        ");
            else
                printf(" x%-6.3f", r);
            printf(" ");
            bar(lg < 0 ? 0 : lg, 36);
            printf("\n");
            if (!isnan(r) && r > 0) {
                prod *= r;
                steps++;
            }
        }
        if (steps > 0) {
            double per = pow(prod, 1.0 / steps);
            printf("\n  bars are on a log scale (6 decades). On average each step back multiplies the gradient by %.3f;\n",
                   per);
            if (per < 0.95)
                printf("  after %d steps only %.2g of it is left: a VANISHING gradient. A vanilla RNN therefore learns\n"
                       "  short-range patterns easily and long-range ones hardly at all - the problem LSTMs fix\n"
                       "  with an additive cell state.\n",
                       steps, prod);
            else if (per > 1.05)
                printf("  it GROWS here (x%.2g after %d steps): kept up over long sequences that is an EXPLODING\n"
                       "  gradient - what gradient clipping (the --clip norm) protects training from.\n",
                       prod, steps);
            else
                printf("  it stays roughly the same size here (neither vanishing nor exploding over %d steps).\n", steps);
            printf("  (each step multiplies by W_hh%s and by tanh' = 1 - h%s <= 1, which is why it usually shrinks)\n",
                   sym("ᵀ", "^T"), sym("²", "^2"));
        }
    }

    /* ---------------------------------------------- clipping and update */
    {
        const nmz_trace_event *clip = find(rec, NMZ_TRACE_UPDATE, "clip", -1, 0);
        size_t i;
        int header = 0;
        section(applied ? "GRADIENT CLIPPING AND WEIGHT UPDATE" : "GRADIENT CLIPPING AND WEIGHT UPDATE (preview only: nothing is changed; use `step`)");
        if (clip) {
            double n = 0, mxn = 0, sc = 1;
            nmz_trace_scalar_get(clip, "norm", &n);
            nmz_trace_scalar_get(clip, "max_norm", &mxn);
            nmz_trace_scalar_get(clip, "scale", &sc);
            if (sc < 1.0)
                printf("  global gradient norm sqrt(sum of all g%s) = %s > clip %s  ->  every gradient x %s\n",
                       sym("²", "^2"), nm(n), nm(mxn), nm(sc));
            else
                printf("  global gradient norm sqrt(sum of all g%s) = %s <= clip %s  ->  unchanged\n", sym("²", "^2"),
                       nm(n), nm(mxn));
        } else {
            printf("  (gradient clipping is off)\n");
        }
        for (i = 0; i < nmz_trace_recorder_count(rec); i++) {
            const nmz_trace_event *ev = nmz_trace_recorder_get(rec, i);
            if (ev->phase != NMZ_TRACE_UPDATE || !strcmp(ev->layer_type, "clip")) continue;
            if (!header && nmz_trace_find(ev, "m")) pr_adam_header(ev);
            header = 1;
            if (nmz_trace_find(ev, "m"))
                pr_adam_update(ev, o);
            else
                print_sgd_update(ev, o);
        }
    }
}
