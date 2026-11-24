/* print_attn.c — what a Transformer's attention does, with real numbers.
 *
 *   trace --text "..." on a model with attention prints:
 *     1. the input, every character's embedding e_t and position vector p_t
 *     2. ONE QUERY POSITION, one layer, one head: the query q, every earlier
 *        key k_j, the raw score q·k_j/√d, which later positions the mask
 *        blocks, the softmax weights as bars labelled with the characters,
 *        why the winner wins (the largest terms of q·k), and the weighted sum
 *        of the values
 *     3. the same position in every head of that layer
 *     4. a heatmap of query × key positions with the characters on both axes
 *     5. what every head mostly attends to — inferred from the measured
 *        weights (previous character, itself, start of the word, …)
 *     6. the residual stream at that position: how much each block's
 *        attention and feed-forward change it
 *     7. the next-character prediction, the direct gradient path, the update
 *   `show attention` prints 4 and 5 for any text (DESIGN decisions 99-101).
 *
 * Arabic on the heatmap axes: one character per column. Terminals that apply
 * the bidirectional algorithm would reverse a run of Arabic letters and join
 * them into one word, so every Arabic letter is followed by ZWNJ (U+200C: do
 * not join) and LRM (U+200E: left-to-right), and lines start with LRM — both
 * are zero-width. Combining marks (harakat) get a dotted circle, so they take
 * one column. --ascii prints '.' for every non-ASCII character instead. */
#include "cli.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static const char *nm(double v) {
    static char pool[64][32];
    static int k = 0;
    char *b = pool[k++ % 64];
    return fmt_num(b, sizeof pool[0], v);
}

static const char *n3(double v) {
    static char pool[64][24];
    static int k = 0;
    char *b = pool[k++ % 64];
    if (v != 0 && (fabs(v) < 1e-3 || fabs(v) >= 1e4))
        snprintf(b, sizeof pool[0], "%.2e", v);
    else
        snprintf(b, sizeof pool[0], "%.3f", v);
    return b;
}

static void section(const char *title) {
    printf("\n");
    print_rule();
    printf("%s\n", title);
}

/* ------------------------------------------------------------ characters */

static int is_arabic(uint32_t cp) {
    return (cp >= 0x0600 && cp <= 0x06FF) || (cp >= 0x0750 && cp <= 0x077F) || (cp >= 0x08A0 && cp <= 0x08FF) ||
           (cp >= 0xFB50 && cp <= 0xFDFF) || (cp >= 0xFE70 && cp <= 0xFEFF);
}

static int is_mark(uint32_t cp) {
    return (cp >= 0x0300 && cp <= 0x036F) || (cp >= 0x064B && cp <= 0x065F) || cp == 0x0670 ||
           (cp >= 0x06D6 && cp <= 0x06ED);
}

static uint32_t cp_of(const nmz_vocab *v, int id) { return v && id >= 0 && id < v->size ? v->codepoints[id] : '?'; }

/* exactly one display column */
void attn_put_char(const nmz_vocab *v, int id) {
    uint32_t cp = cp_of(v, id);
    char u[5];
    if (print_is_ascii()) {
        putchar(cp == ' ' ? '_' : cp == '\n' ? '~' : cp > 0x20 && cp < 0x7f ? (int)cp : '.');
        return;
    }
    if (cp == ' ') fputs("␣", stdout);
    else if (cp == '\n') fputs("⏎", stdout);
    else if (cp < 0x20 || cp == 0x7f) putchar('.');
    else if (is_mark(cp)) {
        nmz_utf8_encode(cp, u);
        printf("◌%s\xe2\x80\x8e", u);
    } else if (is_arabic(cp)) {
        nmz_utf8_encode(cp, u);
        printf("%s\xe2\x80\x8c\xe2\x80\x8e", u); /* ZWNJ + LRM: no joining, no reordering */
    } else {
        nmz_utf8_encode(cp, u);
        fputs(u, stdout);
    }
}

void attn_line_start(void) {
    if (!print_is_ascii()) fputs("\xe2\x80\x8e", stdout); /* LRM: the line is left-to-right */
}

/* a character in quotes for prose, e.g. 'e' (Arabic kept LTR-safe) */
static void put_quoted(const nmz_vocab *v, int id) {
    putchar('\'');
    attn_put_char(v, id);
    putchar('\'');
}

static int is_sep(uint32_t cp) {
    return cp == ' ' || cp == '\n' || cp == '\t' || (cp < 0x80 && cp > 0x20 && !((cp >= '0' && cp <= '9') ||
                                                                                (cp >= 'A' && cp <= 'Z') ||
                                                                                (cp >= 'a' && cp <= 'z') || cp == '\'')) ||
           cp == 0x060C || cp == 0x061B || cp == 0x061F || cp == 0x06D4 || cp == 0xAB || cp == 0xBB;
}

/* ------------------------------------------------------- finding events */

/* the n-th (1-based) layer with an attention event; *idx gets its net index */
static const nmz_trace_event *attn_event(const nmz_trace_recorder *rec, int n, int *count) {
    size_t i;
    int seen = -1, k = 0;
    const nmz_trace_event *hit = NULL;
    for (i = 0; i < nmz_trace_recorder_count(rec); i++) {
        const nmz_trace_event *ev = nmz_trace_recorder_get(rec, i);
        if (ev->phase != NMZ_TRACE_FORWARD || strcmp(ev->layer_type, "attention")) continue;
        if (ev->layer != seen) {
            seen = ev->layer;
            k++;
            if (k == n) hit = ev;
        }
    }
    if (count) *count = k;
    return hit;
}

static const nmz_trace_event *find_ev(const nmz_trace_recorder *rec, nmz_trace_phase ph, const char *type, int nth) {
    size_t i;
    for (i = 0; i < nmz_trace_recorder_count(rec); i++) {
        const nmz_trace_event *ev = nmz_trace_recorder_get(rec, i);
        if (ev->phase == ph && !strcmp(ev->layer_type, type) && nth-- == 0) return ev;
    }
    return NULL;
}

static double norm(const float *v, int n) {
    double s = 0;
    int i;
    for (i = 0; i < n; i++) s += (double)v[i] * v[i];
    return sqrt(s);
}

static void vec(const float *v, int n, int show) {
    int i;
    printf("[");
    for (i = 0; i < n && i < show; i++) printf("%s%s", i ? ", " : "", n3(v[i]));
    printf(n > show ? ", ...]" : "]");
}

static void wbar(double w, int width) {
    int n = (int)(w * width + 0.5), i;
    for (i = 0; i < n; i++) fputs(sym("█", "#"), stdout);
    if (n == 0 && w > 0.005) fputs(sym("▏", "|"), stdout);
}

/* ------------------------------------------------------------- patterns */

enum { P_SELF, P_PREV, P_PREV2, P_SAME, P_WORD, P_SEP, P_FIRST, P_FAR, N_PAT };
static const char *const k_pat[N_PAT] = {"itself",
                                         "the previous character",
                                         "the character two back",
                                         "earlier copies of the same character",
                                         "the start of the current word",
                                         "the last space or punctuation",
                                         "the first position of the window",
                                         "far back (more than 16 positions)"};

typedef struct head_acc {
    int layer, heads;
    double *mass, *base, *ent, *top; /* [heads × N_PAT], [heads × N_PAT], [heads], [heads] */
    long n;
} head_acc;

struct attn_pats {
    head_acc *a;
    int n;
};

attn_pats *attn_pats_new(void) { return calloc(1, sizeof(attn_pats)); }

void attn_pats_free(attn_pats *ap) {
    int i;
    if (!ap) return;
    for (i = 0; i < ap->n; i++) {
        free(ap->a[i].mass);
        free(ap->a[i].base);
        free(ap->a[i].ent);
        free(ap->a[i].top);
    }
    free(ap->a);
    free(ap);
}

/* key positions of pattern p for query t: 1 if j belongs to it */
static int in_pat(int p, const nmz_vocab *v, const int *ids, int t, int j, int word, int sep) {
    switch (p) {
    case P_SELF: return j == t;
    case P_PREV: return j == t - 1;
    case P_PREV2: return j == t - 2;
    case P_SAME: return j < t && ids[j] == ids[t];
    case P_WORD: return j == word && word >= 0;
    case P_SEP: return j == sep && sep >= 0;
    case P_FIRST: return j == 0;
    case P_FAR: return j < t - 16;
    default: (void)v; return 0;
    }
}

/* add one attention event over ids[0..T) (one window, positions from 0) */
int attn_pats_add(attn_pats *ap, const nmz_trace_event *ev, const int *ids, const nmz_vocab *v) {
    const nmz_tensor *w = nmz_trace_find(ev, "weights");
    head_acc *a = NULL;
    int i, H, T, Tk, b, h, t, j, p;
    if (!w || w->rank != 4) return 0;
    H = w->shape[1];
    T = w->shape[2];
    Tk = w->shape[3];
    if (Tk != T) return 0; /* a cached pass: positions are not those of ids */
    for (i = 0; i < ap->n; i++)
        if (ap->a[i].layer == ev->layer) a = &ap->a[i];
    if (!a) {
        head_acc *na = realloc(ap->a, (size_t)(ap->n + 1) * sizeof *na);
        if (!na) return 1;
        ap->a = na;
        a = &ap->a[ap->n++];
        memset(a, 0, sizeof *a);
        a->layer = ev->layer;
        a->heads = H;
        a->mass = calloc((size_t)H * N_PAT, sizeof *a->mass);
        a->base = calloc((size_t)H * N_PAT, sizeof *a->base);
        a->ent = calloc((size_t)H, sizeof *a->ent);
        a->top = calloc((size_t)H, sizeof *a->top);
        if (!a->mass || !a->base || !a->ent || !a->top) return 1;
    }
    for (b = 0; b < w->shape[0]; b++) {
        const int *id = ids + (size_t)b * T;
        int word = -1, sep = -1;
        for (t = 0; t < T; t++) {
            uint32_t c = cp_of(v, id[t]);
            if (t > 0 && is_sep(cp_of(v, id[t - 1]))) sep = t - 1;
            if (is_sep(c))
                word = -1; /* a separator is not part of a word */
            else if (t == 0 || is_sep(cp_of(v, id[t - 1])))
                word = t; /* a new word starts here */
            if (t < 2) continue; /* positions 0 and 1 have almost no choice: not counted */
            for (h = 0; h < H; h++) {
                const float *row = w->data + (((size_t)b * H + h) * T + t) * Tk;
                double e = 0, mx = 0;
                for (j = 0; j <= t; j++) {
                    if (row[j] > 0) e -= row[j] * log(row[j]);
                    if (row[j] > mx) mx = row[j];
                    for (p = 0; p < N_PAT; p++)
                        if (in_pat(p, v, id, t, j, word, sep)) {
                            a->mass[h * N_PAT + p] += row[j];
                            a->base[h * N_PAT + p] += 1.0 / (t + 1); /* what uniform attention would give */
                        }
                }
                a->ent[h] += e;
                a->top[h] += mx;
            }
            a->n++;
        }
    }
    return 0;
}

/* the pattern with the largest excess over uniform attention */
static int best_pattern(const head_acc *a, int h, int skip, double *mass, double *base) {
    int p, best = -1;
    double be = -1;
    for (p = 0; p < N_PAT; p++) {
        double m = a->mass[h * N_PAT + p] / a->n, u = a->base[h * N_PAT + p] / a->n;
        if (p == skip) continue;
        if (m - u > be) {
            be = m - u;
            best = p;
            *mass = m;
            *base = u;
        }
    }
    return best;
}

void attn_pats_print(const attn_pats *ap, const nmz_net *net) {
    int i, h;
    if (!ap || ap->n == 0) return;
    printf("  For every head: the share of its attention on each kind of earlier position, averaged over the\n"
           "  queries of this text, against what uniform attention would give. The label is the kind with the\n"
           "  largest excess; \"span\" = e^entropy, the number of positions it effectively looks at.\n\n");
    for (i = 0; i < ap->n; i++) {
        const head_acc *a = &ap->a[i];
        int nth = 0, k;
        for (k = 0; k <= a->layer && k < nmz_net_layer_count(net); k++)
            nth += nmz_layer_has_attention(nmz_net_layer((nmz_net *)net, k));
        if (a->n == 0) continue;
        for (h = 0; h < a->heads; h++) {
            double m1 = 0, u1 = 0, m2 = 0, u2 = 0;
            int p1 = best_pattern(a, h, -1, &m1, &u1), p2 = best_pattern(a, h, p1, &m2, &u2);
            double span = exp(a->ent[h] / a->n), top = a->top[h] / a->n;
            printf("  layer %d head %d  ", nth, h + 1);
            if (m1 - u1 < 0.1)
                printf("spread out: no single kind of position stands out (best: %s %.0f%%, uniform %.0f%%)", k_pat[p1],
                       100 * m1, 100 * u1);
            else
                printf("%-38s %3.0f%% (uniform %2.0f%%)", k_pat[p1], 100 * m1, 100 * u1);
            if (m1 - u1 >= 0.1 && p2 >= 0 && m2 - u2 >= 0.05) printf(";  then %s %.0f%%", k_pat[p2], 100 * m2);
            printf("\n                  span %.1f positions, top weight %.2f on average\n", span, top);
        }
    }
}

/* ------------------------------------------------------------- heatmap */

/* weights [H, T, Tk] of one row b; head h; positions [from, to) on both axes */
void attn_heatmap(const nmz_tensor *w, int b, int h, const int *ids, const nmz_vocab *v, int from, int to) {
    const int H = w->shape[1], T = w->shape[2], Tk = w->shape[3];
    int t, j;
    double mx = 1e-12;
    for (t = from; t < to; t++)
        for (j = from; j <= t && j < to; j++) {
            double a = w->data[(((size_t)b * H + h) * T + t) * Tk + j];
            if (a > mx) mx = a;
        }
    printf("  rows = query positions (the character being read), columns = key positions it may look at;\n"
           "  '%c' = 0 ... '%c' = %s (the largest weight shown); right of the diagonal is masked (the future)\n\n",
           heat_char(0, 0, 1), heat_char(1, 0, 1), n3(mx));
    if (to - from > 10) { /* a ruler: tens and units of the key position */
        attn_line_start();
        printf("  %8s ", "");
        for (j = from; j < to; j++) putchar(j % 10 == 0 ? (char)('0' + (j / 10) % 10) : ' ');
        printf("\n");
    }
    attn_line_start();
    printf("  %8s ", "");
    for (j = from; j < to; j++) attn_put_char(v, ids[j]);
    printf("\n");
    for (t = from; t < to; t++) {
        attn_line_start();
        printf("  %4d ", t);
        attn_put_char(v, ids[t]);
        printf("   ");
        for (j = from; j < to; j++) {
            if (j > t) {
                putchar(' ');
                continue;
            }
            putchar(heat_char(w->data[(((size_t)b * H + h) * T + t) * Tk + j], 0, mx));
        }
        printf("\n");
    }
}

/* ----------------------------------------------------------- the trace */

static int g_rows = 24;

static void one_position(const nmz_trace_event *ev, const nmz_vocab *v, const int *ids, int p, int h, int full) {
    const nmz_tensor *q = nmz_trace_find(ev, "q"), *k = nmz_trace_find(ev, "k"), *vv = nmz_trace_find(ev, "v");
    const nmz_tensor *sc = nmz_trace_find(ev, "scores"), *w = nmz_trace_find(ev, "weights"),
                     *cx = nmz_trace_find(ev, "context");
    double dd = 0, hh = 0;
    int H, d, T, Tk, E, j, shown = 0, *order = NULL, n, best = 0, i;
    const float *qp, *wr, *sr;
    nmz_trace_scalar_get(ev, "head_dim", &dd);
    nmz_trace_scalar_get(ev, "heads", &hh);
    H = (int)hh;
    d = (int)dd;
    T = w->shape[2];
    Tk = w->shape[3];
    E = H * d;
    qp = q->data + (size_t)p * E + (size_t)h * d;
    wr = w->data + ((size_t)h * T + p) * Tk;
    sr = sc->data + ((size_t)h * T + p) * Tk;
    for (j = 1; j <= p; j++)
        if (wr[j] > wr[best]) best = j;
    printf("  query position %d = ", p);
    put_quoted(v, ids[p]);
    printf(" (the model has read up to here and predicts the next character)\n");
    printf("  q = x%sW_q + b_q   (%d numbers per head; head %d's part):  q = ", sym("·", "*"), d, h + 1);
    vec(qp, d, 8);
    printf("\n\n  for every position j <= %d:  score s_j = q%sk_j / %s%d = q%sk_j / %s,  weight a_j = e^s_j / %s e^s_i\n\n",
           p, sym("·", "."), sym("√", "sqrt "), d, sym("·", "."), n3(sqrt((double)d)), sym("Σ", "sum"));
    printf("     j  char  k_j (first 3)                     score s_j   weight a_j\n");
    n = p + 1;
    order = malloc((size_t)n * sizeof *order);
    if (!order) return;
    for (j = 0; j < n; j++) order[j] = j;
    if (!full && n > g_rows) { /* keep the g_rows heaviest, in position order */
        for (i = 0; i < g_rows; i++) {
            int m = i, t2;
            for (j = i + 1; j < n; j++)
                if (wr[order[j]] > wr[order[m]]) m = j;
            t2 = order[i], order[i] = order[m], order[m] = t2;
        }
        for (i = 1; i < g_rows; i++) { /* insertion sort by position */
            int key = order[i];
            for (j = i - 1; j >= 0 && order[j] > key; j--) order[j + 1] = order[j];
            order[j + 1] = key;
        }
        shown = g_rows;
    } else {
        shown = n;
    }
    for (i = 0; i < shown; i++) {
        const float *kj;
        char kb[64];
        j = order[i];
        kj = k->data + (size_t)j * E + (size_t)h * d;
        snprintf(kb, sizeof kb, "[%s, %s, %s%s]", n3(kj[0]), d > 1 ? n3(kj[1]) : "", d > 2 ? n3(kj[2]) : "",
                 d > 3 ? ", ..." : "");
        attn_line_start();
        printf("  %4d  ", j);
        attn_put_char(v, ids[j]);
        printf("     %-32s %9s   %8s  ", kb, n3(sr[j]), n3(wr[j]));
        wbar(wr[j], 30);
        printf(j == best ? "  <- largest\n" : "\n");
    }
    if (shown < n) printf("  (%d smaller weights not shown; --full lists all %d positions)\n", n - shown, n);
    if (p + 1 < T) {
        printf("  %4d..%-4d masked: ", p + 1, T - 1);
        for (j = p + 1; j < T && j < p + 21; j++) attn_put_char(v, ids[j]);
        printf("%s -> score -%s, weight 0 (a position may not look at the future)\n", T > p + 21 ? "..." : "",
               sym("∞", "inf"));
    }
    free(order);
    { /* why the winner wins: the largest terms of q·k */
        const float *kb = k->data + (size_t)best * E + (size_t)h * d;
        int top[3] = {-1, -1, -1}, m, r;
        printf("\n  why ");
        put_quoted(v, ids[best]);
        printf(" at %d: q%sk = %s q_d%sk_d = %s; the largest terms:", best, sym("·", "."), sym("Σ", "sum"),
               sym("·", "*"), n3(sr[best] * sqrt((double)d)));
        for (r = 0; r < 3 && r < d; r++) {
            int bm = -1;
            for (m = 0; m < d; m++) {
                if (m == top[0] || m == top[1]) continue;
                if (bm < 0 || fabs((double)qp[m] * kb[m]) > fabs((double)qp[bm] * kb[bm])) bm = m;
            }
            top[r] = bm;
            printf("%s d=%d: %s%s%s = %s", r ? "," : "", bm + 1, n3(qp[bm]), sym("·", "*"), n3(kb[bm]),
                   n3((double)qp[bm] * kb[bm]));
        }
        printf("\n");
    }
    { /* the weighted sum */
        const float *c = cx->data + (size_t)p * E + (size_t)h * d;
        int top[3] = {-1, -1, -1}, r, m;
        printf("\n  context c = %s_j a_j%sv_j  (the values, mixed by the weights) = ", sym("Σ", "sum"), sym("·", "*"));
        vec(c, d, 8);
        printf("\n  the largest contributions a_j%sv_j:\n", sym("·", "*"));
        for (r = 0; r < 3 && r <= p; r++) {
            int bm = -1;
            for (m = 0; m <= p; m++) {
                if (m == top[0] || m == top[1]) continue;
                if (bm < 0 || wr[m] > wr[bm]) bm = m;
            }
            top[r] = bm;
            {
                const float *vj = vv->data + (size_t)bm * E + (size_t)h * d;
                float tmp[8];
                int q2;
                for (q2 = 0; q2 < d && q2 < 8; q2++) tmp[q2] = wr[bm] * vj[q2];
                attn_line_start();
                printf("    %s %s%s v_%d = ", n3(wr[bm]), sym("·", "*"), "", bm);
                vec(tmp, d < 8 ? d : 8, 4);
                printf("   (");
                attn_put_char(v, ids[bm]);
                printf(")\n");
            }
        }
    }
}

static void all_heads(const nmz_trace_event *ev, const nmz_vocab *v, const int *ids, int p) {
    const nmz_tensor *w = nmz_trace_find(ev, "weights");
    int H = w->shape[1], T = w->shape[2], Tk = w->shape[3], h, r, j;
    for (h = 0; h < H; h++) {
        const float *wr = w->data + ((size_t)h * T + p) * Tk;
        int top[3] = {-1, -1, -1};
        attn_line_start();
        printf("  head %d:", h + 1);
        for (r = 0; r < 3 && r <= p; r++) {
            int bm = -1;
            for (j = 0; j <= p; j++) {
                if (j == top[0] || j == top[1]) continue;
                if (bm < 0 || wr[j] > wr[bm]) bm = j;
            }
            top[r] = bm;
            printf("   ");
            put_quoted(v, ids[bm]);
            printf("@%d %s", bm, n3(wr[bm]));
        }
        printf("\n");
    }
}

static void residual(const nmz_trace_recorder *rec, int p) {
    int k;
    const nmz_trace_event *b;
    printf("  each block adds to the vector it receives (the residual stream): h = x + attn, y = h + ffn\n\n");
    printf("  %-8s %8s %10s %8s %10s %8s %10s %12s\n", "block", "|x|", "+|attn|", "|h|", "+|ffn|", "|y|",
           "cos(x,y)", "attn share");
    for (k = 0; (b = find_ev(rec, NMZ_TRACE_FORWARD, "block", k)) != NULL; k++) {
        const nmz_tensor *x = nmz_trace_find(b, "x"), *ao = nmz_trace_find(b, "attn_out"), *hh = nmz_trace_find(b, "h"),
                         *fo = nmz_trace_find(b, "ffn_out"), *y = nmz_trace_find(b, "y");
        int D = x->shape[2], i;
        const float *xv = x->data + (size_t)p * D, *yv = y->data + (size_t)p * D;
        double na = norm(ao->data + (size_t)p * D, D), nf = norm(fo->data + (size_t)p * D, D), dot = 0;
        for (i = 0; i < D; i++) dot += (double)xv[i] * yv[i];
        printf("  %-8d %8s %10s %8s %10s %8s %10s %11.0f%%\n", k + 1, n3(norm(xv, D)), n3(na),
               n3(norm(hh->data + (size_t)p * D, D)), n3(nf), n3(norm(yv, D)), n3(dot / (norm(xv, D) * norm(yv, D) + 1e-30)),
               100.0 * na / (na + nf + 1e-30));
    }
    printf("  (|.| = length of the vector; cos(x,y) near 1 = the block only nudged its direction)\n");
}

void print_attn_trace(const nmz_trace_recorder *rec, const nmz_vocab *v, const print_opts *o, int pos, int attn_n,
                      int head, const double *grad_e, const nmz_net *net) {
    const nmz_trace_event *emb = find_ev(rec, NMZ_TRACE_FORWARD, "embedding", 0);
    const nmz_trace_event *pe = find_ev(rec, NMZ_TRACE_FORWARD, "posenc", 0);
    const nmz_trace_event *loss = find_ev(rec, NMZ_TRACE_LOSS, "loss", 0), *ev, *soft = NULL;
    const nmz_tensor *idt, *e, *P = NULL, *tg = NULL, *pr = NULL;
    int T, D, t, k, *ids, *tgt, n_attn = 0, H, p, h;
    double L = NAN, hh = 0;
    char title[200];
    if (!emb) {
        printf("(no embedding layer: nothing to show)\n");
        return;
    }
    g_rows = o->full ? 1 << 30 : 24;
    idt = nmz_trace_find(emb, "ids");
    e = nmz_trace_find(emb, "e");
    T = idt->shape[1];
    D = e->shape[2];
    ids = malloc((size_t)T * sizeof *ids);
    tgt = malloc((size_t)T * sizeof *tgt);
    if (!ids || !tgt) {
        free(ids);
        free(tgt);
        return;
    }
    for (t = 0; t < T; t++) ids[t] = (int)idt->data[t];
    if (pe) P = nmz_trace_find(pe, "P");
    for (k = 0; (ev = find_ev(rec, NMZ_TRACE_FORWARD, "activation", k)) != NULL; k++)
        if (ev->detail && !strcmp(ev->detail, "softmax")) soft = ev;
    if (soft) pr = nmz_trace_find(soft, "a");
    if (loss) {
        tg = nmz_trace_find(loss, "t");
        nmz_trace_scalar_get(loss, "L", &L);
    }
    for (t = 0; t < T; t++) {
        tgt[t] = -1;
        if (tg)
            for (k = 0; k < v->size; k++)
                if (tg->data[(size_t)t * v->size + k] > 0.5f) tgt[t] = k;
    }
    ev = attn_event(rec, attn_n > 0 ? attn_n : 1, &n_attn);
    if (!ev) {
        printf("(no attention layer number %d; the model has %d)\n", attn_n, n_attn);
        free(ids);
        free(tgt);
        return;
    }
    nmz_trace_scalar_get(ev, "heads", &hh);
    H = (int)hh;
    p = pos >= 0 && pos < T ? pos : T - 1;
    if (head >= 0 && head < H) {
        h = head;
    } else { /* the most decisive head at that position */
        const nmz_tensor *w = nmz_trace_find(ev, "weights");
        double bm = -1;
        int j;
        h = 0;
        for (k = 0; k < H; k++)
            for (j = 0; j <= p; j++)
                if (w->data[((size_t)k * T + p) * w->shape[3] + j] > bm) bm = w->data[((size_t)k * T + p) * w->shape[3] + j], h = k;
    }

    printf("\n");
    attn_line_start();
    printf("input :");
    for (t = 0; t < T; t++) {
        putchar(' ');
        attn_put_char(v, ids[t]);
    }
    if (tg) {
        printf("\n");
        attn_line_start();
        printf("target:");
        for (t = 0; t < T; t++) {
            putchar(' ');
            if (tgt[t] >= 0) attn_put_char(v, tgt[t]);
        }
    }
    printf("\n(the target is the input shifted by one character: every position predicts the next one, at once)\n");

    section("EMBEDDING + POSITION   x_t = e_t + p_t: what the character is, plus where it is");
    printf("  e_t = W_emb[id_t] (%d numbers)%s\n", D,
           pe ? (strcmp(pe->detail, "learned") ? ";  p_t = sinusoidal position code (fixed)" : ";  p_t = learned row t of P")
              : ";  (no position encoding)");
    for (t = 0; t < T && t < (o->full ? T : 12); t++) {
        attn_line_start();
        printf("  t=%-3d ", t);
        attn_put_char(v, ids[t]);
        printf("  e = ");
        vec(e->data + (size_t)t * D, D, 4);
        if (P) {
            printf("   p = ");
            vec(P->data + (size_t)t * D, D, 4);
        }
        printf("\n");
    }
    if (!o->full && T > 12) printf("  %s %d more positions (use --full)\n", sym("…", "..."), T - 12);

    snprintf(title, sizeof title, "ATTENTION AT ONE POSITION   layer %d of %d, head %d of %d (--layer, --head, --pos)",
             attn_n > 0 ? attn_n : 1, n_attn, h + 1, H);
    section(title);
    one_position(ev, v, ids, p, h, o->full);

    snprintf(title, sizeof title, "ALL HEADS OF LAYER %d AT POSITION %d   the three positions each head weighs most",
             attn_n > 0 ? attn_n : 1, p);
    section(title);
    all_heads(ev, v, ids, p);

    snprintf(title, sizeof title, "ATTENTION HEATMAP   layer %d, head %d", attn_n > 0 ? attn_n : 1, h + 1);
    section(title);
    {
        int from = !o->full && T > 64 ? T - 64 : 0;
        attn_heatmap(nmz_trace_find(ev, "weights"), 0, h, ids, v, from, T);
        if (from) printf("  (the last 64 positions; --full shows all %d)\n", T);
    }

    section("WHAT EACH HEAD ATTENDS TO   inferred from the weights on this text (`show attention --data FILE`\n"
            "for a longer text)");
    {
        attn_pats *ap = attn_pats_new();
        size_t i;
        if (ap) {
            for (i = 0; i < nmz_trace_recorder_count(rec); i++) {
                const nmz_trace_event *a = nmz_trace_recorder_get(rec, i);
                if (a->phase == NMZ_TRACE_FORWARD && !strcmp(a->layer_type, "attention")) attn_pats_add(ap, a, ids, v);
            }
            attn_pats_print(ap, net);
            attn_pats_free(ap);
        }
    }

    snprintf(title, sizeof title, "THE RESIDUAL STREAM AT POSITION %d", p);
    section(title);
    residual(rec, p);

    if (pr) {
        int V = pr->shape[2], idx[5], r, j;
        const float *row = pr->data + (size_t)p * V;
        section("THE PREDICTION   softmax of the output at that position");
        for (r = 0; r < 5 && r < V; r++) {
            idx[r] = -1;
            for (j = 0; j < V; j++) {
                int used = 0, q;
                for (q = 0; q < r; q++) used |= idx[q] == j;
                if (!used && (idx[r] < 0 || row[j] > row[idx[r]])) idx[r] = j;
            }
            attn_line_start();
            printf("  ");
            put_quoted(v, idx[r]);
            printf("  %s  ", n3(row[idx[r]]));
            wbar(row[idx[r]], 30);
            printf("%s\n", tgt[p] == idx[r] ? "   <- the actual next character" : "");
        }
        if (!isnan(L))
            printf("  L = mean over the %d positions of -log p(target) = %s   (perplexity %s)\n", T, nm(L), nm(exp(L)));
    }

    if (grad_e) {
        double ref = grad_e[T - 1] > 0 ? grad_e[T - 1] : 1;
        const nmz_tensor *w1 = nmz_trace_find(attn_event(rec, 1, NULL), "weights");
        section("THE GRADIENT PATH   the loss of the LAST prediction alone, at each position's input vector x_t");
        printf("  |dL/dx_t| relative to the last position. Attention connects the last position to every earlier\n"
               "  one in one step, so the gradient does not have to pass through the positions in between (an RNN\n"
               "  multiplies it by W_hh and tanh' once per step). How much arrives follows the attention weights.\n\n");
        printf("  %5s  %-4s %10s   %s\n", "back", "char", "|dL/dx_t|", "layer-1 weight of the last position (mean over heads)");
        for (k = 0; k < T; k++) {
            int tt = T - 1 - k, hq;
            double a = 0;
            if (!o->full && k > 12 && k != T - 1 && k % 8) continue;
            if (w1)
                for (hq = 0; hq < w1->shape[1]; hq++)
                    a += w1->data[((size_t)hq * T + (T - 1)) * w1->shape[3] + tt] / w1->shape[1];
            attn_line_start();
            printf("  %5d  ", k);
            attn_put_char(v, ids[tt]);
            printf("    %10s   %s  ", n3(grad_e[tt] / ref), n3(a));
            wbar(a, 30);
            printf("\n");
        }
        if (!o->full && T > 14) printf("  (every 8th step after 12; --full prints all)\n");
    }
    free(ids);
    free(tgt);
}
