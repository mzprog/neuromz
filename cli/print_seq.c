/* print_seq.c — a character model unrolled through time (rnn, lstm, gru).
 *
 *   1. the sequence, and each character's embedding
 *   2. the unrolled recurrent layer: for every time step the input character
 *      and, for one unit j, every formula with real numbers —
 *        rnn:  h_t[j] = tanh(x_t·W_xh[:,j] + h_{t−1}·W_hh[:,j] + b[j])
 *        lstm: the gates i, f, g, o, then c_t and h_t, with bars
 *        gru:  the gates z, r, the candidate ĥ, then h_t
 *      and the top-5 next-character probabilities
 *   3. heatmaps over time: all hidden units (rnn), or the chosen unit's
 *      gates and cell (lstm: i f o g c h; gru: z r ĥ h) plus the forget /
 *      update gate of every unit
 *   4. backpropagation through time: how ∂L/∂h_t (and ∂L/∂c_t) is assembled
 *      at every step and how much of it is passed on
 *   5. "vanishing gradients": the gradient of the LAST step's loss alone,
 *      followed back through time (from a separate backward pass) — for an
 *      LSTM along h and along the cell state c (the "gradient highway")
 *   6. clipping and the weight update
 * Stacked models: one recurrent layer is shown (the first, or --layer N). */
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

/* short number: 3 decimals, for the gate lines */
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

/* one glyph per column (heatmap headers): the character, or '.' */
static void put_col_char(const nmz_vocab *v, int id) {
    const char *c = ch(v, id);
    int w = 0, q;
    for (q = 0; c[q]; q++) w += ((unsigned char)c[q] & 0xC0) != 0x80;
    fputs(w == 1 ? c : ".", stdout);
}

static double norm_row(const float *v, int n) {
    double s = 0;
    int i;
    for (i = 0; i < n; i++) s += (double)v[i] * v[i];
    return sqrt(s);
}

static double norm_t(const nmz_tensor *t) { return t ? norm_row(t->data, (int)t->size) : 0.0; }

int seq_is_recurrent_type(const char *type) {
    return !strcmp(type, "rnn") || !strcmp(type, "lstm") || !strcmp(type, "gru");
}

/* the nth event of a phase/type/step; type NULL = any, layer < 0 = any */
static const nmz_trace_event *find(const nmz_trace_recorder *rec, nmz_trace_phase ph, const char *type, int layer,
                                   int time_step, int nth) {
    size_t i;
    for (i = 0; i < nmz_trace_recorder_count(rec); i++) {
        const nmz_trace_event *e = nmz_trace_recorder_get(rec, i);
        if (e->phase == ph && (!type || !strcmp(e->layer_type, type)) && (layer < 0 || e->layer == layer) &&
            e->time_step == time_step && nth-- == 0)
            return e;
    }
    return NULL;
}

/* the forward summary event of the chosen recurrent layer (layer < 0: the first) */
static const nmz_trace_event *find_recurrent(const nmz_trace_recorder *rec, int layer) {
    size_t i;
    for (i = 0; i < nmz_trace_recorder_count(rec); i++) {
        const nmz_trace_event *e = nmz_trace_recorder_get(rec, i);
        if (e->phase == NMZ_TRACE_FORWARD && e->time_step == -1 && seq_is_recurrent_type(e->layer_type) &&
            (layer < 0 || e->layer == layer))
            return e;
    }
    return NULL;
}

static void bar(double frac, int width) {
    int n = (int)(frac * width + 0.5), i;
    for (i = 0; i < n && i < width; i++) fputs(print_is_ascii() ? "#" : "█", stdout);
}

/* a value in [0, 1] as a full-width bar: filled part + light rest */
static void gate_bar(double v, int width) {
    int n = (int)(v * width + 0.5), i;
    if (n < 0) n = 0;
    if (n > width) n = width;
    for (i = 0; i < width; i++) fputs(i < n ? sym("█", "#") : sym("·", "."), stdout);
}

/* a signed value in [−m, m] around a centre mark */
static void signed_bar(double v, double m, int half) {
    int n = m > 0 ? (int)(fabs(v) / m * half + 0.5) : 0, i;
    if (n > half) n = half;
    for (i = half; i > 0; i--) fputs(v < 0 && i <= n ? sym("█", "#") : " ", stdout);
    fputs("|", stdout);
    for (i = 1; i <= half; i++) fputs(v > 0 && i <= n ? sym("█", "#") : " ", stdout);
}

static void section(const char *title) {
    printf("\n");
    print_rule();
    printf("%s\n", title);
    print_rule();
}

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

/* the unit whose output changes most over the sequence */
static int busiest_unit(const nmz_tensor *h) {
    int T = h->shape[1], H = h->shape[2], j, t, best = 0;
    double best_range = -1;
    for (j = 0; j < H; j++) {
        double lo = HUGE_VAL, hi = -HUGE_VAL;
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

typedef struct seq_ctx {
    const nmz_trace_recorder *rec;
    const nmz_vocab *v;
    const print_opts *o;
    const nmz_tensor *ids, *p, *tg;
    int T, V, D, H, unit, layer, rows;
} seq_ctx;

static int target_at(const seq_ctx *s, int t) {
    const float *r;
    int k, best = 0;
    if (!s->tg) return -1;
    r = s->tg->data + (size_t)t * s->V;
    for (k = 1; k < s->V; k++)
        if (r[k] > r[best]) best = k;
    return best;
}

static void print_next(const seq_ctx *s, int t) {
    const float *pr;
    int top[5], target, k;
    if (!s->p || s->V <= 0) return;
    pr = s->p->data + (size_t)t * s->V;
    target = target_at(s, t);
    top5(pr, s->V, top, s->V < 5 ? s->V : 5);
    for (k = 0; k < 5 && k < s->V; k++) {
        printf("        %s %s %5.1f%% ", k == 0 ? "next:" : "     ", chp(s->v, top[k], 4), 100.0 * pr[top[k]]);
        bar(pr[top[k]], 30);
        printf("%s\n", top[k] == target ? "   <- target" : "");
    }
    if (target >= 0) {
        double nll = -log(pr[target] > 1e-30 ? pr[target] : 1e-30);
        printf("        target %s: p = %s, loss -log p = %s\n", ch(s->v, target), nm(pr[target]), nm(nll));
    }
}

/* x·W[:, col] for a [n, G] matrix */
static double dotcol(const float *x, int n, const nmz_tensor *W, int col) {
    double a = 0;
    int i, G = W->shape[1];
    for (i = 0; i < n; i++) a += (double)x[i] * W->data[(size_t)i * G + col];
    return a;
}

/* ------------------------------------------------------------------ rnn */

static void unrolled_rnn(const seq_ctx *s, const nmz_trace_event *rnn) {
    const nmz_tensor *h = nmz_trace_find(rnn, "h"), *Wxh = nmz_trace_find(rnn, "W_xh"), *Whh = nmz_trace_find(rnn, "W_hh"),
                     *b = nmz_trace_find(rnn, "b");
    int t, H = s->H, unit = s->unit;
    section("RNN UNROLLED THROUGH TIME   the same weights at every step; h carries the past forward");
    printf("  h_t = tanh(x_t%sW_xh + h_{t-1}%sW_hh + b)      shown for hidden unit j = %d of %d%s\n", sym("·", "*"),
           sym("·", "*"), unit + 1, H, s->o->at_set ? "" : " (the unit that changes most here; choose with --unit J)");
    printf("  x_t = the layer's input at step t, h_{-1} = 0 (or the carried state)\n");
    for (t = 0; t < s->rows; t++) {
        const nmz_trace_event *st = find(s->rec, NMZ_TRACE_FORWARD, "rnn", s->layer, t, 0);
        const nmz_tensor *xt = st ? nmz_trace_find(st, "x_t") : NULL, *hp = st ? nmz_trace_find(st, "h_prev") : NULL;
        double ax, ah, zj;
        if (!xt || !hp) continue;
        ax = dotcol(xt->data, (int)xt->size, Wxh, unit);
        ah = dotcol(hp->data, H, Whh, unit);
        zj = ax + ah + b->data[unit];
        printf("\n  t=%-3d input %s h_t[%d] = tanh(%s + %s + %s) = tanh(%s) = %s\n", t,
               chp(s->v, (int)s->ids->data[t], 4), unit + 1, nm(ax), nm(ah), nm(b->data[unit]), nm(zj),
               nm(h->data[(size_t)t * H + unit]));
        printf("        x_t%sW_xh[:,j]   h_{t-1}%sW_hh[:,j]   b[j]\n", sym("·", "*"), sym("·", "*"));
        print_next(s, t);
    }
    if (s->rows < s->T) printf("\n  %s %d more steps (use --full)\n", sym("…", "..."), s->T - s->rows);
}

/* ----------------------------------------------------------------- lstm */

static void gate_line(const char *name, const char *what, const char *fn, double ax, double ah, double bj, double v) {
    double a = ax + ah + bj;
    int w = 0, q;
    char f[16];
    for (q = 0; fn[q]; q++) w += ((unsigned char)fn[q] & 0xC0) != 0x80; /* display width of σ / sig / tanh */
    snprintf(f, sizeof f, "%*s%s", 4 - w, "", fn);
    printf("     %-2s %-11s = %s(%8s + %8s + %7s) = %s(%7s) = %6s ", name, what, f, n3(ax), n3(ah), n3(bj), f, n3(a),
           n3(v));
    if (strcmp(fn, "tanh") != 0)
        gate_bar(v, 20);
    else
        signed_bar(v, 1.0, 10);
    printf("\n");
}

static void unrolled_lstm(const seq_ctx *s, const nmz_trace_event *ev) {
    const nmz_tensor *Wx = nmz_trace_find(ev, "W_x"), *Wh = nmz_trace_find(ev, "W_h"), *b = nmz_trace_find(ev, "b");
    const nmz_tensor *c = nmz_trace_find(ev, "c");
    int t, H = s->H, j = s->unit;
    double cmax = 1e-9;
    const char *sg = sym("σ", "sig");
    for (t = 0; t < s->T; t++)
        if (fabs(c->data[(size_t)t * H + j]) > cmax) cmax = fabs(c->data[(size_t)t * H + j]);
    section("LSTM UNROLLED THROUGH TIME   gates decide what the cell forgets (f), writes (i, g) and shows (o)");
    printf("  for cell j = %d of %d%s\n", j + 1, H,
           s->o->at_set ? "" : " (the cell whose output changes most here; choose with --unit J)");
    printf("  each gate = %s(x_t%sW_x[:,gate j] + h_{t-1}%sW_h[:,gate j] + bias[gate j]); %s is between 0 (closed) "
           "and 1 (open)\n",
           sg, sym("·", "*"), sym("·", "*"), sg);
    printf("  c_t = f%sc_{t-1} + i%sg  is the memory, h_t = o%stanh(c_t) what the cell shows\n", sym("·", "*"),
           sym("·", "*"), sym("·", "*"));
    printf("  the three numbers in each gate: x_t%sW_x, h_{t-1}%sW_h, bias\n", sym("·", "*"), sym("·", "*"));
    for (t = 0; t < s->rows; t++) {
        const nmz_trace_event *st = find(s->rec, NMZ_TRACE_FORWARD, "lstm", s->layer, t, 0);
        const nmz_tensor *xt, *hp, *cp, *gi, *gf, *gg, *go, *ct, *ht;
        int D;
        if (!st) continue;
        xt = nmz_trace_find(st, "x_t");
        hp = nmz_trace_find(st, "h_prev");
        cp = nmz_trace_find(st, "c_prev");
        gi = nmz_trace_find(st, "i");
        gf = nmz_trace_find(st, "f");
        gg = nmz_trace_find(st, "g");
        go = nmz_trace_find(st, "o");
        ct = nmz_trace_find(st, "c_t");
        ht = nmz_trace_find(st, "h_t");
        if (!xt || !hp || !cp || !gi || !gf || !gg || !go || !ct || !ht) continue;
        D = (int)xt->size;
        printf("\n  t=%-3d input %s\n", t, chp(s->v, (int)s->ids->data[t], 4));
        gate_line("i", "input gate", sg, dotcol(xt->data, D, Wx, j), dotcol(hp->data, H, Wh, j), b->data[j], gi->data[j]);
        gate_line("f", "forget gate", sg, dotcol(xt->data, D, Wx, H + j), dotcol(hp->data, H, Wh, H + j), b->data[H + j],
                  gf->data[j]);
        gate_line("g", "candidate", "tanh", dotcol(xt->data, D, Wx, 2 * H + j), dotcol(hp->data, H, Wh, 2 * H + j),
                  b->data[2 * H + j], gg->data[j]);
        gate_line("o", "output gate", sg, dotcol(xt->data, D, Wx, 3 * H + j), dotcol(hp->data, H, Wh, 3 * H + j),
                  b->data[3 * H + j], go->data[j]);
        printf("     c_t = f%sc_{t-1} + i%sg = %s%s%s + %s%s%s = %s + %s = %s   ", sym("·", "*"), sym("·", "*"),
               n3(gf->data[j]), sym("·", "*"), n3(cp->data[j]), n3(gi->data[j]), sym("·", "*"), n3(gg->data[j]),
               n3((double)gf->data[j] * cp->data[j]), n3((double)gi->data[j] * gg->data[j]), n3(ct->data[j]));
        signed_bar(ct->data[j], cmax, 10);
        printf("\n     h_t = o%stanh(c_t) = %s%s%s = %s\n", sym("·", "*"), n3(go->data[j]), sym("·", "*"),
               n3(tanh(ct->data[j])), n3(ht->data[j]));
        print_next(s, t);
    }
    if (s->rows < s->T) printf("\n  %s %d more steps (use --full)\n", sym("…", "..."), s->T - s->rows);
    printf("\n  (c bars are scaled to this cell's largest |c| here, %s)\n", n3(cmax));
}

/* ------------------------------------------------------------------ gru */

static void unrolled_gru(const seq_ctx *s, const nmz_trace_event *ev) {
    const nmz_tensor *Wx = nmz_trace_find(ev, "W_x"), *Wh = nmz_trace_find(ev, "W_h"), *b = nmz_trace_find(ev, "b");
    int t, H = s->H, j = s->unit, i;
    const char *sg = sym("σ", "sig");
    section("GRU UNROLLED THROUGH TIME   an update gate z mixes the old state with a candidate");
    printf("  for unit j = %d of %d%s\n", j + 1, H,
           s->o->at_set ? "" : " (the unit whose output changes most here; choose with --unit J)");
    printf("  z, r = %s(x_t%sW_x + h_{t-1}%sW_h + b)   h~ = tanh(x_t%sW_x + (r%sh_{t-1})%sW_h + b)   "
           "h_t = z%sh_{t-1} + (1-z)%sh~\n",
           sg, sym("·", "*"), sym("·", "*"), sym("·", "*"), sym("⊙", "*"), sym("·", "*"), sym("·", "*"),
           sym("·", "*"));
    printf("  z = 1 keeps the old state, z = 0 replaces it; r = 0 lets the candidate ignore the past\n");
    for (t = 0; t < s->rows; t++) {
        const nmz_trace_event *st = find(s->rec, NMZ_TRACE_FORWARD, "gru", s->layer, t, 0);
        const nmz_tensor *xt, *hp, *z, *r, *n, *ht;
        double arh = 0;
        int D;
        if (!st) continue;
        xt = nmz_trace_find(st, "x_t");
        hp = nmz_trace_find(st, "h_prev");
        z = nmz_trace_find(st, "z");
        r = nmz_trace_find(st, "r");
        n = nmz_trace_find(st, "h~");
        ht = nmz_trace_find(st, "h_t");
        if (!xt || !hp || !z || !r || !n || !ht) continue;
        D = (int)xt->size;
        for (i = 0; i < H; i++) arh += (double)r->data[i] * hp->data[i] * Wh->data[(size_t)i * 3 * H + 2 * H + j];
        printf("\n  t=%-3d input %s\n", t, chp(s->v, (int)s->ids->data[t], 4));
        gate_line("z", "update gate", sg, dotcol(xt->data, D, Wx, j), dotcol(hp->data, H, Wh, j), b->data[j], z->data[j]);
        gate_line("r", "reset gate", sg, dotcol(xt->data, D, Wx, H + j), dotcol(hp->data, H, Wh, H + j), b->data[H + j],
                  r->data[j]);
        gate_line("h~", "candidate", "tanh", dotcol(xt->data, D, Wx, 2 * H + j), arh, b->data[2 * H + j], n->data[j]);
        printf("     h_t = z%sh_{t-1} + (1-z)%sh~ = %s%s%s + %s%s%s = %s   ", sym("·", "*"), sym("·", "*"),
               n3(z->data[j]), sym("·", "*"), n3(hp->data[j]), n3(1.0 - z->data[j]), sym("·", "*"), n3(n->data[j]),
               n3(ht->data[j]));
        signed_bar(ht->data[j], 1.0, 10);
        printf("\n");
        print_next(s, t);
    }
    if (s->rows < s->T) printf("\n  %s %d more steps (use --full)\n", sym("…", "..."), s->T - s->rows);
}

/* ------------------------------------------------------------- heatmaps */

static void heat_header(const seq_ctx *s, int label_w) {
    int t;
    printf("  %*s ", label_w, "");
    for (t = 0; t < s->T; t++) put_col_char(s->v, (int)s->ids->data[t]);
    printf("\n");
}

static void heat_row(const char *label, int label_w, const float *v, int T, int stride, double lo, double hi) {
    int t;
    printf("  %-*s ", label_w, label);
    for (t = 0; t < T; t++) putchar(heat_char(v[(size_t)t * stride], lo, hi));
    printf("\n");
}

static void heatmaps(const seq_ctx *s, const nmz_trace_event *ev) {
    const nmz_tensor *h = nmz_trace_find(ev, "h");
    int H = s->H, j, units = s->o->full || H <= 24 ? H : 24;
    if (!strcmp(ev->layer_type, "rnn")) {
        section("HIDDEN STATE OVER TIME   rows = hidden units, columns = time steps");
        printf("  '%c' = -1 ... '%c' = 0 ... '%c' = +1   (tanh range)\n", heat_char(-1, -1, 1), heat_char(0, -1, 1),
               heat_char(1, -1, 1));
        heat_header(s, 5);
        for (j = 0; j < units; j++) {
            char lbl[16];
            snprintf(lbl, sizeof lbl, "%s%4d", j == s->unit ? ">" : " ", j + 1);
            heat_row(lbl, 5, h->data + j, s->T, H, -1, 1);
        }
        if (units < H) printf("  %s %d more units (use --full)\n", sym("…", "..."), H - units);
        return;
    }
    {
        const nmz_tensor *g = nmz_trace_find(ev, "gates"), *c = nmz_trace_find(ev, "c");
        int lstm = !strcmp(ev->layer_type, "lstm"), G = lstm ? 4 * H : 3 * H, u = s->unit;
        double cm = 1e-9;
        int t;
        char title[160];
        snprintf(title, sizeof title, "GATE ACTIVITY OVER TIME   %s %d: one row per gate, columns = time steps",
                 lstm ? "cell" : "unit", u + 1);
        section(title);
        printf("  gates: '%c' = 0 (closed) ... '%c' = 1 (open);   %s: '%c' = -m ... '%c' = 0 ... '%c' = +m\n",
               heat_char(0, 0, 1), heat_char(1, 0, 1), lstm ? "g, c, h" : "h~, h", heat_char(-1, -1, 1),
               heat_char(0, -1, 1), heat_char(1, -1, 1));
        heat_header(s, 14);
        if (lstm) {
            for (t = 0; t < s->T; t++)
                if (fabs(c->data[(size_t)t * H + u]) > cm) cm = fabs(c->data[(size_t)t * H + u]);
            heat_row("i input gate", 14, g->data + u, s->T, G, 0, 1);
            heat_row("f forget gate", 14, g->data + H + u, s->T, G, 0, 1);
            heat_row("o output gate", 14, g->data + 3 * H + u, s->T, G, 0, 1);
            heat_row("g candidate", 14, g->data + 2 * H + u, s->T, G, -1, 1);
            heat_row("c cell", 14, c->data + u, s->T, H, -cm, cm);
            heat_row("h output", 14, h->data + u, s->T, H, -1, 1);
            printf("  (c is scaled to its largest |c| here: m = %s; g and h: m = 1)\n", n3(cm));
        } else {
            heat_row("z update gate", 14, g->data + u, s->T, G, 0, 1);
            heat_row("r reset gate", 14, g->data + H + u, s->T, G, 0, 1);
            heat_row("h~ candidate", 14, g->data + 2 * H + u, s->T, G, -1, 1);
            heat_row("h state", 14, h->data + u, s->T, H, -1, 1);
        }
        printf("\n  %s of every %s (rows) over time: '%c' = 0 ... '%c' = 1%s\n", lstm ? "FORGET GATE f" : "UPDATE GATE z",
               lstm ? "cell" : "unit", heat_char(0, 0, 1), heat_char(1, 0, 1),
               lstm ? "  (light rows forget, dark rows remember)" : "  (dark rows keep their state)");
        heat_header(s, 5);
        for (j = 0; j < units; j++) {
            char lbl[16];
            snprintf(lbl, sizeof lbl, "%s%4d", j == u ? ">" : " ", j + 1);
            heat_row(lbl, 5, g->data + (lstm ? H : 0) + j, s->T, G, 0, 1);
        }
        if (units < H) printf("  %s %d more units (use --full)\n", sym("…", "..."), H - units);
    }
}

/* ------------------------------------------------------------------ BPTT */

static void bptt_table(const seq_ctx *s, const char *type) {
    int t, lstm = !strcmp(type, "lstm"), gru = !strcmp(type, "gru");
    section("BACKPROPAGATION THROUGH TIME   from the last step back to the first");
    printf("  at every step:  dL/dh_t = (from the output at t) + (from step t+1)\n");
    if (lstm) {
        printf("                  dL/dc_t = dL/dc_{t+1}%sf_{t+1} (from step t+1) + dL/dh_t%so%s(1 - tanh%s(c_t))\n",
               sym("⊙", "*"), sym("⊙", "*"), sym("⊙", "*"), sym("²", "^2"));
        printf("                  dL/dc_{t-1} = dL/dc_t %s f    <- only the forget gate in between: the highway\n",
               sym("⊙", "*"));
        printf("                  dL/dh_{t-1} = dL/da %s W_h%s     (a = the inputs of the 4 gates)\n", sym("·", "*"),
               sym("ᵀ", "^T"));
    } else if (gru) {
        printf("                  dL/dh_{t-1} = dL/dh_t %s z (the direct path) + the paths through z, r and h~ (W_h)\n",
               sym("⊙", "*"));
    } else {
        printf("                  %s_t = dL/dh_t %s (1 - h_t%s)            (through tanh)\n", sym("δ", "delta"),
               sym("⊙", "*"), sym("²", "^2"));
        printf("                  dL/dh_{t-1} = %s_t %s W_hh%s           (passed back to the previous step)\n",
               sym("δ", "delta"), sym("·", "*"), sym("ᵀ", "^T"));
    }
    printf("  and every step adds to the SAME weight gradients\n\n");
    if (lstm)
        printf("  %4s %-5s %11s %11s %11s %11s %11s %11s\n", "t", "char", "|from out|", "|from t+1|", "|dL/dh_t|",
               "|dL/dc_t|", "|to h_t-1|", "|to c_t-1|");
    else if (gru)
        printf("  %4s %-5s %11s %11s %11s %11s %11s\n", "t", "char", "|from out|", "|from t+1|", "|dL/dh_t|",
               "|direct|", "|to h_t-1|");
    else
        printf("  %4s %-5s %11s %11s %11s %11s %11s\n", "t", "char", "|from out|", "|from t+1|", "|dL/dh_t|",
               print_is_ascii() ? "|delta_t|" : "|δ_t|", "|to t-1|");
    for (t = s->T - 1; t >= 0; t--) {
        const nmz_trace_event *st = find(s->rec, NMZ_TRACE_BACKWARD, type, s->layer, t, 0);
        if (!st) continue;
        if (s->o->max_rows > 0 && s->T - 1 - t >= s->o->max_rows) {
            printf("  %s %d more steps (use --full)\n", sym("…", "..."), t + 1);
            break;
        }
        printf("  %4d %s %11s %11s %11s", t, chp(s->v, (int)s->ids->data[t], 5),
               nm(norm_t(nmz_trace_find(st, "dL/dh_out"))), nm(norm_t(nmz_trace_find(st, "dL/dh_next"))),
               nm(norm_t(nmz_trace_find(st, "dL/dh"))));
        if (lstm)
            printf(" %11s %11s %11s\n", nm(norm_t(nmz_trace_find(st, "dL/dc"))),
                   nm(norm_t(nmz_trace_find(st, "dL/dh_prev"))), nm(norm_t(nmz_trace_find(st, "dL/dc_prev"))));
        else if (gru)
            printf(" %11s %11s\n", nm(norm_t(nmz_trace_find(st, "dL/dh_direct"))),
                   nm(norm_t(nmz_trace_find(st, "dL/dh_prev"))));
        else
            printf(" %11s %11s\n", nm(norm_t(nmz_trace_find(st, "dL/dz"))),
                   nm(norm_t(nmz_trace_find(st, "dL/dh_prev"))));
    }
    printf("  (|v| = L2 norm; at t = 0 the gradient to the initial state is dropped: that is where truncation cuts)\n");
}

/* --------------------------------------------------- gradient survival */

void seq_survival_free(seq_survival *sv) {
    free(sv->h);
    free(sv->c);
    free(sv->gate_prod);
    free(sv->x);
    free(sv->n_keep);
    memset(sv, 0, sizeof *sv);
}

int seq_survival_collect(const nmz_trace_recorder *rec, int layer, int T, seq_survival *sv) {
    const nmz_trace_event *fw = find_recurrent(rec, layer), *emb;
    size_t k;
    int t;
    memset(sv, 0, sizeof *sv);
    if (!fw || T < 1) return 1;
    sv->T = T;
    sv->layer = fw->layer;
    snprintf(sv->kind, sizeof sv->kind, "%s", fw->layer_type);
    sv->h = calloc((size_t)T, sizeof *sv->h);
    sv->x = calloc((size_t)T, sizeof *sv->x);
    if (!sv->h || !sv->x) return 1;
    if (strcmp(sv->kind, "rnn") != 0) {
        const nmz_tensor *g = nmz_trace_find(fw, "gates");
        int lstm = !strcmp(sv->kind, "lstm"), H, G, j;
        sv->gate_prod = calloc((size_t)T, sizeof *sv->gate_prod);
        sv->n_keep = calloc((size_t)T, sizeof *sv->n_keep);
        if (lstm) sv->c = calloc((size_t)T, sizeof *sv->c);
        if (!sv->gate_prod || !sv->n_keep || (lstm && !sv->c) || !g) return 1;
        G = g->shape[2];
        H = lstm ? G / 4 : G / 3;
        sv->units = H;
        /* mean over units of the product of f (lstm) or z (gru) of the later steps */
        for (j = 0; j < H; j++) {
            double prod = 1.0;
            for (t = T - 1; t >= 0; t--) {
                sv->gate_prod[t] += prod / H;
                sv->n_keep[t] += prod >= 0.5;
                prod *= g->data[(size_t)t * G + (lstm ? H : 0) + j];
            }
        }
    }
    for (k = 0; k < nmz_trace_recorder_count(rec); k++) {
        const nmz_trace_event *e = nmz_trace_recorder_get(rec, k);
        if (e->phase != NMZ_TRACE_BACKWARD || e->layer != sv->layer || e->time_step < 0 || e->time_step >= T) continue;
        sv->h[e->time_step] = norm_t(nmz_trace_find(e, "dL/dh"));
        if (sv->c) sv->c[e->time_step] = norm_t(nmz_trace_find(e, "dL/dc"));
    }
    emb = find(rec, NMZ_TRACE_BACKWARD, "embedding", -1, -1, 0);
    if (emb) {
        const nmz_tensor *de = nmz_trace_find(emb, "dL/de");
        if (de && de->rank == 3 && de->shape[1] == T)
            for (t = 0; t < T; t++) sv->x[t] = norm_row(de->data + (size_t)t * de->shape[2], de->shape[2]);
    }
    return 0;
}

double seq_survival_rate(const double *v, int T, double *left) {
    if (left) *left = NAN;
    if (!v || T < 2 || !(v[T - 1] > 0) || !(v[0] > 0)) return NAN;
    if (left) *left = v[0] / v[T - 1];
    return pow(v[0] / v[T - 1], 1.0 / (T - 1));
}

/* fraction of a log axis lo..hi */
static double log_frac(double v, double lo, double hi) {
    double f = v > 0 && hi > lo ? (log10(v) - log10(lo)) / (log10(hi) - log10(lo)) : 0;
    return f < 0 ? 0 : f > 1 ? 1 : f;
}

/* a log-scale bar padded to `width` columns */
static void log_bar_pad(double v, double lo, double hi, int width) {
    int n = (int)(log_frac(v, lo, hi) * width + 0.5), i;
    for (i = 0; i < width; i++) fputs(i < n ? sym("█", "#") : " ", stdout);
}

static void vanishing(const seq_ctx *s, const seq_survival *sv) {
    int t, T = sv->T, lstm = sv->c != NULL, gru = !strcmp(sv->kind, "gru"), w = lstm ? 18 : 30;
    double mx = 0, mn = HUGE_VAL, lo, decades, per_h, per_c = NAN, left_h, left_c = NAN;
    section(lstm ? "VANISHING GRADIENTS?   the loss of the LAST character only, followed back along h and along c"
                 : "VANISHING / EXPLODING GRADIENTS   the loss of the LAST character only, followed back in time");
    printf("  |dL_last/dh_t|: how strongly the last prediction can still teach step t%s\n\n",
           lstm ? "; |dL_last/dc_t|: the same along the cell state" : "");
    for (t = 0; t < T; t++) {
        double a = sv->h[t], b = lstm ? sv->c[t] : 0;
        if (a > mx) mx = a;
        if (b > mx) mx = b;
        if (a > 0 && a < mn) mn = a;
        if (b > 0 && b < mn) mn = b;
    }
    lo = mn < mx / 10 ? mn : mx / 10;
    decades = mx > 0 && lo > 0 ? log10(mx) - log10(lo) : 1;
    printf("  %-6s %-4s %10s %8s  %-*s", "step", "char", "|dL/dh_t|", "x/step", w, "(log scale)");
    if (lstm) printf(" %10s %8s  %-*s", "|dL/dc_t|", "x/step", w, "(log scale)");
    if (lstm || gru) printf(" %s", lstm ? sym("mean Π f", "mean prod f") : sym("mean Π z", "mean prod z"));
    printf("\n");
    for (t = T - 1; t >= 0; t--) {
        double r = t < T - 1 && sv->h[t + 1] > 0 ? sv->h[t] / sv->h[t + 1] : NAN;
        if (s->o->max_rows > 0 && T - 1 - t >= 2 * s->o->max_rows) {
            printf("  %s\n", sym("…", "..."));
            break;
        }
        printf("  t=%-4d %s %10s ", t, chp(s->v, (int)s->ids->data[t], 4), nm(sv->h[t]));
        if (isnan(r)) printf("%8s  ", "");
        else printf(" x%-6.3f  ", r);
        log_bar_pad(sv->h[t], lo, mx, w);
        if (lstm) {
            double rc = t < T - 1 && sv->c[t + 1] > 0 ? sv->c[t] / sv->c[t + 1] : NAN;
            printf(" %10s ", nm(sv->c[t]));
            if (isnan(rc)) printf("%8s  ", "");
            else printf(" x%-6.3f  ", rc);
            log_bar_pad(sv->c[t], lo, mx, w);
        }
        if (lstm || gru) printf(" %s", n3(sv->gate_prod[t]));
        printf("\n");
    }
    per_h = seq_survival_rate(sv->h, T, &left_h);
    if (lstm) per_c = seq_survival_rate(sv->c, T, &left_c);
    if (isnan(per_h)) return;
    printf("\n  bars: log scale over %.1f decade(s). Each step back multiplies |dL/dh| by %.3f on average", decades,
           per_h);
    if (lstm) printf(" and |dL/dc| by %.3f", per_c);
    printf(";\n  after %d steps %.2g of the h-gradient is left", T - 1, left_h);
    if (lstm) printf(", %.2g of the c-gradient", left_c);
    printf(".\n");
    if (!strcmp(sv->kind, "rnn")) {
        if (per_h < 0.95)
            printf("  That is a VANISHING gradient: a vanilla RNN learns short-range patterns easily and long-range\n"
                   "  ones hardly at all - the problem the LSTM's additive cell state fixes (try `compare`).\n");
        else if (per_h > 1.05)
            printf("  It GROWS here: kept up over long sequences that is an EXPLODING gradient - what gradient\n"
                   "  clipping (the --clip norm) protects training from.\n");
        else
            printf("  It stays roughly the same size here (neither vanishing nor exploding).\n");
        printf("  (each step multiplies by W_hh%s and by tanh' = 1 - h%s <= 1, which is why it usually shrinks)\n",
               sym("ᵀ", "^T"), sym("²", "^2"));
    } else if (lstm) {
        printf("  Along c, each step back multiplies only by the forget gate f (no weight matrix, no tanh').\n"
               "  The last column, the product of f over the later steps (mean over the cells), is what that path\n"
               "  alone would keep: cells with f near 1 pass the gradient almost untouched - the \"gradient\n"
               "  highway\" a plain RNN does not have.\n");
    } else {
        printf("  The GRU's direct path h_{t-1} -> h_t multiplies the gradient only by the update gate z (last\n"
               "  column: the product of z over the later steps, mean over the units): units with z near 1 keep\n"
               "  both their state and its gradient.\n");
    }
}

/* ------------------------------------------------------------------ main */

void print_seq_trace(const nmz_trace_recorder *rec, const nmz_vocab *v, const print_opts *o, int applied, int unit,
                     int layer, const seq_survival *sv) {
    const nmz_trace_event *emb = find(rec, NMZ_TRACE_FORWARD, "embedding", -1, -1, 0);
    const nmz_trace_event *rl = find_recurrent(rec, layer);
    const nmz_trace_event *soft, *loss = find(rec, NMZ_TRACE_LOSS, "loss", -1, -1, 0);
    const nmz_tensor *e, *h;
    seq_ctx s;
    int t, k;
    double L = NAN;
    memset(&s, 0, sizeof s);
    if (!emb || !rl) {
        printf("(this network has no embedding + recurrent layer; showing nothing unrolled)\n");
        return;
    }
    s.rec = rec;
    s.v = v;
    s.o = o;
    s.V = v ? v->size : 0;
    s.layer = rl->layer;
    s.ids = nmz_trace_find(emb, "ids");
    e = nmz_trace_find(emb, "e");
    h = nmz_trace_find(rl, "h");
    /* the last softmax forward is the output head */
    for (k = 0; (soft = find(rec, NMZ_TRACE_FORWARD, "activation", -1, -1, k)) != NULL; k++)
        if (!strcmp(soft->detail, "softmax")) s.p = nmz_trace_find(soft, "a");
    if (loss) {
        s.tg = nmz_trace_find(loss, "t");
        nmz_trace_scalar_get(loss, "L", &L);
    }
    s.T = h->shape[1];
    s.H = h->shape[2];
    s.D = e->shape[2];
    s.unit = unit >= 0 && unit < s.H ? unit : busiest_unit(h);
    s.rows = o->max_rows > 0 && o->max_rows < s.T ? o->max_rows : s.T;

    printf("\ninput :");
    for (t = 0; t < s.T; t++) printf(" %s", ch(v, (int)s.ids->data[t]));
    if (s.tg) {
        printf("\ntarget:");
        for (t = 0; t < s.T; t++) printf(" %s", ch(v, target_at(&s, t)));
    }
    printf("\n(the target is the input shifted by one character: at every step the net predicts the next one)\n");
    if (find_recurrent(rec, -1) != rl || layer >= 0)
        printf("(showing layer %d (%s) of a stacked model; --layer N picks the N-th recurrent layer)\n", rl->layer,
               rl->layer_type);

    section("EMBEDDING   each character id selects one row of W (a learned vector)");
    printf("  e_t = W[id_t]        W is %d characters x %d numbers\n", s.V, s.D);
    for (t = 0; t < s.rows; t++) {
        printf("  t=%-3d %s id %-4d e = [", t, chp(v, (int)s.ids->data[t], 4), (int)s.ids->data[t]);
        for (k = 0; k < s.D && k < 6; k++) printf("%s%s", k ? ", " : "", nm(e->data[(size_t)t * s.D + k]));
        printf(s.D > 6 ? ", ...]\n" : "]\n");
    }
    if (s.rows < s.T) printf("  %s %d more steps (use --full)\n", sym("…", "..."), s.T - s.rows);

    if (!strcmp(rl->layer_type, "lstm"))
        unrolled_lstm(&s, rl);
    else if (!strcmp(rl->layer_type, "gru"))
        unrolled_gru(&s, rl);
    else
        unrolled_rnn(&s, rl);
    if (!isnan(L))
        printf("\n  L = mean over the %d steps of -log p(target) = %s   (perplexity e^L = %s: as unsure as a choice "
               "among %.1f characters)\n",
               s.T, nm(L), nm(exp(L)), exp(L));
    heatmaps(&s, rl);
    bptt_table(&s, rl->layer_type);
    if (sv && sv->T > 1) vanishing(&s, sv);

    {
        const nmz_trace_event *clip = find(rec, NMZ_TRACE_UPDATE, "clip", -1, -1, 0);
        size_t i;
        int header = 0;
        section(applied ? "GRADIENT CLIPPING AND WEIGHT UPDATE"
                        : "GRADIENT CLIPPING AND WEIGHT UPDATE (preview only: nothing is changed; use `step`)");
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
