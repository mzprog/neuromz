/* print.c — turn trace events into readable formulas and tables.
 *
 * Every number the engine computed for one sample is printed next to the
 * formula that produced it, e.g.
 *     z_1 = b_1 + Σ_i x_i·W_i1 = 0.000 + 1.000·0.512 + 0.000·(-0.231) = 0.5120
 * Tensors are [1, n] (one sample) when called from `trace` / `step`. */
#include "cli.h"

#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

static int g_ascii = 0;

void print_set_ascii(int ascii) { g_ascii = ascii; }
int print_is_ascii(void) { return g_ascii; }
const char *sym(const char *utf8, const char *ascii) { return g_ascii ? ascii : utf8; }

#define S_SUM sym("Σ", "sum")
#define S_DOT sym("·", "*")
#define S_D sym("∂", "d")
#define S_DELTA sym("δ", "delta")
#define S_ETA sym("η", "lr")
#define S_MU sym("μ", "mu")
#define S_HALF sym("½", "1/2")
#define S_SIG sym("σ", "sigmoid")
#define S_LARROW sym("←", "<-")
#define S_RARROW sym("→", "->")
#define S_MINUS sym("−", "-")
#define S_SQ sym("²", "^2")
#define S_ALPHA sym("α", "alpha")
#define S_DOTS sym("…", "...")

const char *fmt_num(char *buf, size_t n, double v) {
    double a = fabs(v);
    if (v == 0.0 || (a >= 1e-4 && a < 1e5))
        snprintf(buf, n, "%.4f", v);
    else
        snprintf(buf, n, "%.3e", v);
    return buf;
}

/* Rotating pool so several numbers can be formatted in one printf. */
static const char *num(double v) {
    static char pool[24][32];
    static int k = 0;
    char *b = pool[k++ % 24];
    return fmt_num(b, sizeof pool[0], v);
}

/* number wrapped in parentheses when negative (for products) */
static const char *pnum(double v) {
    static char pool[24][40];
    static int k = 0;
    char *b = pool[k++ % 24];
    if (v < 0)
        snprintf(b, sizeof pool[0], "(%s)", num(v));
    else
        snprintf(b, sizeof pool[0], "%s", num(v));
    return b;
}

/* ---------------------------------------------------------- string builder */

typedef struct sb {
    char *s;
    size_t len, cap;
} sb;

static void sb_add(sb *b, const char *fmt, ...) {
    va_list ap;
    int need;
    va_start(ap, fmt);
    need = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (need < 0) return;
    if (b->len + (size_t)need + 1 > b->cap) {
        size_t cap = (b->len + (size_t)need + 1) * 2;
        char *p = realloc(b->s, cap);
        if (!p) return;
        b->s = p;
        b->cap = cap;
    }
    va_start(ap, fmt);
    vsnprintf(b->s + b->len, b->cap - b->len, fmt, ap);
    va_end(ap);
    b->len += (size_t)need;
}

static const char *sb_str(sb *b) { return b->s ? b->s : ""; }
static void sb_free(sb *b) {
    free(b->s);
    b->s = NULL;
    b->len = b->cap = 0;
}

/* ------------------------------------------------------------------ helpers */

/* printf pads by bytes; pad UTF-8 text by characters instead */
static int display_width(const char *s) {
    int n = 0;
    for (; *s; s++)
        if (((unsigned char)*s & 0xC0) != 0x80) n++;
    return n;
}

static void rpad(const char *s, int w) {
    int k = w - display_width(s);
    while (k-- > 0) putchar(' ');
    fputs(s, stdout);
}

void print_rule(void) {
    puts(g_ascii ? "------------------------------------------------------------------------"
                 : "────────────────────────────────────────────────────────────────────────");
}

static int width(const nmz_tensor *t) { return t ? nmz_tensor_cols(t) : 0; }
static float at(const nmz_tensor *t, int i) { return t->data[i]; }

/* returns 1 if the row limit was hit (caller prints the ellipsis line) */
static int over_limit(int i, int n, const print_opts *o) {
    if (o->max_rows > 0 && i >= o->max_rows && n > o->max_rows) {
        printf("    %s %d more row(s) not shown (use --full)\n", S_DOTS, n - o->max_rows);
        return 1;
    }
    return 0;
}

static void title(const nmz_trace_event *ev, const char *what) {
    printf("\n[layer %d] %s\n", ev->layer, what);
}

void print_matrix(const char *name, const nmz_tensor *t, int max_rows) {
    int rows, cols, i, j;
    if (!t) return;
    cols = nmz_tensor_cols(t);
    rows = (int)nmz_tensor_rows(t);
    if (name) printf("  %s  (%d x %d)\n", name, rows, cols);
    printf("  %6s", "");
    for (j = 0; j < cols && j < 12; j++) {
        char h[16];
        snprintf(h, sizeof h, "j=%d", j + 1);
        printf(" %10s", h);
    }
    if (cols > 12) printf("  %s", S_DOTS);
    printf("\n");
    for (i = 0; i < rows; i++) {
        if (max_rows > 0 && i >= max_rows) {
            printf("    %s %d more row(s)\n", S_DOTS, rows - max_rows);
            break;
        }
        printf("  i=%-4d", i + 1);
        for (j = 0; j < cols && j < 12; j++) printf(" %10s", num(t->data[(size_t)i * cols + j]));
        if (cols > 12) printf("  %s", S_DOTS);
        printf("\n");
    }
}

/* ------------------------------------------------------------- forward */

static void dense_forward(const nmz_trace_event *ev, const print_opts *o) {
    const nmz_tensor *x = nmz_trace_find(ev, "x"), *W = nmz_trace_find(ev, "W"), *b = nmz_trace_find(ev, "b"),
                     *z = nmz_trace_find(ev, "z");
    int n = width(x), u = width(z), i, j;
    char what[96];
    snprintf(what, sizeof what, "dense %d %s %d   (forward)", n, S_RARROW, u);
    title(ev, what);
    printf("  z_j = b_j + %s_i x_i%sW_ij        x = this layer's input, W_ij = weight from input i to unit j\n",
           S_SUM, S_DOT);
    for (j = 0; j < u; j++) {
        sb line = {NULL, 0, 0};
        if (over_limit(j, u, o)) break;
        sb_add(&line, "  z_%d = %s", j + 1, num(at(b, j)));
        for (i = 0; i < n; i++) {
            if (o->max_terms > 0 && i >= o->max_terms && n > o->max_terms + 1) {
                double rest = 0;
                int k;
                for (k = i; k < n; k++) rest += (double)at(x, k) * W->data[(size_t)k * u + j];
                sb_add(&line, " + [%d more terms = %s]", n - i, num(rest));
                break;
            }
            sb_add(&line, " + %s%s%s", pnum(at(x, i)), S_DOT, pnum(W->data[(size_t)i * u + j]));
        }
        sb_add(&line, " = %s", num(at(z, j)));
        puts(sb_str(&line));
        sb_free(&line);
    }
}

static const char *act_formula(const char *act) {
    if (!strcmp(act, "sigmoid")) return g_ascii ? "a = sigmoid(z) = 1/(1+e^-z)" : "a = σ(z) = 1/(1+e^−z)";
    if (!strcmp(act, "tanh")) return "a = tanh(z)";
    if (!strcmp(act, "relu")) return "a = max(0, z)";
    if (!strcmp(act, "leaky_relu")) return g_ascii ? "a = z if z > 0 else alpha*z" : "a = z if z > 0 else α·z";
    if (!strcmp(act, "linear")) return "a = z";
    return "a = f(z)";
}

static void act_forward(const nmz_trace_event *ev, const print_opts *o) {
    const nmz_tensor *z = nmz_trace_find(ev, "z"), *a = nmz_trace_find(ev, "a");
    int n = width(z), j;
    char what[96];
    snprintf(what, sizeof what, "activation %s   (forward)", ev->detail);
    title(ev, what);
    if (!strcmp(ev->detail, "softmax")) {
        double mx = at(z, 0), sum = 0;
        for (j = 1; j < n; j++)
            if (at(z, j) > mx) mx = at(z, j);
        for (j = 0; j < n; j++) sum += exp((double)at(z, j) - mx);
        printf("  a_k = e^(z_k) / %s_j e^(z_j)   (computed as e^(z_k %s m) with m = max z = %s, for stability)\n",
               S_SUM, S_MINUS, num(mx));
        printf("  %s_j e^(z_j %s m) = %s\n", S_SUM, S_MINUS, num(sum));
        for (j = 0; j < n; j++) {
            if (over_limit(j, n, o)) break;
            printf("  a_%d = e^(%s %s %s) / %s = %s / %s = %s\n", j + 1, num(at(z, j)), S_MINUS, num(mx), num(sum),
                   num(exp((double)at(z, j) - mx)), num(sum), num(at(a, j)));
        }
        return;
    }
    printf("  %s\n", act_formula(ev->detail));
    for (j = 0; j < n; j++) {
        if (over_limit(j, n, o)) break;
        printf("  a_%d = %s(%s) = %s\n", j + 1, !strcmp(ev->detail, "sigmoid") ? S_SIG : ev->detail,
               num(at(z, j)), num(at(a, j)));
    }
}

/* ------------------------------------------------------------------ loss */

static void loss_event(const nmz_trace_event *ev, const print_opts *o) {
    const nmz_tensor *y = nmz_trace_find(ev, "y"), *t = nmz_trace_find(ev, "t");
    int n = width(y), k;
    double L = NAN, sum = 0;
    nmz_trace_scalar_get(ev, "L", &L);
    printf("\nLOSS  (%s)\n", ev->detail);
    if (!strcmp(ev->detail, "mse")) {
        printf("  L = %s %s_k (y_k %s t_k)%s       y = network output, t = target\n", S_HALF, S_SUM, S_MINUS, S_SQ);
        for (k = 0; k < n; k++) {
            double e = (double)at(y, k) - at(t, k);
            sum += e * e;
            if (o->max_rows > 0 && k >= o->max_rows) continue;
            printf("  (y_%d %s t_%d)%s = (%s %s %s)%s = %s\n", k + 1, S_MINUS, k + 1, S_SQ, num(at(y, k)), S_MINUS,
                   num(at(t, k)), S_SQ, num(e * e));
        }
        if (o->max_rows > 0 && n > o->max_rows) printf("    %s %d more\n", S_DOTS, n - o->max_rows);
        printf("  L = %s %s %s = %s\n", S_HALF, S_DOT, num(sum), num(L));
    } else {
        printf("  L = %s%s_k t_k%slog(p_k)       p = softmax output, t = target (one-hot)\n", S_MINUS, S_SUM,
               S_DOT);
        for (k = 0; k < n; k++) {
            if (at(t, k) == 0.0f) continue;
            printf("  %st_%d%slog(p_%d) = %s%s%slog(%s) = %s\n", S_MINUS, k + 1, S_DOT, k + 1, S_MINUS, num(at(t, k)),
                   S_DOT, num(at(y, k)), num(-(double)at(t, k) * log(fmax((double)at(y, k), 1e-300))));
        }
        printf("  L = %s\n", num(L));
    }
}

/* -------------------------------------------------------------- backward */

static void loss_backward(const nmz_trace_event *ev, const print_opts *o) {
    int k, n;
    if (!strcmp(ev->detail, "softmax+cross_entropy")) {
        const nmz_tensor *p = nmz_trace_find(ev, "p"), *t = nmz_trace_find(ev, "t"), *d = nmz_trace_find(ev, "dL/dz");
        n = width(p);
        printf("\n[loss] softmax + cross-entropy, combined   (backward)\n");
        printf("  %s_k = %sL/%sz_k = p_k %s t_k        (the softmax layer's own backward step is folded in here)\n",
               S_DELTA, S_D, S_D, S_MINUS);
        for (k = 0; k < n; k++) {
            if (over_limit(k, n, o)) break;
            printf("  %s_%d = %s %s %s = %s\n", S_DELTA, k + 1, num(at(p, k)), S_MINUS, num(at(t, k)), num(at(d, k)));
        }
    } else {
        const nmz_tensor *y = nmz_trace_find(ev, "y"), *t = nmz_trace_find(ev, "t"), *g = nmz_trace_find(ev, "dL/dy");
        n = width(y);
        printf("\n[loss] %s   (backward)\n", ev->detail);
        printf("  %sL/%sy_k = (y_k %s t_k) / B        B = batch size = 1\n", S_D, S_D, S_MINUS);
        for (k = 0; k < n; k++) {
            if (over_limit(k, n, o)) break;
            printf("  %sL/%sy_%d = %s %s %s = %s\n", S_D, S_D, k + 1, num(at(y, k)), S_MINUS, num(at(t, k)),
                   num(at(g, k)));
        }
    }
}

static void act_backward(const nmz_trace_event *ev, const nmz_trace_event *fwd, const print_opts *o) {
    const nmz_tensor *g = nmz_trace_find(ev, "dL/da"), *fp = nmz_trace_find(ev, "f'(z)"),
                     *d = nmz_trace_find(ev, "dL/dz");
    const nmz_tensor *z = fwd ? nmz_trace_find(fwd, "z") : NULL, *a = fwd ? nmz_trace_find(fwd, "a") : NULL;
    int n = width(g), j;
    char what[96];
    snprintf(what, sizeof what, "activation %s   (backward)", ev->detail);
    title(ev, what);
    if (!strcmp(ev->detail, "softmax")) {
        double dot = 0;
        for (j = 0; j < n && a; j++) dot += (double)at(g, j) * at(a, j);
        printf("  %sL/%sz_k = a_k %s (%sL/%sa_k %s %s_j %sL/%sa_j%sa_j)      %s_j %sL/%sa_j%sa_j = %s\n", S_D, S_D,
               S_DOT, S_D, S_D, S_MINUS, S_SUM, S_D, S_D, S_DOT, S_SUM, S_D, S_D, S_DOT, num(dot));
        for (j = 0; j < n; j++) {
            if (over_limit(j, n, o)) break;
            printf("  %s_%d = %s %s (%s %s %s) = %s\n", S_DELTA, j + 1, a ? num(at(a, j)) : "?", S_DOT, num(at(g, j)),
                   S_MINUS, num(dot), num(at(d, j)));
        }
        return;
    }
    printf("  %s_j = %sL/%sa_j %s f'(z_j)        ", S_DELTA, S_D, S_D, S_DOT);
    if (!strcmp(ev->detail, "sigmoid")) printf("f'(z) = %s(z)%s(1 %s %s(z)) = a%s(1 %s a)\n", S_SIG, S_DOT, S_MINUS, S_SIG, S_DOT, S_MINUS);
    else if (!strcmp(ev->detail, "tanh")) printf("f'(z) = 1 %s tanh%s(z) = 1 %s a%s\n", S_MINUS, S_SQ, S_MINUS, S_SQ);
    else if (!strcmp(ev->detail, "relu")) printf("f'(z) = 1 if z > 0 else 0\n");
    else if (!strcmp(ev->detail, "leaky_relu")) printf("f'(z) = 1 if z > 0 else %s\n", S_ALPHA);
    else printf("f'(z) = 1\n");
    for (j = 0; j < n; j++) {
        if (over_limit(j, n, o)) break;
        if (a && !strcmp(ev->detail, "sigmoid"))
            printf("  f'(z_%d) = %s%s(1 %s %s) = %s\n", j + 1, num(at(a, j)), S_DOT, S_MINUS, num(at(a, j)), num(at(fp, j)));
        else if (a && !strcmp(ev->detail, "tanh"))
            printf("  f'(z_%d) = 1 %s %s%s = %s\n", j + 1, S_MINUS, pnum(at(a, j)), S_SQ, num(at(fp, j)));
        else if (z)
            printf("  f'(z_%d) = f'(%s) = %s\n", j + 1, num(at(z, j)), num(at(fp, j)));
        printf("  %s_%d = %s %s %s = %s\n", S_DELTA, j + 1, num(at(g, j)), S_DOT, pnum(at(fp, j)), num(at(d, j)));
    }
}

static void dense_backward(const nmz_trace_event *ev, const print_opts *o) {
    const nmz_tensor *d = nmz_trace_find(ev, "dL/dz"), *x = nmz_trace_find(ev, "x"), *W = nmz_trace_find(ev, "W"),
                     *dW = nmz_trace_find(ev, "dL/dW"), *db = nmz_trace_find(ev, "dL/db"),
                     *dx = nmz_trace_find(ev, "dL/dx");
    int n = width(x), u = width(d), i, j;
    char what[96];
    snprintf(what, sizeof what, "dense %d %s %d   (backward)", n, S_RARROW, u);
    title(ev, what);
    printf("  %s = %sL/%sz (from the layer above):", S_DELTA, S_D, S_D);
    for (j = 0; j < u && j < 16; j++) printf(" %s", num(at(d, j)));
    printf(u > 16 ? " %s\n" : "\n", S_DOTS);
    printf("  %sL/%sW_ij = x_i %s %s_j", S_D, S_D, S_DOT, S_DELTA);
    if (n > 0 && u > 0)
        printf("        e.g. %sL/%sW_11 = %s %s %s = %s", S_D, S_D, num(at(x, 0)), S_DOT, pnum(at(d, 0)), num(dW->data[0]));
    printf("\n");
    print_matrix(NULL, dW, o->max_rows);
    printf("  %sL/%sb_j = %s_j:", S_D, S_D, S_DELTA);
    for (j = 0; j < u && j < 16; j++) printf(" %s", num(at(db, j)));
    printf(u > 16 ? " %s\n" : "\n", S_DOTS);
    if (dx) {
        printf("  %sL/%sx_i = %s_j W_ij %s %s_j        (passed down: it is %sL/%sa of the layer below%s)\n", S_D, S_D,
               S_SUM, S_DOT, S_DELTA, S_D, S_D, ev->layer == 0 ? "; here: of the input" : "");
        for (i = 0; i < n; i++) {
            sb line = {NULL, 0, 0};
            if (over_limit(i, n, o)) break;
            sb_add(&line, "  %sL/%sx_%d =", S_D, S_D, i + 1);
            for (j = 0; j < u; j++) {
                if (o->max_terms > 0 && j >= o->max_terms && u > o->max_terms + 1) {
                    sb_add(&line, " + [%d more terms]", u - j);
                    break;
                }
                sb_add(&line, "%s%s%s%s", j ? " + " : " ", pnum(W->data[(size_t)i * u + j]), S_DOT, pnum(at(d, j)));
            }
            sb_add(&line, " = %s", num(at(dx, i)));
            puts(sb_str(&line));
            sb_free(&line);
        }
    }
}

/* ---------------------------------------------------------------- update */

void print_sgd_update(const nmz_trace_event *ev, const print_opts *o) {
    const nmz_tensor *w = nmz_trace_find(ev, "value"), *g = nmz_trace_find(ev, "grad"),
                     *v0 = nmz_trace_find(ev, "velocity_prev"), *dl = nmz_trace_find(ev, "delta");
    int n = (int)w->size, e;
    printf("\n[layer %d] %s %s  (%d value%s)\n", ev->layer, ev->layer_type, ev->detail, n, n == 1 ? "" : "s");
    printf("  ");
    rpad(w->rank == 2 ? "i,j" : w->rank == 4 ? "f,c,ky,kx" : "j", 10);
    rpad("w old", 13);
    rpad(g_ascii ? "dL/dw" : "∂L/∂w", 13);
    rpad("v old", 13);
    rpad(g_ascii ? "v new=delta" : "v new = Δ", 13);
    rpad("w new", 13);
    printf("\n");
    for (e = 0; e < n; e++) {
        char idx[48];
        if (over_limit(e, n, o)) break;
        fmt_index(idx, sizeof idx, w, (size_t)e);
        printf("  %10s %12s %12s %12s %12s %12s\n", idx, num(w->data[e]), num(g->data[e]), num(v0->data[e]),
               num(dl->data[e]), num((double)w->data[e] + dl->data[e]));
    }
}

/* -------------------------------------------------------------- dispatch */

/* Prints a whole recorded pass. `fwd` lookups pair each backward event with
 * the forward event of the same layer (for z and a values). */
void print_trace_all(const nmz_trace_recorder *rec, const print_opts *o, int applied) {
    size_t i, n = nmz_trace_recorder_count(rec);
    nmz_trace_phase last = (nmz_trace_phase)-1;
    pr_focus_reset();
    for (i = 0; i < n; i++) {
        const nmz_trace_event *ev = nmz_trace_recorder_get(rec, i);
        if (ev->phase != last && ev->phase != NMZ_TRACE_LOSS) {
            printf("\n");
            print_rule();
            if (ev->phase == NMZ_TRACE_FORWARD) puts("FORWARD PASS   input -> output");
            if (ev->phase == NMZ_TRACE_BACKWARD) puts("BACKWARD PASS   output -> input: every gradient is computed first");
            if (ev->phase == NMZ_TRACE_UPDATE) {
                double lr = 0, mu = 0;
                nmz_trace_scalar_get(ev, "lr", &lr);
                if (nmz_trace_find(ev, "m")) {
                    printf("WEIGHT UPDATE%s\n", applied ? "" : " (preview only: not applied; use `step`)");
                    pr_adam_header(ev);
                } else {
                    nmz_trace_scalar_get(ev, "momentum", &mu);
                    printf("WEIGHT UPDATE%s   %s = %g, %s = %g\n",
                           applied ? "" : " (preview only: not applied; use `step`)", S_ETA, lr, S_MU, mu);
                    printf("  v %s %s%sv %s %s%s%sL/%sw        w %s w + v\n", S_LARROW, S_MU, S_DOT, S_MINUS, S_ETA,
                           S_DOT, S_D, S_D, S_LARROW);
                }
            }
            print_rule();
        }
        last = ev->phase;
        switch (ev->phase) {
        case NMZ_TRACE_FORWARD:
            if (!strcmp(ev->layer_type, "dense")) dense_forward(ev, o);
            else if (!strcmp(ev->layer_type, "activation")) {
                if (is_map(nmz_trace_find(ev, "z"))) pr_act_map_forward(ev, o);
                else act_forward(ev, o);
            } else if (!strcmp(ev->layer_type, "normalize")) pr_normalize(ev, o);
            else if (!strcmp(ev->layer_type, "conv2d")) pr_conv_forward(ev, o);
            else if (!strcmp(ev->layer_type, "maxpool2d")) pr_pool_forward(ev, o);
            else if (!strcmp(ev->layer_type, "flatten")) pr_flatten(ev, 0);
            else if (!strcmp(ev->layer_type, "dropout")) pr_dropout(ev, 0);
            else printf("\n[layer %d] %s (forward)\n", ev->layer, ev->layer_type);
            break;
        case NMZ_TRACE_LOSS: loss_event(ev, o); break;
        case NMZ_TRACE_BACKWARD:
            if (ev->layer < 0) loss_backward(ev, o);
            else if (!strcmp(ev->layer_type, "dense")) dense_backward(ev, o);
            else if (!strcmp(ev->layer_type, "conv2d")) pr_conv_backward(ev, o);
            else if (!strcmp(ev->layer_type, "maxpool2d")) pr_pool_backward(ev, o);
            else if (!strcmp(ev->layer_type, "flatten")) pr_flatten(ev, 1);
            else if (!strcmp(ev->layer_type, "dropout")) pr_dropout(ev, 1);
            else if (!strcmp(ev->layer_type, "activation") && is_map(nmz_trace_find(ev, "dL/da"))) pr_act_map_backward(ev, o);
            else if (!strcmp(ev->layer_type, "activation")) {
                const nmz_trace_event *fwd = NULL;
                size_t k;
                for (k = 0; k < i; k++) {
                    const nmz_trace_event *f = nmz_trace_recorder_get(rec, k);
                    if (f->phase == NMZ_TRACE_FORWARD && f->layer == ev->layer) fwd = f;
                }
                act_backward(ev, fwd, o);
            }
            break;
        case NMZ_TRACE_UPDATE:
            if (nmz_trace_find(ev, "m")) pr_adam_update(ev, o);
            else if (!strcmp(ev->layer_type, "clip")) {
                double gn = 0, mx = 0, sc = 1;
                nmz_trace_scalar_get(ev, "norm", &gn);
                nmz_trace_scalar_get(ev, "max_norm", &mx);
                nmz_trace_scalar_get(ev, "scale", &sc);
                printf("\n[clip] global gradient norm %s %s %s  ->  %s\n", num(gn), sc < 1 ? ">" : "<=", num(mx),
                       sc < 1 ? "every gradient scaled down" : "unchanged");
            } else print_sgd_update(ev, o);
            break;
        }
    }
}
