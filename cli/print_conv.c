/* print_conv.c — trace printers for image layers (conv2d, maxpool2d,
 * flatten, dropout, normalize, activations on feature maps), the Adam
 * update table, and ASCII heatmaps.
 *
 * Feature maps are [1, C, H, W] (one sample). For each spatial layer one
 * output position is "in focus": --at C,Y,X (1-based, clamped to the layer's
 * map), otherwise channel 1 at the centre of the map. */
#include "cli.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define D sym("∂", "d")
#define DOT sym("·", "*")
#define SUM sym("Σ", "sum")
#define DELTA sym("δ", "delta")
#define ARROW sym("→", "->")

static const char *n4(double v) { /* rotating number pool, as in print.c */
    static char pool[32][32];
    static int k = 0;
    char *b = pool[k++ % 32];
    return fmt_num(b, sizeof pool[0], v);
}

/* compact 7-char number for matrices */
static const char *c7(double v) {
    static char pool[64][16];
    static int k = 0;
    char *b = pool[k++ % 64];
    double a = fabs(v);
    if (v == 0.0)
        snprintf(b, sizeof pool[0], "0");
    else if (a >= 0.001 && a < 1000)
        snprintf(b, sizeof pool[0], "%.3f", v);
    else
        snprintf(b, sizeof pool[0], "%.1e", v);
    return b;
}

static const nmz_tensor *item(const nmz_trace_event *ev, const char *name) { return nmz_trace_find(ev, name); }

static int scalar_i(const nmz_trace_event *ev, const char *name, int dflt) {
    double v;
    return nmz_trace_scalar_get(ev, name, &v) ? (int)v : dflt;
}

/* The focus position of a layer: --at (clamped), else the position of the
 * largest value of channel 1 in the forward output ("where the filter fires").
 * Remembered per layer, so the backward pass shows the same position. */
#define MAX_FOCUS 512
static int g_focus_known[MAX_FOCUS], g_focus[MAX_FOCUS][3];

void pr_focus_reset(void) { memset(g_focus_known, 0, sizeof g_focus_known); }

static void focus_layer(int layer, const print_opts *o, const nmz_tensor *t, int *c, int *y, int *x) {
    int C = t->shape[1], H = t->shape[2], W = t->shape[3], i, best = 0;
    if (o->at_set) {
        *c = o->at_c < C ? o->at_c : C - 1;
        *y = o->at_y < H ? o->at_y : H - 1;
        *x = o->at_x < W ? o->at_x : W - 1;
        return;
    }
    if (layer >= 0 && layer < MAX_FOCUS && g_focus_known[layer]) {
        *c = g_focus[layer][0];
        *y = g_focus[layer][1];
        *x = g_focus[layer][2];
        return;
    }
    for (i = 1; i < H * W; i++)
        if (t->data[i] > t->data[best]) best = i;
    *c = 0;
    *y = best / W;
    *x = best % W;
    if (layer >= 0 && layer < MAX_FOCUS) {
        g_focus_known[layer] = 1;
        g_focus[layer][0] = *c;
        g_focus[layer][1] = *y;
        g_focus[layer][2] = *x;
    }
}

#define focus(o, t, c, y, x) focus_layer(ev->layer, o, t, c, y, x)

static float at4(const nmz_tensor *t, int c, int y, int x) {
    return t->data[((size_t)c * t->shape[2] + (size_t)y) * t->shape[3] + (size_t)x];
}

/* ------------------------------------------------------------- heatmaps */

static const char *ramp(void) { return " .:-=+*#%@"; }

char heat_char(double v, double lo, double hi) {
    const char *r = ramp();
    int n = (int)strlen(r), k;
    if (!(hi > lo)) return r[n / 2];
    k = (int)((v - lo) / (hi - lo) * (n - 1) + 0.5);
    if (k < 0) k = 0;
    if (k > n - 1) k = n - 1;
    return r[k];
}

void print_heatmap(const float *v, int h, int w, int stride_row, const char *indent, int cell) {
    double lo = v[0], hi = v[0];
    int y, x, k;
    for (y = 0; y < h; y++)
        for (x = 0; x < w; x++) {
            double a = v[y * stride_row + x];
            if (a < lo) lo = a;
            if (a > hi) hi = a;
        }
    for (y = 0; y < h; y++) {
        fputs(indent, stdout);
        for (x = 0; x < w; x++) {
            char ch = heat_char(v[y * stride_row + x], lo, hi);
            for (k = 0; k < cell; k++) putchar(ch);
        }
        putchar('\n');
    }
}

/* ---------------------------------------------------------- normalize */

void pr_normalize(const nmz_trace_event *ev, const print_opts *o) {
    const nmz_tensor *x = item(ev, "x"), *xn = item(ev, "x_norm"), *off = item(ev, "offset"), *sc = item(ev, "scale");
    int g, n = (int)off->size;
    size_t e = 0;
    printf("\n[input] normalize (%s)   (forward)\n", ev->detail);
    printf("  x' = (x %s offset_g) %s scale_g        fitted on the training data, one group per %s\n", sym("−", "-"),
           DOT, x->rank == 4 ? "channel" : "feature");
    for (g = 0; g < n && (o->max_rows <= 0 || g < o->max_rows); g++)
        printf("  group %d: offset %s, scale %s%s\n", g + 1, n4(off->data[g]), n4(sc->data[g]),
               !strcmp(ev->detail, "zscore") ? "   (mean, 1/std)" : "   (min, 1/(max-min))");
    if (n > o->max_rows && o->max_rows > 0) printf("    %s %d more groups\n", sym("…", "..."), n - o->max_rows);
    if (x->rank == 4) {
        int c, y, xx;
        focus(o, x, &c, &y, &xx);
        e = ((size_t)c * x->shape[2] + (size_t)y) * x->shape[3] + (size_t)xx;
        printf("  e.g. pixel [c=%d,y=%d,x=%d]: (%s %s %s) %s %s = %s\n", c + 1, y + 1, xx + 1, n4(x->data[e]),
               sym("−", "-"), n4(off->data[c]), DOT, n4(sc->data[c]), n4(xn->data[e]));
    } else {
        printf("  e.g. x_1: (%s %s %s) %s %s = %s\n", n4(x->data[0]), sym("−", "-"), n4(off->data[0]), DOT,
               n4(sc->data[0]), n4(xn->data[0]));
    }
}

/* ------------------------------------------------------------- conv2d */

static void conv_geom(const nmz_trace_event *ev, int *stride, int *pad) {
    *stride = scalar_i(ev, "stride", 1);
    *pad = scalar_i(ev, "pad", 0);
}

/* print up to three KH×KW matrices side by side */
static void side_by_side(const char *t1, const float *a, const char *t2, const float *b, const char *t3,
                         const float *c, int kh, int kw) {
    int y, x, colw = kw * 9;
    printf("    %-*s   %-*s   %s\n", colw, t1, colw, t2, t3 ? t3 : "");
    for (y = 0; y < kh; y++) {
        printf("    ");
        for (x = 0; x < kw; x++) printf("%9s", c7(a[y * kw + x]));
        printf("   ");
        for (x = 0; x < kw; x++) printf("%9s", c7(b[y * kw + x]));
        if (c) {
            printf("   ");
            for (x = 0; x < kw; x++) printf("%9s", c7(c[y * kw + x]));
        }
        printf("\n");
    }
}

/* the input patch of channel ch for output (oy,ox), zeros in the padding */
static void patch(const nmz_tensor *x, int ch, int oy, int ox, int kh, int kw, int s, int p, float *out) {
    int ky, kx, H = x->shape[2], W = x->shape[3];
    for (ky = 0; ky < kh; ky++)
        for (kx = 0; kx < kw; kx++) {
            int iy = oy * s + ky - p, ix = ox * s + kx - p;
            out[ky * kw + kx] = (iy < 0 || iy >= H || ix < 0 || ix >= W) ? 0.0f : at4(x, ch, iy, ix);
        }
}

void pr_conv_forward(const nmz_trace_event *ev, const print_opts *o) {
    const nmz_tensor *x = item(ev, "x"), *W = item(ev, "W"), *b = item(ev, "b"), *z = item(ev, "z");
    int s, p, f, oy, ox, ch, k, C = x->shape[1], F = W->shape[0], kh = W->shape[2], kw = W->shape[3];
    int show = o->full ? C : (C < 3 ? C : 3);
    float *pt = calloc((size_t)kh * kw, sizeof(float)), *pr = calloc((size_t)kh * kw, sizeof(float));
    double total = 0.0;
    conv_geom(ev, &s, &p);
    focus(o, z, &f, &oy, &ox);
    printf("\n[layer %d] conv2d %dx%dx%d %s %dx%dx%d   (forward)   %d filters %dx%d, stride %d, pad %d\n", ev->layer,
           C, x->shape[2], x->shape[3], ARROW, F, z->shape[2], z->shape[3], F, kh, kw, s, p);
    printf("  z[f,y,x] = b[f] + %s_c %s_ky %s_kx x[c, y%ss+ky%sp, x%ss+kx%sp] %s W[f,c,ky,kx]\n", SUM, SUM, SUM, DOT,
           sym("−", "-"), DOT, sym("−", "-"), DOT);
    printf("  in focus: filter %d, output position y=%d, x=%d%s  (choose another with --at F,Y,X)\n", f + 1, oy + 1,
           ox + 1, o->at_set ? "" : ", where this filter responds most");
    if (!pt || !pr) goto out;
    for (ch = 0; ch < C; ch++) {
        const float *wk = W->data + ((size_t)f * C + (size_t)ch) * kh * kw;
        double sub = 0.0;
        patch(x, ch, oy, ox, kh, kw, s, p, pt);
        for (k = 0; k < kh * kw; k++) {
            pr[k] = pt[k] * wk[k];
            sub += (double)pt[k] * wk[k];
        }
        total += sub;
        if (ch < show) {
            printf("  channel %d: rows %d..%d, cols %d..%d of the input%s\n", ch + 1, oy * s - p + 1,
                   oy * s - p + kh, ox * s - p + 1, ox * s - p + kw, p ? " (outside = zero padding)" : "");
            side_by_side("input patch x", pt, "kernel W", wk, sym("x · W (element-wise)", "x * W (element-wise)"), pr, kh, kw);
            printf("    %s of the products = %s\n", SUM, n4(sub));
        }
    }
    if (C > show) printf("  %s %d more channel(s) not shown (use --full)\n", sym("…", "..."), C - show);
    printf("  z = %s (all %d channel%s) + b = %s + %s = %s\n", n4(total), C, C == 1 ? "" : "s", n4(total),
           n4(b->data[f]), n4(at4(z, f, oy, ox)));
out:
    free(pt);
    free(pr);
}

void pr_conv_backward(const nmz_trace_event *ev, const print_opts *o) {
    const nmz_tensor *g = item(ev, "dL/dz"), *x = item(ev, "x"), *W = item(ev, "W"), *dW = item(ev, "dL/dW"),
                     *db = item(ev, "dL/db"), *dx = item(ev, "dL/dx");
    int s, p, f, oy, ox, k, C = x->shape[1], kh = W->shape[2], kw = W->shape[3];
    float *pt = calloc((size_t)kh * kw, sizeof(float)), *pr = calloc((size_t)kh * kw, sizeof(float));
    double d;
    conv_geom(ev, &s, &p);
    focus(o, g, &f, &oy, &ox);
    d = at4(g, f, oy, ox);
    printf("\n[layer %d] conv2d   (backward)   in focus: filter %d, y=%d, x=%d, %s = %sL/%sz = %s\n", ev->layer, f + 1,
           oy + 1, ox + 1, DELTA, D, D, n4(d));
    if (!pt || !pr) goto out;
    patch(x, 0, oy, ox, kh, kw, s, p, pt);
    for (k = 0; k < kh * kw; k++) pr[k] = (float)(d * pt[k]);
    printf("  this position's share of %sL/%sW[f] is %s %s patch (channel 1 shown):\n", D, D, DELTA, DOT);
    side_by_side("input patch x", pt, sym("δ · x", "delta * x"), pr, NULL, NULL, kh, kw);
    for (k = 0; k < kh * kw; k++) pr[k] = (float)(d * W->data[(size_t)f * C * kh * kw + k]);
    printf("  and it sends %s %s W[f] back to the same patch of the input (%sL/%sx, channel 1):\n", DELTA, DOT, D, D);
    side_by_side("kernel W", W->data + (size_t)f * C * kh * kw, sym("δ · W", "delta * W"), pr, NULL, NULL, kh, kw);
    printf("  summed over all %d output positions:\n", g->shape[2] * g->shape[3]);
    printf("    %sL/%sb[%d] = %s %s = %s\n", D, D, f + 1, SUM, DELTA, n4(db->data[f]));
    printf("    %sL/%sW[%d] (channel 1):\n", D, D, f + 1);
    {
        int y, xx;
        for (y = 0; y < kh; y++) {
            printf("    ");
            for (xx = 0; xx < kw; xx++) printf("%9s", c7(dW->data[(size_t)f * C * kh * kw + (size_t)y * kw + xx]));
            printf("\n");
        }
    }
    if (dx) {
        int nz = 0;
        size_t e;
        for (e = 0; e < dx->size; e++) nz += dx->data[e] != 0.0f;
        printf("    %sL/%sx: %d of %zu input values received a gradient%s\n", D, D, nz, dx->size,
               ev->layer == 0 ? " (the input image: nothing to update there)" : "");
    }
out:
    free(pt);
    free(pr);
}

/* ------------------------------------------------------------ maxpool2d */

void pr_pool_forward(const nmz_trace_event *ev, const print_opts *o) {
    const nmz_tensor *x = item(ev, "x"), *y = item(ev, "y"), *am = item(ev, "argmax");
    int size = scalar_i(ev, "size", 2), s = scalar_i(ev, "stride", 2), c, oy, ox, ky, kx, W = x->shape[3];
    int win;
    focus(o, y, &c, &oy, &ox);
    win = (int)at4(am, c, oy, ox);
    printf("\n[layer %d] maxpool2d %dx%dx%d %s %dx%dx%d   (forward)   window %dx%d, stride %d\n", ev->layer,
           x->shape[1], x->shape[2], W, ARROW, y->shape[1], y->shape[2], y->shape[3], size, size, s);
    printf("  y = max of the window; the winner's position is remembered for the backward pass\n");
    printf("  in focus: channel %d, output y=%d, x=%d  -> input rows %d..%d, cols %d..%d\n", c + 1, oy + 1, ox + 1,
           oy * s + 1, oy * s + size, ox * s + 1, ox * s + size);
    for (ky = 0; ky < size; ky++) {
        printf("    ");
        for (kx = 0; kx < size; kx++) {
            int iy = oy * s + ky, ix = ox * s + kx;
            int is_win = iy * W + ix == win;
            printf(is_win ? " [%8s]" : "  %8s ", c7(at4(x, c, iy, ix)));
        }
        printf("\n");
    }
    printf("  winner [ ] at row %d, col %d  ->  y = %s\n", win / W + 1, win % W + 1, n4(at4(y, c, oy, ox)));
}

void pr_pool_backward(const nmz_trace_event *ev, const print_opts *o) {
    const nmz_tensor *g = item(ev, "dL/dy"), *am = item(ev, "argmax"), *dx = item(ev, "dL/dx");
    int c, oy, ox, win, W;
    focus(o, g, &c, &oy, &ox);
    W = dx ? dx->shape[3] : 1;
    win = (int)at4(am, c, oy, ox);
    printf("\n[layer %d] maxpool2d   (backward)\n", ev->layer);
    printf("  the gradient goes only to the winner; the other window values get 0\n");
    printf("  in focus: channel %d, y=%d, x=%d: %sL/%sy = %s  ->  %sL/%sx[row %d, col %d] += %s\n", c + 1, oy + 1,
           ox + 1, D, D, n4(at4(g, c, oy, ox)), D, D, win / W + 1, win % W + 1, n4(at4(g, c, oy, ox)));
    if (dx) {
        size_t e;
        int nz = 0;
        for (e = 0; e < dx->size; e++) nz += dx->data[e] != 0.0f;
        printf("  %d of %zu inputs receive a gradient (one per window)\n", nz, dx->size);
    }
}

/* ------------------------------------------------------ flatten, dropout */

void pr_flatten(const nmz_trace_event *ev, int backward) {
    const nmz_tensor *a = item(ev, backward ? "dL/dx" : "x"), *b = item(ev, backward ? "dL/dy" : "y");
    if (!backward && a->rank == 4)
        printf("\n[layer %d] flatten %dx%dx%d %s %d   (forward)   the same numbers in one row: index = c%s%d + y%s%d + x\n",
               ev->layer, a->shape[1], a->shape[2], a->shape[3], ARROW, b->shape[1], DOT,
               a->shape[2] * a->shape[3], DOT, a->shape[3]);
    else if (!backward)
        printf("\n[layer %d] flatten   (forward)   %zu values, unchanged\n", ev->layer, a->size);
    else
        printf("\n[layer %d] flatten   (backward)   gradients reshaped back to the feature-map shape\n", ev->layer);
}

void pr_dropout(const nmz_trace_event *ev, int backward) {
    double rate = 0, training = 0;
    const nmz_tensor *m = item(ev, "mask");
    nmz_trace_scalar_get(ev, "rate", &rate);
    nmz_trace_scalar_get(ev, "training", &training);
    printf("\n[layer %d] dropout (rate %g)   (%s)\n", ev->layer, rate, backward ? "backward" : "forward");
    if (!training) {
        printf("  evaluation mode: values pass through unchanged (trace runs in evaluation mode; `step` trains)\n");
        return;
    }
    if (m) {
        size_t e, kept = 0;
        for (e = 0; e < m->size; e++) kept += m->data[e] != 0.0f;
        if (!backward)
            printf("  training: each value kept with probability %g and scaled by 1/(1-p) = %s; %zu of %zu kept\n",
                   1 - rate, n4(1.0 / (1.0 - rate)), kept, m->size);
        else
            printf("  %sL/%sx = %sL/%sy %s mask (dropped values get no gradient)\n", D, D, D, D, DOT);
    }
}

/* ----------------------------------------------- activations on maps */

int is_map(const nmz_tensor *t) { return t && t->rank == 4; }

void pr_act_map_forward(const nmz_trace_event *ev, const print_opts *o) {
    const nmz_tensor *z = item(ev, "z"), *a = item(ev, "a");
    int c, y, x;
    size_t e, zero = 0;
    focus(o, z, &c, &y, &x);
    for (e = 0; e < a->size; e++) zero += a->data[e] == 0.0f;
    printf("\n[layer %d] activation %s   (forward)   applied to all %zu values of the %dx%dx%d maps\n", ev->layer,
           ev->detail, z->size, z->shape[1], z->shape[2], z->shape[3]);
    printf("  in focus [c=%d,y=%d,x=%d]: a = %s(%s) = %s", c + 1, y + 1, x + 1, ev->detail, n4(at4(z, c, y, x)),
           n4(at4(a, c, y, x)));
    if (!strcmp(ev->detail, "relu")) printf("      (%zu of %zu values are 0 after relu)", zero, a->size);
    printf("\n");
}

void pr_act_map_backward(const nmz_trace_event *ev, const print_opts *o) {
    const nmz_tensor *g = item(ev, "dL/da"), *fp = item(ev, "f'(z)"), *d = item(ev, "dL/dz");
    int c, y, x;
    focus(o, g, &c, &y, &x);
    printf("\n[layer %d] activation %s   (backward)   %s = %sL/%sa %s f'(z), element-wise\n", ev->layer, ev->detail,
           DELTA, D, D, DOT);
    if (fp)
        printf("  in focus [c=%d,y=%d,x=%d]: %s %s %s = %s\n", c + 1, y + 1, x + 1, n4(at4(g, c, y, x)), DOT,
               n4(at4(fp, c, y, x)), n4(at4(d, c, y, x)));
}

/* ------------------------------------------------------------ Adam table */

void fmt_index(char *buf, size_t n, const nmz_tensor *t, size_t e) {
    int d, w = 0;
    size_t rest = e, idx[NMZ_MAX_RANK];
    for (d = t->rank - 1; d >= 0; d--) {
        idx[d] = rest % (size_t)t->shape[d];
        rest /= (size_t)t->shape[d];
    }
    buf[0] = '\0';
    for (d = 0; d < t->rank && w >= 0 && (size_t)w < n; d++) w += snprintf(buf + w, n - (size_t)w, d ? ",%zu" : "%zu", idx[d] + 1);
}

void pr_adam_header(const nmz_trace_event *ev) {
    double lr = 0, b1 = 0, b2 = 0, eps = 0, t = 0;
    nmz_trace_scalar_get(ev, "lr", &lr);
    nmz_trace_scalar_get(ev, "beta1", &b1);
    nmz_trace_scalar_get(ev, "beta2", &b2);
    nmz_trace_scalar_get(ev, "epsilon", &eps);
    nmz_trace_scalar_get(ev, "t", &t);
    printf("  Adam step t = %g   %s = %g, %s1 = %g, %s2 = %g, %s = %g\n", t, sym("η", "lr"), lr, sym("β", "beta"), b1,
           sym("β", "beta"), b2, sym("ε", "eps"), eps);
    printf("  m %s %s1%sm + (1%s%s1)%sg      v %s %s2%sv + (1%s%s2)%sg%s\n", sym("←", "<-"), sym("β", "beta"), DOT,
           sym("−", "-"), sym("β", "beta"), DOT, sym("←", "<-"), sym("β", "beta"), DOT, sym("−", "-"), sym("β", "beta"),
           DOT, sym("²", "^2"));
    printf("  m%s = m/(1%s%s1^t)   v%s = v/(1%s%s2^t)   %s = %s%s%sm%s/(%sv%s + %s)\n", sym("̂", "_hat"), sym("−", "-"),
           sym("β", "beta"), sym("̂", "_hat"), sym("−", "-"), sym("β", "beta"), sym("Δ", "delta"), sym("−", "-"),
           sym("η", "lr"), DOT, sym("̂", "_hat"), sym("√", "sqrt "), sym("̂", "_hat"), sym("ε", "eps"));
}

void pr_adam_update(const nmz_trace_event *ev, const print_opts *o) {
    const nmz_tensor *w = item(ev, "value"), *g = item(ev, "grad"), *m = item(ev, "m"), *v = item(ev, "v"),
                     *mh = item(ev, "m_hat"), *vh = item(ev, "v_hat"), *dl = item(ev, "delta");
    size_t e, n = w->size;
    printf("\n[layer %d] %s %s  (%zu value%s)\n", ev->layer, ev->layer_type, ev->detail, n, n == 1 ? "" : "s");
    printf("  %10s %10s %10s %10s %10s %10s %10s %10s %10s\n", "index", "w old", "grad", "m", "v", "m_hat", "v_hat",
           "delta", "w new");
    for (e = 0; e < n; e++) {
        char idx[48];
        if (o->max_rows > 0 && e >= (size_t)o->max_rows) {
            printf("    %s %zu more value(s) not shown (use --full)\n", sym("…", "..."), n - (size_t)o->max_rows);
            break;
        }
        fmt_index(idx, sizeof idx, w, e);
        printf("  %10s %10s %10s %10s %10s %10s %10s %10s %10s\n", idx, c7(w->data[e]), c7(g->data[e]),
               c7(m->data[e]), c7(v->data[e]), c7(mh->data[e]), c7(vh->data[e]), c7(dl->data[e]),
               c7((double)w->data[e] + dl->data[e]));
    }
}
