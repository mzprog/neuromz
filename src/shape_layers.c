/* shape_layers.c — MaxPool2D, Flatten and Dropout: layers without weights.
 *
 * MaxPool2D  x [C,H,W] → y [C,OH,OW], OH = (H − K)/S + 1 (no padding)
 *            y = max over the K×K window; the index of the winner is
 *            recorded ("argmax", flat index iy·W+ix in the input plane;
 *            ties go to the first maximum in row-major order).
 *            backward: the whole gradient goes to the winner, 0 elsewhere.
 * Flatten    x [d1,d2,...] → y [d1·d2·...]; data is already contiguous,
 *            so forward and backward are plain copies.
 * Dropout    training: y = x · m, m = 0 with probability p, else 1/(1−p)
 *            ("inverted" dropout: evaluation needs no rescaling)
 *            evaluation: y = x.   backward: dx = dy · m (or dy). */
#include "internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int no_params(const nmz_layer *l) {
    (void)l;
    return 0;
}

static nmz_param no_param(nmz_layer *l, int i) {
    nmz_param p = {NULL, NULL, NULL};
    (void)l;
    (void)i;
    return p;
}

static nmz_status no_init(nmz_layer *l, nmz_rng *rng, const nmz_layer *next) {
    (void)l;
    (void)rng;
    (void)next;
    return NMZ_OK;
}

static void shape_str(char *buf, size_t n, int rank, const int *shape) {
    int i, w = 0;
    buf[0] = '\0';
    for (i = 0; i < rank && w >= 0 && (size_t)w < n; i++) w += snprintf(buf + w, n - (size_t)w, i ? "x%d" : "%d", shape[i]);
}

/* ================================================================ MaxPool */

typedef struct pool_layer {
    int size, stride;
    int C, H, W, OH, OW;
    nmz_tensor argmax; /* [B, C, OH, OW], flat index in the input plane */
} pool_layer;

static nmz_status pool_setup(nmz_layer *l, int in_rank, const int *in_shape, int *out_rank, int *out_shape) {
    pool_layer *p = l->self;
    if (in_rank != 3) return nmz__fail(NMZ_ERR_SHAPE, "maxpool2d needs [channels, height, width] input");
    p->C = in_shape[0];
    p->H = in_shape[1];
    p->W = in_shape[2];
    if (p->H < p->size || p->W < p->size)
        return nmz__fail(NMZ_ERR_SHAPE, "maxpool2d window %d is larger than the input %dx%d", p->size, p->H, p->W);
    p->OH = (p->H - p->size) / p->stride + 1;
    p->OW = (p->W - p->size) / p->stride + 1;
    *out_rank = 3;
    out_shape[0] = p->C;
    out_shape[1] = p->OH;
    out_shape[2] = p->OW;
    return NMZ_OK;
}

static void pool_emit(const nmz_ctx *ctx, const pool_layer *p, nmz_trace_phase ph, int n, const nmz_trace_item *it) {
    nmz_trace_scalar sc[2];
    sc[0].name = "size";
    sc[0].value = p->size;
    sc[1].name = "stride";
    sc[1].value = p->stride;
    nmz_trace_emit(ctx, ph, "maxpool2d", "", 2, sc, n, it);
}

/* The CPU kernels (the v2.0.1 loops). */
nmz_status nmz__cpu_maxpool_forward(const nmz_backend *be, int planes, int H, int W, int size, int stride, int OH,
                                    int OW, const float *x, float *y, float *argmax) {
    int bc, oy, ox, ky, kx;
    size_t in_plane = (size_t)H * W, out_plane = (size_t)OH * OW;
    (void)be;
    for (bc = 0; bc < planes; bc++) {
        const float *xp = x + (size_t)bc * in_plane;
        float *yp = y + (size_t)bc * out_plane, *ap = argmax + (size_t)bc * out_plane;
        for (oy = 0; oy < OH; oy++)
            for (ox = 0; ox < OW; ox++) {
                int best = (oy * stride) * W + ox * stride;
                for (ky = 0; ky < size; ky++)
                    for (kx = 0; kx < size; kx++) {
                        int idx = (oy * stride + ky) * W + ox * stride + kx;
                        if (xp[idx] > xp[best]) best = idx;
                    }
                yp[oy * OW + ox] = xp[best];
                ap[oy * OW + ox] = (float)best;
            }
    }
    return NMZ_OK;
}

nmz_status nmz__cpu_maxpool_backward(const nmz_backend *be, int planes, int H, int W, int size, int stride, int OH,
                                     int OW, const float *dy, const float *argmax, float *dx) {
    int bc;
    size_t in_plane = (size_t)H * W, out_plane = (size_t)OH * OW, o;
    (void)be;
    (void)size;
    (void)stride; /* the winners are recorded: a scatter needs no window */
    memset(dx, 0, (size_t)planes * in_plane * sizeof(float));
    for (bc = 0; bc < planes; bc++) {
        const float *gp = dy + (size_t)bc * out_plane, *ap = argmax + (size_t)bc * out_plane;
        float *dp = dx + (size_t)bc * in_plane;
        for (o = 0; o < out_plane; o++) dp[(size_t)ap[o]] += gp[o]; /* overlapping windows add up */
    }
    return NMZ_OK;
}

static nmz_status pool_forward(nmz_layer *l, const nmz_tensor *x, nmz_tensor *y, const nmz_ctx *ctx) {
    pool_layer *p = l->self;
    const nmz_backend *be = ctx->backend;
    const float *px;
    float *py, *pa;
    nmz_status s = nmz_tensor_reshape(&p->argmax, y->rank, y->shape);
    if (s != NMZ_OK) return s;
    px = nmz__r(be, x);
    py = nmz__w(be, y);
    pa = nmz__w(be, &p->argmax);
    if (!px || !py || !pa) return nmz__op_failed();
    if ((s = be->maxpool_forward(be, x->shape[0] * p->C, p->H, p->W, p->size, p->stride, p->OH, p->OW, px, py, pa)) !=
        NMZ_OK)
        return s;
    if (NMZ_TRACING(ctx)) {
        nmz_trace_item it[3] = {{"x", x}, {"y", y}, {"argmax", &p->argmax}};
        pool_emit(ctx, p, NMZ_TRACE_FORWARD, 3, it);
    }
    return NMZ_OK;
}

static nmz_status pool_backward(nmz_layer *l, const nmz_tensor *x, const nmz_tensor *y, const nmz_tensor *dy,
                                nmz_tensor *dx, const nmz_ctx *ctx) {
    pool_layer *p = l->self;
    const nmz_backend *be = ctx->backend;
    const float *pdy, *pa;
    float *pdx;
    nmz_status s;
    (void)y;
    if (!dx) return NMZ_OK;
    pdy = nmz__r(be, dy);
    pa = nmz__r(be, &p->argmax);
    pdx = nmz__w(be, dx);
    if (!pdy || !pa || !pdx) return nmz__op_failed();
    if ((s = be->maxpool_backward(be, x->shape[0] * p->C, p->H, p->W, p->size, p->stride, p->OH, p->OW, pdy, pa, pdx)) !=
        NMZ_OK)
        return s;
    if (NMZ_TRACING(ctx)) {
        nmz_trace_item it[3] = {{"dL/dy", dy}, {"argmax", &p->argmax}, {"dL/dx", dx}};
        pool_emit(ctx, p, NMZ_TRACE_BACKWARD, 3, it);
    }
    return NMZ_OK;
}

static nmz_status pool_save(const nmz_layer *l, nmz_writer *w) {
    const pool_layer *p = l->self;
    nmz_write_u32(w, (uint32_t)p->size);
    return nmz_write_u32(w, (uint32_t)p->stride);
}

static nmz_status pool_load(nmz_layer **out, nmz_reader *r) {
    uint32_t size = 0, stride = 0;
    nmz_read_u32(r, &size);
    if (nmz_read_u32(r, &stride) != NMZ_OK) return r->err;
    if (size == 0 || size > 1000 || stride == 0 || stride > 1000)
        return nmz__fail(NMZ_ERR_FORMAT, "corrupt maxpool2d config");
    *out = nmz_layer_maxpool2d((int)size, (int)stride);
    return *out ? NMZ_OK : NMZ_ERR_NOMEM;
}

static void pool_describe(const nmz_layer *l, char *buf, size_t n) {
    const pool_layer *p = l->self;
    snprintf(buf, n, "maxpool2d %dx%dx%d->%dx%dx%d (%dx%d, stride %d)", p->C, p->H, p->W, p->C, p->OH, p->OW,
             p->size, p->size, p->stride);
}

static void pool_destroy(nmz_layer *l) {
    pool_layer *p = l->self;
    if (p) nmz_tensor_free(&p->argmax);
    free(p);
}

static int pool_buffers(nmz_layer *l, nmz_tensor **out, int max) {
    pool_layer *p = l->self;
    nmz_tensor *all[] = {&p->argmax};
    return nmz__list(out, max, (int)(sizeof all / sizeof all[0]), all);
}

const nmz_layer_ops nmz__maxpool2d_ops = {"maxpool2d", pool_setup, no_init,   pool_forward,  pool_backward, no_params,
                                          no_param,    pool_save,  pool_load, pool_describe, pool_destroy, NULL,
                                          NMZ_LAYER_BACKEND, NULL, pool_buffers};

const nmz_tensor *nmz__maxpool_argmax(const nmz_layer *l) { return &((const pool_layer *)l->self)->argmax; }

nmz_layer *nmz_layer_maxpool2d(int size, int stride) {
    pool_layer *p;
    nmz_layer *l;
    if (stride == 0) stride = size;
    if (size <= 0 || size > 1000 || stride <= 0 || stride > 1000) {
        nmz__fail(NMZ_ERR_ARG, "maxpool2d needs a positive window size and stride");
        return NULL;
    }
    p = calloc(1, sizeof *p);
    if (!p) {
        nmz__fail(NMZ_ERR_NOMEM, "out of memory");
        return NULL;
    }
    p->size = size;
    p->stride = stride;
    l = nmz_layer_alloc(&nmz__maxpool2d_ops, p);
    if (!l) free(p);
    return l;
}

/* ================================================================ Flatten */

typedef struct flatten_layer {
    int dummy;
} flatten_layer;

static nmz_status flat_setup(nmz_layer *l, int in_rank, const int *in_shape, int *out_rank, int *out_shape) {
    size_t n;
    nmz_status s = nmz__shape_size(in_rank, in_shape, &n);
    (void)l;
    if (s != NMZ_OK) return s;
    if (n > (size_t)NMZ_MAX_DIM) return nmz__fail(NMZ_ERR_SHAPE, "flatten: too many values");
    *out_rank = 1;
    out_shape[0] = (int)n;
    return NMZ_OK;
}

/* a 1-row copy in the backend's memory */
static nmz_status copy_all(const nmz_backend *be, const nmz_tensor *src, nmz_tensor *dst) {
    const float *ps = nmz__r(be, src);
    float *pd = nmz__w(be, dst);
    if (!src->size) return NMZ_OK;
    if (!ps || !pd) return nmz__op_failed();
    return be->copy(be, 1, src->size, ps, src->size, pd, src->size);
}

static nmz_status flat_forward(nmz_layer *l, const nmz_tensor *x, nmz_tensor *y, const nmz_ctx *ctx) {
    nmz_status s;
    (void)l;
    if ((s = copy_all(ctx->backend, x, y)) != NMZ_OK) return s;
    if (NMZ_TRACING(ctx)) {
        nmz_trace_item it[2] = {{"x", x}, {"y", y}};
        nmz_trace_emit(ctx, NMZ_TRACE_FORWARD, "flatten", "", 0, NULL, 2, it);
    }
    return NMZ_OK;
}

static nmz_status flat_backward(nmz_layer *l, const nmz_tensor *x, const nmz_tensor *y, const nmz_tensor *dy,
                                nmz_tensor *dx, const nmz_ctx *ctx) {
    nmz_status s;
    (void)l;
    (void)x;
    (void)y;
    if (!dx) return NMZ_OK;
    if ((s = copy_all(ctx->backend, dy, dx)) != NMZ_OK) return s;
    if (NMZ_TRACING(ctx)) {
        nmz_trace_item it[2] = {{"dL/dy", dy}, {"dL/dx", dx}};
        nmz_trace_emit(ctx, NMZ_TRACE_BACKWARD, "flatten", "", 0, NULL, 2, it);
    }
    return NMZ_OK;
}

static nmz_status flat_save(const nmz_layer *l, nmz_writer *w) {
    (void)l;
    (void)w;
    return NMZ_OK;
}

static nmz_status flat_load(nmz_layer **out, nmz_reader *r) {
    (void)r;
    *out = nmz_layer_flatten();
    return *out ? NMZ_OK : NMZ_ERR_NOMEM;
}

static void flat_describe(const nmz_layer *l, char *buf, size_t n) {
    char in[64];
    shape_str(in, sizeof in, l->in_rank, l->in_shape);
    snprintf(buf, n, "flatten %s->%d", in, l->out_rank ? l->out_shape[0] : 0);
}

static void flat_destroy(nmz_layer *l) { free(l->self); }

const nmz_layer_ops nmz__flatten_ops = {"flatten", flat_setup, no_init,   flat_forward,  flat_backward, no_params,
                                        no_param,  flat_save,  flat_load, flat_describe, flat_destroy, NULL,
                                        NMZ_LAYER_BACKEND, NULL, NULL};

nmz_layer *nmz_layer_flatten(void) {
    flatten_layer *f = calloc(1, sizeof *f);
    nmz_layer *l;
    if (!f) {
        nmz__fail(NMZ_ERR_NOMEM, "out of memory");
        return NULL;
    }
    l = nmz_layer_alloc(&nmz__flatten_ops, f);
    if (!l) free(f);
    return l;
}

/* ================================================================ Dropout */

typedef struct dropout_layer {
    float rate;
    int was_training; /* mode of the last forward (decides backward) */
    nmz_tensor mask;  /* 0 or 1/(1−p), training only */
} dropout_layer;

static nmz_status drop_setup(nmz_layer *l, int in_rank, const int *in_shape, int *out_rank, int *out_shape) {
    int i;
    (void)l;
    *out_rank = in_rank;
    for (i = 0; i < in_rank; i++) out_shape[i] = in_shape[i];
    return NMZ_OK;
}

static nmz_status drop_forward(nmz_layer *l, const nmz_tensor *x, nmz_tensor *y, const nmz_ctx *ctx) {
    dropout_layer *d = l->self;
    const nmz_backend *be = ctx->backend;
    nmz_status s;
    d->was_training = ctx->training && d->rate > 0.0f;
    if (d->was_training) {
        const float *px;
        float *pm, *py;
        if ((s = nmz_tensor_reshape(&d->mask, x->rank, x->shape)) != NMZ_OK) return s;
        px = nmz__r(be, x);
        pm = nmz__w(be, &d->mask);
        py = nmz__w(be, y);
        if (!px || !pm || !py) return nmz__op_failed();
        if ((s = be->dropout_mask(be, x->size, d->rate, ctx->rng, pm)) != NMZ_OK ||
            (s = be->mul(be, x->size, px, pm, py)) != NMZ_OK)
            return s;
    } else if ((s = copy_all(be, x, y)) != NMZ_OK) {
        return s;
    }
    if (NMZ_TRACING(ctx)) {
        nmz_trace_scalar sc[2] = {{"rate", d->rate}, {"training", d->was_training}};
        nmz_trace_item it[3] = {{"x", x}, {"y", y}, {"mask", &d->mask}};
        nmz_trace_emit(ctx, NMZ_TRACE_FORWARD, "dropout", "", 2, sc, d->was_training ? 3 : 2, it);
    }
    return NMZ_OK;
}

static nmz_status drop_backward(nmz_layer *l, const nmz_tensor *x, const nmz_tensor *y, const nmz_tensor *dy,
                                nmz_tensor *dx, const nmz_ctx *ctx) {
    dropout_layer *d = l->self;
    const nmz_backend *be = ctx->backend;
    nmz_status s;
    (void)x;
    (void)y;
    if (!dx) return NMZ_OK;
    if (d->was_training) {
        const float *pdy = nmz__r(be, dy), *pm = nmz__r(be, &d->mask);
        float *pdx = nmz__w(be, dx);
        if (!pdy || !pm || !pdx) return nmz__op_failed();
        if ((s = be->mul(be, dy->size, pdy, pm, pdx)) != NMZ_OK) return s;
    } else if ((s = copy_all(be, dy, dx)) != NMZ_OK) {
        return s;
    }
    if (NMZ_TRACING(ctx)) {
        nmz_trace_scalar sc[2] = {{"rate", d->rate}, {"training", d->was_training}};
        nmz_trace_item it[3] = {{"dL/dy", dy}, {"dL/dx", dx}, {"mask", &d->mask}};
        nmz_trace_emit(ctx, NMZ_TRACE_BACKWARD, "dropout", "", 2, sc, d->was_training ? 3 : 2, it);
    }
    return NMZ_OK;
}

static nmz_status drop_save(const nmz_layer *l, nmz_writer *w) {
    return nmz_write_f32(w, ((const dropout_layer *)l->self)->rate);
}

static nmz_status drop_load(nmz_layer **out, nmz_reader *r) {
    float rate = 0;
    if (nmz_read_f32(r, &rate) != NMZ_OK) return r->err;
    if (!(rate >= 0.0f && rate < 1.0f)) return nmz__fail(NMZ_ERR_FORMAT, "corrupt dropout rate");
    *out = nmz_layer_dropout(rate);
    return *out ? NMZ_OK : NMZ_ERR_NOMEM;
}

static void drop_describe(const nmz_layer *l, char *buf, size_t n) {
    snprintf(buf, n, "dropout (rate %g, training only)", (double)((const dropout_layer *)l->self)->rate);
}

static void drop_destroy(nmz_layer *l) {
    dropout_layer *d = l->self;
    if (d) nmz_tensor_free(&d->mask);
    free(d);
}

static int drop_buffers(nmz_layer *l, nmz_tensor **out, int max) {
    dropout_layer *d = l->self;
    nmz_tensor *all[] = {&d->mask};
    return nmz__list(out, max, (int)(sizeof all / sizeof all[0]), all);
}

const nmz_layer_ops nmz__dropout_ops = {"dropout", drop_setup, no_init,   drop_forward,  drop_backward, no_params,
                                        no_param,  drop_save,  drop_load, drop_describe, drop_destroy, NULL,
                                        NMZ_LAYER_BACKEND, NULL, drop_buffers};

nmz_layer *nmz_layer_dropout(float rate) {
    dropout_layer *d;
    nmz_layer *l;
    if (!(rate >= 0.0f && rate < 1.0f)) {
        nmz__fail(NMZ_ERR_ARG, "dropout rate must be in [0, 1)");
        return NULL;
    }
    d = calloc(1, sizeof *d);
    if (!d) {
        nmz__fail(NMZ_ERR_NOMEM, "out of memory");
        return NULL;
    }
    d->rate = rate;
    l = nmz_layer_alloc(&nmz__dropout_ops, d);
    if (!l) free(d);
    return l;
}
