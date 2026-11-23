/* dense.c — fully connected layer.
 *
 *   forward:   z_j = Σ_i x_i · W_ij + b_j          (W is [inputs, units])
 *   backward:  δ_j = ∂L/∂z_j  (given)
 *              ∂L/∂W_ij += Σ_rows x_i · δ_j
 *              ∂L/∂b_j  += Σ_rows δ_j
 *              ∂L/∂x_i   = Σ_j W_ij · δ_j
 *
 * Dense acts on the last dimension; all leading dimensions are rows, so the
 * same layer works on [B, F] now and on [B, T, F] sequences later. */
#include "internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct dense_layer {
    int units;
    int n_in;
    nmz_init init;     /* as requested (may be AUTO) */
    nmz_init resolved; /* what was actually used     */
    nmz_tensor W, b, dW, db;
    double *acc; /* units doubles, forward accumulator */
} dense_layer;

static nmz_status dense_setup(nmz_layer *l, int in_rank, const int *in_shape, int *out_rank,
                              int *out_shape) {
    dense_layer *d = l->self;
    int wshape[2], i;
    nmz_status s;
    if (in_rank < 1) return nmz__fail(NMZ_ERR_SHAPE, "dense needs at least rank-1 input");
    d->n_in = in_shape[in_rank - 1];
    wshape[0] = d->n_in;
    wshape[1] = d->units;
    if ((s = nmz_tensor_init(&d->W, 2, wshape)) != NMZ_OK) return s;
    if ((s = nmz_tensor_init(&d->dW, 2, wshape)) != NMZ_OK) return s;
    if ((s = nmz_tensor_init(&d->b, 1, &d->units)) != NMZ_OK) return s;
    if ((s = nmz_tensor_init(&d->db, 1, &d->units)) != NMZ_OK) return s;
    free(d->acc);
    d->acc = malloc((size_t)d->units * sizeof *d->acc);
    if (!d->acc) return nmz__fail(NMZ_ERR_NOMEM, "out of memory");
    *out_rank = in_rank;
    for (i = 0; i < in_rank - 1; i++) out_shape[i] = in_shape[i];
    out_shape[in_rank - 1] = d->units;
    return NMZ_OK;
}

static nmz_status dense_init(nmz_layer *l, nmz_rng *rng, const nmz_layer *next) {
    dense_layer *d = l->self;
    double limit;
    size_t i;
    nmz_init init = d->init;
    if (init == NMZ_INIT_AUTO) {
        int k = next ? nmz_layer_activation_kind(next) : -1;
        init = (k == NMZ_ACT_RELU || k == NMZ_ACT_LEAKY_RELU) ? NMZ_INIT_HE : NMZ_INIT_XAVIER;
    }
    d->resolved = init;
    if (init == NMZ_INIT_HE)
        limit = sqrt(6.0 / d->n_in); /* Var = 2/fan_in */
    else
        limit = sqrt(6.0 / (d->n_in + d->units)); /* Var = 2/(fan_in+fan_out) */
    for (i = 0; i < d->W.size; i++) d->W.data[i] = (float)((2.0 * nmz_rng_uniform(rng) - 1.0) * limit);
    nmz_tensor_zero(&d->b);
    return NMZ_OK;
}

static nmz_status dense_forward(nmz_layer *l, const nmz_tensor *x, nmz_tensor *y,
                                const nmz_ctx *ctx) {
    dense_layer *d = l->self;
    size_t rows = nmz_tensor_rows(x), r;
    int i, j, n = d->n_in, u = d->units;
    for (r = 0; r < rows; r++) {
        const float *xr = x->data + r * (size_t)n;
        float *yr = y->data + r * (size_t)u;
        for (j = 0; j < u; j++) d->acc[j] = d->b.data[j];
        for (i = 0; i < n; i++) {
            const float *wi = d->W.data + (size_t)i * u;
            double xi = xr[i];
            for (j = 0; j < u; j++) d->acc[j] += xi * wi[j];
        }
        for (j = 0; j < u; j++) yr[j] = (float)d->acc[j];
    }
    if (NMZ_TRACING(ctx)) {
        nmz_trace_item it[4] = {{"x", x}, {"W", &d->W}, {"b", &d->b}, {"z", y}};
        nmz_trace_emit(ctx, NMZ_TRACE_FORWARD, "dense", "", 0, NULL, 4, it);
    }
    return NMZ_OK;
}

static nmz_status dense_backward(nmz_layer *l, const nmz_tensor *x, const nmz_tensor *y,
                                 const nmz_tensor *dy, nmz_tensor *dx, const nmz_ctx *ctx) {
    dense_layer *d = l->self;
    size_t rows = nmz_tensor_rows(x), r;
    int i, j, n = d->n_in, u = d->units;
    (void)y;
    for (r = 0; r < rows; r++) {
        const float *xr = x->data + r * (size_t)n;
        const float *gr = dy->data + r * (size_t)u;
        for (j = 0; j < u; j++) d->db.data[j] += gr[j];
        for (i = 0; i < n; i++) {
            float *dwi = d->dW.data + (size_t)i * u;
            float xi = xr[i];
            for (j = 0; j < u; j++) dwi[j] += xi * gr[j];
        }
        if (dx) {
            float *dxr = dx->data + r * (size_t)n;
            for (i = 0; i < n; i++) {
                const float *wi = d->W.data + (size_t)i * u;
                double s = 0.0;
                for (j = 0; j < u; j++) s += (double)wi[j] * gr[j];
                dxr[i] = (float)s;
            }
        }
    }
    if (NMZ_TRACING(ctx)) {
        nmz_trace_item it[5] = {{"dL/dz", dy}, {"x", x}, {"W", &d->W}, {"dL/dW", &d->dW},
                                {"dL/db", &d->db}};
        int n_items = 5;
        nmz_trace_item all[6];
        for (i = 0; i < n_items; i++) all[i] = it[i];
        if (dx) {
            all[n_items].name = "dL/dx";
            all[n_items].t = dx;
            n_items++;
        }
        nmz_trace_emit(ctx, NMZ_TRACE_BACKWARD, "dense", "", 0, NULL, n_items, all);
    }
    return NMZ_OK;
}

static int dense_param_count(const nmz_layer *l) {
    (void)l;
    return 2;
}

static nmz_param dense_param(nmz_layer *l, int i) {
    dense_layer *d = l->self;
    nmz_param p = {NULL, NULL, NULL};
    if (i == 0) {
        p.name = "W";
        p.value = &d->W;
        p.grad = &d->dW;
    } else if (i == 1) {
        p.name = "b";
        p.value = &d->b;
        p.grad = &d->db;
    }
    return p;
}

static nmz_status dense_save(const nmz_layer *l, nmz_writer *w) {
    const dense_layer *d = l->self;
    nmz_write_u32(w, (uint32_t)d->units);
    nmz_write_u32(w, (uint32_t)d->init);
    return nmz_write_u32(w, (uint32_t)d->resolved);
}

static nmz_status dense_load(nmz_layer **out, nmz_reader *r) {
    uint32_t units = 0, init = 0, resolved = 0;
    nmz_read_u32(r, &units);
    nmz_read_u32(r, &init);
    if (nmz_read_u32(r, &resolved) != NMZ_OK) return r->err;
    if (units == 0 || units > NMZ_MAX_DIM || init >= NMZ_INIT_COUNT || resolved >= NMZ_INIT_COUNT)
        return nmz__fail(NMZ_ERR_FORMAT, "corrupt dense layer config");
    *out = nmz_layer_dense((int)units, (nmz_init)init);
    if (!*out) return NMZ_ERR_NOMEM;
    ((dense_layer *)(*out)->self)->resolved = (nmz_init)resolved;
    return NMZ_OK;
}

static void dense_describe(const nmz_layer *l, char *buf, size_t n) {
    const dense_layer *d = l->self;
    const char *init = nmz_init_name(d->resolved);
    if (d->init == NMZ_INIT_AUTO) init = d->resolved == NMZ_INIT_AUTO ? "auto" : init;
    snprintf(buf, n, "dense %d->%d (%s init)", d->n_in, d->units, init);
}

static void dense_destroy(nmz_layer *l) {
    dense_layer *d = l->self;
    if (!d) return;
    nmz_tensor_free(&d->W);
    nmz_tensor_free(&d->b);
    nmz_tensor_free(&d->dW);
    nmz_tensor_free(&d->db);
    free(d->acc);
    free(d);
}

const nmz_layer_ops nmz__dense_ops = {
    "dense", dense_setup, dense_init, dense_forward, dense_backward, dense_param_count,
    dense_param, dense_save, dense_load, dense_describe, dense_destroy, NULL, 0};

int nmz_layer_dense_init(const nmz_layer *l) {
    if (!l || l->ops != &nmz__dense_ops) return -1;
    return (int)((const dense_layer *)l->self)->resolved;
}

int nmz_layer_init_used(const nmz_layer *l) {
    if (l && l->ops == &nmz__conv2d_ops) return nmz__conv2d_init_used(l);
    if (l && (l->ops == &nmz__embedding_ops || nmz_layer_is_recurrent(l))) return nmz__recurrent_init_used(l);
    return nmz_layer_dense_init(l);
}

nmz_layer *nmz_layer_dense(int units, nmz_init init) {
    dense_layer *d;
    nmz_layer *l;
    if (units <= 0 || units > NMZ_MAX_DIM) {
        nmz__fail(NMZ_ERR_ARG, "dense layer needs a positive number of units (got %d)", units);
        return NULL;
    }
    if ((int)init < 0 || init >= NMZ_INIT_COUNT) {
        nmz__fail(NMZ_ERR_ARG, "invalid init %d", (int)init);
        return NULL;
    }
    if (init == NMZ_INIT_ORTHOGONAL) {
        nmz__fail(NMZ_ERR_ARG, "orthogonal init is for recurrent layers (rnn, lstm, gru), not dense");
        return NULL;
    }
    d = calloc(1, sizeof *d);
    if (!d) {
        nmz__fail(NMZ_ERR_NOMEM, "out of memory");
        return NULL;
    }
    d->units = units;
    d->init = init;
    d->resolved = init;
    l = nmz_layer_alloc(&nmz__dense_ops, d);
    if (!l) free(d);
    return l;
}
