/* activation.c — activation functions, their derivatives, and the
 * Activation layer (a = f(z), element-wise; softmax over the last dim). */
#include "internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *const k_act_names[NMZ_ACT_COUNT] = {
    "linear", "sigmoid", "tanh", "relu", "leaky_relu", "softmax"};

const char *nmz_act_name(nmz_act a) {
    if ((int)a < 0 || a >= NMZ_ACT_COUNT) return "?";
    return k_act_names[a];
}

nmz_status nmz_act_parse(const char *s, nmz_act *out) {
    int i;
    if (!s || !out) return nmz__fail(NMZ_ERR_ARG, "NULL argument");
    for (i = 0; i < NMZ_ACT_COUNT; i++)
        if (strcmp(s, k_act_names[i]) == 0) {
            *out = (nmz_act)i;
            return NMZ_OK;
        }
    if (strcmp(s, "lrelu") == 0 || strcmp(s, "leaky") == 0) { *out = NMZ_ACT_LEAKY_RELU; return NMZ_OK; }
    if (strcmp(s, "identity") == 0 || strcmp(s, "none") == 0) { *out = NMZ_ACT_LINEAR; return NMZ_OK; }
    return nmz__fail(NMZ_ERR_ARG,
                     "unknown activation '%s' (expected linear, sigmoid, tanh, relu, "
                     "leaky_relu, softmax)", s);
}

/* Numerically stable logistic function. */
static double sigmoid_d(double z) {
    if (z >= 0) return 1.0 / (1.0 + exp(-z));
    {
        double e = exp(z);
        return e / (1.0 + e);
    }
}

float nmz_act_apply(nmz_act a, float z, float alpha) {
    switch (a) {
    case NMZ_ACT_LINEAR: return z;
    case NMZ_ACT_SIGMOID: return (float)sigmoid_d(z);
    case NMZ_ACT_TANH: return (float)tanh(z);
    case NMZ_ACT_RELU: return z > 0.0f ? z : 0.0f;
    case NMZ_ACT_LEAKY_RELU: return z > 0.0f ? z : alpha * z;
    case NMZ_ACT_SOFTMAX:
    case NMZ_ACT_COUNT: break;
    }
    return z;
}

/* f'(z). For relu/leaky relu the (undefined) derivative at exactly 0 is taken
 * from the left: 0 resp. alpha. */
float nmz_act_derivative(nmz_act a, float z, float alpha) {
    switch (a) {
    case NMZ_ACT_LINEAR: return 1.0f;
    case NMZ_ACT_SIGMOID: {
        double s = sigmoid_d(z);
        return (float)(s * (1.0 - s));
    }
    case NMZ_ACT_TANH: {
        double t = tanh(z);
        return (float)(1.0 - t * t);
    }
    case NMZ_ACT_RELU: return z > 0.0f ? 1.0f : 0.0f;
    case NMZ_ACT_LEAKY_RELU: return z > 0.0f ? 1.0f : alpha;
    case NMZ_ACT_SOFTMAX:
    case NMZ_ACT_COUNT: break;
    }
    return 1.0f;
}

/* ------------------------------------------------------------ the layer */

typedef struct act_layer {
    nmz_act act;
    float alpha;
    nmz_tensor fprime; /* f'(z), filled only when tracing */
} act_layer;

nmz_act nmz__activation_layer_act(const nmz_layer *l) { return ((const act_layer *)l->self)->act; }
float nmz__activation_layer_alpha(const nmz_layer *l) { return ((const act_layer *)l->self)->alpha; }

static nmz_status act_setup(nmz_layer *l, int in_rank, const int *in_shape, int *out_rank,
                            int *out_shape) {
    int i;
    if (in_rank < 1) return nmz__fail(NMZ_ERR_SHAPE, "activation needs at least rank-1 input");
    (void)l;
    *out_rank = in_rank;
    for (i = 0; i < in_rank; i++) out_shape[i] = in_shape[i];
    return NMZ_OK;
}

static nmz_status act_init(nmz_layer *l, nmz_rng *rng, const nmz_layer *next) {
    (void)l; (void)rng; (void)next;
    return NMZ_OK;
}

static void softmax_rows(const nmz_tensor *x, nmz_tensor *y) {
    size_t rows = nmz_tensor_rows(x), r;
    int k, cols = nmz_tensor_cols(x);
    for (r = 0; r < rows; r++) {
        const float *zr = x->data + r * (size_t)cols;
        float *ar = y->data + r * (size_t)cols;
        double mx = zr[0], sum = 0.0;
        for (k = 1; k < cols; k++)
            if (zr[k] > mx) mx = zr[k];
        for (k = 0; k < cols; k++) sum += exp((double)zr[k] - mx); /* shift for stability */
        for (k = 0; k < cols; k++) ar[k] = (float)(exp((double)zr[k] - mx) / sum);
    }
}

static nmz_status act_forward(nmz_layer *l, const nmz_tensor *x, nmz_tensor *y,
                              const nmz_ctx *ctx) {
    act_layer *a = l->self;
    size_t i;
    if (a->act == NMZ_ACT_SOFTMAX) {
        softmax_rows(x, y);
    } else {
        for (i = 0; i < x->size; i++) y->data[i] = nmz_act_apply(a->act, x->data[i], a->alpha);
    }
    if (NMZ_TRACING(ctx)) {
        nmz_trace_item it[2] = {{"z", x}, {"a", y}};
        nmz_trace_emit(ctx, NMZ_TRACE_FORWARD, "activation", nmz_act_name(a->act), 0, NULL, 2, it);
    }
    return NMZ_OK;
}

static nmz_status act_backward(nmz_layer *l, const nmz_tensor *x, const nmz_tensor *y,
                               const nmz_tensor *dy, nmz_tensor *dx, const nmz_ctx *ctx) {
    act_layer *a = l->self;
    size_t i;
    int tracing = NMZ_TRACING(ctx);
    if (!dx) return NMZ_OK; /* no params, nothing else to do */
    if (a->act == NMZ_ACT_SOFTMAX) {
        /* Jacobian-vector product: dz_i = a_i (g_i − Σ_j g_j a_j) */
        size_t rows = nmz_tensor_rows(y), r;
        int k, cols = nmz_tensor_cols(y);
        for (r = 0; r < rows; r++) {
            const float *ar = y->data + r * (size_t)cols, *gr = dy->data + r * (size_t)cols;
            float *dr = dx->data + r * (size_t)cols;
            double dot = 0.0;
            for (k = 0; k < cols; k++) dot += (double)gr[k] * ar[k];
            for (k = 0; k < cols; k++) dr[k] = (float)(ar[k] * ((double)gr[k] - dot));
        }
        if (tracing) {
            nmz_trace_item it[2] = {{"dL/da", dy}, {"dL/dz", dx}};
            nmz_trace_emit(ctx, NMZ_TRACE_BACKWARD, "activation", "softmax", 0, NULL, 2, it);
        }
        return NMZ_OK;
    }
    if (tracing) {
        nmz_status s = nmz_tensor_reshape(&a->fprime, x->rank, x->shape);
        if (s != NMZ_OK) return s;
    }
    for (i = 0; i < x->size; i++) {
        float d = nmz_act_derivative(a->act, x->data[i], a->alpha);
        if (tracing) a->fprime.data[i] = d;
        dx->data[i] = dy->data[i] * d;
    }
    if (tracing) {
        nmz_trace_item it[3] = {{"dL/da", dy}, {"f'(z)", &a->fprime}, {"dL/dz", dx}};
        nmz_trace_emit(ctx, NMZ_TRACE_BACKWARD, "activation", nmz_act_name(a->act), 0, NULL, 3, it);
    }
    return NMZ_OK;
}

static int act_param_count(const nmz_layer *l) {
    (void)l;
    return 0;
}

static nmz_param act_param(nmz_layer *l, int i) {
    nmz_param p = {NULL, NULL, NULL};
    (void)l; (void)i;
    return p;
}

static nmz_status act_save(const nmz_layer *l, nmz_writer *w) {
    const act_layer *a = l->self;
    nmz_write_u32(w, (uint32_t)a->act);
    return nmz_write_f32(w, a->alpha);
}

static nmz_status act_load(nmz_layer **out, nmz_reader *r) {
    uint32_t act = 0;
    float alpha = 0.0f;
    nmz_read_u32(r, &act);
    if (nmz_read_f32(r, &alpha) != NMZ_OK) return r->err;
    if (act >= NMZ_ACT_COUNT) return nmz__fail(NMZ_ERR_FORMAT, "unknown activation id %u", act);
    *out = nmz_layer_activation((nmz_act)act, alpha);
    return *out ? NMZ_OK : NMZ_ERR_NOMEM;
}

static void act_describe(const nmz_layer *l, char *buf, size_t n) {
    const act_layer *a = l->self;
    if (a->act == NMZ_ACT_LEAKY_RELU)
        snprintf(buf, n, "activation leaky_relu (alpha=%g)", (double)a->alpha);
    else
        snprintf(buf, n, "activation %s", nmz_act_name(a->act));
}

static void act_destroy(nmz_layer *l) {
    act_layer *a = l->self;
    if (a) nmz_tensor_free(&a->fprime);
    free(a);
}

const nmz_layer_ops nmz__activation_ops = {
    "activation", act_setup, act_init, act_forward, act_backward, act_param_count, act_param,
    act_save, act_load, act_describe, act_destroy, NULL, 0};

nmz_layer *nmz_layer_activation(nmz_act act, float alpha) {
    act_layer *a;
    nmz_layer *l;
    if ((int)act < 0 || act >= NMZ_ACT_COUNT) {
        nmz__fail(NMZ_ERR_ARG, "invalid activation %d", (int)act);
        return NULL;
    }
    if (!(alpha == alpha) || alpha < 0.0f || alpha >= 1.0f) {
        nmz__fail(NMZ_ERR_ARG, "leaky_relu alpha must be in [0, 1)");
        return NULL;
    }
    a = calloc(1, sizeof *a);
    if (!a) {
        nmz__fail(NMZ_ERR_NOMEM, "out of memory");
        return NULL;
    }
    a->act = act;
    a->alpha = act == NMZ_ACT_LEAKY_RELU ? (alpha > 0.0f ? alpha : NMZ_LEAKY_DEFAULT) : 0.0f;
    l = nmz_layer_alloc(&nmz__activation_ops, a);
    if (!l) free(a);
    return l;
}
