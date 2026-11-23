/*
 * neuromz_layer.h — API for writing new layer types.
 *
 * A layer is a vtable (nmz_layer_ops) plus private state. The engine talks to
 * layers only through this table, so Conv, RNN, Embedding, Attention, ... can
 * be added without changing net.c or train.c. See docs/DESIGN.md §2.3 and §3.
 */
#ifndef NEUROMZ_LAYER_H
#define NEUROMZ_LAYER_H

#include "neuromz.h"

#ifdef __cplusplus
extern "C" {
#endif

/* A parameter and its gradient, always exposed together. */
typedef struct nmz_param {
    const char *name;
    nmz_tensor *value;
    nmz_tensor *grad;
} nmz_param;

/* Per-call context handed to forward/backward. */
typedef struct nmz_ctx {
    nmz_trace_fn trace; /* NULL when tracing is off */
    void *trace_user;
    long step;
    int layer_index;
    int training; /* 1 during training passes, 0 for evaluation */
} nmz_ctx;

/* Little-endian byte writer/reader used by save_config/load_config. */
typedef struct nmz_writer nmz_writer;
typedef struct nmz_reader nmz_reader;
nmz_status nmz_write_u32(nmz_writer *w, uint32_t v);
nmz_status nmz_write_f32(nmz_writer *w, float v);
nmz_status nmz_write_f64(nmz_writer *w, double v);
nmz_status nmz_read_u32(nmz_reader *r, uint32_t *v);
nmz_status nmz_read_f32(nmz_reader *r, float *v);
nmz_status nmz_read_f64(nmz_reader *r, double *v);

typedef struct nmz_layer_ops {
    const char *type;
    /* Shape inference from the per-sample input shape; allocate params. */
    nmz_status (*setup)(nmz_layer *l, int in_rank, const int *in_shape,
                        int *out_rank, int *out_shape);
    /* Initialise parameters. `next` is the following layer or NULL. */
    nmz_status (*init_params)(nmz_layer *l, nmz_rng *rng, const nmz_layer *next);
    /* y = f(x). y has already been shaped [B, out_shape...] by the engine. */
    nmz_status (*forward)(nmz_layer *l, const nmz_tensor *x, nmz_tensor *y,
                          const nmz_ctx *ctx);
    /* dy = ∂L/∂y. ADD parameter gradients into the grad tensors. Write
     * dx = ∂L/∂x (already shaped like x) unless dx is NULL. */
    nmz_status (*backward)(nmz_layer *l, const nmz_tensor *x, const nmz_tensor *y,
                           const nmz_tensor *dy, nmz_tensor *dx, const nmz_ctx *ctx);
    int (*param_count)(const nmz_layer *l);
    nmz_param (*param)(nmz_layer *l, int i);
    nmz_status (*save_config)(const nmz_layer *l, nmz_writer *w);
    nmz_status (*load_config)(nmz_layer **out, nmz_reader *r);
    void (*describe)(const nmz_layer *l, char *buf, size_t n);
    void (*destroy)(nmz_layer *l);
} nmz_layer_ops;

struct nmz_layer {
    const nmz_layer_ops *ops;
    void *self;
    int in_rank;
    int in_shape[NMZ_MAX_RANK];
    int out_rank;
    int out_shape[NMZ_MAX_RANK];
};

/* Allocate an nmz_layer shell around ops/self (NULL on OOM). */
nmz_layer *nmz_layer_alloc(const nmz_layer_ops *ops, void *self);

/* Emit a trace event if tracing is on. Cheap when ctx->trace is NULL; wrap
 * expensive preparation in `if (NMZ_TRACING(ctx))`. */
#define NMZ_TRACING(ctx) ((ctx) != NULL && (ctx)->trace != NULL)
void nmz_trace_emit(const nmz_ctx *ctx, nmz_trace_phase phase, const char *layer_type,
                    const char *detail, int n_scalars, const nmz_trace_scalar *scalars,
                    int n_items, const nmz_trace_item *items);

/* Registry used by the loader: type name → ops. Built-in layers are always
 * registered. Returns NULL if unknown. */
const nmz_layer_ops *nmz_layer_ops_find(const char *type);

#ifdef __cplusplus
}
#endif

#endif /* NEUROMZ_LAYER_H */
