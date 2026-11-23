/* internal.h — private declarations shared by libneuromz sources. */
#ifndef NMZ_INTERNAL_H
#define NMZ_INTERNAL_H

#include "neuromz.h"
#include "neuromz_layer.h"

#include <stdarg.h>

#define NMZ_LEAKY_DEFAULT 0.01f
#define NMZ_MAX_DIM 100000000  /* sanity limit for a single dimension */
#define NMZ_MAX_ELEMS ((size_t)1 << 31)
#define NMZ_MAX_VOCAB 65536

/* Record a formatted message as the last error and return `s`. */
nmz_status nmz__fail(nmz_status s, const char *fmt, ...)
#if defined(__GNUC__)
    __attribute__((format(printf, 2, 3)))
#endif
    ;

char *nmz__strdup(const char *s);
/* Product of dims with overflow/limit checks. */
nmz_status nmz__shape_size(int rank, const int *shape, size_t *out);
int nmz__shape_equal(int rank_a, const int *a, int rank_b, const int *b);

/* ----- byte I/O (io.c) */
struct nmz_writer {
    FILE *f;
    uint32_t crc;
    nmz_status err;
};
struct nmz_reader {
    FILE *f;
    uint32_t crc;
    nmz_status err;
};
uint32_t nmz__crc32(uint32_t crc, const void *data, size_t n);

/* ----- built-in layer ops */
extern const nmz_layer_ops nmz__dense_ops;
extern const nmz_layer_ops nmz__activation_ops;
extern const nmz_layer_ops nmz__conv2d_ops;
extern const nmz_layer_ops nmz__maxpool2d_ops;
extern const nmz_layer_ops nmz__flatten_ops;
extern const nmz_layer_ops nmz__dropout_ops;
extern const nmz_layer_ops nmz__embedding_ops;
extern const nmz_layer_ops nmz__rnn_ops;
extern const nmz_layer_ops nmz__lstm_ops;
extern const nmz_layer_ops nmz__gru_ops;
int nmz__recurrent_init_used(const nmz_layer *l);
int nmz__gated_init_used(const nmz_layer *l); /* lstm or gru */
int nmz__lstm_init_used(const nmz_layer *l);
float nmz__lstm_forget_bias(const nmz_layer *l);
/* recurrent.c helpers shared by rnn/lstm/gru */
nmz_status nmz__orthogonal_block(float *W, int n, int ld, int col0, nmz_rng *rng);
nmz_status nmz__recurrent_init(nmz_init init, nmz_init *resolved, nmz_tensor *Wx, nmz_tensor *Wh, int D, int H,
                               int gates, nmz_rng *rng);
/* copy step t of a [B, T, F] tensor into a [B, F] tensor; _tc: only the
 * columns col0 … col0+n−1, into [B, n] */
nmz_status nmz__slice_t(const nmz_tensor *src, int t, nmz_tensor *dst);
nmz_status nmz__slice_tc(const nmz_tensor *src, int t, int col0, int n, nmz_tensor *dst);
int nmz__conv2d_init_used(const nmz_layer *l);
const nmz_tensor *nmz__maxpool_argmax(const nmz_layer *l);

/* Activation layer accessors */
nmz_act nmz__activation_layer_act(const nmz_layer *l);
float nmz__activation_layer_alpha(const nmz_layer *l);

/* ----- losses (loss.c). Tensors are treated as rows × last-dim.
 * `z` is the softmax input for cross-entropy (may be NULL → uses p). */
nmz_status nmz__loss_value(nmz_loss kind, const nmz_tensor *y, const nmz_tensor *t,
                           const nmz_tensor *z, double *out);
/* MSE: g = ∂L/∂y = (y − t)/rows.
 * CE (fused with softmax): g = ∂L/∂z = (p·Σt − t)/rows. */
nmz_status nmz__loss_grad(nmz_loss kind, const nmz_tensor *y, const nmz_tensor *t,
                          nmz_tensor *g);

/* ----- the network object (net.c) */
typedef struct nmz__param_ref {
    int layer;
    int index;
} nmz__param_ref;

struct nmz_net {
    int in_rank;
    int in_shape[NMZ_MAX_RANK];

    int n_layers;
    int cap_layers;
    nmz_layer **layers;

    nmz_tensor *acts;  /* n_layers + 1: acts[0] input copy, acts[i+1] layer i out */
    nmz_tensor *dacts; /* n_layers + 1: ∂L/∂acts[i]                              */

    int n_params; /* parameter tensors (not scalars) */
    nmz__param_ref *param_refs;
    /* optimizer state, one pair per parameter tensor:
     * SGD: opt_m = velocity ; Adam: opt_m = first moment, opt_v = second */
    nmz_tensor *opt_m, *opt_v;
    int opt_kind; /* nmz_optimizer the state belongs to, -1 = none yet */
    long opt_t;   /* Adam step counter */
    nmz_tensor scratch[6]; /* trace-only temporaries */

    /* input normalization (DESIGN decision 34) */
    nmz_norm norm;
    int norm_fitted;
    int norm_groups;
    float *norm_offset, *norm_scale;
    nmz_tensor in1, out1; /* predict() single-sample buffers */

    nmz_loss loss;
    int built;
    int have_forward;
    int have_grads;
    int training;
    nmz_rng rng;
    long step;
    nmz_train_opts train;

    int n_classes;
    char **class_names;

    nmz_trace_fn trace;
    void *trace_user;

    int carry;            /* recurrent layers continue from their state */
    nmz_vocab *vocab;     /* text models: the character vocabulary      */
};

/* Is the loss fused with a final softmax layer? */
int nmz__net_fused_ce(const nmz_net *net);
/* Does the net end in a softmax activation layer? */
int nmz__net_fused_ce_possible(const nmz_net *net);
/* Forward with explicit training flag. */
nmz_status nmz__net_forward(nmz_net *net, const nmz_tensor *x, int training,
                            const nmz_tensor **y);
/* Allocate optimizer state etc. after layers exist and params are allocated. */
nmz_status nmz__net_finish_build(nmz_net *net);
nmz_status nmz__net_check_dataset(const nmz_net *net, const nmz_dataset *ds);
nmz_status nmz__train_opts_validate(const nmz_train_opts *o);

/* generate.c: temperature + top-k sampling from logits z[0..V); probs is
 * scratch for V doubles and receives the final distribution. */
int nmz__sample_next(const float *z, int V, double temperature, int top_k, nmz_rng *rng, double *probs);

/* normalize.c */
int nmz__norm_group_count(const nmz_net *net);
/* group of element e (0-based within one sample) */
int nmz__norm_group_of(const nmz_net *net, size_t e);
/* x' = (x − offset)·scale over a batch tensor (in place) */
void nmz__norm_apply(const nmz_net *net, nmz_tensor *t);
/* g_raw = g_norm · scale (in place), for ∂L/∂input */
void nmz__norm_scale_grad(const nmz_net *net, nmz_tensor *g);
nmz_status nmz__norm_alloc(nmz_net *net, int groups);
void nmz__norm_free(nmz_net *net);

#endif
