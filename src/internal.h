/* internal.h — private declarations shared by libneuromz sources. */
#ifndef NMZ_INTERNAL_H
#define NMZ_INTERNAL_H

#include "neuromz.h"
#include "neuromz_backend.h"
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

/* ----- backends and device copies of tensors (backend.c, DESIGN §11) */
extern const nmz_backend nmz__cpu_backend;         /* the reference: v2.0.1's loops      */
extern const nmz_backend nmz__cpu_generic_backend; /* the same ops, no fused kernels     */
#ifdef NMZ_HAVE_CUDA
const nmz_backend *nmz__cuda_backend(void);        /* src/cuda/backend_cuda.cu           */
#endif
enum { NMZ__BOTH = 0, NMZ__HOST = 1, NMZ__DEVICE = 2 }; /* which copy is newer */
struct nmz_tensor_dev {
    const nmz_backend *be; /* the device backend that owns ptr */
    float *ptr;            /* device memory (NULL until first needed)    */
    size_t cap;            /* floats at ptr                               */
    int where;             /* NMZ__HOST / NMZ__DEVICE: that copy is newer */
};
/* the slow path of the accessors below: binds t to a device backend,
 * allocates and moves data as needed; NULL on failure (message set) */
float *nmz__acquire(const nmz_backend *be, nmz_tensor *t, unsigned mode);
/* back to host only: newest data to the host, device copy freed */
void nmz__tensor_to_host(nmz_tensor *t);
/* write one element in every copy (gradcheck perturbations) */
nmz_status nmz__tensor_set(nmz_tensor *t, size_t i, float v);
/* Pointers into t's data in be's memory. Host tensors on host backends
 * take the first branch: one test, nothing else (the CPU's hot path). */
static inline const float *nmz__r(const nmz_backend *be, const nmz_tensor *t) {
    if (!t) return NULL;
    if (!t->dev && be->host_memory) return t->data;
    return nmz__acquire(be, (nmz_tensor *)t, NMZ_READ);
}
static inline float *nmz__w(const nmz_backend *be, nmz_tensor *t) {
    if (!t) return NULL;
    if (!t->dev && be->host_memory) return t->data;
    return nmz__acquire(be, t, NMZ_WRITE);
}
static inline float *nmz__u(const nmz_backend *be, nmz_tensor *t) {
    if (!t) return NULL;
    if (!t->dev && be->host_memory) return t->data;
    return nmz__acquire(be, t, NMZ_UPDATE);
}
/* the layers' `buffers` entries: copy ts[0..n) to out (at most max) */
static inline int nmz__list(nmz_tensor **out, int max, int n, nmz_tensor *const *ts) {
    int i;
    for (i = 0; i < n && i < max; i++) out[i] = ts[i];
    return n;
}
/* give a tensor's device memory back; keep = bring its newest data to the
 * host first (else the contents are dropped: caches) */
void nmz__tensor_drop_device(nmz_tensor *t, int keep);
/* t = 0 in be's memory */
nmz_status nmz__zero_on(const nmz_backend *be, nmz_tensor *t);
/* host access from engine code (read / overwrite / update) */
#define nmz__hr(t) nmz__r(&nmz__cpu_backend, (t))
#define nmz__hw(t) nmz__w(&nmz__cpu_backend, (t))
#define nmz__hu(t) nmz__u(&nmz__cpu_backend, (t))

/* CPU reference kernels that live next to their layers (the CPU backend's
 * table points at them, backend_cpu.c); signatures as in neuromz_backend.h */
nmz_status nmz__cpu_dense_forward(const nmz_backend *be, size_t rows, int n_in, int units, const float *x,
                                  const float *W, const float *b, float *y);
nmz_status nmz__cpu_dense_backward(const nmz_backend *be, size_t rows, int n_in, int units, const float *x,
                                   const float *W, const float *dy, float *dW, float *db, float *dx);
nmz_status nmz__cpu_conv_forward(const nmz_backend *be, const nmz_conv_shape *s, int naive, const float *x,
                                 const float *W, const float *b, float *y, float *col);
nmz_status nmz__cpu_conv_backward(const nmz_backend *be, const nmz_conv_shape *s, int naive, const float *x,
                                  const float *W, const float *dy, float *dW, float *db, float *dx, double *acc,
                                  float *col, float *dcol);
nmz_status nmz__cpu_rnn_forward(const nmz_backend *be, int B, int T, int D, int H, const float *x, const float *h0,
                                const float *Wxh, const float *Whh, const float *b, float *y, float *z);
nmz_status nmz__cpu_rnn_backward(const nmz_backend *be, int B, int T, int D, int H, const float *x, const float *y,
                                 const float *h0, const float *Wxh, const float *Whh, const float *dy, float *dWxh,
                                 float *dWhh, float *db, float *dx, float *dh_next, const nmz_seq_record *rec);
nmz_status nmz__cpu_lstm_forward(const nmz_backend *be, int B, int T, int D, int H, const float *x, const float *h0,
                                 const float *c0, const float *Wx, const float *Wh, const float *b, float *y,
                                 float *gates, float *c);
nmz_status nmz__cpu_lstm_backward(const nmz_backend *be, int B, int T, int D, int H, const float *x, const float *y,
                                  const float *h0, const float *c0, const float *gates, const float *c,
                                  const float *Wx, const float *Wh, const float *dy, float *dWx, float *dWh,
                                  float *db, float *dx, float *dh_next, float *dc_next, const nmz_seq_record *rec);
nmz_status nmz__cpu_gru_forward(const nmz_backend *be, int B, int T, int D, int H, const float *x, const float *h0,
                                const float *Wx, const float *Wh, const float *b, float *y, float *gates);
nmz_status nmz__cpu_gru_backward(const nmz_backend *be, int B, int T, int D, int H, const float *x, const float *y,
                                 const float *h0, const float *gates, const float *Wx, const float *Wh,
                                 const float *dy, float *dWx, float *dWh, float *db, float *dx, float *dh_next,
                                 const nmz_seq_record *rec);
nmz_status nmz__cpu_attention_forward(const nmz_backend *be, int B, int H, int T, int Tk, int Tp, int d, int E,
                                      const float *qkv, const float *kv, float *P, float *S, float *ctx);
nmz_status nmz__cpu_attention_backward(const nmz_backend *be, int B, int H, int T, int Tk, int Tp, int d, int E,
                                       const float *qkv, const float *kv, const float *P, const float *dctx,
                                       float *dP, float *dqkv, float *dkv);
nmz_status nmz__cpu_layernorm_forward(const nmz_backend *be, size_t rows, int D, float eps, const float *x,
                                      const float *g, const float *b, float *y, float *xhat, float *mean,
                                      float *rstd);
nmz_status nmz__cpu_layernorm_backward(const nmz_backend *be, size_t rows, int D, const float *dy, const float *xhat,
                                       const float *rstd, const float *g, float *dx, float *dg, float *db,
                                       float *dxhat);
nmz_status nmz__cpu_embed_forward(const nmz_backend *be, size_t n, int V, int D, const float *ids, const float *W,
                                  float *y);
nmz_status nmz__cpu_embed_backward(const nmz_backend *be, size_t n, int V, int D, const float *ids, const float *dy,
                                   float *dW);
nmz_status nmz__cpu_maxpool_forward(const nmz_backend *be, int planes, int H, int W, int size, int stride, int OH,
                                    int OW, const float *x, float *y, float *argmax);
nmz_status nmz__cpu_maxpool_backward(const nmz_backend *be, int planes, int H, int W, int OH, int OW,
                                     const float *dy, const float *argmax, float *dx);
nmz_status nmz__cpu_posenc_forward(const nmz_backend *be, int B, int T, int D, const int *off, const float *x,
                                   const float *P, float *y);
nmz_status nmz__cpu_posenc_backward(const nmz_backend *be, int B, int T, int D, const int *off, const float *dy,
                                    float *dP);
nmz_status nmz__cpu_act_forward(const nmz_backend *be, nmz_act act, float alpha, size_t n, const float *z, float *a);
nmz_status nmz__cpu_act_backward(const nmz_backend *be, nmz_act act, float alpha, size_t n, const float *z,
                                 const float *dy, float *dz, float *fprime);
nmz_status nmz__cpu_softmax_forward(const nmz_backend *be, size_t rows, int cols, const float *z, float *p);
nmz_status nmz__cpu_softmax_backward(const nmz_backend *be, size_t rows, int cols, const float *p, const float *dy,
                                     float *dz);
nmz_status nmz__cpu_loss_value(const nmz_backend *be, nmz_loss kind, size_t rows, int cols, const float *y,
                               const float *t, const float *z, double *out);
nmz_status nmz__cpu_loss_grad(const nmz_backend *be, nmz_loss kind, size_t rows, int cols, const float *y,
                              const float *t, float *g);
/* layers pick the fused kernel when the backend has one */
#define NMZ_FUSED(be, op) ((be)->op != NULL)
/* status for an op that got NULL for data it needs (an acquire failed and
 * set the message), or a generic one */
nmz_status nmz__op_failed(void);
/* file names are UTF-8 on every system (error.c) */
FILE *nmz__fopen(const char *path, const char *mode);
int nmz__remove(const char *path);
int nmz__rename(const char *from, const char *to);
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
extern const nmz_layer_ops nmz__posenc_ops;
extern const nmz_layer_ops nmz__layernorm_ops;
extern const nmz_layer_ops nmz__attention_ops;
extern const nmz_layer_ops nmz__ffn_ops;
extern const nmz_layer_ops nmz__block_ops;
extern const nmz_layer_ops nmz__tied_output_ops;
int nmz__recurrent_init_used(const nmz_layer *l);
/* transformer helpers (attention.c, transformer.c) */
void nmz__attention_dims(const nmz_layer *l, int *heads, int *head_dim, int *head_dim_req);
nmz_layer *nmz__block_sub(const nmz_layer *l, int i); /* 0 ln1, 1 attention, 2 ln2, 3 ffn */
int nmz__tied_embedding(const nmz_layer *l);          /* index of the shared embedding, or -1 */
size_t nmz__kink_signature(const nmz_layer *l, float *out); /* relu sides inside ffn/block (gradcheck) */
int nmz__gated_init_used(const nmz_layer *l); /* lstm or gru */
int nmz__lstm_init_used(const nmz_layer *l);
float nmz__lstm_forget_bias(const nmz_layer *l);
/* recurrent.c helpers shared by rnn/lstm/gru */
nmz_status nmz__orthogonal_block(float *W, int n, int ld, int col0, nmz_rng *rng);
nmz_status nmz__recurrent_init(nmz_init init, nmz_init *resolved, nmz_tensor *Wx, nmz_tensor *Wh, int D, int H,
                               int gates, nmz_rng *rng);
/* Σ a[j]·b[j] in double with four independent partial sums: the same
 * precision as one double accumulator, but the additions do not wait for
 * each other (the recurrent backward passes spend most of their time here). */
static inline double nmz__dot(const float *a, const float *b, int n) {
    double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int j = 0;
    for (; j + 4 <= n; j += 4) {
        s0 += (double)a[j] * b[j];
        s1 += (double)a[j + 1] * b[j + 1];
        s2 += (double)a[j + 2] * b[j + 2];
        s3 += (double)a[j + 3] * b[j + 3];
    }
    for (; j < n; j++) s0 += (double)a[j] * b[j];
    return (s0 + s1) + (s2 + s3);
}

/* copy step t of a [B, T, F] tensor into a [B, F] tensor; _tc: only the
 * columns col0 … col0+n−1, into [B, n] */
nmz_status nmz__slice_t(const nmz_tensor *src, int t, nmz_tensor *dst);
/* recurrent layers' generic path (recurrent.c): a_t (cols c0 … c0+n) += h_{t−1}·W_h;
 * h_{t−1} for every step [B, T, H]; trace records [B, T, widths[k] or G] */
nmz_status nmz__rec_product(const nmz_backend *be, int B, int T, int t, int H, int G, int c0, int n, const float *h0,
                            const float *y, const float *Wh, float *a);
nmz_status nmz__rec_prev(const nmz_backend *be, int B, int T, int H, const float *h0, const float *y, float *prev);
nmz_status nmz__rec_alloc(const nmz_backend *be, nmz_tensor *t, int n, int B, int T, int G, nmz_seq_record *rec,
                          float **out[], const int *widths);
nmz_status nmz__slice_tc(const nmz_tensor *src, int t, int col0, int n, nmz_tensor *dst);
int nmz__conv2d_init_used(const nmz_layer *l);
const nmz_tensor *nmz__maxpool_argmax(const nmz_layer *l);

/* the parameter pair without the public accessors' host synchronisation */
nmz_param nmz__param(const nmz_layer *l, int i);
/* token ids of an embedding layer's input are integers in range (checked on
 * the host before a device computes, since a kernel cannot report it) */
nmz_status nmz__check_ids(const nmz_layer *embedding, const float *ids, size_t n);

/* Activation layer accessors */
nmz_act nmz__activation_layer_act(const nmz_layer *l);
float nmz__activation_layer_alpha(const nmz_layer *l);

/* ----- losses (loss.c). Tensors are treated as rows × last-dim.
 * `z` is the softmax input for cross-entropy (may be NULL → uses p). */
nmz_status nmz__loss_value(const nmz_backend *be, nmz_loss kind, const nmz_tensor *y, const nmz_tensor *t,
                           const nmz_tensor *z, double *out);
/* MSE: g = ∂L/∂y = (y − t)/rows.
 * CE (fused with softmax): g = ∂L/∂z = (p·Σt − t)/rows. */
nmz_status nmz__loss_grad(const nmz_backend *be, nmz_loss kind, const nmz_tensor *y, const nmz_tensor *t,
                          nmz_tensor *g);

/* ----- the network object (net.c) */
typedef struct nmz__param_ref {
    int layer;
    int index;
} nmz__param_ref;

struct nmz_net {
    const nmz_backend *be; /* where the net computes (follows the selection) */
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
    nmz_tensor norm_t; /* offset | scale copied for the backend (when norm_ver changes) */
    long norm_ver, norm_t_ver;
    nmz_tensor in1, out1; /* predict() single-sample buffers */
    nmz_tensor tgt;       /* a caller's target, copied for a device backend */

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
    char *data_path;      /* training data source (decision 111)        */
    uint64_t *item_hash;  /* sorted FNV-1a hashes of training items     */
    int n_item_hash;
};
/* net.c: store an already sorted, distinct hash set (takes ownership) */
void nmz__net_set_item_hashes(nmz_net *net, uint64_t *h, int n);

/* The backend the net computes on now (follows nmz_backend_select). */
const nmz_backend *nmz__net_backend(nmz_net *net);
/* After running out of device memory: free the activations and layer
 * caches on the device (they are rebuilt at the size the next call needs) */
void nmz__net_drop_buffers(nmz_net *net);

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
nmz_status nmz__norm_alloc(nmz_net *net, int groups);
void nmz__norm_free(nmz_net *net);

#endif
