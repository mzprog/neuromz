/*
 * neuromz.h — NeuroMZ v2 public API.
 *
 * An educational neural-network engine in C11. Every public name starts with
 * nmz_ / NMZ_. The library never prints, never exits, and keeps no global
 * state apart from a thread-local "last error" message.
 *
 * See docs/DESIGN.md for the design and README.md for a tutorial.
 */
#ifndef NEUROMZ_H
#define NEUROMZ_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NMZ_VERSION_MAJOR 2
#define NMZ_VERSION_MINOR 0
#define NMZ_VERSION_PATCH 0
#define NMZ_VERSION_STRING "2.0.0"

#define NMZ_MAX_RANK 4
#define NMZ_TRACE_MAX_ITEMS 8

/* ------------------------------------------------------------------ errors */

typedef enum nmz_status {
    NMZ_OK = 0,
    NMZ_ERR_NOMEM = -1,       /* allocation failed                         */
    NMZ_ERR_ARG = -2,         /* invalid argument (NULL, out of range, ...) */
    NMZ_ERR_SHAPE = -3,       /* tensor/layer shapes do not fit together   */
    NMZ_ERR_STATE = -4,       /* call not valid in the current state       */
    NMZ_ERR_IO = -5,          /* file could not be opened/read/written     */
    NMZ_ERR_FORMAT = -6,      /* file or CSV content is malformed          */
    NMZ_ERR_UNSUPPORTED = -7, /* valid request the engine cannot do yet    */
    NMZ_ERR_NUMERIC = -8      /* NaN/Inf appeared (e.g. learning rate too high) */
} nmz_status;

/* Short constant name of a status, e.g. "NMZ_ERR_SHAPE". */
const char *nmz_status_str(nmz_status s);
/* Human-readable message for the most recent failure on this thread
 * ("" if none). Valid until the next failing call on the same thread. */
const char *nmz_last_error(void);

/* ----------------------------------------------------------------- tensors */

/* Contiguous row-major float tensor. shape[0] is the batch dimension when
 * the tensor holds a batch. */
typedef struct nmz_tensor {
    int rank;
    int shape[NMZ_MAX_RANK];
    size_t size;     /* product of shape (1 for rank 0) */
    size_t capacity; /* allocated floats                */
    float *data;
} nmz_tensor;

#define NMZ_TENSOR_EMPTY {0, {0, 0, 0, 0}, 0, 0, NULL}

/* Allocate a zero-filled tensor. `t` must be empty or previously freed. */
nmz_status nmz_tensor_init(nmz_tensor *t, int rank, const int *shape);
/* Change shape; reallocates only if the new size exceeds capacity. New
 * contents are unspecified. Works on an empty tensor too. */
nmz_status nmz_tensor_reshape(nmz_tensor *t, int rank, const int *shape);
/* Make `dst` a deep copy of `src` (reusing dst's allocation when possible). */
nmz_status nmz_tensor_copy(nmz_tensor *dst, const nmz_tensor *src);
void nmz_tensor_zero(nmz_tensor *t);
void nmz_tensor_free(nmz_tensor *t);
/* Number of rows when the last dimension is treated as features. */
size_t nmz_tensor_rows(const nmz_tensor *t);
/* Size of the last dimension (1 for rank 0). */
int nmz_tensor_cols(const nmz_tensor *t);

/* --------------------------------------------------------------------- rng */

/* PCG32 (O'Neill 2014): small, fast, identical output on every platform. */
typedef struct nmz_rng {
    uint64_t state;
    uint64_t inc;
} nmz_rng;

void nmz_rng_seed(nmz_rng *r, uint64_t seed, uint64_t stream);
uint32_t nmz_rng_u32(nmz_rng *r);
uint32_t nmz_rng_below(nmz_rng *r, uint32_t bound); /* uniform in [0, bound) */
double nmz_rng_uniform(nmz_rng *r);                 /* uniform in [0, 1)     */
double nmz_rng_normal(nmz_rng *r);                  /* standard normal       */

/* ------------------------------------------------------------------- enums */

typedef enum nmz_act {
    NMZ_ACT_LINEAR = 0,
    NMZ_ACT_SIGMOID,
    NMZ_ACT_TANH,
    NMZ_ACT_RELU,
    NMZ_ACT_LEAKY_RELU,
    NMZ_ACT_SOFTMAX,
    NMZ_ACT_COUNT
} nmz_act;

typedef enum nmz_init {
    NMZ_INIT_AUTO = 0, /* He before relu/leaky_relu, Xavier otherwise */
    NMZ_INIT_XAVIER,   /* Glorot uniform: ±sqrt(6/(fan_in+fan_out))  */
    NMZ_INIT_HE,       /* He uniform:     ±sqrt(6/fan_in)            */
    NMZ_INIT_COUNT
} nmz_init;

typedef enum nmz_loss {
    NMZ_LOSS_MSE = 0,       /* L = (1/rows) Σ_rows ½ Σ_k (y_k − t_k)²          */
    NMZ_LOSS_CROSS_ENTROPY, /* L = −(1/rows) Σ_rows Σ_k t_k log p_k (softmax)  */
    NMZ_LOSS_COUNT
} nmz_loss;

const char *nmz_act_name(nmz_act a);
const char *nmz_init_name(nmz_init i);
const char *nmz_loss_name(nmz_loss l);
/* Parse names (case-sensitive, a few aliases such as "ce", "lrelu"). */
nmz_status nmz_act_parse(const char *s, nmz_act *out);
nmz_status nmz_init_parse(const char *s, nmz_init *out);
nmz_status nmz_loss_parse(const char *s, nmz_loss *out);

/* Scalar activation and its derivative w.r.t. its input z (not softmax). */
float nmz_act_apply(nmz_act a, float z, float alpha);
float nmz_act_derivative(nmz_act a, float z, float alpha);

/* ------------------------------------------------------------------ layers */

typedef struct nmz_layer nmz_layer; /* see neuromz_layer.h */

/* Factories return NULL on failure (see nmz_last_error()). */
nmz_layer *nmz_layer_dense(int units, nmz_init init);
/* alpha is the negative slope for NMZ_ACT_LEAKY_RELU (ignored otherwise;
 * pass 0 to use the default 0.01). */
nmz_layer *nmz_layer_activation(nmz_act act, float alpha);
/* Convolution over per-sample [C, H, W] input: `filters` kernels of
 * kh × kw, zero padding `pad` (or NMZ_PAD_SAME: output keeps H×W; needs
 * stride 1 and an odd kernel). NMZ_INIT_AUTO = He unless a tanh/sigmoid
 * activation follows. Output [filters, OH, OW], OH = (H + 2·pad − kh)/stride + 1. */
#define NMZ_PAD_SAME (-1)
nmz_layer *nmz_layer_conv2d(int filters, int kh, int kw, int stride, int pad, nmz_init init);
/* How conv2d computes (same results up to float rounding): im2col (default;
 * receptive fields copied into a matrix, then matrix products) or the naive
 * nested loops that mirror the formula. Not saved in model files. */
typedef enum nmz_conv_algo { NMZ_CONV_IM2COL = 0, NMZ_CONV_NAIVE } nmz_conv_algo;
nmz_status nmz_layer_conv2d_set_algo(nmz_layer *l, nmz_conv_algo algo);
/* Max pooling over size × size windows (stride 0 = size), no padding. */
nmz_layer *nmz_layer_maxpool2d(int size, int stride);
/* [d1, d2, ...] → [d1·d2·...] (needed between conv/pool and dense layers). */
nmz_layer *nmz_layer_flatten(void);
/* Inverted dropout: in training each value is zeroed with probability `rate`
 * and the rest scaled by 1/(1−rate); identity in evaluation. */
nmz_layer *nmz_layer_dropout(float rate);
/* Free a layer that was NOT added to a net. */
void nmz_layer_free(nmz_layer *l);

const char *nmz_layer_type(const nmz_layer *l);
/* One-line description, e.g. "dense 2→4 (xavier)". */
void nmz_layer_describe(const nmz_layer *l, char *buf, size_t n);
int nmz_layer_param_count(const nmz_layer *l);
/* Parameter i: name ("W", "b"), value and gradient tensors. */
const char *nmz_layer_param_name(const nmz_layer *l, int i);
nmz_tensor *nmz_layer_param_value(nmz_layer *l, int i);
nmz_tensor *nmz_layer_param_grad(nmz_layer *l, int i);
/* For activation layers: which function (else -1). */
int nmz_layer_activation_kind(const nmz_layer *l);
/* For dense layers: the init actually used (AUTO resolved after build), else -1. */
int nmz_layer_dense_init(const nmz_layer *l);
/* Same for any layer with initialised weights (dense, conv2d), else -1. */
int nmz_layer_init_used(const nmz_layer *l);
/* Output size per sample (product of the per-sample output shape). */
int nmz_layer_output_size(const nmz_layer *l);
int nmz_layer_input_size(const nmz_layer *l);

/* --------------------------------------------------------------- the net */

typedef struct nmz_net nmz_net;
typedef struct nmz_dataset nmz_dataset; /* defined below */

/* Training configuration (also stored inside each model). */
typedef struct nmz_epoch_info {
    int epoch;           /* 1-based                                  */
    double loss;         /* loss over the whole training set          */
    double accuracy;     /* NaN when not meaningful (see DESIGN §10.23) */
    float learning_rate; /* η used in this epoch (after the schedule) */
} nmz_epoch_info;

typedef struct nmz_batch_info {
    int epoch;     /* 1-based                               */
    int batch;     /* 1-based index within the epoch        */
    int n_batches; /* mini-batches per epoch                */
    double loss;   /* loss of this mini-batch (before update) */
} nmz_batch_info;

/* Return non-zero from a callback to stop training early. */
typedef int (*nmz_epoch_fn)(const nmz_epoch_info *info, void *user);
typedef int (*nmz_batch_fn)(const nmz_batch_info *info, void *user);

typedef enum nmz_optimizer {
    NMZ_OPT_SGD = 0, /* v ← μ·v − η·g ; w ← w + v                                 */
    NMZ_OPT_ADAM,    /* m ← β1·m + (1−β1)·g ; v ← β2·v + (1−β2)·g²
                        w ← w − η·m̂/(√v̂ + ε), m̂ = m/(1−β1^t), v̂ = v/(1−β2^t)      */
    NMZ_OPT_COUNT
} nmz_optimizer;

typedef enum nmz_schedule {
    NMZ_SCHED_CONSTANT = 0, /* η_e = η                                               */
    NMZ_SCHED_STEP,         /* η_e = η · γ^floor((e−1)/step_size)                     */
    NMZ_SCHED_COSINE,       /* η_e = η_min + ½(η − η_min)(1 + cos(π·(e−1)/max_epochs)) */
    NMZ_SCHED_COUNT
} nmz_schedule;

typedef struct nmz_train_opts {
    int max_epochs;      /* hard limit                                  */
    int batch_size;      /* 1 = online SGD, >= n = full batch           */
    float learning_rate; /* η (initial value when a schedule is used)   */
    float momentum;      /* μ in [0, 1); 0 = plain SGD (SGD only)       */
    int shuffle;         /* reshuffle samples every epoch (1/0)         */
    double target_loss;  /* stop when whole-set loss <= this            */
    nmz_optimizer optimizer;
    float beta1, beta2, epsilon; /* Adam (defaults 0.9, 0.999, 1e-8)    */
    nmz_schedule schedule;
    int step_size;       /* NMZ_SCHED_STEP: epochs between decays       */
    float gamma;         /* NMZ_SCHED_STEP: decay factor                */
    float min_lr;        /* NMZ_SCHED_COSINE: final learning rate       */
    nmz_epoch_fn on_epoch;
    nmz_batch_fn on_batch;
    void *user;          /* passed to both callbacks                    */
} nmz_train_opts;

typedef struct nmz_train_result {
    int epochs;      /* epochs actually run                          */
    double loss;     /* final whole-set loss                          */
    double accuracy; /* final accuracy, NaN when not meaningful       */
    int converged;   /* 1 if loss <= target_loss                      */
    int stopped;     /* 1 if the on_epoch callback asked to stop      */
} nmz_train_result;

/* Library defaults (not tuned for any dataset). */
nmz_train_opts nmz_train_defaults(void);
/* Learning rate for epoch e (1-based) under o's schedule. */
float nmz_schedule_lr(const nmz_train_opts *o, int epoch);
const char *nmz_optimizer_name(nmz_optimizer o);
const char *nmz_schedule_name(nmz_schedule s);
nmz_status nmz_optimizer_parse(const char *s, nmz_optimizer *out);
nmz_status nmz_schedule_parse(const char *s, nmz_schedule *out);

/* Input normalization, fitted on training data and stored with the model:
 * x' = (x − offset_g) · scale_g, one (offset, scale) per group g: per
 * channel for [C,H,W] inputs, per feature otherwise.
 *   min-max: offset = min, scale = 1/(max − min)   → values in [0, 1]
 *   z-score: offset = mean, scale = 1/std          → mean 0, std 1
 * (scale = 1 when max = min or std = 0). */
typedef enum nmz_norm { NMZ_NORM_NONE = 0, NMZ_NORM_MINMAX, NMZ_NORM_ZSCORE, NMZ_NORM_COUNT } nmz_norm;
const char *nmz_norm_name(nmz_norm n);
nmz_status nmz_norm_parse(const char *s, nmz_norm *out);

/* Create an empty network for samples of the given per-sample shape. */
nmz_status nmz_net_create(nmz_net **out, int input_rank, const int *input_shape,
                          uint64_t seed);
void nmz_net_free(nmz_net *net);

/* Append a layer. The net ALWAYS takes ownership, also on failure. */
nmz_status nmz_net_add(nmz_net *net, nmz_layer *layer);
/* Convenience: Dense(units, init) followed by Activation(act) (no activation
 * layer is added for NMZ_ACT_LINEAR). */
nmz_status nmz_net_add_dense(nmz_net *net, int units, nmz_act act, nmz_init init);
nmz_status nmz_net_set_loss(nmz_net *net, nmz_loss loss);
/* Freeze the architecture, validate it and initialise parameters. Called
 * implicitly by forward/train/save; explicit calls are allowed. */
nmz_status nmz_net_build(nmz_net *net);
/* Re-initialise all parameters (new random weights) with a new seed. */
nmz_status nmz_net_reinit(nmz_net *net, uint64_t seed);

int nmz_net_layer_count(const nmz_net *net);
nmz_layer *nmz_net_layer(nmz_net *net, int i);
int nmz_net_input_size(const nmz_net *net);  /* per sample */
int nmz_net_output_size(const nmz_net *net); /* per sample */
nmz_loss nmz_net_loss_kind(const nmz_net *net);
size_t nmz_net_param_count(const nmz_net *net); /* total scalar parameters */
int nmz_net_is_built(const nmz_net *net);
long nmz_net_step(const nmz_net *net); /* number of backward passes so far */
/* Stored training configuration (saved with the model); never NULL for a
 * valid net. Callbacks in it are not saved. */
nmz_train_opts *nmz_net_train_config(nmz_net *net);

/* Request a normalization; it is fitted by the next nmz_net_train (or
 * explicitly with nmz_net_fit_normalization) and then frozen. Forward and
 * predict apply it automatically. NMZ_NORM_NONE removes it. */
nmz_status nmz_net_set_normalization(nmz_net *net, nmz_norm kind);
nmz_status nmz_net_fit_normalization(nmz_net *net, const nmz_dataset *ds);
/* Kind, whether fitted, and the per-group parameters (NULL if not fitted). */
nmz_norm nmz_net_normalization(const nmz_net *net, int *fitted, int *groups, const float **offset,
                               const float **scale);

/* Class names for classification outputs (copied). */
nmz_status nmz_net_set_class_names(nmz_net *net, int n, const char *const *names);
int nmz_net_class_count(const nmz_net *net);
const char *nmz_net_class_name(const nmz_net *net, int i);

/* ----------------------------------------------------------------- compute */

/* Forward a batch x of shape [B, input_shape...]. *y points to the net's
 * output buffer (valid until the next forward). */
nmz_status nmz_net_forward(nmz_net *net, const nmz_tensor *x, const nmz_tensor **y);
/* Same with an explicit mode: training = 1 activates dropout (and consumes
 * the net's PRNG); nmz_net_forward uses evaluation mode (training = 0). */
nmz_status nmz_net_forward_mode(nmz_net *net, const nmz_tensor *x, int training, const nmz_tensor **y);
/* Single sample convenience: x has input_size floats, y receives output_size. */
nmz_status nmz_net_predict(nmz_net *net, const float *x, float *y);
/* Loss of the last forward pass against target t (same shape as output). */
nmz_status nmz_net_compute_loss(nmz_net *net, const nmz_tensor *t, double *loss);
/* Backward pass for the last forward: zeroes all gradients, then computes
 * every gradient (nothing is updated). Optionally returns the loss. */
nmz_status nmz_net_backward(nmz_net *net, const nmz_tensor *t, double *loss);
/* Apply one SGD/momentum step using the current gradients:
 *   v ← μ·v − η·g ;  w ← w + v */
nmz_status nmz_net_update(nmz_net *net, float learning_rate, float momentum);
/* Emit the UPDATE trace events nmz_net_update would emit, without changing
 * any weight or velocity (for "what would this step do?" teaching views). */
nmz_status nmz_net_preview_update(nmz_net *net, float learning_rate, float momentum);
/* One optimizer step with o's optimizer (SGD+momentum or Adam) at the given
 * learning rate. Optimizer state (velocity or Adam moments and step t) is
 * kept in the net; switching optimizer resets it. */
nmz_status nmz_net_optimizer_step(nmz_net *net, const nmz_train_opts *o, float learning_rate);
nmz_status nmz_net_optimizer_preview(nmz_net *net, const nmz_train_opts *o, float learning_rate);
/* Which optimizer the stored state belongs to (-1: none yet) and Adam's t. */
int nmz_net_optimizer_state(const nmz_net *net, long *t);
/* ∂L/∂input of the last backward pass ([B, input_shape...]). */
const nmz_tensor *nmz_net_input_grad(const nmz_net *net);
/* Output of layer i from the last forward (i = -1: the input copy). */
const nmz_tensor *nmz_net_activation(const nmz_net *net, int i);

/* ----------------------------------------------------------------- dataset */

/* ------------------------------------------------------------------- text */

/* Character vocabulary over Unicode code points (not bytes). */
typedef struct nmz_vocab {
    int size;
    uint32_t *codepoints; /* sorted ascending; id = position */
    uint64_t *counts;     /* occurrences in the text it was built from */
    int items;            /* 1: line-per-item text; '\n' ends an item */
} nmz_vocab;

#define NMZ_REPLACEMENT_CHAR 0xFFFDu
/* Decode UTF-8 (RFC 3629: no overlong forms, surrogates or values above
 * U+10FFFF). Each invalid byte becomes U+FFFD and is counted in *invalid.
 * *out is malloc'd (free with free()). */
nmz_status nmz_utf8_decode(const char *s, size_t n, uint32_t **out, size_t *out_n, size_t *invalid);
/* Encode one code point; writes up to 4 bytes + NUL, returns the byte count
 * (0 for an invalid code point). */
int nmz_utf8_encode(uint32_t cp, char out[5]);
/* Vocabulary of the distinct code points in cps[0..n) with their counts. */
nmz_status nmz_vocab_build(nmz_vocab **out, const uint32_t *cps, size_t n);
nmz_status nmz_vocab_create(nmz_vocab **out, int size, const uint32_t *codepoints, const uint64_t *counts,
                            int items);
nmz_status nmz_vocab_copy(nmz_vocab **out, const nmz_vocab *v);
int nmz_vocab_index(const nmz_vocab *v, uint32_t cp); /* -1 if absent */
void nmz_vocab_free(nmz_vocab *v);

typedef struct nmz_dataset {
    int n_samples;
    int n_inputs;
    int n_outputs;
    float *x;           /* n_samples × n_inputs, row-major  */
    float *y;           /* n_samples × n_outputs, row-major — or, when
                           y_index is set, n_samples × time_steps class ids */
    int n_classes;      /* > 0 when targets were one-hot encoded from labels */
    char **class_names; /* n_classes strings, or NULL      */
    int sample_rank;    /* shape of one input sample, e.g. [1,28,28] for */
    int sample_shape[NMZ_MAX_RANK]; /* images; 0 = flat (n_inputs) */
    /* sequences (text): x holds token ids [n_samples × time_steps], y the ids
     * of the next tokens; n_outputs is the vocabulary size and targets are
     * expanded to one-hot rows only when a batch is assembled */
    int time_steps; /* 0 for non-sequence data */
    int y_index;    /* 1: y holds class ids, one per time step (or sample) */
    nmz_vocab *vocab;
    size_t n_tokens;    /* characters in the text after normalisation */
    size_t invalid_utf8; /* bytes replaced by U+FFFD while decoding */
} nmz_dataset;

typedef struct nmz_text_opts {
    int seq_len;            /* time steps per training sequence (default 64)  */
    int items;              /* 1: one item per line (names); '\n' ends an item,
                               empty lines are dropped and the text starts
                               with '\n' so every item has a start context */
    const nmz_vocab *vocab; /* encode with this vocabulary (e.g. the model's);
                               NULL builds one from the text */
} nmz_text_opts;

nmz_text_opts nmz_text_defaults(void);
/* Read a UTF-8 text file into non-overlapping sequences: input = ids
 * [i·T, i·T+T), target = the same window shifted by one character.
 * Normalisation: a UTF-8 BOM is dropped, CR LF and CR become LF. */
nmz_status nmz_dataset_load_text(nmz_dataset **out, const char *path, const nmz_text_opts *opts);

typedef struct nmz_csv_opts {
    int n_targets;   /* trailing columns that are targets (default 1)       */
    int n_inputs;    /* if > 0: the first n_inputs columns are inputs and all
                        remaining columns are targets (overrides n_targets) */
    int header;      /* 1 yes, 0 no, -1 auto-detect (default)               */
    int categorical; /* 1: the single target column holds class labels and
                        is one-hot encoded; 0: numeric; -1: auto (default:
                        categorical when any label is not a number)          */
    char delimiter;  /* default ','                                         */
} nmz_csv_opts;

nmz_csv_opts nmz_csv_defaults(void);
nmz_status nmz_dataset_load_csv(nmz_dataset **out, const char *path,
                                const nmz_csv_opts *opts);
/* Copying constructor (y may be NULL for unlabelled data: zero targets). */
nmz_status nmz_dataset_create(nmz_dataset **out, int n_samples, int n_inputs,
                              int n_outputs, const float *x, const float *y);
void nmz_dataset_free(nmz_dataset *ds);
/* MNIST-style IDX files (big-endian; images: unsigned bytes [n, rows, cols],
 * labels: unsigned bytes [n]). Pixels are kept as raw 0..255 floats (use
 * normalization); labels are one-hot with class names "0", "1", ...
 * limit > 0 reads only the first `limit` samples. sample shape [1,rows,cols]. */
nmz_status nmz_dataset_load_idx(nmz_dataset **out, const char *images_path, const char *labels_path,
                                int limit);
/* Keep only the first n samples (n >= 1). */
nmz_status nmz_dataset_truncate(nmz_dataset *ds, int n);

/* ---------------------------------------------------------------- training */

/* Train with mini-batch SGD + momentum until the whole-set loss reaches
 * opts->target_loss or opts->max_epochs is hit. opts may be NULL: the net's
 * stored training configuration is used. */
nmz_status nmz_net_train(nmz_net *net, const nmz_dataset *ds,
                         const nmz_train_opts *opts, nmz_train_result *result);
/* Whole-set loss and accuracy (accuracy NaN when not meaningful). */
nmz_status nmz_net_evaluate(nmz_net *net, const nmz_dataset *ds, double *loss,
                            double *accuracy);

/* -------------------------------------------------------------- quick mode */

/* The defaults "quick mode" picks for a network with the given layer sizes
 * (sizes[0] = inputs, sizes[n-1] = outputs), with a one-line reason each. */
typedef struct nmz_quick {
    nmz_act hidden_act;
    const char *why_hidden_act;
    nmz_act output_act;
    const char *why_output_act;
    nmz_init init; /* resolved for the hidden activation */
    const char *why_init;
    nmz_loss loss;
    const char *why_loss;
    nmz_train_opts train;
    const char *why_learning_rate;
    const char *why_momentum;
    const char *why_batch;
    const char *why_epochs;
    const char *why_target_loss;
} nmz_quick;

nmz_status nmz_quick_defaults(int n_sizes, const int *sizes, nmz_quick *out);
/* Only the training fields (train, why_learning_rate ... why_target_loss)
 * for a given loss; nmz_quick_defaults uses this too. */
nmz_status nmz_quick_train_defaults(nmz_loss loss, nmz_quick *out);
/* Build a fully configured net from layer sizes using nmz_quick_defaults.
 * `quick` (may be NULL) receives the choices that were made. */
nmz_status nmz_net_create_quick(nmz_net **out, int n_sizes, const int *sizes,
                                uint64_t seed, nmz_quick *quick);

/* ---------------------------------------------------------------- file I/O */

/* Portable .nmz format (DESIGN §7). Builds the net if needed. Writes
 * format 3; optimizer state (velocity / Adam moments) is included when the
 * net has been trained, unless NMZ_SAVE_NO_OPTIMIZER is given. Files of
 * format 2 and 3 can be loaded. */
#define NMZ_SAVE_NO_OPTIMIZER 1u
nmz_status nmz_net_save(nmz_net *net, const char *path);
nmz_status nmz_net_save_ex(nmz_net *net, const char *path, unsigned flags);
nmz_status nmz_net_load(nmz_net **out, const char *path);

/* ------------------------------------------------------------------- trace */

typedef enum nmz_trace_phase {
    NMZ_TRACE_FORWARD = 0,
    NMZ_TRACE_LOSS,
    NMZ_TRACE_BACKWARD,
    NMZ_TRACE_UPDATE
} nmz_trace_phase;

typedef struct nmz_trace_item {
    const char *name;
    const nmz_tensor *t;
} nmz_trace_item;

#define NMZ_TRACE_MAX_SCALARS 8

/* A named number attached to an event: "L" (loss), "lr", "momentum", ... */
typedef struct nmz_trace_scalar {
    const char *name;
    double value;
} nmz_trace_scalar;

typedef struct nmz_trace_event {
    nmz_trace_phase phase;
    long step;              /* 1-based number of the pass the event belongs to */
    int layer;              /* layer index, -1 for loss events              */
    int time_step;          /* -1 unless a sequence layer emits per step    */
    const char *layer_type; /* "dense", "activation", "loss", ...           */
    const char *detail;     /* activation/loss name, or param name for updates */
    int n_scalars;          /* loss events: "L"; update events: "lr", "momentum" */
    nmz_trace_scalar scalars[NMZ_TRACE_MAX_SCALARS];
    int n_items;
    nmz_trace_item items[NMZ_TRACE_MAX_ITEMS];
} nmz_trace_event;

typedef void (*nmz_trace_fn)(const nmz_trace_event *ev, void *user);

/* Install (or with fn == NULL remove) a trace callback. Tensors inside an
 * event are only valid during the callback. */
void nmz_net_set_trace(nmz_net *net, nmz_trace_fn fn, void *user);

const char *nmz_trace_phase_name(nmz_trace_phase p);
/* Find an item by name in an event (NULL if absent). */
const nmz_tensor *nmz_trace_find(const nmz_trace_event *ev, const char *name);
/* Find a scalar by name; returns 1 and stores it in *out if present. */
int nmz_trace_scalar_get(const nmz_trace_event *ev, const char *name, double *out);
/* Write one event as one JSON line (no trailing comma, ends with '\n'). */
nmz_status nmz_trace_event_write_json(const nmz_trace_event *ev, FILE *f);

/* A recorder deep-copies every event it receives. Use
 * nmz_net_set_trace(net, nmz_trace_recorder_fn, rec). */
typedef struct nmz_trace_recorder nmz_trace_recorder;
nmz_status nmz_trace_recorder_create(nmz_trace_recorder **out);
void nmz_trace_recorder_free(nmz_trace_recorder *rec);
void nmz_trace_recorder_fn(const nmz_trace_event *ev, void *rec);
void nmz_trace_recorder_clear(nmz_trace_recorder *rec);
size_t nmz_trace_recorder_count(const nmz_trace_recorder *rec);
const nmz_trace_event *nmz_trace_recorder_get(const nmz_trace_recorder *rec, size_t i);
/* NMZ_ERR_NOMEM if a copy failed while recording (events were dropped). */
nmz_status nmz_trace_recorder_status(const nmz_trace_recorder *rec);

/* ---------------------------------------------------------- gradient check */

typedef struct nmz_gradcheck_result {
    size_t checked;      /* number of scalars compared                 */
    double max_abs_err;  /* max |analytic − numeric|                   */
    double max_rel_err;  /* max |a − n| / max(|a| + |n|, 1e-8)         */
    int worst_layer;     /* -1 means the input gradient                 */
    char worst_param[16];/* "W", "b", "x"                               */
    size_t worst_index;
    double worst_analytic;
    double worst_numeric;
    size_t failures;     /* scalars with |a−n| > atol + rtol·max(|a|,|n|) */
    size_t kinks;        /* checks skipped: a relu changed side or a max-pool
                            winner changed within ±eps, so the loss has no
                            derivative there and the difference quotient is
                            meaningless (not counted in `checked`)        */
} nmz_gradcheck_result;

/* Compare backprop gradients (all params and the input) with central
 * differences (L(w+eps) − L(w−eps)) / 2eps. Parameters are restored.
 * Tolerances: atol + rtol·max(|a|,|n|); pass 0 for defaults (1e-4, 2e-2). */
nmz_status nmz_net_gradcheck(nmz_net *net, const nmz_tensor *x, const nmz_tensor *t,
                             double eps, double atol, double rtol,
                             nmz_gradcheck_result *result);
/* Same, choosing the mode: training = 1 (dropout active; the PRNG state is
 * replayed before every forward pass so each pass sees the same mask) or
 * training = 0 (evaluation mode). nmz_net_gradcheck uses training = 1. */
nmz_status nmz_net_gradcheck_mode(nmz_net *net, const nmz_tensor *x, const nmz_tensor *t,
                                  double eps, double atol, double rtol, int training,
                                  nmz_gradcheck_result *result);

/* ----------------------------------------------------------------- summary */

/* Write a human-readable multi-line summary of the architecture. Returns the
 * number of characters that would have been written (like snprintf). */
int nmz_net_summary(const nmz_net *net, char *buf, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* NEUROMZ_H */
