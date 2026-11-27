/*
 * neuromz_backend.h — the device backend interface (DESIGN §11).
 *
 * A backend is a table of functions: memory management and the numeric
 * kernels ("ops") the layers are written in. Every layer computes through
 * the backend in its nmz_ctx, so the same layer code runs on the CPU and on
 * a GPU. The CPU backend is the reference: its ops are the loops of v2.0.1,
 * so results on the CPU are bit-identical to that release.
 *
 * Who needs this header:
 *   - authors of a new backend (HIP, Vulkan, OpenCL, ...): implement the
 *     table and add it to the registry in src/backend.c (DESIGN §11.6);
 *   - authors of a layer that should run on every backend: get pointers
 *     with nmz_tensor_ptr() and call ctx->backend->ops (a layer that reads
 *     tensor->data directly still works: the engine moves its data to the
 *     host around each call, see NMZ_LAYER_BACKEND in neuromz_layer.h).
 *
 * Conventions for every op:
 *   - matrices are row-major floats; pointers are memory of THIS backend
 *     (host memory for the CPU backend, device memory otherwise), obtained
 *     with nmz_tensor_ptr();
 *   - ops may run asynchronously; results written through `double *` (or
 *     `float *` host) scalar pointers are valid only after be->sync(be);
 *   - "+=" in a comment means the op adds into its output;
 *   - an op returns NMZ_OK or an error (with nmz_last_error() set).
 * Required ops must be provided by every backend. Fused kernels (marked
 * optional) may be NULL: the layer then composes the same math from the
 * required ops (that is what the CUDA backend does).
 */
#ifndef NEUROMZ_BACKEND_H
#define NEUROMZ_BACKEND_H

#include "neuromz.h"

#ifdef __cplusplus
extern "C" {
#endif

#define NMZ_BACKEND_ABI 1 /* bumped when the table below changes */

typedef struct nmz_backend nmz_backend;

/* Access modes for nmz_tensor_ptr. */
#define NMZ_READ 1u   /* the op only reads the tensor                        */
#define NMZ_WRITE 2u  /* the op overwrites every element (old values unused) */
#define NMZ_UPDATE 3u /* the op reads and writes (e.g. adds into a gradient)  */

/* Pointer to t's data in `be`'s memory, valid until the next access of t
 * through another backend or a host access. A tensor keeps one copy per
 * memory and moves data only when the copy asked for is out of date
 * (DESIGN §11.3). Returns NULL for t == NULL, or on failure (out of device
 * memory: nmz_last_error() says how much was needed). */
float *nmz_tensor_ptr(const nmz_backend *be, nmz_tensor *t, unsigned mode);

/* B rows of n numbers starting at p, row r at p + r·ld (a [B, n] slice of
 * a larger matrix, e.g. time step t of a [B, T, n] sequence: p = base + t·n,
 * ld = T·n). */
typedef struct nmz_rows {
    float *p;
    int ld;
} nmz_rows;

/* Trace-only per-step values of a recurrent backward pass, each [B, T, H]
 * (da: [B, T, G], G = gates·H). NULL members are not wanted. */
typedef struct nmz_seq_record {
    int T;
    float *dh_next; /* ∂L/∂h_t arriving from step t+1              */
    float *dh;      /* ∂L/∂h_t (from above + from t+1)             */
    float *da;      /* ∂L/∂(gate inputs) (RNN: ∂L/∂z)              */
    float *dh_prev; /* ∂L/∂h_{t−1}, sent to step t−1                */
    float *dc_next; /* LSTM: ∂L/∂c_t from step t+1 (already × f)   */
    float *dc;      /* LSTM: ∂L/∂c_t                                */
    float *dc_prev; /* LSTM: ∂L/∂c_{t−1} = ∂L/∂c_t ⊙ f              */
    float *direct;  /* GRU: ∂L/∂h_t ⊙ z (the direct path)          */
} nmz_seq_record;

/* Shapes of one conv2d call: x [B, C, H, W], W [F, C, kh, kw], y [B, F, OH, OW]. */
typedef struct nmz_conv_shape {
    int B, C, H, W, F, kh, kw, stride, pad, OH, OW;
} nmz_conv_shape;

/* GEMM flags (as for nmz_matmul): TRANS_A = A is stored [K, M], TRANS_B = B
 * is stored [N, K] (not both), ACCUMULATE = C += A·B instead of C = A·B. */

struct nmz_backend {
    int abi;              /* NMZ_BACKEND_ABI                                    */
    const char *name;     /* "cpu", "cuda", ...                                 */
    const char *summary;  /* one line for listings                               */
    int host_memory;      /* 1: computes in host memory (tensors never move)    */
    void *impl;           /* the backend's own state                            */

    /* ---- lifecycle (device backends initialise lazily, on first selection) */
    /* NMZ_OK when usable; otherwise NMZ_ERR_UNSUPPORTED and why (in why[n]). */
    nmz_status (*init)(const nmz_backend *be, char *why, size_t n);
    void (*info)(const nmz_backend *be, nmz_device_info *out);
    /* optional: cap the memory alloc may hand out (0 = the backend's default) */
    nmz_status (*memory_limit)(const nmz_backend *be, size_t bytes);

    /* ---- memory (unused when host_memory) */
    void *(*alloc)(const nmz_backend *be, size_t bytes); /* NULL + message on failure */
    void (*release)(const nmz_backend *be, void *p);
    nmz_status (*upload)(const nmz_backend *be, void *dev, const void *host, size_t bytes);
    nmz_status (*download)(const nmz_backend *be, void *host, const void *dev, size_t bytes);
    nmz_status (*sync)(const nmz_backend *be); /* wait; scalar outputs become valid */

    /* ---- element-wise and data movement */
    nmz_status (*fill)(const nmz_backend *be, size_t n, float v, float *x);
    /* dst[r][c] = src[r][c] for a rows × cols block (row strides lds, ldd) */
    nmz_status (*copy)(const nmz_backend *be, size_t rows, size_t cols, const float *src, size_t lds, float *dst,
                       size_t ldd);
    nmz_status (*add)(const nmz_backend *be, size_t n, const float *a, const float *b, float *out); /* a + b */
    nmz_status (*mul)(const nmz_backend *be, size_t n, const float *a, const float *b, float *out); /* a · b */

    /* ---- matrix products and reductions */
    /* C[M,N] (+)= op(A)·op(B), row strides lda, ldb, ldc (the stored rows) */
    nmz_status (*gemm)(const nmz_backend *be, unsigned flags, int M, int N, int K, const float *A, int lda,
                       const float *B, int ldb, float *C, int ldc);
    nmz_status (*add_bias)(const nmz_backend *be, size_t rows, int cols, float *y, const float *b); /* y[r] += b */
    /* y[a][b][c] += bias[b] for y [A, B, C] (conv bias per filter) */
    nmz_status (*add_bias3)(const nmz_backend *be, int A, int B, int C, float *y, const float *bias);
    /* out[j] += Σ_r a[r][j] (summed in double) */
    nmz_status (*colsum_add)(const nmz_backend *be, size_t rows, int cols, const float *a, float *out);
    /* out[r] += Σ_j a[r][j] (summed in double) */
    nmz_status (*rowsum_add)(const nmz_backend *be, int rows, size_t cols, const float *a, float *out);
    /* out[b][a][c] = in[a][b][c] for in [A, B, C] */
    nmz_status (*transpose3)(const nmz_backend *be, int A, int B, int C, const float *in, float *out);

    /* ---- activations, softmax, losses */
    nmz_status (*act_forward)(const nmz_backend *be, nmz_act act, float alpha, size_t n, const float *z, float *a);
    /* dz = dy · f'(z) (dz may alias dy); fprime (optional) receives f'(z) */
    nmz_status (*act_backward)(const nmz_backend *be, nmz_act act, float alpha, size_t n, const float *z,
                               const float *dy, float *dz, float *fprime);
    nmz_status (*softmax_forward)(const nmz_backend *be, size_t rows, int cols, const float *z, float *p);
    /* dz = p ⊙ (dy − Σ dy·p), per row */
    nmz_status (*softmax_backward)(const nmz_backend *be, size_t rows, int cols, const float *p, const float *dy,
                                   float *dz);
    /* mean loss over rows into *out (valid after sync); z = softmax input
     * for cross-entropy (NULL: computed from y) */
    nmz_status (*loss_value)(const nmz_backend *be, nmz_loss kind, size_t rows, int cols, const float *y,
                             const float *t, const float *z, double *out);
    /* MSE: g = (y − t)/rows; CE fused with softmax: g = (p·Σt − t)/rows */
    nmz_status (*loss_grad)(const nmz_backend *be, nmz_loss kind, size_t rows, int cols, const float *y,
                            const float *t, float *g);

    /* ---- convolution and pooling */
    /* col [C·kh·kw, B·OH·OW]: column b·OH·OW + p holds the receptive field of
     * output position p of sample b (zero padding) */
    nmz_status (*im2col)(const nmz_backend *be, const nmz_conv_shape *s, const float *x, float *col);
    /* dx = the sum of every column entry back at the input it came from */
    nmz_status (*col2im)(const nmz_backend *be, const nmz_conv_shape *s, const float *col, float *dx);
    /* planes = B·C; argmax = winner's flat index iy·W + ix (first max wins) */
    nmz_status (*maxpool_forward)(const nmz_backend *be, int planes, int H, int W, int size, int stride, int OH,
                                  int OW, const float *x, float *y, float *argmax);
    nmz_status (*maxpool_backward)(const nmz_backend *be, int planes, int H, int W, int size, int stride, int OH,
                                   int OW, const float *dy, const float *argmax, float *dx);

    /* ---- dropout, embeddings, positions */
    /* mask = 0 with probability rate, else 1/(1−rate); draws from rng (the
     * CPU draws one number per element, a device may draw a seed) */
    nmz_status (*dropout_mask)(const nmz_backend *be, size_t n, float rate, nmz_rng *rng, float *mask);
    /* y[i] = W[ids[i]] (rows of D); ids are floats, exact integers < V */
    nmz_status (*embed_forward)(const nmz_backend *be, size_t n, int V, int D, const float *ids, const float *W,
                                float *y);
    nmz_status (*embed_backward)(const nmz_backend *be, size_t n, int V, int D, const float *ids, const float *dy,
                                 float *dW); /* dW[ids[i]] += dy[i] */
    /* y[b][t] = x[b][t] + P[off[b] + t]; off is a HOST array of B offsets */
    nmz_status (*posenc_forward)(const nmz_backend *be, int B, int T, int D, const int *off, const float *x,
                                 const float *P, float *y);
    nmz_status (*posenc_backward)(const nmz_backend *be, int B, int T, int D, const int *off, const float *dy,
                                  float *dP); /* dP[off[b] + t] += dy[b][t] */

    /* ---- LayerNorm and attention */
    nmz_status (*layernorm_forward)(const nmz_backend *be, size_t rows, int D, float eps, const float *x,
                                    const float *g, const float *b, float *y, float *xhat, float *mean, float *rstd);
    /* dg += Σ dy·x̂, db += Σ dy; dx (optional) =; dxhat (optional, trace) */
    nmz_status (*layernorm_backward)(const nmz_backend *be, size_t rows, int D, const float *dy, const float *xhat,
                                     const float *rstd, const float *g, float *dx, float *dg, float *db,
                                     float *dxhat);
    /* the core of causal multi-head attention for every (row, head):
     * qkv [B, T, 3E] (q at columns h·d), kv [B, Tk, 2E] (k | v), Tp cached
     * positions (query t sees keys 0 … Tp+t); P [B, H, T, Tk] (0 where
     * masked), S (optional, trace) raw scores, ctx [B, T, E] */
    nmz_status (*attention_forward)(const nmz_backend *be, int B, int H, int T, int Tk, int Tp, int d, int E,
                                    const float *qkv, const float *kv, float *P, float *S, float *ctx);
    /* dP = ∂L/∂scores (out); dqkv (q columns) and dkv += (zero them first) */
    nmz_status (*attention_backward)(const nmz_backend *be, int B, int H, int T, int Tk, int Tp, int d, int E,
                                     const float *qkv, const float *kv, const float *P, const float *dctx, float *dP,
                                     float *dqkv, float *dkv);

    /* ---- recurrent gate math: one time step of B rows (nmz_rows slices).
     * The layers do the matrix products around them with gemm. */
    nmz_status (*rnn_cell_forward)(const nmz_backend *be, int B, int H, nmz_rows z, nmz_rows h); /* h = tanh(z) */
    /* dh = dy + dh_next; dz = dh ⊙ (1 − h²); rec (optional) at step t */
    nmz_status (*rnn_cell_backward)(const nmz_backend *be, int B, int H, nmz_rows h, nmz_rows dy, nmz_rows dh_next,
                                    nmz_rows dz, const nmz_seq_record *rec, int t);
    /* a: gate inputs [B, 4H] (i f g o) in, gates after σ/tanh out (in place);
     * c = f⊙c_prev + i⊙g; h = o⊙tanh(c) */
    nmz_status (*lstm_cell_forward)(const nmz_backend *be, int B, int H, nmz_rows a, nmz_rows c_prev, nmz_rows c,
                                    nmz_rows h);
    /* da [B, 4H] out; dc_next in/out (becomes ∂L/∂c_{t−1}) */
    nmz_status (*lstm_cell_backward)(const nmz_backend *be, int B, int H, nmz_rows gates, nmz_rows c,
                                     nmz_rows c_prev, nmz_rows dy, nmz_rows dh_next, nmz_rows dc_next, nmz_rows da,
                                     const nmz_seq_record *rec, int t);
    /* GRU (gates z r h~ in a [B, 3H]): step 1: z, r = σ(a) in place,
     * rh = r ⊙ h_prev; step 2 (after a_h~ += rh·W_hh~): h~ = tanh(a_h~) in
     * place, h = z⊙h_prev + (1−z)⊙h~ */
    nmz_status (*gru_cell_forward1)(const nmz_backend *be, int B, int H, nmz_rows a, nmz_rows h_prev, nmz_rows rh);
    nmz_status (*gru_cell_forward2)(const nmz_backend *be, int B, int H, nmz_rows a, nmz_rows h_prev, nmz_rows h);
    /* step 1: dh = dy + dh_next; da_z, da_h~ out; direct = dh⊙z; rh = r⊙h_prev
     * step 2 (after drh = da_h~·W_hh~ᵀ): da_r out; dh_next = direct + drh⊙r
     * (the layer then adds [da_z da_r]·[W_hz W_hr]ᵀ) */
    nmz_status (*gru_cell_backward1)(const nmz_backend *be, int B, int H, nmz_rows gates, nmz_rows h_prev,
                                     nmz_rows dy, nmz_rows dh_next, nmz_rows da, nmz_rows direct, nmz_rows rh,
                                     const nmz_seq_record *rec, int t);
    nmz_status (*gru_cell_backward2)(const nmz_backend *be, int B, int H, nmz_rows gates, nmz_rows h_prev,
                                     nmz_rows drh, nmz_rows direct, nmz_rows da, nmz_rows dh_next);

    /* ---- optimizers and clipping */
    /* v_out = (fresh ? 0 : μ·v_prev) − η·g (v_out may alias v_prev) */
    nmz_status (*sgd_velocity)(const nmz_backend *be, size_t n, float lr, float mu, int fresh, const float *g,
                               const float *v_prev, float *v_out);
    /* Adam(W) in place: m, v moments, w −= η·(m̂/(√v̂ + ε) + λ·w) */
    nmz_status (*adam_update)(const nmz_backend *be, size_t n, double lr, double b1, double b2, double eps,
                              double bc1, double bc2, double decay, float *w, const float *g, float *m, float *v);
    /* the same step computed aside (trace, preview): new m, v, m̂, v̂, Δ */
    nmz_status (*adam_preview)(const nmz_backend *be, size_t n, double lr, double b1, double b2, double eps,
                               double bc1, double bc2, double decay, int fresh, const float *w, const float *g,
                               const float *m, const float *v, float *m_out, float *v_out, float *mhat, float *vhat,
                               float *delta);
    /* Σ of x_k[e]² over all k tensors, in double, into *out (after sync) */
    nmz_status (*sumsq)(const nmz_backend *be, int k, float *const *x, const size_t *n, double *out);
    nmz_status (*scale)(const nmz_backend *be, int k, float *const *x, const size_t *n, double f); /* x ·= f */

    /* ---- the training loop */
    /* x = (x − offset_g)·scale_g (grad = 0) or x ·= scale_g (grad = 1), with
     * g = ((e mod per) / div) mod groups; offset, scale in this memory */
    nmz_status (*norm_apply)(const nmz_backend *be, size_t n, size_t per, size_t div, int groups,
                             const float *offset, const float *scale, float *x, int grad);
    /* dst[r] = src[idx[r]] (rows of `width` floats; idx floats in this memory) */
    nmz_status (*gather_rows)(const nmz_backend *be, int rows, size_t width, const float *idx, const float *src,
                              float *dst);
    /* out [n, V] = one-hot rows of the ids (exact integers < V) */
    nmz_status (*onehot)(const nmz_backend *be, size_t n, int V, const float *ids, float *out);
    /* mode 1: argmax(y) == argmax(t); 2: every output on the same side of
     * 0.5; *out (after sync) = the number of correct rows */
    nmz_status (*count_correct)(const nmz_backend *be, int mode, size_t rows, int cols, const float *y,
                                const float *t, double *out);

    /* ---- optional fused kernels (NULL: composed from the ops above). A
     * kernel may also decline a call it does not handle (a hidden size too
     * large for one thread block, say) by returning NMZ_ERR_UNSUPPORTED
     * before touching anything; the layer then composes the math. */
    nmz_status (*dense_forward)(const nmz_backend *be, size_t rows, int n_in, int units, const float *x,
                                const float *W, const float *b, float *y);
    nmz_status (*dense_backward)(const nmz_backend *be, size_t rows, int n_in, int units, const float *x,
                                 const float *W, const float *dy, float *dW, float *db, float *dx);
    /* scratch: col and dcol hold C·kh·kw·OH·OW floats, acc F·C·kh·kw doubles */
    nmz_status (*conv_forward)(const nmz_backend *be, const nmz_conv_shape *s, int naive, const float *x,
                               const float *W, const float *b, float *y, float *col);
    nmz_status (*conv_backward)(const nmz_backend *be, const nmz_conv_shape *s, int naive, const float *x,
                                const float *W, const float *dy, float *dW, float *db, float *dx, double *acc,
                                float *col, float *dcol);
    /* z (optional, trace) = pre-activations [B, T, H] */
    nmz_status (*rnn_forward)(const nmz_backend *be, int B, int T, int D, int H, const float *x, const float *h0,
                              const float *Wxh, const float *Whh, const float *b, float *y, float *z);
    /* dh_next: [B, H] work space */
    nmz_status (*rnn_backward)(const nmz_backend *be, int B, int T, int D, int H, const float *x, const float *y,
                               const float *h0, const float *Wxh, const float *Whh, const float *dy, float *dWxh,
                               float *dWhh, float *db, float *dx, float *dh_next, const nmz_seq_record *rec);
    nmz_status (*lstm_forward)(const nmz_backend *be, int B, int T, int D, int H, const float *x, const float *h0,
                               const float *c0, const float *Wx, const float *Wh, const float *b, float *y,
                               float *gates, float *c);
    nmz_status (*lstm_backward)(const nmz_backend *be, int B, int T, int D, int H, const float *x, const float *y,
                                const float *h0, const float *c0, const float *gates, const float *c, const float *Wx,
                                const float *Wh, const float *dy, float *dWx, float *dWh, float *db, float *dx,
                                float *dh_next, float *dc_next, const nmz_seq_record *rec);
    nmz_status (*gru_forward)(const nmz_backend *be, int B, int T, int D, int H, const float *x, const float *h0,
                              const float *Wx, const float *Wh, const float *b, float *y, float *gates);
    nmz_status (*gru_backward)(const nmz_backend *be, int B, int T, int D, int H, const float *x, const float *y,
                               const float *h0, const float *gates, const float *Wx, const float *Wh, const float *dy,
                               float *dWx, float *dWh, float *db, float *dx, float *dh_next,
                               const nmz_seq_record *rec);
};

/* Add a backend to the registry (selectable by name with
 * nmz_backend_select). `be` must stay valid for the life of the program.
 * The built-in backends are always registered. */
nmz_status nmz_backend_register(const nmz_backend *be);
/* The backend that nets compute on (the selected one, CPU by default). */
const nmz_backend *nmz_backend_current(void);
/* For backend code: record the message nmz_last_error() returns and give
 * back s (e.g. return nmz_backend_error(NMZ_ERR_NOMEM, "...") from alloc). */
nmz_status nmz_backend_error(nmz_status s, const char *fmt, ...)
#if defined(__GNUC__)
    __attribute__((format(printf, 2, 3)))
#endif
    ;

#ifdef __cplusplus
}
#endif

#endif /* NEUROMZ_BACKEND_H */
