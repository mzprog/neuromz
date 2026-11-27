# NeuroMZ v2 — Design

NeuroMZ v2 is an **educational** neural-network engine in C11. Its job is to be
*correct and inspectable* first, fast second. A beginner can build and train a
network with one command; an advanced student can configure every layer and
training knob; and after (or during) training the engine can expose every
intermediate value — `z`, `a`, `δ`, gradients, weight updates — so the math
can be printed now and visualised later.

Status of this document: written before implementation, then corrected where
the implementation taught us something (see the git history of this file).
Phase 1 (dense networks) added decisions 27–33; phase 2 (CNN, Adam,
normalization, MNIST) added decisions 34–50; phase 3 (text, Embedding,
RNN, generation) added decisions 51–70; phase 4 (LSTM, GRU, orthogonal
init, gate traces, gradient-survival comparison, Arabic prose) added
decisions 71–90; phase 5 (a decoder-only Transformer, AdamW, warm-up,
the blocked matrix product, the KV cache, attention traces) added
decisions 91–110. Release 2.1 (device backends: the backend layer, the
CPU and CUDA backends, devices in the program) added §11 and decisions
127–157. Release 2.2 (threads, models described in words in the library,
the program's remote mode; the server and the protocol have their own
design documents) added §11.8 and decisions 158–165.

---

## 1. Directory layout

```
include/
  neuromz.h            public API (everything a library user needs)
  neuromz_layer.h      layer-author API (vtable, params, trace emit, byte I/O)
  neuromz_backend.h    backend-author API (the table of ops, tensor pointers)
src/                   libneuromz — platform-independent C11, libc + libm only
  error.c              status strings, thread-local last-error message
  tensor.c             contiguous float tensor
  rng.c                PCG32 PRNG (seedable, reproducible everywhere)
  activation.c         activation functions + derivatives, Activation layer
  dense.c              Dense (fully connected) layer
  conv2d.c             Conv2D (im2col by default, naive loops as reference)
  shape_layers.c       MaxPool2D, Flatten, Dropout
  normalize.c          input normalization, LR schedules, optimizer/schedule names
  loss.c               MSE, softmax+cross-entropy
  net.c                nmz_net: layer stack, forward/backward, optimizer step
  train.c              epochs, shuffling, mini-batches, stop rule, evaluation
  trace.c              trace events, recorder, JSON-lines export
  gradcheck.c          numerical gradient check (public: it is a teaching tool)
  io.c                 portable .nmz save/load, CRC32, byte writer/reader
  dataset.c            dataset type, CSV loader, IDX (MNIST) loader
  text.c               UTF-8 decoding, code-point vocabularies, text datasets
  recurrent.c          Embedding, RNN (time loop + BPTT inside the layer),
                       orthogonal init and helpers shared with lstm.c/gru.c
  lstm.c               LSTM (gates i f g o, cell state c, BPTT along h and c)
  gru.c                GRU (Cho et al. 2014: gates z r, candidate h~)
  generate.c           sampling text: temperature, top-k, seeded PCG32,
                       batches of rows (finished rows leave the batch);
                       Transformers: KV cache, context window, sliding
  matmul.c             C = A·B, cache-blocked with a register tile, the
                       naive reference, optional OpenMP (-DNMZ_OPENMP=ON)
  attention.c          causal multi-head self-attention (+ KV cache)
  transformer.c        position encoding, LayerNorm, feed-forward, the
                       pre-LN block (composite layer), tied output
  defaults.c           "quick mode" defaults and the reasons for them
  backend.c            backend registry and selection; device copies of
                       tensors (§11)
  backend_cpu.c        the CPU backend: the ops (v2.0.1's loops) and the
                       "cpu-generic" table without fused kernels
  internal.h           private helpers shared by the library sources
tests/                 test_*.c (unit, gradient checks, convergence, I/O),
                       data/*.nmz (model files of earlier phases), package/
                       (a find_package consumer); all run under sanitizers
examples/              xor.csv, xor_and.csv, iris.csv (+ 120/30 train/test
                       split), names_en.txt, names_ar.txt, DATASETS.md,
                       api_xor.c, api_mnist.c (CTest regressions),
                       bench_conv.c, bench_matmul.c, bench_device.c
scripts/get_mnist.*    downloads MNIST into data/ (gitignored), verifies
                       MD5 + SHA-256 (.sh: Unix-like systems, .ps1: Windows)
scripts/get_text.*     downloads Alice (Gutenberg #11) and Kalila wa-Dimna
                       (Arabic Wikisource, fixed revisions) into data/,
                       verifies SHA-256 of every download and of the result;
                       strip_wikitext.awk removes markup (ported in .ps1)
cmake/                 neuromzConfig.cmake.in (find_package support)
docs/DESIGN.md         this file
legacy/                v1 sources (2017–2020), unchanged, tag `v1-legacy`
CMakeLists.txt         library, tests, examples; test, sanitize, bench,
                       data, data-text targets; install + package config
```

The `neuromz` program is its own repository since v2.0.0,
[neuromz-cli](https://github.com/mzprog/neuromz-cli) (`src/` there; it
was `cli/` here, and its history came along — decision 121). It uses only
the public headers. Build trees are CMake's (`build/` by convention, the
sanitizer tree `build-san/`); `-DNMZ_OPENMP=ON` builds with OpenMP.

The library never includes POSIX headers, never touches the terminal, never
prints, and never calls `exit()`/`abort()`; its one platform difference is
how a UTF-8 file name is opened on Windows (decision 113). The CLI is the
only place allowed to do the rest.

---

## 2. Core data model

### 2.1 Tensor

```c
#define NMZ_MAX_RANK 4
typedef struct nmz_tensor {
    int    rank;                 /* 0..4                                  */
    int    shape[NMZ_MAX_RANK];  /* shape[0] is the batch dim when batched */
    size_t size;                 /* product of shape                      */
    size_t capacity;             /* allocated floats (>= size)            */
    float *data;                 /* contiguous, row-major (C order)       */
    struct nmz_tensor_dev *dev;  /* device copy (2.1, §11.3), or NULL     */
} nmz_tensor;
```

* One dtype: `float` (32-bit). Integer token ids for future embedding layers
  are stored as floats; they are exact up to 2^24 = 16.7M, far beyond any
  vocabulary this project will use.
* Row-major and **batch-first**: Dense uses `[B, F]`, sequence layers will use
  `[B, T, F]`, convolution `[B, C, H, W]`. Rank 4 covers all planned layers.
* `nmz_tensor_reshape` re-uses the allocation when it is large enough, so the
  per-layer activation buffers are allocated once and re-used every batch.

### 2.2 Network handle (no globals)

```c
typedef struct nmz_net nmz_net;   /* opaque */
```

An `nmz_net` owns: the per-sample input shape, a stack of layers, one
activation tensor per layer boundary (`acts[0]` = input copy, `acts[i+1]` =
output of layer `i`), one gradient tensor per boundary (`dacts[i]` =
∂L/∂acts[i]), the loss kind, the PRNG, optimizer state (one velocity tensor per
parameter), the stored training configuration, optional class names, the
trace callback, and a step counter. Any number of networks can coexist; the
library has no mutable global state except the thread-local error message.

### 2.3 Layer interface (vtable)

```c
typedef struct nmz_param { const char *name; nmz_tensor *value; nmz_tensor *grad; } nmz_param;

typedef struct nmz_layer_ops {
    const char *type;                                  /* "dense", "activation", ... */
    /* shape inference + allocation, from the per-sample input shape */
    nmz_status (*setup)(nmz_layer *, int in_rank, const int *in_shape,
                        int *out_rank, int *out_shape);
    /* (re)initialise parameters; `next` lets AUTO init look ahead       */
    nmz_status (*init_params)(nmz_layer *, nmz_rng *, const nmz_layer *next);
    nmz_status (*forward)(nmz_layer *, const nmz_tensor *x, nmz_tensor *y,
                          const nmz_ctx *);
    /* dy = ∂L/∂y in; ADDS into param grads; writes dx = ∂L/∂x (dx may be NULL) */
    nmz_status (*backward)(nmz_layer *, const nmz_tensor *x, const nmz_tensor *y,
                           const nmz_tensor *dy, nmz_tensor *dx, const nmz_ctx *);
    int        (*param_count)(const nmz_layer *);
    nmz_param  (*param)(nmz_layer *, int i);           /* value + grad pair   */
    nmz_status (*save_config)(const nmz_layer *, nmz_writer *);
    nmz_status (*load_config)(nmz_layer **out, nmz_reader *);
    void       (*describe)(const nmz_layer *, char *buf, size_t n);
    void       (*destroy)(nmz_layer *);
    /* optional (phase 3): carried recurrent state, and NMZ_LAYER_* flags */
    nmz_tensor *(*state)(nmz_layer *);
    unsigned   flags;                            /* NMZ_LAYER_DISCRETE_INPUT */
} nmz_layer_ops;

struct nmz_layer {
    const nmz_layer_ops *ops;
    void *self;                                   /* layer-private state */
    int in_rank,  in_shape[NMZ_MAX_RANK];         /* per-sample shapes   */
    int out_rank, out_shape[NMZ_MAX_RANK];
};
```

The engine only ever talks to layers through this table. Adding Conv, RNN,
Embedding or Attention means writing one new `.c` file with an `ops` table and
adding one line to the layer registry (used by the file loader to map a type
name to its `load_config`). Nothing in `net.c` or `train.c` changes.

"params" and "grads" are exposed together as `nmz_param` pairs so a value can
never be paired with the wrong gradient; the optimizer, the saver, the
gradient checker and the trace all iterate over the same list.

Release 2.1 appended three things for device backends (§11): the field
`nmz_ctx.backend` (where to compute), the flag `NMZ_LAYER_BACKEND` (the
layer computes through it) and the optional entry `buffers` (the layer's
caches, so their device memory can be given back).

---

## 3. Sequences: embeddings, RNN and BPTT (implemented in phase 3)

What was planned before phase 3 held up; this is what exists now.

* **The layer owns its time loop.** The RNN layer receives `[B, T, D]` and
  returns all hidden states `[B, T, H]`. Its forward pass loops over `t`
  and keeps what backward needs (the h_t are its output; the initial state
  h₀ in its private cache). Its backward pass walks `t = T−1 … 0`
  (backpropagation through time) and **adds** into the shared weight
  gradients. The engine sees one forward and one backward call.
* **Any-length time dimension.** A per-sample shape may start with
  `NMZ_ANY_LENGTH` (0): text nets are created with input `[T]` and accept
  `[B, T]` batches for any T — 64 in training, 1 per step when generating.
  Every layer computes from the actual tensor shapes; the engine replaces
  the 0 by the batch's T when it shapes the activation buffers; sizes such
  as `nmz_net_output_size` are per time step.
* **Gradient accumulation convention** (engine zeroes, layers add) is what
  makes the shared RNN weights work with no special case.
* **Carried state (truncated BPTT)** uses two optional vtable fields that
  were appended to `nmz_layer_ops` (old layer tables set them to `NULL, 0`):
  `state(l)` returns the tensor the layer carries between forward passes
  (the RNN's last h, `[B, H]`; empty = start from zeros), and `flags`.
  `nmz_ctx.carry` tells the layer to start from / update that state. The
  engine resets it (`nmz_net_reset_state`) at every epoch start and around
  evaluation and generation; gradcheck snapshots and restores it before
  every pass, so the carried state is a constant input like x. The gradient
  that would flow into the carried state is dropped — that is the
  truncation, with window = `seq_len` (k1 = k2 in the usual notation). No
  `detach_state` entry was needed: state is copied values, never a graph.
* **Embedding** takes `[B, T]` token ids (floats, exact < 2^24), returns
  `[B, T, D]`, has `NMZ_LAYER_DISCRETE_INPUT` (no input gradient; gradcheck
  does not perturb ids) and writes only the gradient rows of tokens that
  occur. The optimizer still updates the full matrix (decision 59).
* **The per-step output head needed no wrapper:** Dense acts on the last
  dimension and softmax + cross-entropy work on rows, so
  `rnn → dense(V) → softmax` on `[B, T, H]` is a "time-distributed" head
  with B·T rows. The loss is the mean over all B·T next-character
  predictions (nats per character); perplexity = e^loss.
* **Trace events carry `time_step`:** the RNN emits one summary event per
  pass (`time_step` = −1: x, h₀, W_xh, W_hh, b, all h) and one event per
  step (x_t, h_prev, z_t, h_t forward; dL/dh_out, dL/dh_next, dL/dh, dL/dz,
  dL/dh_prev backward), so the CLI can draw the unrolled network.
* **Attention / residual connections** (still planned) stay composite
  layers inside the sequential stack.

### 3.1 LSTM and GRU (implemented in phase 4, no engine change)

The plan written before phase 4 (an `lstm.c` with an ops table and one
registry line) held; this is what exists.

| piece | LSTM (`lstm.c`) | GRU (`gru.c`) |
|---|---|---|
| shapes | `[T, D]` → `[T, H]`, like the RNN | same |
| parameters | `W_x` `[D, 4H]`, `W_h` `[H, 4H]`, `b` `[4H]`, gates side by side in the order i, f, g, o | `W_x` `[D, 3H]`, `W_h` `[H, 3H]`, `b` `[3H]`, order z, r, h~ |
| forward | `a = x_t·W_x + h_{t−1}·W_h + b`; i, f, o = σ, g = tanh; `c_t = f⊙c_{t−1} + i⊙g`; `h_t = o⊙tanh(c_t)`; the gates and c are cached | z, r = σ(…); `h~ = tanh(x_t·W_xh~ + (r⊙h_{t−1})·W_hh~ + b_h~)`; `h_t = z⊙h_{t−1} + (1−z)⊙h~` (decision 73) |
| backward | BPTT inside the layer carrying ∂L/∂h and ∂L/∂c; `∂L/∂c_{t−1} = ∂L/∂c_t ⊙ f` | carrying ∂L/∂h; the direct part `∂L/∂h_t ⊙ z` is traced separately |
| init | forget-gate bias = 1 (configurable); AUTO = each H×H gate block of `W_h` orthogonal, `W_x` Glorot | AUTO = orthogonal blocks, Glorot `W_x`, biases 0 |
| carried state | one `[B, 2H]` tensor, h \| c side by side | `[B, H]` |
| trace | per step `x_t h_prev c_prev i f g o c_t h_t`; summary `x h0 c0 W_x W_h b gates c h` | per step `x_t h_prev z r h~ h_t`; summary `x h0 W_x W_h b gates h` |
| file | config: units, init, resolved init, f32 forget bias | units, init, resolved init |

What did **not** change: `net.c`, `train.c` and `gradcheck.c`. Any-length
time, carry/reset/state, gradcheck's snapshot of the carried state,
clipping, validation, early stopping, the vocabulary and generation were
written against the vtable in phase 3 and work for the new layers as they
are. Stacking (`lstm:64 dropout:0.2 lstm:64`) needed nothing either: a
recurrent layer's output `[T, H]` is a valid input for the next one, and
Dropout is element-wise. Two things outside the layer files did change:

* `NMZ_TRACE_MAX_ITEMS` went from 8 to 12 — one LSTM step has nine values
  to show. It is a capacity constant of the event struct, not engine
  logic.
* Batched generation (decision 80) removes finished rows from every
  layer's carried state. It relies on the contract that a state tensor has
  one row per stream (`[B, …]`), which all three recurrent layers follow;
  it does not need to know what the row contains.

### 3.2 The Transformer (implemented in phase 5)

A decoder-only, GPT-style character model is a stack the engine already
knows how to run:

```
embedding → posenc → block → block → … → layernorm → tied_output (or dense) → softmax
```

| layer | shapes | parameters | what it does |
|---|---|---|---|
| `posenc` | `[T, D]` → `[T, D]` | learned: `P [max_len, D]`; sinusoidal: none | `y_t = x_t + P[offset + t]`; `max_len` is the context window |
| `layernorm` | any → same | `g`, `b` `[D]` | per row: `(x − μ)/√(σ² + ε) ⊙ g + b` |
| `attention` | `[T, D]` → `[T, D]` | `W_qkv [D, 3E]`, `b_qkv`, `W_o [E, D]`, `b_o` | causal multi-head self-attention, E = heads × head_dim |
| `ffn` | `[T, D]` → `[T, D]` | `W1 [D, F]`, `b1`, `W2 [F, D]`, `b2` | `act(x·W1 + b1)·W2 + b2`, act = gelu or relu |
| `block` | `[T, D]` → `[T, D]` | the 12 of its parts, named `ln1.g`, `attn.W_qkv`, … | `h = x + drop(attn(LN1(x)))`, `y = h + drop(ffn(LN2(h)))` |
| `tied_output` | `[T, D]` → `[T, V]` | `b [V]` (+ the embedding's `W`, shared) | `z = x·Eᵀ + b` |

**The block is a composite layer** (decision 6 had planned residual wiring
this way): it owns four ordinary layers (`layernorm`, `attention`,
`layernorm`, `ffn`), calls them through their ops tables with its own
buffers, adds the residual connections and exposes their parameters under
prefixed names. Each part is also a layer of its own, so each has its own
gradient check, and the CLI can build an attention or feed-forward layer
alone. Trace events of the parts reach the net's callback with `detail` set
to their role (`ln1`, `attn`, `ln2`, `ffn`); the block adds its own event
with the residual stream (`x`, `attn_out`, `h`, `ffn_out`, `y`).

**What changed outside the new layer files**, each the smallest change that
does the job:

* One optional vtable field, `bind(layer, layers, n, index)`, called when
  the net is built (and after loading). Weight tying needs one layer to use
  another layer's parameter; `setup` only sees shapes and `init_params`
  only the next layer. The tied output finds the nearest embedding before
  it and keeps pointers to its `W` and gradient. The engine's rule "zero
  every gradient, then layers add" makes the shared gradient right with no
  special case: the output adds `∂L/∂zᵀ·x` and the embedding adds its rows.
* One layer flag, `NMZ_LAYER_NO_DECAY` (position tables), and AdamW in the
  optimizer (decision 94).
* A per-batch learning rate in the training loop (decision 95); per-epoch
  schedules are unchanged.
* `nmz_net_activation_grad` (∂L/∂ of any layer's output), a read-only
  accessor like `nmz_net_activation`.
* Gradcheck's kink detector also looks at the ReLU inside a feed-forward
  layer (decision 45 extended).

`net.c`'s forward/backward loops, the dataset/batching code, early
stopping, clipping, the file format's structure and the trace mechanism did
not change. The KV cache is the carried state of phase 3 (decision 98).

## 4. Trace system

Goal: show every number the math produces, at near-zero cost when unused.

### 4.1 Mechanism

```c
typedef void (*nmz_trace_fn)(const nmz_trace_event *ev, void *user);
void nmz_net_set_trace(nmz_net *, nmz_trace_fn fn, void *user);  /* NULL = off */
```

Every layer gets an `nmz_ctx` holding the callback pointer. When it is `NULL`
(the default) the only cost is one pointer comparison per layer per pass: no
allocation, no copying, no formatting. When it is set, layers fill a small
stack-allocated event that *points at* their existing tensors and call the
callback synchronously. Pointers are valid only during the callback.

`nmz_trace_recorder` is a ready-made callback that deep-copies events so they
can be inspected later (e.g. by a GUI). `nmz_trace_event_write_json` writes one
event as one JSON line, so a run can be streamed to a `.jsonl` file.

### 4.2 Event format

```c
typedef enum { NMZ_TRACE_FORWARD, NMZ_TRACE_LOSS, NMZ_TRACE_BACKWARD,
               NMZ_TRACE_UPDATE } nmz_trace_phase;
typedef struct { const char *name; const nmz_tensor *t; } nmz_trace_item;
typedef struct nmz_trace_event {
    nmz_trace_phase phase;
    long        step;        /* backward-pass counter of the net (1, 2, ...) */
    int         layer;       /* layer index, -1 for the loss               */
    int         time_step;   /* -1 unless a sequence layer emits per step   */
    const char *layer_type;  /* "dense", "activation", "loss", ...         */
    const char *detail;      /* "sigmoid", "mse", "softmax+cross_entropy",
                                "W"/"b" for updates, ...                   */
    int         n_scalars;   /* named numbers: "L" for loss events,         */
    nmz_trace_scalar scalars[4]; /* "lr", "momentum" for update events  */
    int         n_items;
    nmz_trace_item items[NMZ_TRACE_MAX_ITEMS];
} nmz_trace_event;
```

Item names are a fixed, documented vocabulary (tensors are `[B, …]`):

| Event | items |
|---|---|
| dense forward | `x` (input a⁽ˡ⁻¹⁾), `W`, `b`, `z` |
| activation forward | `z`, `a` |
| loss | `y` (prediction), `t` (target); scalar `L` |
| loss backward | `dL/dy` — or, fused softmax+CE, `dL/dz` (= p − t) |
| activation backward | `dL/da`, `f'(z)` (omitted for softmax), `dL/dz` |
| dense backward | `dL/dz` (= δ), `x`, `dL/dW`, `dL/db`, `dL/dx` |
| normalize forward (layer −1) | `x` (raw), `offset`, `scale` (per group), `x_norm`; `detail` = minmax/zscore |
| conv2d forward | `x`, `W` [F,C,KH,KW], `b`, `z`; scalars `stride`, `pad` |
| conv2d backward | `dL/dz`, `x`, `W`, `dL/dW`, `dL/db`, `dL/dx` |
| maxpool2d forward | `x`, `y`, `argmax` (winner's flat index iy·W+ix per output); scalars `size`, `stride` |
| maxpool2d backward | `dL/dy`, `argmax`, `dL/dx` |
| flatten | forward `x`, `y`; backward `dL/dy`, `dL/dx` |
| embedding | forward `ids`, `W` [V,D], `e`; backward `dL/de`, `ids`, `dL/dW` (only rows of occurring tokens non-zero) |
| rnn (time_step −1) | forward `x`, `h0`, `W_xh`, `W_hh`, `b`, `h` (all steps); backward `dL/dW_xh`, `dL/dW_hh`, `dL/db`, `W_hh`, `dL/dx`; scalar `steps` |
| rnn (time_step t) | forward `x_t`, `h_prev`, `z_t`, `h_t`; backward `dL/dh_out` (from the layer above at t), `dL/dh_next` (from step t+1), `dL/dh` (their sum), `dL/dz` (after tanh′), `dL/dh_prev` (sent to t−1) |
| lstm (time_step −1) | forward `x`, `h0`, `c0`, `W_x`, `W_h`, `b`, `gates` ([B,T,4H] after σ/tanh, order i f g o), `c`, `h`; scalars `steps`, `forget_bias`; backward `dL/dW_x`, `dL/dW_h`, `dL/db`, `W_h`, `dL/dx` |
| lstm (time_step t) | forward `x_t`, `h_prev`, `c_prev`, `i`, `f`, `g`, `o`, `c_t`, `h_t`; backward `dL/dh_out`, `dL/dh_next`, `dL/dh`, `dL/dc_next` (from step t+1, already × f_{t+1}), `dL/dc`, `dL/da` ([B,4H], the gate inputs), `dL/dh_prev`, `dL/dc_prev` (= dL/dc ⊙ f) |
| gru (time_step −1) | forward `x`, `h0`, `W_x`, `W_h`, `b`, `gates` ([B,T,3H], order z r h~), `h`; backward as lstm |
| gru (time_step t) | forward `x_t`, `h_prev`, `z`, `r`, `h~`, `h_t`; backward `dL/dh_out`, `dL/dh_next`, `dL/dh`, `dL/da` ([B,3H]), `dL/dh_direct` (= dL/dh ⊙ z, the direct path), `dL/dh_prev` |
| posenc | forward `x`, `P` (the rows used, `[T, D]`), `y`; scalars `offset`, `max_len`; `detail` = learned / sinusoidal; backward `dL/dy`, `dL/dP`, `dL/dx` |
| layernorm | forward `x`, `mean`, `rstd` (1/√(σ²+ε), per row), `x_hat`, `g`, `b`, `y`; scalar `epsilon`; backward `dL/dy`, `dL/dx_hat`, `dL/dg`, `dL/db`, `dL/dx` |
| attention | forward `x`, `W_qkv`, `q` `[B,T,E]`, `k`, `v` `[B,Tk,E]` (Tk = cached + new positions), `scores` `[B,H,T,Tk]` (q·k/√d; 0 where masked), `weights` (softmax; 0 where masked), `context` `[B,T,E]`, `W_o`, `y`; scalars `heads`, `head_dim`, `past`; backward `dL/dy`, `dL/dcontext`, `dL/dscores`, `dL/dq`, `dL/dk`, `dL/dv`, `dL/dx` |
| ffn | forward `x`, `W1`, `z`, `a`, `W2`, `y`; backward `dL/dy`, `dL/da`, then `dL/dz`, `dL/dx` |
| block | forward `x`, `attn_out` (after dropout: what is added), `h`, `ffn_out`, `y`; scalar `dropout`; backward `dL/dy`, `dL/dh`, `dL/dx`. Its parts emit their own events with `detail` = `ln1`/`attn`/`ln2`/`ffn` |
| tied_output | forward `x`, `W` (the embedding's matrix), `b`, `z`; scalar `embedding_layer`; backward `dL/dz`, `dL/dW` (the shared gradient so far), `dL/dx` |
| clip (update phase, layer −1) | scalars `norm` (before), `max_norm`, `scale`; `detail` = clipped / unchanged |
| dropout | forward `x`, `y`, `mask` (training only); backward `dL/dy`, `dL/dx`, `mask`; scalars `rate`, `training` |
| update, SGD (per param) | `value` (before), `grad`, `velocity_prev`, `delta` (= new velocity); scalars `lr`, `momentum`; `detail` = param name |
| update, Adam (per param) | `value` (before), `grad`, `m`, `v` (new moments), `m_hat`, `v_hat`, `delta`; scalars `lr`, `beta1`, `beta2`, `epsilon`, `t`; AdamW adds `weight_decay` (the λ applied to that parameter, 0 when excluded) |

The CLI's `trace`/`step` commands turn these into tables with formulas, e.g.
`z_j = Σ_i x_i·W_ij + b_j`. A future GUI consumes the same events (live via the
callback, or offline via the JSON lines).

---

## 5. Public API (summary)

All names use the `nmz_` / `NMZ_` prefix. The full header is
`include/neuromz.h`; this is the shape of it.

```c
/* errors */      nmz_status; nmz_status_str(); nmz_last_error();
/* tensors */     nmz_tensor_init/reshape/copy/zero/free
/* rng */         nmz_rng_seed/u32/below/uniform/normal
/* names */       nmz_act_name/nmz_act_parse, nmz_init_*, nmz_loss_*
/* build */       nmz_net_create(&net, rank, shape, seed); nmz_net_free(net);
                  nmz_layer_conv2d(F, kh, kw, stride, pad|NMZ_PAD_SAME, init)
                  nmz_layer_maxpool2d(size, stride); nmz_layer_flatten();
                  nmz_layer_dropout(rate); nmz_layer_conv2d_set_algo(l, algo)
                  nmz_layer_embedding(V, D, init); nmz_layer_rnn(H, init)
                  nmz_layer_lstm(H, init, forget_bias); nmz_layer_gru(H, init)
                  (NMZ_INIT_ORTHOGONAL: recurrent layers only; AUTO = it)
                  nmz_layer_posenc(LEARNED|SINUSOIDAL, max_len); nmz_layer_layernorm()
                  nmz_layer_attention(heads, head_dim); nmz_layer_ffn(F, act)
                  nmz_layer_block(heads, head_dim, F, act, dropout)
                  nmz_layer_tied_output(V); nmz_layer_has_attention
                  nmz_net_context_length; NMZ_ACT_GELU
                  nmz_net_set_normalization(net, NMZ_NORM_ZSCORE|MINMAX|NONE)
                  nmz_net_add(net, layer)             /* takes ownership      */
                  nmz_net_add_dense(net, units, act, init) /* Dense+Activation */
                  nmz_layer_dense(units, init); nmz_layer_activation(act, alpha)
                  nmz_net_set_loss(net, loss); nmz_net_build(net);
                  nmz_net_create_quick(&net, n_sizes, sizes, seed, &quick)
                  nmz_quick_defaults / nmz_quick_train_defaults (choices + reasons)
/* inspect */     nmz_net_layer_count/layer/input_size/output_size/param_count,
                  nmz_layer_type/describe/param_count/param_*, nmz_net_summary,
                  nmz_layer_activation_kind, nmz_layer_dense_init
/* compute */     nmz_net_forward(net, x, &y); nmz_net_predict(net, x, y);
                  nmz_net_backward(net, t, &loss); nmz_net_update(net, lr, mu);
                  nmz_net_preview_update(net, lr, mu) /* trace only, no change */
                  nmz_net_optimizer_step(net, &opts, lr)  /* SGD, Adam or AdamW */
                  nmz_net_optimizer_preview(net, &opts, lr)
                  nmz_net_forward_mode(net, x, training, &y)
                  nmz_net_compute_loss(net, t, &loss)
/* text */        nmz_utf8_decode/encode, nmz_vocab_*, nmz_text_normalize (flags
                  NMZ_TEXT_STRIP_DIACRITICS | NMZ_TEXT_UNIFY_ALEF),
                  nmz_dataset_load_text, nmz_net_set_vocab/vocab,
                  nmz_net_generate, nmz_net_generate_batch (row r = seed + r;
                  gen opts context, no_cache for Transformers)
/* train */       nmz_train_opts (max_epochs, batch_size, learning_rate, momentum,
                  shuffle, target_loss, optimizer, beta1/beta2/epsilon,
                  schedule/step_size/gamma/min_lr, weight_decay/decay_all,
                  warmup_steps, on_epoch/on_batch callbacks);
                  nmz_schedule_lr(&opts, epoch); nmz_schedule_lr_step(&opts,
                  batch, total, per_epoch); nmz_net_fit_normalization
                  nmz_net_train(net, ds, &opts, &result);
                  nmz_net_train_val(net, ds, val, &opts, &result) /* separate validation set */
                  nmz_net_evaluate(net, ds, &loss, &accuracy);
                  nmz_net_train_config(net)  /* defaults stored with the model */
/* data */        nmz_dataset_load_csv(&ds, path, &csv_opts); nmz_dataset_create;
                  nmz_dataset_load_idx(&ds, images, labels, limit);
                  nmz_dataset_truncate; nmz_dataset_free
/* io */          nmz_net_save(net, path); nmz_net_save_ex(net, path, flags);
                  nmz_net_load(&net, path)
/* trace */       nmz_net_set_trace; nmz_trace_recorder_*; nmz_trace_event_write_json
/* inspect */     nmz_net_activation_grad(net, i); nmz_layer_param_decays(l, i, all)
/* matmul */      nmz_matmul / nmz_matmul_naive(M, N, K, A, B, C, TRANS_A|TRANS_B|
                  ACCUMULATE); nmz_threads()
/* devices */     nmz_backend_select("auto"|"cpu"|"cuda"); nmz_backend_info(&info);
                  nmz_backend_query(name, &info); nmz_backend_count/name;
                  (neuromz_backend.h) nmz_backend_register, nmz_tensor_ptr
/* teaching */    nmz_net_gradcheck(net, x, t, eps, atol, rtol, &result);
                  nmz_net_gradcheck_mode(..., training, &result)
```

Ownership: `nmz_net_add` always takes ownership of the layer (it frees it on
failure too), so callers never leak a layer on an error path. Anything
returned through an out-parameter (`nmz_net **`, `nmz_dataset **`) is owned by
the caller and freed with the matching `*_free`.

---

## 6. Error handling

* Every fallible function returns `nmz_status` (`NMZ_OK == 0`, errors < 0):
  `NMZ_ERR_NOMEM, ARG, SHAPE, STATE, IO, FORMAT, UNSUPPORTED, NUMERIC`.
* Constructors return objects through out-parameters so they can also return
  a status. Layer factories (`nmz_layer_dense`, …) return `NULL` on failure.
* A human-readable message for the most recent failure on the current thread
  is available from `nmz_last_error()` (thread-local where C11 threads are
  available). The CLI prints it; the library never prints.
* No `exit()`, `abort()` or `assert()` in the library. Arguments are
  validated at the API boundary (`NULL` net, wrong sizes, layers added after
  build, …) and turn into `NMZ_ERR_ARG`/`NMZ_ERR_STATE`.
* Training watches for NaN/Inf loss and stops with `NMZ_ERR_NUMERIC` instead
  of silently producing garbage (typical cause: learning rate too high).

---

## 7. File format (`.nmz`, format version 3; version 2 is still read)

All integers are fixed-width **little-endian**; floats are IEEE-754 binary32
(`f32`) / binary64 (`f64`) stored little-endian. Strings are `u32 length` +
bytes (no terminator, max 4096). The writer never dumps C structs, so padding,
`sizeof(long)` and host endianness do not matter.

```
offset  field
0       magic        8 bytes: 89 4E 4D 5A 0D 0A 1A 0A  ("\x89NMZ\r\n\x1a\n")
8       u32          format version = 3 (2 is accepted)
12      u32          flags = 0 (reserved)
        u32          input rank R, then R × u32 input dims (per sample)
        u32          loss kind (0 = mse, 1 = cross_entropy)
        u64, u64     PRNG state, PRNG increment (resume reproducibly)
        u64          step counter
        train config u32 max_epochs, u32 batch_size, f32 learning_rate,
                     f32 momentum, u32 shuffle, f64 target_loss
   [v3] train ext.   u32 optimizer, f32 beta1, f32 beta2, f32 epsilon,
                     u32 schedule, u32 step_size, f32 gamma, f32 min_lr
        u32          class-name count C, then C × string
        u32          layer count N, then per layer:
                       string   type ("dense", "activation", "conv2d",
                                "maxpool2d", "flatten", "dropout",
                                "embedding", "rnn", "lstm", "gru",
                                "posenc", "layernorm", "attention",
                                "ffn", "block", "tied_output")
                       u32      config byte length, then config bytes
                       u32      parameter count P, then per parameter:
                                string name, u32 rank, rank × u32 dims,
                                product(dims) × f32
   [v3] u32          section count S, then per section:
                       string tag, u32 byte length, bytes
                     "normalize": u32 kind, u32 fitted, u32 groups G,
                                  G × f32 offset, G × f32 scale
                     "optimizer": u32 kind, u64 t, u32 n_params, then per
                                  parameter u32 count, count × f32 (velocity
                                  or Adam m), and for Adam count × f32 v
                     "vocab":     u32 size V, u32 items, then V × (u32 code
                                  point, u64 count), code points ascending
                     "train_ext": f32 val_fraction, f32 clip_norm,
                                  u32 carry_state, u32 seq_len, u32 patience
                     "text_norm": u32 NMZ_TEXT_* flags (written only when
                                  non-zero, after "vocab")
                     "train_ext2": f32 weight_decay, u32 decay_all,
                                  u32 warmup_steps (written only when one
                                  is non-zero, so older models stay
                                  byte-identical)
end     u32          CRC-32 (IEEE, reflected 0xEDB88320) of every byte above
```

The magic follows PNG's trick: the high byte catches 7-bit transfers, the
CR-LF/LF pair catches newline translation, `0x1A` stops `type` on Windows.
The loader checks magic, version, every count against sane limits, every
parameter shape against what the layer expects after `setup`, every section
length, and the CRC. Unknown section tags are skipped, so later versions can
add sections without breaking this reader. A version-2 file loads with SGD,
a constant schedule, no normalization and no optimizer state
(`tests/data/v2_xor.nmz` is a real phase-1 file used as a fixture).
Phase 5 added values, not structure: optimizer 2 = AdamW (its section
stores m and v like Adam), schedule 3 = warm-up + cosine, the layer types
above and the `train_ext2` section. `tests/data/v3_phase3_names_ar.nmz` and
`tests/data/v3_phase4_names_en.nmz` (written by the last phase-3 and
phase-4 commits) must load and generate the same text.

---

## 8. CLI

(The program lives in [neuromz-cli](https://github.com/mzprog/neuromz-cli)
since v2.0.0; this section stays here because the decisions below refer to
it.) `neuromz` with no arguments starts an interactive shell (prompt only when
stdin is a terminal, so scripts can be piped in). `neuromz CMD ARGS...` runs
one command; `-m FILE` loads a model first and `-o FILE` saves it after, so
one-shot use composes in shell scripts.

Layer spec grammar (one argument per layer after the input size):
`SIZE[:ACTIVATION[:INIT]]`, activation `leaky_relu=ALPHA` allowed.
`new 2 4 1` is quick mode; `new 4 8:relu:he 3:softmax --loss ce` is advanced
mode — the same command, quick mode is just "everything defaulted", and every
defaulted choice is printed with a one-line reason.

Image layers use the same style: the first field names the layer, the rest
are activation/init names or `key=value` options:
`conv:FILTERS:K[:ACT][:INIT][:stride=S][:pad=P|same]`, `pool:K[:stride=S]`,
`flatten`, `dropout:RATE`, and `dense:N` as a long form of `N`. The input may
be an image shape (`1x28x28`). Presets (`new --preset mnist-cnn|mnist-mlp`)
are named lists of such specs plus training options, each with its reason.
Datasets are named by file (CSV, IDX) or as `mnist` (`mnist:train`,
`mnist:test`); `--sample N` picks one sample for `predict`/`trace`/`step`/
`gradcheck`/`show sample`.

Text models use the input `text` (with `--data FILE` to build the
vocabulary at creation time, so layer sizes are known) and the specs
`embed:D`, `rnn:H`, `lstm:H[:forget_bias=F]`, `gru:H` (inits auto =
orthogonal, xavier, he) and `vocab[:ACT]` (a dense output with one unit per
character); recurrent layers stack, with `dropout:R` between them.
`--strip-diacritics` / `--unify-alef` set the Arabic normalisation of the
vocabulary. `generate` samples from them; `trace --text TEXT [--layer N]`
draws a recurrent layer unrolled through time with every gate;
`show gates` summarises the gates over a text; `compare A.nmz B.nmz …`
puts the gradient survival of several trained models side by side;
`train --val-file FILE` validates on a separate set.

Transformers use the same style: `pos:learned:CTX` / `pos:sin:CTX`,
`block[:heads=H][:head_dim=K][:ff=F][:gelu|relu][:dropout=R]`, the parts
`attn[:heads=H][:head_dim=K]`, `ffn[:F][:gelu|relu]`, `ln`, and
`vocab:tied`. Presets `names-gpt`, `char-gpt`, `arabic-gpt`. `trace --text
T [--pos P] [--layer N] [--head H]` shows attention at one position with
real numbers; `show attention` a heatmap and the head patterns over any
text; `show params` the parameters per component; `generate --context N
--no-cache`; `compare` accepts Transformers.

Remote mode (2.2, decisions 161–165; protocol 2 since program 2.3,
decisions 166–168): profiles in `client.conf` (`--profile NAME`, `--host`,
`--token`); the same commands against a
[neuromz-server](https://github.com/mzprog/neuromz-server) over the
[NeuroMZ protocol](https://github.com/mzprog/neuromz-protocol).

Devices (2.1, decisions 154–157): `--device auto|cpu|cuda` for one command
or the shell, `set --device D` in the shell, `train … --device D` for one
run, `show device`; `new` and `train` print the device with the other
settings, tagged `[auto]` (with the reason) or `[yours]`.

---

## 9. What v2 fixes from v1

| v1 problem | v2 answer |
|---|---|
| multi-output backprop updated only column 0 | per-layer `δ` tensors, matrix-form `dW = xᵀδ`; test `xor_and` |
| weights updated before δ propagated | backward computes every gradient, update is a separate pass |
| global `neuromz` state | `nmz_net*` handle, no globals |
| softsign/softplus/gaussian derivatives wrong | numerical gradient check for every activation and loss |
| biased init, no `srand` | Xavier/He uniform from a seeded PCG32 |
| no epoch limit, stop on last sample | full-dataset loss per epoch + `max_epochs` |
| raw struct dump file | portable little-endian format with magic, version, CRC |
| heap overflows, use-after-free, NULL derefs | tests (and, in neuromz-cli, the CLI smoke test) run under ASan/UBSan (`--target sanitize`, and in CI) |
| fixed 32-byte token buffer | dynamic line reader/tokenizer |

---

## 10. Decisions

Each non-obvious choice and why.

1. **float storage, double accumulation.** Tensors are `float` (what real
   engines and future GPU/embedded work use), but dot products and loss sums
   accumulate in `double`. This costs little and makes the numerical gradient
   check reliable at float precision.
2. **`W` is stored `[inputs, outputs]` and `z = x·W + b`.** With batch-first
   rows this is one row-vector product per sample, and `W_ij` reads "weight
   from input *i* to neuron *j*" — the textbook notation and the v1 notation.
3. **Activation is its own layer.** It keeps Dense simple, lets any layer be
   followed by any activation, and makes `z` and `a` separate, inspectable
   values. `nmz_net_add_dense(net, units, act, init)` adds the pair in one
   call, and the CLI shows the pair as one "layer" to beginners.
4. **MSE is defined as L = (1/B) Σ_rows ½ Σ_k (y_k − t_k)².** The ½ makes
   ∂L/∂y = (y − t)/B, the textbook δ, matching what a student derives by hand.
   (It is half of the "mean of squared errors" when there is one output.)
5. **Cross-entropy is only offered fused with softmax.** The fused gradient
   ∂L/∂z = (p − t)/B is exact and numerically stable; a separate CE layer
   would divide by tiny probabilities. Choosing CE without a softmax output is
   rejected at build time with a clear message. A standalone softmax backward
   (Jacobian-vector product) still exists for softmax+MSE.
6. **Top level is a sequential stack**, with composite layers for residual
   wiring (section 3). A general graph engine would be overkill for the
   planned models and much harder to trace for a student.
7. **Layers add into gradients; the engine zeroes them** (section 3).
8. **Parameters and gradients are exposed as pairs** (`nmz_param`).
9. **Token ids are stored as floats** (one dtype keeps the engine and trace
   simple; exact up to 2^24).
10. **PCG32** (O'Neill) as PRNG: tiny, fast, statistically good, identical
    output on every platform, unlike `rand()`. Bounded integers use Lemire's
    rejection method (no modulo bias); normals use Box–Muller.
11. **Glorot (Xavier) uniform and He uniform.** Same variance as the normal
    variants (2/(fan_in+fan_out) and 2/fan_in) but bounded, so no outlier
    weights in small teaching nets. Biases start at 0 (symmetry is already
    broken by the weights).
12. **`NMZ_INIT_AUTO` is resolved at build time** by looking at the next layer:
    He before relu/leaky_relu, Xavier otherwise. That is why parameter
    initialisation is a separate `init_params` step run by `nmz_net_build`
    (called implicitly by the first forward/train/save).
13. **Quick-mode defaults**: hidden `tanh` (zero-centred, smooth, does not die
    like ReLU in 2–4 unit nets), Xavier init; one output → `sigmoid` + MSE;
    two or more outputs → `softmax` + cross-entropy (assumes one-hot classes;
    multi-label needs `:sigmoid --loss mse`, and quick mode says so); SGD with
    momentum 0.9; learning rate, batch size and epochs chosen by
    `nmz_quick_defaults` and printed with reasons. Defaults live in the
    library (`defaults.c`) so library users and the CLI get the same ones.
14. **The library never prints.** Progress goes through `on_epoch` and trace
    callbacks; errors through status + `nmz_last_error()`.
15. **Thread-local last-error buffer** rather than an error object parameter
    on every call: keeps the API small for beginners, still thread-safe.
16. **Classical momentum** `v ← μ·v − η·g; w ← w + v`. With μ = 0 it is plain
    SGD, so one code path serves both and the trace always shows `velocity`
    and `delta`.
17. **Optimizer velocity is not saved** in the model file. A loaded model is
    for inference or fresh fine-tuning; saving optimizer state is deferred
    with Adam.
18. **Training config and class names are saved with the model**, so
    `neuromz train -m iris.nmz data.csv` reuses what `new` chose, and
    `predict` can answer "setosa" instead of "class 0".
19. **Stop rule** = loss over the whole training set, computed after each
    epoch in evaluation mode, `≤ target_loss` → converged; otherwise stop at
    `max_epochs`. One extra forward pass per epoch is a fair price for a rule
    that means what it says (v1 looked only at the last sample).
20. **One PRNG per net** drives both initialisation and shuffling, and its
    state is saved, so "same seed + same commands" ⇒ identical results.
21. **Layer types are strings in the file**, not numeric ids, so future layers
    never collide over id numbers.
22. **CRC-32 footer**: cheap, catches truncation and corruption, which the v1
    loader silently accepted.
23. **Accuracy** is reported only when it means something: argmax match for
    softmax outputs, all-outputs-match at threshold 0.5 when every target is
    0 or 1, otherwise "n/a" (regression).
24. **`nmz_net_gradcheck` is public**, not just a test helper: "check my
    backprop numerically" is itself a lesson, and the CLI exposes it.
25. **v2 is developed on branch `v2`**; `master` still points at v1. Merging is
    the owner's call.
26. **CLI is one-shot + shell**, not only a shell: one-shot commands with
    `-m/-o` are scriptable and testable; the shell keeps v1's interactive feel.

Added during implementation:

27. **Default seed is 42**, not the clock: a beginner who repeats a lesson gets
    the same numbers as the README and the teacher.
28. **Quick-mode hyper-parameters were measured, not guessed.** For
    softmax+cross-entropy, lr 0.05 (the first guess) stalled at 66.7% on some
    Iris seeds, lr 0.02 converged on 30/30; the Iris training loss plateaus at
    ≈0.06–0.07, so the CE target is 0.07 (lower mostly memorises the data).
    MSE/sigmoid uses lr 0.5, target 0.001 (XOR: 20/20 seeds). See the README
    results table and `examples/benchmark.sh`.
29. **One step number per pass.** Forward events carry `step + 1` and the
    backward pass increments `step`, so all events of one forward/backward/
    update pass share a number (a GUI can group them).
30. **`nmz_net_preview_update`** emits the update events without touching
    weights or velocity, so `trace` can show "what the next step would do"
    without the CLI needing private access to optimizer state.
31. **CSV column split comes from the network** (`n_inputs`): the first
    `input_size` columns are inputs and the rest targets, so `train` needs no
    extra flags for multi-output data.
32. **Evaluation passes are not traced** (`nmz_net_evaluate` suspends the
    callback): the whole-set loss check after each epoch is bookkeeping, not
    part of the learning math a student follows.
33. **Saving is atomic**: the model is written to `FILE.tmp` and renamed, so a
    failed save never destroys the previous good file.

Added in phase 2 (CNN, Adam, normalization, MNIST):

34. **Normalization belongs to the net, not to a layer.** It has fitted (not
    learned) parameters; as a layer the optimizer, gradcheck and the
    parameter list would all need a "non-trainable" notion. As net state it
    is fitted once at the first `train` (then frozen, so test data is scaled
    with training statistics), saved as a section, applied by forward and
    predict, traced as a `normalize` event (layer −1), and the input
    gradient is reported w.r.t. the raw input.
35. **Normalization groups**: one per channel for `[C,H,W]` inputs (per-pixel
    statistics on MNIST would divide by zero on the always-black border and
    are not what CNNs use), one per feature otherwise (Iris-style columns).
36. **Optimizer state lives in the net, one pair of tensors per parameter**
    (velocity, or Adam's m and v) plus Adam's t. Switching optimizer resets
    it. It is saved by default (an optional section), so training resumes
    exactly; `save --no-optimizer` / `NMZ_SAVE_NO_OPTIMIZER` drops it.
    Supersedes decision 17.
37. **Adam trace values are the weight *before* the step** plus m, v, m̂, v̂
    and Δ, like the SGD events; the printer derives "w new = w + Δ". Traced
    updates compute everything into scratch tensors first, emit, then apply,
    so preview and apply show identical numbers.
38. **Schedules are per epoch** (constant, step, cosine), computed by
    `nmz_schedule_lr`, reported in `nmz_epoch_info.learning_rate`. Per-batch
    warm-up etc. are deferred: per-epoch values are easy to show and reason
    about.
39. **Each epoch's shuffle starts from the identity permutation**, so its order
    depends only on the PRNG state. Found by the resume test: 5 + 5 epochs
    (also across save/load) must equal 10 epochs, and did not while the
    permutation was carried from epoch to epoch.
40. **File format 3 = format 2 + extra header fields + optional sections.**
    New data goes into tagged, length-prefixed sections that unknown readers
    skip; the version number only changes when the fixed part changes.
41. **Conv shape convention**: per-sample `[C,H,W]`, batch `[B,C,H,W]`,
    weights `[F,C,KH,KW]`; it is cross-correlation (no kernel flip), as in
    every deep-learning library. Max-pool has no padding; ties go to the
    first maximum in row-major order; overlapping windows add gradients.
42. **Conv AUTO init is He** unless a tanh/sigmoid activation follows (then
    Glorot) — "He by default" for the usual conv+relu.
43. **Flatten is explicit in the library, automatic in the CLI.** Dense acts
    on the last dimension (rows convention), so Dense after `[C,H,W]` would
    silently mix only the W axis; the library leaves the choice to the
    caller, while the CLI inserts (and announces) a flatten before a dense
    layer that follows conv/pool.
44. **Dropout uses the net's PRNG through `nmz_ctx.rng`**, so masks are
    reproducible; it is "inverted" (scale 1/(1−p) in training, identity in
    evaluation). `trace` runs in evaluation mode and changes nothing; `step`
    runs a real training step (dropout active). Gradcheck replays the PRNG
    state before every forward, so dropout can be checked in training mode.
45. **Gradcheck skips kinks.** If a ±ε perturbation flips a relu input's sign
    or a max-pool winner, the loss is not differentiable between the two
    points and the difference quotient is meaningless; such checks are
    counted as `kinks`, not failures. Found when gradchecking a trained CNN
    from the CLI: one dense unit sat 0.006 from its kink while its input was
    6.9, so ε = 0.001 crossed it for 144 weights at once.
46. **im2col is the default conv implementation; the naive loops stay** as the
    readable reference (`nmz_layer_conv2d_set_algo`). Both pass the same
    gradchecks and agree to 1e-5. With `-O3` (now the default CFLAGS) im2col
    is 2.1–2.4× faster on the MNIST CNN; at `-O2` only ~1.1× because GCC 12's
    -O2 cost model does not vectorise the inner loops. The im2col forward sums
    in float (≤ C·KH·KW terms per output; vectorisable), the weight gradient
    uses 8 partial sums per dot product and a double batch accumulator.
47. **IDX pixels stay raw (0..255)** in the dataset; scaling is the model's
    normalization, so `predict` and `show sample` take raw images. Class
    counts come from all labels in the file, not from a `--limit` subset
    (a bug found when `show sample 1` saw only 8 classes).
48. **MNIST download**: two public mirrors (the original host now refuses
    automated downloads), MD5 sums from torchvision as the independent
    reference, data in `data/` (gitignored), `NEUROMZ_DATA` to relocate.
49. **Trace focus for feature maps**: one output position per spatial layer,
    by default where channel 1's output is largest ("where the filter
    fires"), remembered per layer so the backward pass shows the same
    position; `--at F,Y,X` (1-based) overrides. A centred default showed
    blank background patches on most digits.
50. **Presets are CLI data** (layer specs + options + one reason each), not a
    library API: they are a teaching/UI concern, and library users write the
    same network in five calls (see README).

Added in phase 3 (text, Embedding, RNN, generation):

51. **Tokens are Unicode code points, never bytes**, and the vocabulary is
    sorted by code point (deterministic ids). Invalid UTF-8 is never fatal:
    each invalid byte becomes U+FFFD and is counted and reported. One U+FFFD
    per byte (rather than per "maximal subpart") is simpler and still
    resynchronises at the next valid byte.
52. **Text normalisation is minimal**: drop a BOM, turn CR LF / CR into LF.
    No Unicode normalisation of the data — the model learns the text as it
    is written (the Arabic list was NFC-normalised when it was compiled).
53. **The end-of-item token is the newline itself.** In item mode (one item
    per line) empty lines are dropped and the text starts with '\n', so
    every item, including the first, follows an end token; generation of
    one item = prime '\n', sample until '\n'. No separate special token
    keeps the vocabulary = the characters of the file.
54. **Sequence datasets store ids, not one-hot targets**; one-hot rows are
    built per batch. The book as one-hot targets would need ~145k × 75 floats
    per copy; ids need 145k.
55. **Any-length time dimension (decision in §3)** instead of a fixed T in the
    model: the same weights train on 64-step windows and generate one step
    at a time.
56. **No time-distributed wrapper**: Dense's rows convention already is one.
57. **Optional vtable fields `state` and `flags` were appended** to
    `nmz_layer_ops` instead of a parallel "stateful layer" interface; the
    engine never needs to know which layers are recurrent.
58. **RNN init: Glorot for W_xh and W_hh** — W_hh entries then have variance
    1/H, a spectral radius near 1. Orthogonal init is better known for RNNs
    but adds a QR decomposition to the library for little gain at this size.
59. **Sparse embedding gradient, dense optimizer step** = exact Adam
    ("lazy" Adam would freeze rows absent from a batch even though their
    momentum is non-zero). Tested against a hand-computed Adam step for such
    a row. At character scale the dense step costs nothing.
60. **Truncated BPTT window = seq_len; carry arranges B parallel streams**
    (row r of batch k+1 continues row r of batch k), without shuffling, and
    the state is reset every epoch. Without carry, sequences start from
    h = 0 and are shuffled.
61. **A model trained with carried state is evaluated with carried state**
    (the split read in order as one stream). Found by measuring: on Alice,
    the stateless evaluation reported validation loss 1.671 after 20 epochs
    for the carry model vs 1.615 for the shuffled one; evaluated as it is
    run, the carry model reached 1.591 by epoch 16 — better, not worse.
62. **Validation = the last fraction of the samples** (`val_fraction`),
    generic for tabular, image and text data. For text, sequences don't
    overlap, so the tail of the sequences is the tail of the text.
    Normalisation is fitted on the training split only.
63. **Early stopping (`patience`) restores the best weights** and
    re-measures; the optimizer state is not rolled back (training after it
    continues from the restored weights with the latest moments). Found
    needed on the Arabic list (~2.7k characters), which overfits after
    ~7–15 epochs.
64. **Clipping by global L2 norm**, after the backward pass and before the
    optimizer, default 5 for recurrent nets (off otherwise), traced as a
    `clip` event so the CLI can show "‖g‖ = 7.3 > 5 → ×0.68".
65. **The vocabulary is built at `new`** (`--data FILE`), so embedding and
    output sizes are known when the layers are created; it is stored in a
    `vocab` section, and training text is encoded with the model's
    vocabulary (unknown characters are an error naming the code point).
    Text-training settings went into a `train_ext` section, keeping format 3.
66. **Generation has its own PCG32 stream** seeded by `--seed` (it never
    disturbs the training PRNG); temperature divides the logits before the
    softmax; top-k keeps ties at the k-th probability; `--count N` collects N
    distinct items by retrying with seed+attempt; `--data FILE` marks items
    already in the training data (`--new-only` skips them).
67. **The RNN trace shows one hidden unit in full arithmetic** (by default
    the one that changes most over the sequence) plus a heatmap of all
    units; and a *second* backward pass with only the last step's loss
    isolates the gradient that travels through time, so its per-step
    factor (vanishing < 1 < exploding) can be read off directly.
68. **`trace` runs in evaluation mode and changes nothing; `step` is a
    training step** (applies clipping and the update) — as for CNNs.
69. **Datasets**: English names from the US Census 1990 lists (public
    domain), Arabic names compiled by hand (CC0), Alice from Project
    Gutenberg through its mirrors with a SHA-256 check (www.gutenberg.org
    refuses automated downloads). Sources in `examples/DATASETS.md`.
70. **Arabic is printed in logical order as UTF-8**; shaping and right-to-
    left ordering are left to the terminal (many do it imperfectly). Where
    the trace puts one character per column the letters appear in their
    isolated forms, which is what a character model actually sees.

Added in phase 4 (LSTM, GRU, gate traces, gradient survival, Arabic prose):

71. **One matrix per gate group, gates side by side** (`W_x [D, 4H]`,
    `W_h [H, 4H]`, order i f g o; GRU `[·, 3H]`, order z r h~): one
    vector-matrix product per step computes every gate; the trace slices the
    columns back into named gates. Four separate matrices would read more
    like the textbook but quadruple the parameter list for no gain.
72. **Forget-gate bias starts at 1** (Gers et al. 2000; Jozefowicz et al.
    2015): with f ≈ σ(1) = 0.73 a new cell keeps most of its state and the
    gradient along c survives from the first epoch. Configurable
    (`lstm:H:forget_bias=F`, `nmz_layer_lstm(H, init, F)`), saved in the
    layer config; `new` explains the resulting σ(F).
73. **GRU variant: the original (Cho et al. 2014)**, reset gate applied to
    h_{t−1} *before* the product with W_h, one bias per gate, and
    h_t = z⊙h_{t−1} + (1−z)⊙h~ (z = 1 keeps the state, as in the paper and
    PyTorch). PyTorch/cuDNN apply r after the product (r⊙(h·W + b_h), two
    biases) because it fuses into one matrix product per step; that is a
    speed choice, and the original reads more directly: "r decides how much
    of the past the candidate may see". The trace and the hand-computed
    test spell the formula out.
74. **Orthogonal initialisation is the default for recurrent matrices**
    (rnn, lstm, gru: AUTO = each H×H gate block of W_h orthogonal, W_x
    Glorot). It is a Gaussian matrix orthonormalised by modified
    Gram–Schmidt (two passes, in double) — the Q of a QR decomposition, so
    no linear-algebra library is needed (decision 58 had deferred it for
    that reason). All singular values are 1, so repeated multiplication by
    W_h neither shrinks nor blows up h; Glorot W_hh has spectral radius ≈ 1
    but singular values spread from ≈ 0 to ≈ 2. `xavier` keeps the phase-3
    initialisation; dense, conv and embedding layers reject `orthogonal`
    (their matrices are not applied repeatedly). Tested: WᵀW = I within 1e-7.
    Compatibility is one-way: every phase-3 file loads, but a phase-3 build
    rejects a new RNN whose init field says `orthogonal` ("corrupt rnn
    config") — as it rejects any layer type it does not know.
75. **The engine did not change** (DESIGN §3.1). The only edits outside
    the new layer files: `NMZ_TRACE_MAX_ITEMS` 8 → 12, the layer registry
    lines, helpers moved into recurrent.c for sharing, and batch generation
    (decision 80), which needed the "one state row per stream" contract to
    be written down.
76. **The LSTM's carried state is one `[B, 2H]` tensor, h | c.** The
    engine and gradcheck treat it as opaque. The gradient through c_{t−1}
    is tested directly: the trace's `dL/dc_prev` at t = 0 (the gradient
    that truncation drops) is compared with central differences of the loss
    in the carried c — the cell-state path checked numerically.
77. **Trace names**: the LSTM/GRU per-step events keep the RNN's `dL/dh`,
    `dL/dh_prev`, `dL/dh_out`, `dL/dh_next`, so the BPTT table works
    unchanged, and add `dL/dc`, `dL/dc_next`, `dL/dc_prev` (LSTM) and
    `dL/dh_direct` (GRU). The gradient of the gate inputs is `dL/da`, not
    `dL/dz`, because z is the GRU's update gate.
78. **Gradient survival is measured with a separate backward pass where
    only the last step has a loss** (as in phase 3), reported relative to
    the last step, so models whose gradients have different overall sizes
    compare directly. For an LSTM it is reported along h and along c, plus
    the mean over cells of Π f over the later steps — what the cell path
    alone would pass back (Π z for the GRU's direct path). `compare` loads
    up to six models, encodes the same text with each model's vocabulary
    and normalisation, and shows a table, log-scale sparklines and a
    summary (per-step factor, what is left after 10 and all steps, steps
    until below 1 %).
79. **`show gates` summarises gates over a text with the state carried**
    (evaluation mode, 256-character chunks of one stream) through a trace
    callback that accumulates instead of recording, so 20 000 characters
    cost one forward pass and no copies. The "half-life" ln 0.5 / ln f̄ is an
    intuition, not a measurement: how long a cell with that mean forget gate
    would keep half its content if nothing were written.
80. **Batch generation: row r uses seed + r** and is bit-identical to a
    single run with that seed (every layer computes rows independently), so
    `generate --count N` gives exactly the phase-3 output. Rows that finish
    leave the batch: their row of every layer's carried state is removed.
    Found by measuring: without that, a batch keeps stepping until its
    longest item ends, and 500 names took 4× longer than one at a time (and
    the CLI asked for 1.5× the rows it needed, to allow for duplicates;
    now it asks for exactly the missing number, round after round).
    Measured now: 500 names 37–42 ms vs 38–43 ms one at a time, 50 × 300
    characters of Alice 218–235 ms vs 226–228 ms — the same. Batching turns
    15 000 forward calls into 300, but on one thread the arithmetic per row
    is unchanged; reusing each weight row for all rows of the batch (loop
    order) is what would make it faster, and is deferred.
81. **Arabic normalisation is a vocabulary option**, off by default:
    `NMZ_TEXT_STRIP_DIACRITICS` (harakat U+064B–U+0652, superscript alef
    U+0670, tatweel U+0640) and `NMZ_TEXT_UNIFY_ALEF` (أ إ آ ٱ → ا). The
    flags live in `nmz_vocab`, so they are applied wherever the model reads
    text (training data, `--val-file`, prime, trace, compare), and are saved
    as a `text_norm` section written only when non-zero, so models without
    it are byte-identical to phase 3. Off by default because it changes
    the task: the model no longer predicts the marks, and unified alef
    loses the hamza spelling.
82. **The Arabic prose comes from Wikisource at fixed page revisions**
    (`oldid`): a revision never changes, so its SHA-256 is stable even when
    the page is edited later, and no copy of the text has to be hosted.
    The markup is removed by a 25-line awk script run with `LC_ALL=C`
    (byte-wise, the same in gawk, mawk and busybox). Rejected sources:
    Gutenberg (almost no Arabic), the Tanzil Qur'an text (no modification
    allowed, and imitation scripture is inappropriate), OpenITI
    (non-commercial licence).
83. **`--val-file` = `nmz_net_train_val`**: validation on a separate data
    set; then all of the training file trains and `val_fraction` is ignored.
    It is a per-run option, not stored in the model.
84. **Comparisons use the same parameter budget**: rnn:128 ≈ lstm:64 ≈
    gru:75 (within ±3 % for every vocabulary used here). An LSTM cell has
    four gates, so "the same number of units" would give it 4× the
    weights; the presets `names-lstm` and `char-lstm` are the matched
    versions of `names` and `char-rnn`.
85. **The phase-3 regression tests pin the RNN to `xavier`**, so their
    exact strings still check the phase-3 model, and a model file written by
    the last phase-3 commit (b0047fc) is a fixture that must load and
    generate the same names. Found when the new orthogonal default changed
    one test's generated item: it came from a context (zero state + '\n')
    that never occurs in that test's training text — the old pass was luck.
86. **Learning rates were swept per architecture, not shared.** Gated
    layers take a larger step: on English names LSTM 0.005 → 2.038 and
    0.01 → 2.028, GRU 2.061 → 2.056; on Alice LSTM 0.004/0.006/0.01 →
    1.648/1.639/1.621 after 20 epochs. The RNN is best at its phase-3
    values (names 0.003 → 2.118 vs 0.006 → 2.108, within noise; Alice
    0.004 → 1.638 vs 0.008 → 1.667). Hence `names-lstm`/`char-lstm` use
    0.01, and `char-lstm` allows 30 epochs (the LSTM was still improving
    at 20). `arabic-text` (two stacked LSTMs of 96) uses 0.005, chosen
    from these results rather than measured on its own.
87. **Dropout between stacked recurrent layers is the ordinary Dropout
    layer**: an independent mask for every value of [B, T, H], applied to
    the outputs that go up the stack, never to the recurrent connection.
    "Variational" dropout (one mask per sequence, also on h_{t−1}) needs a
    hook inside the recurrent layer; deferred.
88. **`compare` reports the per-step factor over all steps *and* from k ≥ 16
    on.** Found by looking at the first trained models: one average hid the
    lesson. On Alice the LSTM's cell-state gradient shrinks ×0.859 per step
    on average — barely better than the RNN's ×0.803 — because the trained
    LSTM concentrates on the last few characters; from 16 steps back on it
    is ×0.942 against ×0.820, and the count of cells with Π f ≥ 0.5 over
    the whole window (12 of 64) shows where that comes from.
89. **Four partial sums in the recurrent backward dot products**
    (`nmz__dot`). The input and h_{t−1} gradients are row dot products
    accumulated in double; with one accumulator every addition waits for
    the previous one. Four independent sums (still double, so decision 1
    holds) cut the LSTM backward by about a third (gprof, Alice).
90. **The results are reported as measured, including the unflattering
    ones**: the gated layers win on all four texts at equal size, by
    0.2 nats on the 2.7 k-character Arabic names but only 0.01–0.06 on the
    books; orthogonal initialisation made no measurable difference for the
    RNN (names 2.118 vs 2.112 with Glorot, Alice 1.638 vs 1.634); the GRU
    hardly uses its direct path on Alice yet is the best model there.
    Next-character prediction depends mostly on the last few characters;
    the long-range gradient the cell state preserves is visible in
    `compare` but worth little loss at this scale.

Added in phase 5 (a decoder-only Transformer, AdamW, warm-up, the KV cache,
attention traces):

91. **Every Transformer building block is an ordinary layer, and the block
    is a composite of them** (§3.2). The alternative, one monolithic
    "transformer" layer, would hide LayerNorm, attention and the
    feed-forward net from the gradient check and from the CLI. The engine
    changes are listed in §3.2; each is an optional field, a flag or an
    accessor that old layers and old files never see.
92. **A blocked matrix product that is bit-identical to the naive loops.**
    `nmz_matmul` keeps the sum order of `c_ij = Σ_k a_ik·b_kj` (increasing
    k, one float addition at a time) and only changes which c_ij is worked
    on when: blocks of B stay in the cache, and a 1 × 32 tile of C stays in
    registers across a block of k (32 sums = 8 SSE registers; GCC
    vectorises it at -O3 without -ffast-math). Transposed operands are
    copied once into row order. Measured on one core of an i7-7500U
    (`bench_matmul`, the shapes of a D = 64 model): 15–19 GFLOP/s against
    1.7–1.9 for the naive loops (7–10×; 35× for the weight gradient, where
    the naive loop reads A by columns). A 4 × 8 tile came out at 4–6
    GFLOP/s (GCC kept it on the stack), the plain row loop at 11–12. Bit
    equality is what makes the KV cache test exact (decision 98): a row of
    C depends only on that row of A, never on how many rows are computed
    together. The sums are in float, as in the im2col convolution
    (decision 46); softmax, LayerNorm statistics and the attention scores'
    dot products stay in double (decision 1).
93. **Pre-LayerNorm blocks, `h = x + attn(LN(x))`, not post-LN
    `h = LN(x + attn(x))`** (the original Transformer). With pre-LN the
    residual path from the output back to the embedding is the identity:
    `∂L/∂x = ∂L/∂h + (the sub-layer's contribution)` at every block, so
    the gradient reaches the first block unchanged whatever the sub-layers
    do. Post-LN puts a normalisation on that path at every block; its
    gradients near the output are larger at initialisation and it needs a
    careful warm-up to train at all (Xiong et al. 2020). GPT-2 and most
    later models use pre-LN; the price is a final LayerNorm before the
    output, because the last block's output is never normalised.
94. **AdamW with decoupled weight decay, on weight matrices only by
    default.** `Δ = −η·(m̂/(√v̂ + ε) + λ·w)`: the decay is not divided by
    √v̂, so every weight shrinks at the same relative rate (Loshchilov &
    Hutter 2019); L2 regularisation inside Adam's gradient would decay
    weights with large gradients less. Excluded: vectors (biases,
    LayerNorm gains and biases: decaying a gain towards 0 fights the
    normalisation) and lookup tables (embedding rows and position rows are
    selected, not multiplied; decay would pull rarely seen characters
    towards 0 between their updates). The rule is generic — "rank ≥ 2 and
    the layer is not discrete-input or NO_DECAY" — so no layer has to list
    its parameters. `--decay-all` decays everything; λ defaults to 0.1 for
    Transformers. Tested against a hand-computed step, with an unused
    embedding row staying put.
95. **Warm-up + cosine per batch** (decision 38 had deferred it). Batch s of
    S: `η·s/W` during the W warm-up batches, then a cosine from η towards
    η_min; the last batch uses a rate one step above η_min, so no update is
    wasted at 0. W defaults to 5% of all updates (at least 10), computed by
    `new` from the text's length, the batch size and the epochs. Early
    Adam steps are noisy (v̂ rests on a few gradients), and in a fresh
    Transformer large early steps can saturate the attention softmax; a
    short ramp avoids both. The epoch schedules are unchanged.
96. **GELU is the exact x·Φ(x) with `erf`**, not GPT-2's tanh
    approximation: one libm call either way, and the formula a student
    looks up. It is also an ordinary activation (`dense:64:gelu`).
97. **Attention: q, k, v side by side in one W_qkv** (as the LSTM gates,
    decision 71), biases on all projections, Glorot initialisation (each of
    q, k, v is a D → E projection), scores scaled by 1/√d. The causal mask
    is the loop bound `j ≤ t`, not an added −∞ matrix: masked weights are
    exactly 0 and no gradient reaches the future (tested bit-exactly: the
    output at t does not change when later inputs do). The weights
    `[B, H, T, T]` are kept for backward and the trace: memory grows with
    T² (16 × 4 × 128 × 128 floats = 4 MB for the presets).
98. **The KV cache is the carried state of phase 3.** An attention layer's
    state is `[B, Tc, 2E]` (keys | values of every position seen), the
    position encoding's state is its offset; `nmz_net_reset_state`,
    gradcheck's snapshot and the batch generator's row compaction work on
    them unchanged. One new position per step costs one row of every layer
    instead of the whole window. Because every query is computed by the same
    code over the same keys, and the matrix product of a row does not
    depend on the other rows (decision 92), cached generation is
    bit-identical to recomputing the window at every step (`no_cache`,
    tested with slides, small windows, long primes, items and batches).
    Positions are absolute, so a cached key cannot be shifted when the text
    outgrows the window: the window then slides by half (the newest
    context/2 characters are re-encoded from position 0, one full pass),
    and single steps continue — a sliding-by-one window would need a full
    pass at every step. Training never carries a cache: an attention layer
    refuses carry in training mode, and `new` refuses `--carry`.
99. **Positions: learned (default) or sinusoidal.** Learned rows start at
    N(0, 0.02²), small next to the character vectors (Glorot, ≈ 0.1), as in
    GPT-2: position is a hint the model strengthens as it needs. The
    sinusoidal table is the original Transformer's, fixed, with no
    parameters. `max_len` is the context window (`nmz_net_context_length`);
    longer inputs are an error, and generation slides (decision 98).
100. **Weight tying through `bind`, one parameter for two uses.** `vocab:tied`
    replaces the V × D output matrix by the embedding's transpose: 2 400 of
    char-gpt's 32 043 weights would otherwise be a second table, and a
    character's input vector and output direction are learned together
    (Press & Wolf 2017). The shared matrix counts once, is saved once and
    follows the embedding's AdamW rule (not decayed by default). Tested:
    the output side adds into the embedding gradient (a row for a token
    absent from the batch gets a gradient), save/load keeps the tie, and
    the whole tied model passes the gradient check.
101. **Head patterns are inferred from the measured weights, never from a
    label table.** For every head and query (positions ≥ 2) the weight
    falling on eight kinds of earlier position is summed — itself, the
    previous character, two back, earlier copies of the same character,
    the start of the current word, the last space or punctuation, the
    first position, more than 16 back — and compared with what uniform
    attention over the visible positions would give. The label is the kind
    with the largest excess (≥ 10 points; otherwise "spread out"), shown
    with its share, the uniform share, the runner-up, the span e^entropy
    (how many positions the head effectively looks at) and the mean top
    weight. Categories overlap (the previous character can be the word
    start); the excess over uniform keeps short windows from flattering
    "the first position".
102. **The attention trace answers "which characters, and why" at one
    position**: q, every k, the raw scores, the mask, the weights as bars
    labelled with the characters, the three largest terms of q·k for the
    winner, the weighted sum and its three largest contributions. Default
    position: the last; default head: the one with the largest single
    weight there (the most decisive). The residual stream table gives the
    length of the vector before and after each block and of what attention
    and the feed-forward net added. The gradient-path view and `compare`
    use |∂L/∂x_t| at the embedding output from a backward pass with only
    the last position's loss — the same measurement as for the recurrent
    models, which for a Transformer follows the attention weights, not the
    distance.
103. **Arabic on heatmap axes: one column per character.** Terminals that
    apply the Unicode bidirectional algorithm reverse runs of Arabic
    letters and join them, which would scramble an axis. Each Arabic
    letter is followed by ZWNJ (no joining) and LRM (left-to-right), each
    line starts with LRM, combining marks get a dotted circle — all
    zero-width, so the columns stay aligned in terminals with and without
    bidi support. `--ascii` prints `.` for every non-ASCII character (and
    `_`/`~` for space/newline); the row labels keep the position numbers.
104. **Transformer training defaults**: no carried state (each window is
    complete), shuffled windows, sequence length = the context window
    (every position of a window is one prediction), clipping at 1 (Adam
    with a warm-up rarely needs it; it guards the first steps), AdamW
    λ = 0.1, warm-up + cosine to a tenth of the peak rate.
105. **OpenMP is optional and changes no result.** `make OMP=1` splits the
    row blocks of every matrix product and the (row, head) pairs of
    attention among threads; each number is still computed by one thread
    in the same order, so the OpenMP build gives bit-identical results
    (all tests pass in both builds, including the KV-cache equality).
    Small products stay single-threaded (an `if` clause), so generation
    does not pay thread start-up per character. Measured on the 2-core /
    4-thread i7-7500U: matrix products 1.1–2× faster with 2 threads, a
    training epoch of a 113 k-parameter model 16.2 s → 11.2 s; 4 threads
    (hyper-threads) were slower than 2 (14.6 s).
106. **The Transformer presets match the phase-4 parameter budgets, and
    their rates were swept.** `names-gpt` (24 085 weights) sits next to
    `names-lstm` (25 029), `char-gpt` (32 043) next to `char-lstm`
    (32 107): two blocks of 32 with 4 heads, tied output. The learning rate
    was the decisive knob: on Alice the peak rate 0.001 / 0.002 / 0.003 /
    0.005 / 0.01 / 0.02 gave validation loss 2.30 / 2.02 / 1.82 / 1.72 /
    1.67 / 1.66 after 30 epochs (batch 16) — tiny Transformers want
    large steps, ten times Adam's usual 0.001. Batch 8 (twice the updates
    per epoch) gave 1.633, sinusoidal positions 1.637, dropout 0.1 1.70
    (worse: at this size the model underfits). On the English names 0.01
    with batch 16 was best (2.109; 0.003 2.166–2.176), and dropout 0.1
    helped there (2.097), as that list is small enough to overfit within
    20 epochs. Hence names-gpt: 0.01, batch 16, dropout 0.1; char-gpt: 0.01,
    batch 8, no dropout.
107. **A file written by the last phase-4 commit (348bcaf) is a fixture**:
    an embed → lstm → dropout → gru → dense names model with Adam state. It
    must load, keep its optimizer state and generate exactly the strings
    the phase-4 library generated (recorded with the phase-4 build). All
    phase-5 additions to the format are values and an optional section, so
    the older files are read unchanged; files that use AdamW, the new
    schedule or the new layers are not readable by a phase-4 build (as in
    decision 74).
108. **The results are reported as measured** (README "Results"): at equal
    size the Transformer does not beat the recurrent models on these
    texts — Alice 1.633 against the LSTM's 1.614 and the GRU's 1.580,
    Kalila 1.940 against 1.951 and 1.938, the English names 2.097 against
    2.028 / 2.056 — and it is clearly worse on the 504 Arabic names (2.77
    against 2.46), where it overfits after nine epochs. It also trains
    1.2–4× more slowly per run. With more capacity and dropout it wins on
    Alice (213 k weights, 1.564); on Kalila the 162 k arabic-gpt (1.883)
    stays slightly behind phase 4's two stacked LSTMs (130 k, 1.863). The
    gradient path is the clear difference: 63 characters back the
    Transformer receives 0.86% (Alice) / 0.29% (Kalila) of the last
    prediction's gradient, against 7.1e-5 / 4.9e-7 along an LSTM's cell
    state and 4.6e-7 / 1.8e-5 for the GRU. At character level and this
    scale the next character depends mostly on the last few, so that reach
    buys little loss — the same conclusion as decision 90, one step
    further.
109. **The heads found are reported, not hoped for.** Every trained model's
    first layer turned into local n-gram detectors (previous character,
    two back, itself, the start of the word), the clearest a head with
    50–64% of its weight on the previous character; the deeper layers'
    heads spread over 14–22 positions. A word-start head appeared in the
    Arabic model's second layer (where Arabic attaches its prefixes). No
    induction ("what came next last time") head formed at these sizes —
    the pattern is measured for every head and stayed within 2 points of
    uniform.
110. **The benchmark is deterministic and was run once, sequentially**, so
    the times are uncontended and the losses reproducible to every digit
    (the LSTM and GRU reproduced phase 4 exactly). The exploration runs
    behind the presets (learning rates, batch, dropout, positions, larger
    configurations) were run two at a time and are reported in decisions
    106 and README, not as benchmark times.

### Phase 6: first public release (v2.0.0)

111. **A model remembers what it was trained on.** `train FILE` stores the
    absolute path of FILE and, for line-per-item text (names), the set of
    training items as sorted 64-bit FNV-1a hashes (`nmz_net_set_data_source`,
    a new optional `data_source` section of the file). `generate --new-only`
    and the "(in the training data)" note then work without `--data`, and
    `show gates` / `show attention` without a text use a 2000-character
    excerpt of the file. Hashes rather than the items keep a model file
    small (8 bytes per name) and do not copy the data into it; a false
    "known" needs a 64-bit collision (about 10⁻¹¹ for 20 000 names). Items
    are compared after the model's own text normalisation and with the
    `--prime` included (before, a primed name was compared without its
    prime and never matched). `--data FILE` still overrides. Older readers
    skip the section (decision 13), and older files load with no source.
112. **`--optimizer adamw` chosen by hand gets Adam's learning rate (0.001)
    and weight decay 0.01** (PyTorch's default) instead of SGD's 0.5 and no
    decay; the Transformer defaults (0.1, decision 94) are unchanged.
113. **The operating system is reached from two places only.**
    `cli/platform.c` holds everything the program needs: terminal detection
    (`isatty` / `_isatty`), terminal width, UTF-8 console output and UTF-8
    `argv` on Windows (`SetConsoleOutputCP(CP_UTF8)`, `CommandLineToArgvW`),
    file names (`_wfopen`), absolute paths (`realpath` / `_wfullpath`) and a
    monotonic clock (`clock_gettime` / `QueryPerformanceCounter`). The
    library stays plain C11 except for file names: on Windows `fopen` reads
    a `char *` name in the ANSI code page, so `nmz__fopen`, `nmz__rename`
    and `nmz__remove` convert the UTF-8 name to UTF-16 with the library's
    own decoder and call `_wfopen` / `_wrename` / `_wremove` — no
    `<windows.h>` in the library. Names are UTF-8 on every system.
114. **CMake 3.21 replaces the Makefile.** 3.21 is the first version that
    passes `/std:c11` to MSVC; everything else (presets, FetchContent,
    `PROJECT_IS_TOP_LEVEL`) is older. The old `make` targets are CMake
    targets: `test` (CTest; a `build` fixture brings the tree up to date
    first, so `--target test` never runs stale binaries), `sanitize` (a
    second tree `build-san/` with ASan + UBSan, then its tests), `bench`,
    `data`, `data-text`. OpenMP is the option `NMZ_OPENMP` on every compiler
    (a compiler without OpenMP gives a warning and a single-threaded
    build); sanitizers are `NMZ_SANITIZE`.
115. **One warning set per compiler family, errors on by default.**
    GCC/Clang: the Makefile's set plus `-Wvla` (MSVC has no VLAs; the code
    had none). MSVC: `/W4 /WX /utf-8` without C4244/C4267/C4305, the
    implicit-narrowing warnings that GCC and Clang only give under
    `-Wconversion`, which was never enabled; `_CRT_SECURE_NO_WARNINGS`
    because the "secure" `fopen_s` family is not portable C11. `/utf-8`
    because the CLI's sources contain Greek letters and Arabic.
116. **The data scripts exist twice, with the same checksums.**
    `get_mnist.sh` / `get_text.sh` for Unix-like systems, `get_mnist.ps1` /
    `get_text.ps1` for Windows (PowerShell 5.1 and 7, built-in cmdlets and
    .NET only); `--target data` picks the right one. Both now also check
    the SHA-256 of what they produce (the decompressed IDX files, the
    prepared `alice.txt` and `kalila.txt`), so a different awk or a
    different port cannot silently change the data the results in README
    were measured on. The PowerShell port of `strip_wikitext.awk` was
    checked against the awk output with a regex port in Python (same
    backtracking semantics as .NET); it runs in the manual `data` workflow,
    not in CI.
117. **Training output says what each number is.** The epoch line labels
    its accuracy `train accuracy` or `val accuracy` (with validation the
    column was the validation accuracy, while the final line reported the
    training accuracy — the "37.7% vs 33.5%" of manual testing). A note
    under the header says that epoch lines re-evaluate whole sets after
    the epoch's updates and that the live bar shows the last batch's loss;
    the final line says "on the training set" and, after early stopping,
    "re-evaluated with the kept weights". The live bar is never wider than
    the terminal and is erased exactly (its drawn width is remembered):
    an 83-column bar erased with 79 spaces left "…0.0s     .0s" behind.
    The training header prints only the active optimizer's settings.
118. **Weight-update tables show the 8 largest |Δ| by default**, largest
    first, after one summary line per matrix (count, mean |Δ|, max |Δ| and
    where); `--full` lists every weight in index order. Δ (an update) is
    now distinct from δ (a backpropagated error) in these tables.
119. **Preset descriptions quote measured times** (one core of the
    i7-7500U, two epochs each): mnist-cnn ~30 s per epoch (the "about a
    minute (naive convolution)" predates im2col), mnist-mlp ~20 s, names /
    names-lstm ~2 s, names-gpt ~1.5 s, char-rnn / char-lstm / char-gpt
    ~9–10 s, arabic-text ~1 min, arabic-gpt ~50 s.
120. **No floating-point contraction (`-ffp-contract=off`).** With FMA,
    `c += a·b` is rounded once instead of twice, and a compiler may fuse it
    in one loop and not in another: on Apple Clang / arm64 the blocked
    matmul then differed from the naive loops in the last bit (and one
    convolution gradient check fell just outside its tolerance). GCC
    already does not contract in ISO C mode; Clang does by default, so the
    flag is set for GCC and Clang. MSVC does not contract under its
    default `/fp:precise`. Bit-identical results across code paths are
    what the KV-cache and blocked-matmul tests rely on (decisions 92, 98);
    across platforms the results agree as well where the libm agrees (the
    XOR quick-mode run converges in 172 epochs with GCC on Linux and with
    MSVC on Windows).
121. **The CLI moved to its own repository with its history**
    ([neuromz-cli](https://github.com/mzprog/neuromz-cli)), the first step
    of splitting the project into `neuromz` (the library, later its CPU and
    CUDA backends), `neuromz-protocol`, `neuromz-server`, `neuromz-cli` and
    `neuromz-web`. `git filter-repo` kept every commit that touched
    `cli/`, the smoke test, the fixtures and the example data and scripts
    the program uses, with `cli/` renamed to `src/`. What stays here: the
    library, its tests, the C examples and benchmarks (`api_xor`,
    `api_mnist`, `bench_conv`, `bench_matmul`), the data scripts (one copy
    of the checksums; the CLI's `data` targets call them), DESIGN and the
    `v1-legacy` tag. The CLI-driven benchmark scripts and the CLI
    walkthroughs of the README moved with the program; this README keeps
    the build, the C API, the performance numbers and a summary of the
    results. The small example datasets are in both repositories (the
    library's tests and the program's smoke test both read them).
122. **The program pins the library by tag.** `neuromz-cli` fetches the
    library with CMake's FetchContent at `NMZ_LIBRARY_TAG` (default the
    release it was tested with, `v2.0.0`), so a checkout of a CLI release
    always builds the same library; `-DNMZ_LOCAL_PATH=../neuromz` builds
    against a working copy and `-DNMZ_USE_INSTALLED=ON` against an
    installed package (`find_package(neuromz 2.0)`, provided by
    `cmake --install`: headers, the static library, `neuromzConfig.cmake`
    with OpenMP as a dependency when the library was built with it). The
    CLI's CI has one job against the library's `master`, to see breakage
    before a release. Library and CLI share version numbers. When the
    library is a subproject, its tests and examples are off and its
    warnings are not errors (a newer compiler must not break a user's
    build); its own CI keeps `-Werror`.
123. **The library's regressions run without the CLI.** CTest runs the unit
    tests, `examples/api_xor.c` (fails if quick-mode or hand-built XOR does
    not converge) and `examples/api_mnist.c` (the mnist-cnn architecture,
    one epoch on 20 000 images, at least 95% test accuracy; 96.6% measured;
    exit 77 = skipped when the data has not been downloaded), and
    `tests/package` checks in CI that an installed library is usable via
    `find_package`. XOR+AND, Iris and the text models stay covered by the
    unit tests (decisions 85, 107 and the fixtures).
124. **Release v2.0.0, MIT licence.** Both repositories carry the MIT
    licence (copyright Mohamad Zbib), a CHANGELOG in the Keep a Changelog
    format, and the same version number. The library's `master` gets `v2`
    merged (no squash: the phase history is the documentation of how the
    engine grew); tags `v2.0.0` in both repositories. Before publishing,
    both repositories were checked for tracked datasets, models (only the
    small `tests/data/*.nmz` fixtures are tracked), build outputs and
    secrets. The contact address in `legacy/` is the v1 code's own and was
    already public.
125. **The GitHub repository is `mzprog/neuromz`** (renamed from
    `NeuroMZ` with `gh repo rename` at the release; GitHub redirects the
    old URL, and the names of the planned repositories are lower case too).
    Both repositories have a `.gitattributes` with `* text=auto eol=lf`:
    Git for Windows checks text out with CRLF by default, which changed the
    bytes of the example data on Windows — the library's loader already
    reads CRLF as LF, but the CLI's text excerpts (`show gates --data`,
    `compare`) did not; they now do, and checkouts are identical anyway.
    The PowerShell scripts must be ASCII (CI checks it): Windows
    PowerShell 5.1 reads a script without a BOM in the ANSI code page, and
    the Arabic words of one regex had been written as letters instead of
    `\uXXXX` escapes; hashing uses .NET because `Get-FileHash` does not
    load in 5.1 started from 7.
126. **A patch release, not a moved tag.** The PowerShell problems above
    were found by the first run of the manual `data` workflow, minutes
    after `v2.0.0` was published. Published tags are not rewritten:
    `v2.0.1` carries the fix, and `neuromz-cli` 2.0.1 pins it (its data
    targets run the library's scripts from the fetched release).

### Release 2.1: device backends (Step A: the backend layer and the CPU backend)

127. **A backend is a table of ops; fused kernels are optional.** The table
    (`nmz_backend`, `neuromz_backend.h`) has the memory functions and about
    fifty required ops the layers are written in: GEMM with transpose
    flags and row strides, element-wise ops, activations and their
    derivatives, softmax and the fused cross-entropy, reductions, im2col /
    col2im, pooling, dropout masks, embeddings and positions, LayerNorm,
    the attention core, one time step of each recurrent cell, the
    optimizer updates, the clipping norm, normalisation and the training
    loop's gathers. Ten fused layer kernels (dense, conv2d, rnn, lstm and
    gru, forward and backward) may be NULL: a layer then composes the same
    math from the required ops. Each layer therefore has two paths, the
    fused call and the composition; the composition is what a GPU runs,
    written once in the layer for every device. The granularity was chosen
    by the CPU: v2.0.1's dense, convolution and recurrent loops keep
    intermediate sums in double that a GEMM-then-activation split would
    round to float in between, so bit-identity needs them as one kernel;
    every other layer's v2.0.1 code already was a sequence of GEMMs and
    simple loops, which became ops one for one. A recurrent sequence is
    one fused kernel, as in cuDNN: a backend can then pick its own
    schedule (the CPU walks each row through all its steps; the
    composition does the input product of all steps at once, one
    recurrent product and one cell op per step, and the weight gradients
    of all steps as single GEMMs after the loop).
128. **The CPU backend is v2.0.1, bit for bit.** Its ops are the old loops,
    moved into `backend_cpu.c` or left next to their layer as `nmz__cpu_*`
    kernels; the CPU path through a layer is the old arithmetic with the
    old operation order. Verified three ways: the 87 unit tests pass with
    unchanged numbers; a probe program (public API only) that trains a
    model of every layer type (XOR quick mode, Iris with Adam/patience and
    with SGD/clipping/step schedule, a CNN with both convolution
    algorithms, RNN/LSTM/GRU with carried states and dropout, a tied
    Transformer with AdamW and warm-up, the Transformer parts), records a
    traced step and a gradient check, generates text (KV cache and no
    cache) and loads the four fixtures gives the same output as v2.0.1 and
    11 byte-identical model files — in the single-threaded and the OpenMP
    build; and the benchmarks are no slower (im2col 4.32 → 4.17 s, naive
    convolution 13.2 → 10.6 s, a char-LSTM epoch 6.80 → 6.53 s, a
    char-GPT epoch 8.64 → 8.45 s: plain pointers into the kernels let MSVC
    optimise the inner loops better).
129. **A tensor keeps its host data and, once a device computes on it, a
    device copy plus a note of which is newer** (Caffe's synced memory).
    `nmz_tensor` gained one pointer, `dev`; `data` stays host memory.
    `nmz__r/w/u(be, t)` (public: `nmz_tensor_ptr`) return the tensor's
    data in a backend's memory for reading, overwriting or updating, move
    data only when that copy is out of date, and mark the other copy
    stale after a write; for a host tensor on the CPU they are one test.
    A device copy is allocated on first use; `nmz_tensor_reshape` drops one
    that became too small (contents are unspecified on growth anyway),
    `nmz_tensor_copy` and `nmz_tensor_zero` work where the newest data is.
    Device-only tensors were the alternative, and would have broken the
    public API: `nmz_layer_param_value(l, i)->data[k] = …` (the unit tests
    set hand-computed weights that way), `nmz_net_forward`'s output,
    trace events and the CLI all read `data` directly.
130. **What the public API hands out is current on the host at the time of
    the call.** `nmz_net_forward`/`_mode` (the output), `nmz_net_activation`,
    `nmz_net_activation_grad`, `nmz_net_input_grad`, every trace item and
    `nmz_layer_param_value`/`_grad` bring their tensor to the host; the
    parameter accessors also mark the host copy as the newer one, because
    the caller may write. A pointer kept across later computation can be
    stale: ask again. Internally the engine uses `nmz__param` (no
    synchronisation), so counting parameters, saving or clipping never
    moves data that does not have to move. A caller's input and target
    are copied into the net, never given device copies of their own (the
    caller owns those structs and may free them with `free`).
131. **Layers declare `NMZ_LAYER_BACKEND`.** All built-in layers compute
    through `ctx->backend` (a field appended to `nmz_ctx`). A layer table
    without the flag — one written for v2.0 that reads `->data` — still
    runs on a device: the engine brings its input, output gradient and
    parameters to the host before the call, marks its output, input
    gradient and parameter gradients as changed there, and gives it the
    CPU backend in its context. Tested with such a layer inside a net on a
    device (the result equals the host run bit for bit).
132. **The selection is process-wide, the default is the CPU.** Selecting a
    backend is the library's one piece of global state besides the
    thread-local error message: a device is a property of the process
    (one CUDA context), and `nmz_backend_select` is called once at start
    by a program (`--device`) more often than per net. A net computes on
    the backend selected when it runs: entry points that compute look at
    the selection, and a net that moves to the CPU brings every tensor
    home and frees its device memory. Model files never record a device.
    "auto" picks CUDA when it is built in and a usable GPU is found and
    otherwise the CPU, and its note says which and why ("auto: no usable
    CUDA device (…), using the CPU"): the library does not print, so the
    one clear message is the note, which the CLI prints.
133. **The generic compositions and the device data path are tested on
    every CI runner, without a GPU.** Two backends do it: `cpu-generic`
    (the CPU's ops with no fused kernels: the compositions a GPU runs) is
    built in and selectable; a fake device registered by the test through
    the public API (`nmz_backend_register`) uses the same ops on memory of
    its own, with explicit copies, memory poisoned with NaN when
    allocated, and an optional limit. cpu-generic agrees with the CPU to
    2.4·10⁻⁷ relative on every layer type (outputs, all gradients); the
    fake device must give cpu-generic's bits exactly — any missed or stale
    copy would show — for passes, gradient checks, training with the data
    set on the device, moving a net mid-training, generation with and
    without the KV cache, traced values and a host-only layer.
134. **Token ids are checked on the host before a device sees them.** The
    CPU's embedding kernel reports "token id 7.5 is not an integer in
    0..64"; a kernel cannot. On a device the net checks a caller's input
    on the host before copying it, and training checks the whole data set
    once before it goes to the device.
135. **Running out of device memory is a clean error, and the net lets go
    of its buffers.** An allocation that fails returns `NMZ_ERR_NOMEM`
    with a message that says how much was needed, what is in use and what
    to try (a smaller batch); nothing crashes and the device stays usable.
    Training and evaluation that fail this way free the activations and
    every layer's caches on the device, so a retry with a smaller batch in
    the same process starts from nothing. For that the layer table gained
    one more optional entry, `buffers(l, out, max)`: the layer's tensors
    other than its parameters (caches, work space, carried state). The
    same list lets a net that moves back to the CPU free all its device
    memory.
136. **On a device the training data stays on the device.** A run copies
    the data set (and a separate validation set) to the device once, the
    epoch's shuffled order once per epoch, and gathers every batch there
    (`gather_rows`, `onehot` for text targets); evaluation counts correct
    predictions on the device (`count_correct`). What comes back during an
    epoch is the batch loss (and, with clipping, the gradient norm), which
    the loop needs for its NaN check, the clipping decision and the
    `on_batch` report; the epoch's metrics; nothing else. A data set too
    large for the device falls back to gathering batches on the host and
    copying them one at a time.
137. **Traces work on any device, with the same events in the same order.**
    `nmz_trace_emit` brings every item to the host before the callback.
    The recurrent layers used to emit their per-step backward events from
    inside the time loop; the loop now lives in a kernel, so the kernels
    record those values in `[B, T, ·]` buffers (`nmz_seq_record`) and the
    layer emits the same events, with the same numbers, after the loop.
    The fake device's traced numbers equal cpu-generic's.

### Release 2.1, Steps B and C: the CUDA backend and its tests

138. **The CUDA backend is a build option and one CUDA file behind a C
    ABI.** `-DNMZ_CUDA=ON` (off by default) compiles
    `src/cuda/backend_cuda.cu` with nvcc. The C sources see one function,
    `nmz__cuda_backend()`, declared `extern "C"`, which returns the filled
    table; nothing of CUDA appears in a public header (`nmz_device_info` is
    a plain struct), so a program using the library is still plain C and
    needs no CUDA header. The file is CUDA C++ only as far as nvcc needs it
    (kernel launches, `__global__`); it uses no C++ library. Rejected: the
    driver API with PTX loaded by hand (no `<<< >>>`, a lot of loading code
    for nothing) and NVRTC (a JIT compiler as a run-time dependency).
139. **Architectures: the building machine's GPU by default, machine code
    plus PTX for each listed one.** `NMZ_CUDA_ARCHITECTURES` defaults to
    what `nvidia-smi --query-gpu=compute_cap` reports, else 75 (Turing,
    the oldest architecture CUDA 13 compiles for). Every entry is compiled
    to SASS and PTX, so a newer GPU runs the library through the driver's
    JIT (a slower first start, then full speed). A binary for others:
    `-DNMZ_CUDA_ARCHITECTURES="75;86;89;120"` (Turing, Ampere, Ada,
    consumer Blackwell; add 80 and 90 for data-centre parts). The lowest
    entry is compiled in as `NMZ_CUDA_MIN_CC`; `init` refuses an older GPU
    with a message naming the option, and launches a probe kernel, so a GPU
    the binary has no code for is found at selection rather than in the
    first training step.
140. **A CUDA build starts on any machine, and selection never crashes.**
    The CUDA runtime is linked statically (it finds the driver at run time
    and reports a missing or old one as an error code); cuBLAS is loaded
    with `LoadLibrary`/`dlopen` when the backend initialises
    (`cublas64_13.dll` / `libcublas.so.13`, the toolkit's major version;
    on Windows also from `%CUDA_PATH%\bin\x64` and `\bin`), so the program
    has no load-time dependency on the driver or on cuBLAS. The few cuBLAS
    declarations used (a handle, an operation enum, a status, three
    functions) are written out instead of including `cublas_v2.h`, so
    building needs only nvcc and the runtime — the CI jobs install exactly
    those. Initialisation is lazy (the first `select` or `query` of
    "cuda") and checks, in order: driver present, driver new enough, a
    device, its compute capability, the probe kernel, cuBLAS, then creates
    the stream, cuBLAS handle and pinned buffers. Any failure becomes the
    reason ("no NVIDIA driver is installed", "the NVIDIA driver supports
    CUDA 12.4, this library needs 13.0: update the driver", "the cuBLAS
    library (cublas64_13.dll) was not found: …"), is remembered, and is
    what "cuda" fails with and what "auto" prints when it falls back.
    Tested on the CI runners, which have no GPU: the CUDA builds pass the
    CPU suite, the `gpu` tests report "skipped" (exit 77), and
    `unit_no_gpu` (the same build with `CUDA_VISIBLE_DEVICES=-1`) checks
    the fallback and its note.
141. **The CUDA backend is tested against the CPU at three levels, with
    stated tolerances.** `test_gpu` (ctest `gpu`, label `gpu`): every
    required op on random inputs and many shapes — sizes 1, odd, prime,
    above the block and warp sizes, row strides, every GEMM flag; every
    layer type's forward and backward against the CPU; the gradient checks
    of every layer type on the GPU; traces (the same events, the values
    within tolerance); the KV cache; moving nets between devices; two GPU
    runs giving the same bits; running out of memory. `test_gpu --parity`
    (ctest `gpu_parity`): XOR, Iris, the MNIST CNN, the names RNN, the
    Alice LSTM and a small Transformer trained on both devices reach the
    same quality. Without a usable GPU both exit 77 and CTest reports them
    skipped. Tolerances, as largest |gpu − cpu| over largest |cpu| of a
    result: 0 for data movement, element-wise +, ×, gathers, one-hot,
    im2col, pooling, normalisation and counts (the same float operations
    in the same order); 1e-6 for reductions and the optimizers (both sum
    in double, in different orders); 2e-6 for transcendental functions,
    softmax, losses, LayerNorm and the recurrent cells (CUDA's `expf`,
    `tanhf`, `erff` are float functions within 2 ulp; the CPU computes in
    double and rounds once) and col2im (another summation order); 1e-5 for
    matrix products and attention (cuBLAS accumulates in its own order and
    blocking); 1e-4 for whole layers and models, where all of these
    compound. The largest errors measured: GEMM 1.2·10⁻⁶, GELU backward
    3.4·10⁻⁷, attention backward 3·10⁻⁷, any layer ≤ 1·10⁻⁶. Results:
    15 tests and 1369 checks pass; parity in the README.
142. **The library caps its device memory at what was free when the device
    was initialised.** On Windows the display driver (WDDM) lets
    `cudaMalloc` succeed far beyond the card — 16 GB on this 6 GB GTX 1660
    Ti — by paging device memory to the host, and training then slows
    to a crawl instead of failing. So `alloc` refuses
    beyond a cap with `NMZ_ERR_NOMEM` and a message (what was needed, what
    is held, the cap, "try a smaller batch size"); the cap defaults to the
    free memory at initialisation and is changed with
    `nmz_backend_memory_limit(bytes)` or `NMZ_CUDA_MEMORY` (MiB) at start.
    Every allocation is recorded in a hash table (pointer → size), which
    gives `mem_used` and `mem_peak` in `nmz_device_info`; these are the
    VRAM numbers in the README. On Linux the cap only makes the error
    message friendlier: the driver fails by itself.
143. **One stream, no waiting until a number is needed.** Kernels, cuBLAS
    calls and copies go to one stream, in order. An op that produces a
    scalar for the host (a loss, a squared norm, a count) queues a copy
    into pinned host memory and records where the value belongs; the value
    is valid after `sync`, which the engine calls when it reads one (up to
    64 may be pending; a 65th syncs first). A training step therefore
    waits for the batch loss and, with clipping, the gradient norm, and
    nothing else; evaluation queues the results of all its chunks and
    waits once per data set.
144. **GEMMs in cuBLAS, everything else in the library's own kernels, all
    of it deterministic.** Row-major products are cuBLAS's column-major
    products with the operands swapped (Cᵀ = Bᵀ·Aᵀ), in FP32 (Turing has
    no TF32 or BF16 tensor cores to tempt anyone). Reductions are two
    passes in double: each block writes a partial sum, one block adds the
    partials in a fixed order, so the same run gives the same bits every
    time. Atomics, whose order varies, appear in two places only: the
    embedding gradient when vocabulary × tokens exceeds 2²⁴ (a
    deterministic version would need a sort; for the models here the
    deterministic kernel runs), and the positional-encoding gradient when
    the rows of a batch start at different positions (generation, not
    training). The test of decision 141 checks that two GPU trainings
    write identical model files.
145. **No fused multiply-add in the kernels (`--fmad=false`)**, as on the
    CPU (decision 120). The optimizer updates then round exactly as the
    CPU's (the op tests find the momentum step bit-identical and Adam,
    computed in double on both, within 1e-6), and the element-wise ops differ
    from the CPU only where CUDA's math library does. cuBLAS keeps its own
    FMA, which is why products get 1e-5.
146. **On CUDA a recurrent sequence is one kernel launch.** The
    composition (decision 127) launches a recurrent GEMM and one to three
    cell kernels per time step; at the sizes of the presets (H = 64,
    batches of 32) each launch does almost nothing and costs 10–15 µs on
    Windows, so the Alice LSTM spent its time in launches. The CUDA
    backend implements the fused sequence kernels: the input product of
    all steps is one cuBLAS GEMM, then one thread block per batch row walks
    all T steps with its state in shared memory, reading W_h from global
    memory (cached in L2), and the backward pass is the mirror image with
    the weight gradients as single GEMMs after the loop. A block holds at
    most 1024 threads, one per gate unit, so the kernels serve RNN with
    H ≤ 1024, GRU with H ≤ 512 and LSTM with H ≤ 256. For larger layers
    they decline (`NMZ_ERR_UNSUPPORTED` before touching anything, a rule
    added to the table's contract) and the layer composes the step-by-step
    version, where the per-step GEMMs are large enough to keep the GPU
    busy. Both paths are tested on the GPU.
147. **Attention uses batched cuBLAS products for training, warp kernels
    for generation.** For query blocks of 16 positions or more, for each
    batch row, the scores of all heads are one strided batched GEMM
    (S = Q·Kᵀ/√d over all keys), a kernel applies the causal softmax
    (zeroing the future), and a second batched GEMM gives P·V; the
    backward pass is four batched GEMMs around a softmax-backward kernel.
    The masked half of S is computed and discarded. Each batched call
    covers one head for all rows (or one row for all heads, when there are
    more heads than rows): a first version made one call per row, which
    at batch 32 meant about 190 launches per layer and step and made the
    small models slower than with the warp kernels. Measured against
    warp kernels for every length: about even for the small models (0.53
    vs 0.47 s per epoch for the 32k Transformer at T = 128, 1.04 vs 1.10 s
    for the 213k one) and 1.7× faster for the 25M one at T = 256 (8.9 vs
    15.2 s of updates per epoch), so the products start at T = 16. Short
    blocks — generation with the KV cache, one new position at a time —
    run one warp per query row in a single launch.
148. **Dropout masks on a device come from a counter-based generator keyed
    by the net's PRNG.** Each mask takes two numbers from the net's PCG32
    and hashes key + element index (SplitMix64's finaliser) in parallel,
    so a run on the GPU is reproducible and the net's random stream
    advances the same way per mask; but the masks are not the CPU's (which
    draws one PCG32 number per element, in order). With dropout the two
    devices therefore train along different random paths of the same
    distribution: parity compares quality, not bits.
149. **On a device, evaluation chunks are no larger than the training
    batch.** A host evaluation runs 64 rows at a time; on the GPU the
    largest Transformer trained at batch 16 and then ran out of memory in
    evaluation, because 64 rows of activations (attention weights grow
    with T²) did not fit beside the optimizer state. Evaluation now uses at
    most the training batch, so what trained also evaluates.
150. **Model files do not know the device.** Saving brings the parameters
    and optimizer moments to the host; loading creates host tensors that
    move to a device on first use. Tested: a model trained on the GPU,
    saved, loaded on the CPU, predicts what the GPU predicts (within 1e-5)
    and continues training on the CPU from the GPU's optimizer state; the
    same loaded net, moved to the GPU, predicts the same. Loading never
    touches a device, so a CPU-trained file is the same case.
151. **CI compiles the CUDA backend on Linux (GCC) and Windows (MSVC,
    Ninja)** with CUDA 13.5's redistributable archives (nvcc, NVVM, the
    runtime, CCCL headers — no installer), with warnings as errors in the
    CUDA code too, and runs the whole test suite there without a GPU
    (decision 140). The GPU tests run on the one machine with a GPU, before
    a release.
152. **What the GPU is worth here, measured, including where it is not.**
    Intel Core i5-8500 (6 cores) and a GTX 1660 Ti (6 GB, Turing, no
    tensor cores), Windows 11, MSVC 19.52, CUDA 13.5, driver 621.40;
    `bench_device`, time per epoch (updates and the evaluation of the
    training and validation sets), README "On the GPU". The GPU is 10× on
    the MNIST CNN (24.3 → 2.35 s), 9.5× on the Alice LSTM (6.67 → 0.70 s),
    16× on the 32k Transformer (8.55 → 0.54 s; 7.6× against 6 OpenMP
    threads), 33× on the 213k one (33.9 → 1.04 s; 13× against 6 threads),
    and about 240× on a 25M-parameter Transformer (8 blocks of 512,
    context 256, batch 16, 3.5 GB of the 6 GB at its peak): 8.9 s of
    updates per epoch against an extrapolated 36 min on one core and
    8.7 min on six. Where it does not help: XOR trains in 0.00 s on the CPU
    and 0.15 s on the GPU, Iris in 0.01 s and 0.42 s — the device's start
    and one launch per op cost more than the arithmetic of such nets, so
    they belong on the CPU (the library's default; "auto" is a choice).
    The small models are launch-bound on the GPU too: the LSTM's 0.70 s
    epoch is 0.12 s of updates and 0.58 s of evaluation, which runs the
    16 text streams step by step (the fused kernel serves training; the
    evaluation passes are short and many), and the 32k Transformer's
    0.54 s is near the floor of a few hundred launches per batch at
    10–15 µs each under WDDM. OpenMP speeds up only what goes through the
    blocked product and the attention heads: the Transformers (2.1–2.5×
    on six threads) but not the CNN or the LSTM, whose CPU kernels are
    single-threaded loops.
153. **Release 2.1.0 for both repositories.** A minor version: everything
    of 2.0 still compiles and behaves the same on the CPU, and the
    additions are the device API, `neuromz_backend.h` and the optional
    layer-table entry. `nmz_tensor`, `nmz_ctx` and `nmz_layer_ops` grew
    by a pointer each, so code built against 2.0 must be recompiled (the
    library is a static library by default, which a program rebuilds with
    it anyway). The CLI 2.1.0 pins `v2.1.0` and adds `--device`.

### Release 2.1, Step D: devices in the program

154. **The program's default is `--device auto`; the library's stays the
    CPU.** A program chooses for its user (decision 132). A plain build has
    no CUDA, so there "auto" is the CPU and every walkthrough's numbers
    stay bit for bit what they were; whoever builds with `-DNMZ_CUDA=ON`
    asked for the GPU, so "auto" takes it when there is a usable one. The
    output never hides the choice: `new` lists `device cuda [auto] NVIDIA
    GeForce GTX 1660 Ti, 6144 MiB: a usable GPU was found (--device cpu
    for the CPU)` with the other settings, `train` prints it in its header,
    and a fallback prints the library's reason ("no usable CUDA device (no
    NVIDIA driver is installed), using the CPU"). An explicit `--device
    cuda` that cannot be used is an error with that reason, not a silent
    fallback.
155. **The device is the session's, not the model's.** Model files record
    no device (decision 150), so `set --device` changes the shell, works
    without a network, and checks the device at once (reporting it or the
    reason it cannot be used, and keeping the old one then). `train …
    --device D` follows `train`'s rule that options override for one run;
    `new … --device D` sets the session's (new creates its network);
    one-shot `--device D` anywhere on the line applies to that command,
    except after `set`, which reports the device itself.
156. **The device is selected only when a command computes.** `help`,
    `load`, `save` or `show params` never initialise a GPU (that costs a
    few hundred milliseconds and a driver call); `new`, `train`, `test`,
    `predict`, `trace`, `step`, `gradcheck`, `generate`, `compare` and the
    `show` views that run the network select it first. `show device` lists
    the device, why it was chosen, the GPU's compute capability and memory
    (total, free, held by the program now and at most, the cap), and which
    devices the build can use.
157. **The program is honest about small networks.** `train` on a GPU
    reports the GPU memory used at the end, and for a network under 5000
    weights notes that the CPU is usually faster (decision 152: Iris
    trains in 0.42 s on the GPU, 0.01 s on the CPU). "auto" does not switch
    devices by size: a choice that depends on a threshold would be harder
    to predict than a note. The smoke test covers devices in every build:
    without a usable GPU it checks the fallback and the clean failure of
    `--device cuda`; with one it trains on the GPU, loads the file on the
    CPU, runs gradient checks and traces on the GPU and exhausts a small
    memory cap — and in a CUDA build with a GPU the whole script, every
    model type and view, runs on the GPU.

### Release 2.2: threads, models in words, remote mode

158. **Nets on several threads at once; the selection and the CUDA state
    per thread.** A server runs jobs in parallel threads, each with its own
    nmz_net. The audit found the library's state already per net, the
    error message thread-local, and three shared things: the backend
    registry (lazy initialisation), the process-wide selection, and the
    CUDA backend's stream, cuBLAS handle, work buffers and pending
    scalars. The registry and the selection now sit behind one lock (SRW
    lock / pthread mutex; the library links Threads on POSIX), and a
    thread may select a backend for itself (`nmz_backend_select_thread`),
    which wins over the process-wide one in that thread only — a worker
    runs its job on its job's device. The CUDA backend shares the device
    context, cuBLAS and the allocation table (a mutex) and gives every
    host thread its own stream, cuBLAS handle, pinned slots and work
    buffers, created on the thread's first op and released when it ends:
    nets on different threads run side by side on the GPU and never see
    each other's buffers. The rules are in `neuromz.h`: a net, dataset,
    vocabulary or recorder belongs to one thread at a time and may move
    between calls (after `nmz_backend_sync` if the last call only queued
    device work). Tested: six nets of four kinds trained in six threads
    give byte-identical model files to sequential runs; three GPU jobs and
    a CPU job at once give the files of sequential runs; a thread's
    selection does not leak; errors stay per thread; eight threads call
    the registry concurrently. CPU results are bit-identical to 2.1.0 (the
    probe of decision 128: 75 lines and 11 model files).
159. **Models described in words belong to the library.** The program's
    `new` — the layer grammar, the presets, the explained defaults — was
    the only place that turned a description into a model, and the
    server must build exactly the same models (and the web client show
    the same explanations). It moved to the library as `nmz_model_new`
    (the arguments of `new`, an explanation line callback, a data-path
    callback so a server can map dataset names to its files, and two
    hooks the program uses for its device line), with
    `nmz_model_format`, the preset list, `nmz_train_option` (the training
    options as the program spells them) and `nmz_train_describe` (the
    configuration in one line). The library still never prints. Moving
    rather than rewriting kept every message: the program's `new` prints
    byte for byte what it printed before (43 cases: presets, specs, text
    and Transformer models, 15 errors; stdout, stderr and status).
160. **Progress and stopping use the callbacks that exist.** `on_batch`
    and `on_epoch` already let a caller stop training (a non-zero return,
    `res.stopped`); a server's cancel returns non-zero from `on_batch`, so
    training stops after the current batch with the weights of the
    batches done. Now tested: stopped after three batches, the weights
    equal a one-epoch run over the same three batches, bit for bit.
161. **In remote mode the current model lives on the server.** The program
    keeps one model; in remote mode it is a model on a neuromz-server,
    named, and commands act on it there: `new` builds it on the server
    (which explains it with the library's lines, decision 159), `load
    FILE` uploads a file and `load NAME` picks a server model, `save FILE`
    downloads it (`-m`/`-o` work the same way in one-shot use), and
    `train`, `test`, `predict`, `generate` and `trace` compute on the
    server. The views (`show`, and the printing of a trace) download the
    model and use the program's own printers, so remote output reads like
    local output; the trace's numbers come from the server's events. Every
    command says which server it used, and the shell's prompt names the
    server and the model.
162. **Profiles in client.conf; local mode unchanged.** `client.conf`
    (Linux `~/.config/neuromz/`, Windows `%APPDATA%\NeuroMZ\`, or
    `$NEUROMZ_CLIENT_CONFIG`) holds named profiles: `mode = local`, or
    `host`, `port`, `token`, `save`; `default = NAME` picks one when no
    `--profile` is given. `--host` and `--token` override. Without a
    file, or with a local profile, the program is exactly what it was:
    the smoke test runs unchanged in local mode and passes.
163. **Data goes to the server once.** A local data file a remote command
    needs (CSV, text, MNIST's IDX pair) is uploaded as a dataset named by
    its file name and SHA-256 (`xor.csv` becomes `xor-csv-` and ten hex
    digits); a file the server already has is not sent again (its
    `sha256` answers). The program counts samples from 1, the protocol
    from 0: the program converts.
164. **Remote training streams the server's events.** `train` starts a job
    with the program's own option arguments (the server parses them with
    the same library parser, so the same options mean the same thing and
    a bad one gets the same message) and prints the server's epoch events
    with the local epoch line. With `save = client` on a stored model the
    result is a new client model (`NAME.jID`), which becomes the current
    model.
165. **Some commands stay local.** `step`, `gradcheck`, `compare`,
    `reinit` and `set` (except `set --device`) change or examine a model
    in ways the protocol 1.0 does not cover; in remote mode they say so
    and suggest `save FILE` and a local profile. Plain HTTP only: TLS is
    a reverse proxy's job (the server's README).

### Program 2.3: the remote mode over protocol 2

The protocol's owner replaced HTTP with a binary protocol of its own
(neuromz-protocol 2.0.0, its decisions P18–P37); neuromz-server 0.2.0
speaks only that. The program follows; the library does not change.

166. **The remote mode speaks protocol 2 through `nmzp`; nothing else
    changes for its user.** The same profiles, flags, commands, messages
    and outputs: a request is a MessagePack map instead of a JSON body, a
    file goes in DATA frames with its SHA-256 instead of an HTTP body, and
    `train` follows the job through a subscription on the same connection
    instead of a second HTTP stream (decision 164 otherwise stands). One
    connection serves a whole run of the program (a shell session keeps
    it, and reconnects if the server closed it as idle). `host` is
    `HOST[:PORT]`; a URL (`http://…`, a protocol 1 address) is refused with
    a message that says so. The token is proved in the handshake and
    never sent; the server's proof is checked, and a program that holds a
    token refuses a server that offers no authentication, since an
    attacker on the path could otherwise pose as the server.

167. **The program has its own version from 2.3.** Until 2.2 the program
    and the library shared version numbers (decision 122). A change of the program alone — its remote mode — should
    not release a library whose code did not change, so the program is
    2.3.0 against library 2.2.0; `neuromz --version` names the program,
    the library and the protocol (`neuromz 2.3.0 (library 2.2.0, remote
    protocol 2.0)`), and `NMZ_LIBRARY_TAG` still pins the library.

168. **Local mode stays byte for byte.** The commands that run locally are
    untouched; their output and the model files they write are compared
    with 2.2.0's in the release checks (the same text, the same bytes).

---

## 11. Device backends (release 2.1)

### 11.1 The layer

```
 layers (dense.c, lstm.c, attention.c, …)       the engine (net.c, train.c)
        │  ctx->backend->gemm(…), ->lstm_cell_forward(…), …       │
        ▼                                                         ▼
 nmz_backend  ── memory: alloc release upload download sync ──────┘
   ├─ "cpu"          the ops = v2.0.1's loops, + fused kernels   (backend_cpu.c)
   ├─ "cpu-generic"  the same ops, no fused kernels              (testing)
   └─ "cuda"         cuBLAS + kernels, device memory             (src/cuda/, §11.7)
```

Every layer computes through the backend in its context; the optimizer,
clipping, normalisation, the loss and the training loop's gathers through
`net->be`. Pointers come from `nmz__r/w/u(be, tensor)`. The ops are listed
in `neuromz_backend.h` with their exact contracts (row-major, strides,
which outputs are added into, which scalars are valid only after `sync`).

### 11.2 The CPU backend and cpu-generic

The CPU table is v2.0.1 (decision 128). `cpu-generic` shares its ops and
leaves the fused kernels out, so on the CPU it runs exactly what a GPU
runs: one input GEMM for all time steps, a recurrent GEMM and a cell op
per step, im2col over the whole batch and one GEMM per pass; it is the
reference the device path is tested against (decision 133).

### 11.3 Tensors on devices

A tensor's `dev` (struct `nmz_tensor_dev`: owning backend, device pointer,
capacity, which copy is newer) appears the first time a device backend
asks for the tensor's data (decision 129). Host code keeps working as long
as it goes through the accessors (`nmz__hr/hw/hu` in the library, the
public accessors outside it, decision 130).

### 11.4 Selection

`nmz_backend_select("auto" | "cpu" | "cuda")`, `nmz_backend_info`,
`nmz_backend_query`, `nmz_backend_count/name` (decision 132). A device
backend's `init` runs once, the first time it is selected or queried, and
its answer (usable, or why not) is remembered.

### 11.5 Training on a device

Decision 136: the data set is copied once, batches are gathered on the
device, and per batch only the loss (and the clipping norm) come back.

### 11.6 Adding a backend (HIP, Vulkan, OpenCL, …)

1. Write `src/<name>/backend_<name>.<ext>` exporting
   `const nmz_backend *nmz__<name>_backend(void)`: fill the table of
   `neuromz_backend.h` — `init` (find the device lazily; return
   `NMZ_ERR_UNSUPPORTED` and the reason when there is none, never crash),
   `info`, the memory functions (`alloc` reports failures with
   `nmz_backend_error(NMZ_ERR_NOMEM, …)`), `sync`, and every required op.
   Leave the fused kernels NULL: the layers then compose them from the
   required ops.
2. Add one `add_slot()` line to `builtins()` in `backend.c`, under a
   compile-time switch, and the sources and the switch to CMake (as
   `NMZ_CUDA` does).
3. Test it the way the CUDA backend is tested (§11.7): every op against
   the CPU on random inputs, the gradient checks of every layer type,
   training parity. The engine, the layers and the file format need no
   change; `nmz_backend_register` also adds a backend from outside the
   library (the test suite's fake device is one).

### 11.7 The CUDA backend

```
 nmz__cuda_backend()            src/cuda/backend_cuda.cu, behind extern "C"
   init      driver → device → compute capability → probe kernel →
             cuBLAS (loaded at run time) → stream, handle, pinned slots
   memory    cudaMalloc under a cap, every block in a table (used, peak)
   GEMM      cuBLAS SGEMM; attention: strided batched SGEMM per batch row
   the rest  grid-stride kernels (element-wise), one warp per row
             (softmax, LayerNorm, losses, short attention), two-pass
             double reductions, one block per row for whole sequences
             (rnn, lstm, gru)
   scalars   queued into pinned memory, valid after sync
```

Building and choosing it: `-DNMZ_CUDA=ON`, `NMZ_CUDA_ARCHITECTURES`
(decision 139), `nmz_backend_select("cuda" | "auto")`. What runs where:
the dense and convolution layers compose cuBLAS products with bias and
activation kernels (im2col over the whole batch and one GEMM per pass);
the recurrent layers use the fused sequence kernels (decision 146);
attention uses batched products for training and warp kernels for
generation (decision 147); the optimizers, clipping, normalisation and the
training loop's gathers are kernels; nothing in an epoch moves to the host
but the batch loss (decision 136). Tests: decision 141. Memory: decision
142. Determinism: decision 144.

Deferred (not in 2.1): half precision and tensor cores (the GTX 1660 Ti
has none); several GPUs; a persistent kernel for whole recurrent stacks;
cuDNN (another large run-time dependency, for convolutions that are
already cuBLAS-bound here); CUDA graphs to cut the remaining launch cost of
short batches.

### 11.8 Threads

The library's own state is per net; what is shared is the backend
registry, the process-wide selection (one lock) and the CUDA device
(decision 158). A thread chooses its device with
`nmz_backend_select_thread`; the CUDA backend gives each thread a stream,
a cuBLAS handle and buffers of its own, so nets on different threads run
side by side, on the CPU and on the GPU, with the results of sequential
runs. A net belongs to one thread at a time and may move between calls.
