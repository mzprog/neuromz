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
decisions 91–110.

---

## 1. Directory layout

```
include/
  neuromz.h            public API (everything a library user needs)
  neuromz_layer.h      layer-author API (vtable, params, trace emit, byte I/O)
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
                       naive reference, optional OpenMP (make OMP=1)
  attention.c          causal multi-head self-attention (+ KV cache)
  transformer.c        position encoding, LayerNorm, feed-forward, the
                       pre-LN block (composite layer), tied output
  defaults.c           "quick mode" defaults and the reasons for them
  internal.h           private helpers shared by the library sources
cli/                   the `neuromz` program (may use POSIX/terminal code)
  cli.h                CLI-internal declarations
  main.c               entry point, REPL, argv dispatch (-m/-o/--ascii), clock/tty
  cmd_new.c            `new`: layer-spec grammar, presets, explanations
  commands.c           show/train/test/predict/save/load/trace/step/...
  util.c               option parsing, dataset resolution (CSV/IDX/mnist)
  print.c              formula printer for dense/activation/loss/SGD events
  print_conv.c         conv/pool/flatten/dropout/normalize/Adam printers, heatmaps
  print_seq.c          a character rnn/lstm/gru unrolled through time: every gate,
                       gate heatmaps, BPTT, vanishing gradients along h and c
  print_attn.c         a Transformer's attention at one position with real
                       numbers, heatmaps, head patterns, the residual stream
  cmd_seq.c            `compare` (gradient survival side by side), `show gates`,
                       `show attention`, `show params`
  line.c               dynamic line reader and tokenizer (no fixed buffers)
tests/                 test_*.c (unit, gradient checks, convergence, I/O),
                       cli_smoke.sh (CLI abuse test); all run under sanitizers
examples/              xor.csv, xor_and.csv, iris.csv (+ 120/30 train/test
                       split), names_en.txt, names_ar.txt, DATASETS.md,
                       api_xor.c, run_examples.sh, benchmark.sh,
                       benchmark_mnist.sh,
                       benchmark_gated.sh, benchmark_transformer.sh,
                       bench_conv.c, bench_matmul.c
scripts/get_mnist.sh   downloads MNIST into data/ (gitignored), verifies MD5
scripts/get_text.sh    downloads Alice (Gutenberg #11) and Kalila wa-Dimna
                       (Arabic Wikisource, fixed revisions) into data/,
                       verifies SHA-256; strip_wikitext.awk removes markup
docs/DESIGN.md         this file
legacy/                v1 sources (2017–2020), unchanged, tag `v1-legacy`
Makefile               lib, cli, test, sanitize, examples, data, bench, clean
```

Build products go to `build/` (`build/libneuromz.a`, `build/neuromz`,
`build/test_neuromz`); the sanitizer build goes to `build/san/`; `make
OMP=1 …` builds with OpenMP into `build/omp/` (and `build/san-omp/`).

The library never includes POSIX headers, never touches the terminal, never
prints, and never calls `exit()`/`abort()`. The CLI is the only place allowed
to do those things.

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

`neuromz` with no arguments starts an interactive shell (prompt only when
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
| heap overflows, use-after-free, NULL derefs | tests + CLI smoke run under ASan/UBSan in `make sanitize` |
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
