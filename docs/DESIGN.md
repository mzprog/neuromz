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
normalization, MNIST) added decisions 34–50.

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
  line.c               dynamic line reader and tokenizer (no fixed buffers)
tests/                 test_*.c (unit, gradient checks, convergence, I/O),
                       cli_smoke.sh (CLI abuse test); all run under sanitizers
examples/              xor.csv, xor_and.csv, iris.csv (+ 120/30 train/test
                       split), api_xor.c, run_examples.sh, benchmark.sh,
                       benchmark_mnist.sh, bench_conv.c
scripts/get_mnist.sh   downloads MNIST into data/ (gitignored), verifies MD5
docs/DESIGN.md         this file
legacy/                v1 sources (2017–2020), unchanged, tag `v1-legacy`
Makefile               lib, cli, test, sanitize, examples, data, bench, clean
```

Build products go to `build/` (`build/libneuromz.a`, `build/neuromz`,
`build/test_neuromz`); the sanitizer build goes to `build/san/`.

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

## 3. Planned: sequences (BPTT) and embeddings

Not implemented in this phase, but the data model is shaped for them:

* **The layer owns its time loop.** An RNN/LSTM layer receives `[B, T, F]`
  and returns `[B, T, H]`. Its forward pass keeps the per-timestep states
  (`h_t`, gates, …) in its private `self` cache; its backward pass walks
  `t = T-1 … 0` (backpropagation through time) and **accumulates** into the
  shared weight gradients. The engine still sees one forward and one backward
  call per layer, so `net.c` needs no change.
* **Gradient accumulation convention.** The engine zeroes every gradient
  before a backward pass and layers *add* into them. Dense would not need
  this, but recurrent layers (weights shared over time) and tied embeddings
  (same matrix used twice) do, so it is the rule from day one.
* **Truncated BPTT / stateful RNNs** will add two optional vtable entries
  (`reset_state`, `detach_state`). Optional entries are `NULL`-able; the engine
  checks before calling, so existing layers are unaffected.
* **Embedding** takes `[B, T]` token ids (floats, exact < 2^24) and returns
  `[B, T, D]`. As the first layer it receives `dx == NULL` and computes no input
  gradient. Its weight gradient is row-sparse but stored dense (fine at
  char-level / tiny-Transformer scale).
* **Losses work on rows.** MSE and cross-entropy treat the last dimension as
  "features/classes" and every leading dimension as rows, so a `[B, T, V]`
  language-model output is just `B·T` rows. This is already how `loss.c` is
  written.
* **Attention / residual connections** break the pure stack. The top level
  stays a sequential stack; non-sequential wiring (residual add, layer norm,
  multi-head split) lives inside *composite layers* (e.g. a
  `transformer_block` layer that owns sub-layers and implements the same
  vtable). The trace system already carries a `layer` index and a `detail`
  string, so composite layers can emit events for their sub-steps.
* **Trace events carry a `time_step` field** (`-1` for non-sequence layers) so
  per-timestep values of an RNN can be shown later without changing the event
  format.
* **Training mode flag.** `nmz_ctx.training` is passed to every layer so
  dropout/batch-norm style layers can behave differently in evaluation.

---

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
| dropout | forward `x`, `y`, `mask` (training only); backward `dL/dy`, `dL/dx`, `mask`; scalars `rate`, `training` |
| update, SGD (per param) | `value` (before), `grad`, `velocity_prev`, `delta` (= new velocity); scalars `lr`, `momentum`; `detail` = param name |
| update, Adam (per param) | `value` (before), `grad`, `m`, `v` (new moments), `m_hat`, `v_hat`, `delta`; scalars `lr`, `beta1`, `beta2`, `epsilon`, `t` |

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
                  nmz_net_optimizer_step(net, &opts, lr)  /* SGD or Adam */
                  nmz_net_optimizer_preview(net, &opts, lr)
                  nmz_net_forward_mode(net, x, training, &y)
                  nmz_net_compute_loss(net, t, &loss)
/* train */       nmz_train_opts (max_epochs, batch_size, learning_rate, momentum,
                  shuffle, target_loss, optimizer, beta1/beta2/epsilon,
                  schedule/step_size/gamma/min_lr, on_epoch/on_batch callbacks);
                  nmz_schedule_lr(&opts, epoch); nmz_net_fit_normalization
                  nmz_net_train(net, ds, &opts, &result);
                  nmz_net_evaluate(net, ds, &loss, &accuracy);
                  nmz_net_train_config(net)  /* defaults stored with the model */
/* data */        nmz_dataset_load_csv(&ds, path, &csv_opts); nmz_dataset_create;
                  nmz_dataset_load_idx(&ds, images, labels, limit);
                  nmz_dataset_truncate; nmz_dataset_free
/* io */          nmz_net_save(net, path); nmz_net_save_ex(net, path, flags);
                  nmz_net_load(&net, path)
/* trace */       nmz_net_set_trace; nmz_trace_recorder_*; nmz_trace_event_write_json
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
                                "maxpool2d", "flatten", "dropout")
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
