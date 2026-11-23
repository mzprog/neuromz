# NeuroMZ v2 — Design

NeuroMZ v2 is an **educational** neural-network engine in C11. Its job is to be
*correct and inspectable* first, fast second. A beginner can build and train a
network with one command; an advanced student can configure every layer and
training knob; and after (or during) training the engine can expose every
intermediate value — `z`, `a`, `δ`, gradients, weight updates — so the math
can be printed now and visualised later.

Status of this document: written before implementation, then corrected where
the implementation taught us something (see the git history of this file and
decisions 27–33, which were added during implementation).

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
  loss.c               MSE, softmax+cross-entropy
  net.c                nmz_net: layer stack, forward/backward, optimizer step
  train.c              epochs, shuffling, mini-batches, stop rule, evaluation
  trace.c              trace events, recorder, JSON-lines export
  gradcheck.c          numerical gradient check (public: it is a teaching tool)
  io.c                 portable .nmz save/load, CRC32, byte writer/reader
  dataset.c            dataset type + CSV loader
  defaults.c           "quick mode" defaults and the reasons for them
  internal.h           private helpers shared by the library sources
cli/                   the `neuromz` program (may use POSIX/terminal code)
  cli.h                CLI-internal declarations
  main.c               entry point, REPL, argv dispatch (-m/-o/--ascii)
  commands.c           new/show/train/test/predict/save/load/trace/step/...
  print.c              table + formula printer for trace events
  line.c               dynamic line reader and tokenizer (no fixed buffers)
tests/                 test_*.c (unit, gradient checks, convergence, I/O),
                       cli_smoke.sh (CLI abuse test); all run under sanitizers
examples/              xor.csv, xor_and.csv, iris.csv (+ 120/30 train/test
                       split), api_xor.c, run_examples.sh, benchmark.sh
docs/DESIGN.md         this file
legacy/                v1 sources (2017–2020), unchanged, tag `v1-legacy`
Makefile               lib, cli, test, sanitize, examples, clean
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
| update (per param) | `value` (before), `grad`, `velocity_prev`, `delta` (= new velocity); scalars `lr`, `momentum`; `detail` = param name |

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
                  nmz_net_compute_loss(net, t, &loss)
/* train */       nmz_train_opts (max_epochs, batch_size, learning_rate, momentum,
                  shuffle, target_loss, on_epoch callback);
                  nmz_net_train(net, ds, &opts, &result);
                  nmz_net_evaluate(net, ds, &loss, &accuracy);
                  nmz_net_train_config(net)  /* defaults stored with the model */
/* data */        nmz_dataset_load_csv(&ds, path, &csv_opts); nmz_dataset_create;
                  nmz_dataset_free
/* io */          nmz_net_save(net, path); nmz_net_load(&net, path)
/* trace */       nmz_net_set_trace; nmz_trace_recorder_*; nmz_trace_event_write_json
/* teaching */    nmz_net_gradcheck(net, x, t, eps, &result)
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

## 7. File format (`.nmz`, format version 2)

All integers are fixed-width **little-endian**; floats are IEEE-754 binary32
(`f32`) / binary64 (`f64`) stored little-endian. Strings are `u32 length` +
bytes (no terminator, max 4096). The writer never dumps C structs, so padding,
`sizeof(long)` and host endianness do not matter.

```
offset  field
0       magic        8 bytes: 89 4E 4D 5A 0D 0A 1A 0A  ("\x89NMZ\r\n\x1a\n")
8       u32          format version = 2
12      u32          flags = 0 (reserved)
        u32          input rank R, then R × u32 input dims (per sample)
        u32          loss kind (0 = mse, 1 = cross_entropy)
        u64, u64     PRNG state, PRNG increment (resume reproducibly)
        u64          step counter
        train config u32 max_epochs, u32 batch_size, f32 learning_rate,
                     f32 momentum, u32 shuffle, f64 target_loss
        u32          class-name count C, then C × string
        u32          layer count N, then per layer:
                       string   type ("dense", "activation", ...)
                       u32      config byte length, then config bytes
                       u32      parameter count P, then per parameter:
                                string name, u32 rank, rank × u32 dims,
                                product(dims) × f32
end     u32          CRC-32 (IEEE, reflected 0xEDB88320) of every byte above
```

The magic follows PNG's trick: the high byte catches 7-bit transfers, the
CR-LF/LF pair catches newline translation, `0x1A` stops `type` on Windows.
The loader checks magic, version, every count against sane limits, every
parameter shape against what the layer expects after `setup`, and the CRC.
Optimizer velocity is *not* saved (decision 17).

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
