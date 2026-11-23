# NeuroMZ v2

**An educational neural-network engine in C.** Build a network in one
command, train it, and then *look inside*: NeuroMZ can print every number of a
forward and backward pass — `z`, `a`, the loss, every `δ`, every gradient and
every weight update — next to the formula that produced it.

* `libneuromz`: a small, dependency-free C11 library (libc + libm only, no globals)
* `neuromz`: a command-line program with a beginner path and an advanced path
* Every derivative is verified against numerical gradients; everything runs clean
  under AddressSanitizer and UndefinedBehaviorSanitizer.

NeuroMZ started in 2017 as a beginner's project. That code is preserved in
[`legacy/`](legacy/) (git tag `v1-legacy`); v2 is a rewrite. The design is in
[`docs/DESIGN.md`](docs/DESIGN.md).

---

## Quick start

```bash
make            # builds build/libneuromz.a and build/neuromz
make test       # unit tests, gradient checks, convergence tests
make sanitize   # the same tests + a CLI smoke test under ASan + UBSan
make examples   # trains XOR, XOR+AND and Iris
```

Requirements: a C11 compiler and `make`. Nothing else.

---

## The beginner path

Give the layer sizes; NeuroMZ picks everything else **and tells you why**:

```
$ ./build/neuromz new 2 4 1 -o xor.nmz
Created a network: 2 → 4 → 1   (17 parameters)
  hidden 2->4      tanh           [default] zero-centred and smooth; in small nets it trains more reliably than relu, whose units can die
    init           xavier         [default] Glorot/Xavier: Var(W) = 2/(fan_in+fan_out), keeps tanh/sigmoid layers in their useful range
  output 4->1      sigmoid        [default] 1 output -> squashed into (0,1), a natural fit for 0/1 targets
    init           xavier         [default] Glorot/Xavier: Var(W) = 2/(fan_in+fan_out), keeps tanh/sigmoid layers in their useful range
  loss             mse            [default] squared error is the textbook pairing for a sigmoid output: delta = (y-t)*f'(z)
  learning rate    0.5            [default] 0.5 suits squared error on sigmoid outputs, whose gradients are small (<= 0.25*error)
  momentum         0.9            [default] 0.9 averages recent gradients: smoother steps and faster progress across plateaus
  batch size       32             [default] mini-batches of 32 (full batch when the dataset is smaller): stable gradients, ...
  max epochs       5000           [default] at most 5000 passes over the data, so training always ends
  target loss      0.001          [default] stop when the average 1/2*error^2 <= 0.001 (outputs within ~0.045 of their targets)
  shuffle          on             [default] visit samples in a new random order every epoch
  seed             42             [default] same seed => same initial weights and same shuffling, every run
Saved to xor.nmz

$ ./build/neuromz -m xor.nmz train examples/xor.csv -o xor.nmz
Training on examples/xor.csv: 4 samples, 2 input(s), 1 output(s)
  lr 0.5, momentum 0.9, batch 32, max epochs 5000, target loss 0.001, shuffle on
  (batch 32 > 4 samples: using the full dataset as one batch)
  epoch      1   loss 0.119855     accuracy  50.0%
  epoch    172   loss 0.000993896  accuracy 100.0%
Converged after 172 epoch(s): loss 0.000993896 <= target 0.001, accuracy 100.0%
Saved to xor.nmz

$ ./build/neuromz -m xor.nmz predict 1 0
output:  0.953803
```

With two or more outputs, quick mode uses softmax + cross-entropy (one class
per sample). A CSV whose last column is a text label is one-hot encoded
automatically, and the class names are stored with the model:

```
$ ./build/neuromz new 4 8 3 -o iris.nmz
$ ./build/neuromz -m iris.nmz train examples/iris_train.csv -o iris.nmz
$ ./build/neuromz -m iris.nmz test examples/iris_test.csv
Tested on examples/iris_test.csv: 30 samples
  loss 0.0458878   accuracy 100.0%
  confusion matrix (rows = true class, columns = predicted)
                     setosa versicolor  virginica
          setosa         10          0          0
      versicolor          0         10          0
       virginica          0          0         10
$ ./build/neuromz -m iris.nmz predict 6.3 2.9 5.6 1.8
output:  setosa 0.0000  versicolor 0.0298  virginica 0.9702
predicted class: virginica (97.0%)
```

Run `neuromz` with no arguments for an interactive shell (same commands; type
`help`). Scripts can be piped into it: `./build/neuromz < lesson.txt`.

## The advanced path

Every layer is `SIZE[:ACTIVATION[:INIT]]`, and every training knob is an
option. What you set is marked `[yours]`; the rest is still explained.

```bash
./build/neuromz new 4 16:relu:he 8:leaky_relu=0.05 3:softmax \
    --loss ce --lr 0.01 --momentum 0.9 --batch 16 --epochs 2000 --target-loss 0.05 --seed 7 -o iris.nmz
```

| | choices |
|---|---|
| activations | `linear`, `sigmoid`, `tanh`, `relu`, `leaky_relu[=ALPHA]`, `softmax` |
| init | `auto` (He before relu/leaky relu, Xavier otherwise), `xavier`, `he` |
| loss | `mse` (½·squared error), `cross_entropy`/`ce` (fused with a softmax output) |
| training | `--lr`, `--momentum`, `--batch`, `--epochs`, `--target-loss`, `--no-shuffle`, `--seed` |

* `train FILE.csv [options]` overrides the stored settings for one run;
  `set --lr 0.1 ...` changes the settings saved with the model.
* `show` / `show weights`, `test FILE.csv`, `predict X...`, `reinit [SEED]`,
  `save [FILE]`, `load FILE`, `clear`, `help [COMMAND]`.
* One-shot use composes in shell scripts: `-m FILE` loads a model before the
  command, `-o FILE` saves it afterwards. `--ascii` replaces Greek letters.

Training is mini-batch SGD with classical momentum (`v ← μv − η∇L; w ← w + v`).
Each epoch shuffles the data (seeded), runs every mini-batch, then measures the
loss over the **whole** training set; training stops when that loss reaches the
target or at the epoch limit. Gradients are always computed for the whole
network before any weight changes.

## The trace command: see the math

`trace` runs **one sample** forward and backward and prints every value with
its formula. Nothing changes; `step` does the same and applies the update.

```
$ ./build/neuromz new 2 2 1 -o tiny.nmz
$ ./build/neuromz -m tiny.nmz trace 1 0 --target 1
FORWARD PASS   input -> output
[layer 0] dense 2 → 2   (forward)
  z_j = b_j + Σ_i x_i·W_ij        x = this layer's input, W_ij = weight from input i to unit j
  z_1 = 0.0000 + 1.0000·1.1884 + 0.0000·0.1973 = 1.1884
  z_2 = 0.0000 + 1.0000·1.1158 + 0.0000·(-0.2618) = 1.1158
[layer 1] activation tanh   (forward)
  a_1 = tanh(1.1884) = 0.8301
  ...
LOSS  (mse)
  L = ½ Σ_k (y_k − t_k)²       y = network output, t = target
  (y_1 − t_1)² = (0.6656 − 1.0000)² = 0.1118
  L = ½ · 0.1118 = 0.0559
BACKWARD PASS   output -> input: every gradient is computed first
[layer 3] activation sigmoid   (backward)
  δ_j = ∂L/∂a_j · f'(z_j)        f'(z) = σ(z)·(1 − σ(z)) = a·(1 − a)
  f'(z_1) = 0.6656·(1 − 0.6656) = 0.2226
  δ_1 = -0.3344 · 0.2226 = -0.0744
[layer 2] dense 2 → 1   (backward)
  ∂L/∂W_ij = x_i · δ_j        e.g. ∂L/∂W_11 = 0.8301 · (-0.0744) = -0.0618
  ∂L/∂x_i = Σ_j W_ij · δ_j        (passed down: it is ∂L/∂a of the layer below)
  ∂L/∂x_1 = 0.8229·(-0.0744) = -0.0612
  ...
WEIGHT UPDATE (preview only: not applied; use `step`)   η = 0.5, μ = 0.9
  v ← μ·v − η·∂L/∂w        w ← w + v
[layer 2] dense W  (2 values)
       i,j        w old        ∂L/∂w        v old    v new = Δ        w new
       1,1       0.8229      -0.0618       0.0000       0.0309       0.8538
```

* `--target` can be a class name: `trace 5.1 3.5 1.4 0.2 --target setosa`.
* With softmax + cross-entropy the trace shows the combined gradient `δ = p − t`.
* `--full` prints every row of large layers; `--json FILE` writes the raw trace
  events as JSON lines (one object per event, with every tensor), ready for a
  plotting script or a future GUI.
* `gradcheck X... --target T...` compares every backprop gradient with
  `(L(w+ε) − L(w−ε)) / 2ε` — a way to *verify* the math, not just read it.

## Using the library from C

```c
#include "neuromz.h"

nmz_net *net;
nmz_dataset *ds;
nmz_train_result res;
int sizes[] = {2, 4, 1};

nmz_dataset_load_csv(&ds, "examples/xor.csv", NULL);
nmz_net_create_quick(&net, 3, sizes, /*seed*/ 42, NULL);    /* beginner defaults */
if (nmz_net_train(net, ds, NULL, &res) != NMZ_OK)
    fprintf(stderr, "%s\n", nmz_last_error());
nmz_net_save(net, "xor.nmz");
nmz_net_free(net);
nmz_dataset_free(ds);
```

Build against `build/libneuromz.a` with `-Iinclude -lm`. The advanced API
(explicit layers, `nmz_train_opts`, forward/backward/update by hand, trace
callbacks and the trace recorder) is shown in
[`examples/api_xor.c`](examples/api_xor.c); all functions are documented in
[`include/neuromz.h`](include/neuromz.h). New layer types implement the vtable
in [`include/neuromz_layer.h`](include/neuromz_layer.h).

## Results

From `sh examples/benchmark.sh ./build/neuromz 20` (seeds 1–20, defaults
unless shown). Iris: 120 training rows, accuracy measured on the 30 held-out
rows of `examples/iris_test.csv`.

| example (seeds 1..20) | converged | median epochs | mean test accuracy |
|---|---|---|---|
| XOR, quick `new 2 4 1` | 20/20 | 176 | – |
| XOR, minimal `new 2 2 1` | 10/20 | 249 | – |
| XOR+AND, `new 2 4 2:sigmoid --loss mse` | 20/20 | 202 | – |
| Iris, quick `new 4 8 3` | 20/20 | 178 | 99.0% |
| Iris, `new 4 16:relu:he 8:leaky_relu=0.05 3:softmax`, lr 0.01, batch 16 | 20/20 | 109 | 98.6% |

The minimal 2-2-1 XOR network fails for half the seeds: with only two hidden
units the loss surface has well-known local minima. Two extra hidden units make
it reliable — a good thing to try with `trace` and `reinit`.

"Converged" means the whole-training-set loss reached the target (0.001 for
MSE, 0.07 for cross-entropy) before 5000 epochs.

## Tests

`make test` runs 23 tests (~100 000 checks) in well under a second:

* **numerical gradient checks** for Dense, every activation (incl. the softmax
  Jacobian), MSE, fused softmax+cross-entropy (one-hot and soft targets), a deep
  mixed network, batch size 1 and rank-2 samples;
* hand-computed forward/backward/SGD/momentum values;
* **XOR convergence over 20 seeds** (quick mode and relu/He);
* **XOR+AND with two outputs** — the case v1 got wrong — including a check that
  the weights into the second output actually change;
* determinism (same seed ⇒ bit-identical training), early stop, NaN detection;
* trace event order, recorder deep copies and JSON export;
* save/load round-trip (bit-identical predictions), little-endian byte layout,
  checksum and truncation detection, v1-file detection;
* CSV parsing and error messages; every API entry point with `NULL`/bad input.

`make sanitize` runs all of that plus `tests/cli_smoke.sh` (66 CLI cases,
including no-network commands, malformed input, a 300 KB line and random binary
input) under ASan + UBSan with leak detection.

## Layout

```
include/neuromz.h        public API          src/        libneuromz (portable C11)
include/neuromz_layer.h  layer-author API    cli/        the neuromz program
docs/DESIGN.md           design + decisions  tests/      test suite + CLI smoke test
examples/                data, C demo, scripts           legacy/     v1 (2017–2020)
```

Model files (`.nmz`) use a portable little-endian format with a magic number,
version and CRC-32 (see DESIGN §7).

## Roadmap

This phase covers dense networks. The data model is already shaped for what
comes next (DESIGN §3): convolution, a character-level RNN and LSTM
(backpropagation through time inside the layer), embeddings, and a tiny
Transformer trained on text; Adam; input normalisation; a GUI that consumes the
trace events.
