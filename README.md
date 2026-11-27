# NeuroMZ

[![CI](https://github.com/mzprog/neuromz/actions/workflows/ci.yml/badge.svg)](https://github.com/mzprog/neuromz/actions/workflows/ci.yml)

**An educational neural-network engine in C11.** A small, dependency-free
library (libc + libm, no globals) whose every forward and backward pass can
be *watched*: each layer reports the numbers it computes — `z`, `a`, the
loss, every `δ`, every gradient and every weight update — as trace events,
and every derivative is checked against numerical gradients.

* Dense, Conv2D (naive and im2col), MaxPool2D, Flatten, Dropout; SGD with
  momentum, Adam, AdamW; step, cosine and warm-up + cosine schedules; input
  normalisation stored with the model
* Embedding, RNN, **LSTM** and **GRU** (backpropagation through time inside
  the layer, truncated BPTT with a carried state, stacking, clipping,
  orthogonal initialisation); UTF-8 text with code-point vocabularies,
  Arabic included
* A decoder-only **Transformer** (position encodings, LayerNorm, causal
  multi-head attention, GELU feed-forward, pre-LN blocks, weight tying), a
  cache-blocked matrix product bit-identical to the naive loops, and a KV
  cache for generation that is bit-identical to recomputation
* Portable model files (little-endian, versioned, CRC-32): files from every
  earlier phase still load
* **Device backends** (2.1): the CPU by default (bit-identical to 2.0)
  and, built with `-DNMZ_CUDA=ON`, NVIDIA GPUs — every layer type trains
  there (cuBLAS + the library's own kernels), 10–240× faster on the models
  below; the public API stays C, traces still work, model files are the
  same on every device, and a CUDA build still runs without a GPU
* **Threads** (2.2): several nets train at once in several threads, each
  on its own device (`nmz_backend_select_thread`), with the results of
  sequential runs — on the CPU and on the GPU; and models **described in
  words** (`nmz_model_new`: the layer grammar and presets of the program's
  `new`, with the explanation of every default)
* Builds with GCC, Clang and MSVC (Windows natively), optional OpenMP;
  tested on Linux, Windows and macOS, under ASan + UBSan

**See it work:** the command-line program [`neuromz`](https://github.com/mzprog/neuromz-cli) is built on
this library — quick start, walkthroughs (MNIST, names, Alice, Arabic prose,
RNN vs LSTM vs GRU, *how attention works, step by step*) and the benchmark
results are in [its README](https://github.com/mzprog/neuromz-cli#readme).
**Over the network:** [`neuromz-server`](https://github.com/mzprog/neuromz-server)
hosts models and runs training and inference jobs on the CPU or a GPU, and
[`neuromz-protocol`](https://github.com/mzprog/neuromz-protocol) is its
protocol — protocol 2, a binary protocol of its own over TCP (frames, a
MessagePack subset, an HMAC challenge, pushed progress; its C library
`nmzp`, shared test vectors and a conformance suite). The program's remote
mode speaks it; a pure-PHP client library (`neuromz-php`, planned) will
let web applications use a server.

NeuroMZ started in 2017 as a beginner's project. That code is preserved in
[`legacy/`](legacy/) (git tag `v1-legacy`); v2 is a rewrite. The design and
every decision behind it are in [`docs/DESIGN.md`](docs/DESIGN.md); changes
in [`CHANGELOG.md`](CHANGELOG.md).

---

## Build and install

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build                       # libneuromz, the tests, the C examples
cmake --build build --target test         # build, then run the tests (= ctest --test-dir build)
cmake --build build --target sanitize     # GCC/Clang: the same under ASan + UBSan (in build-san/)
cmake --build build --target data         # MNIST (~11 MB) into data/, checksums verified
cmake --build build --target data-text    # Alice in Wonderland and Kalila wa-Dimna into data/
cmake --build build --target bench        # slow: matrix-product and convolution benchmarks
cmake --install build --prefix /usr/local # headers, libneuromz, a CMake package
```

Options: `-DNMZ_CUDA=ON` (the GPU backend, [below](#on-an-nvidia-gpu-cuda)),
`-DNMZ_OPENMP=ON` (multi-threaded matrix products and attention heads; same
results), `-DNMZ_SANITIZE=ON`, `-DNMZ_WERROR=OFF`. With MSVC,
use `--config Release` for the build and `ctest -C Release`. Requirements:
CMake ≥ 3.21 and a C11 compiler (GCC, Clang, or MSVC from Visual Studio
2019 16.8 on); the data targets use `sh` with `curl`/`wget` on Unix-like
systems and PowerShell on Windows — both scripts check the same SHA-256
sums, including those of the files they produce. The MNIST regression test
runs only when the data has been downloaded.

Using it from another CMake project:

```cmake
find_package(neuromz 2.2 REQUIRED)          # after cmake --install
target_link_libraries(app PRIVATE neuromz::neuromz)

# or without installing:
include(FetchContent)
FetchContent_Declare(neuromz GIT_REPOSITORY https://github.com/mzprog/neuromz.git GIT_TAG v2.2.0)
FetchContent_MakeAvailable(neuromz)
target_link_libraries(app PRIVATE neuromz::neuromz)
```

### On an NVIDIA GPU (CUDA)

```bash
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DNMZ_CUDA=ON
cmake --build build-cuda
ctest --test-dir build-cuda -L gpu        # every op against the CPU, gradient checks, training parity
```

Building needs the CUDA Toolkit (tested with 13.5: `nvcc` and the CUDA
runtime, which is linked statically). cuBLAS is loaded when a program first
selects the GPU, so a CUDA build starts on any machine: without an NVIDIA
GPU or driver, `"cuda"` fails with the reason and `"auto"` uses the CPU.
To compute on a GPU, a machine needs the NVIDIA driver and cuBLAS
(`cublas64_13.dll` and `cublasLt64_13.dll` on Windows, `libcublas.so.13`
and `libcublasLt.so.13` on Linux: installed with the CUDA Toolkit, or
placed next to the program).
GPUs from compute capability 7.5 (Turing, the oldest CUDA 13 supports).
`-DNMZ_CUDA_ARCHITECTURES` chooses the code compiled in: by default the GPU
of the building machine (else 75); for a binary for others list them, e.g.
`"75;86;89;120"` (Turing, Ampere, Ada, consumer Blackwell; add 80 and 90 for
data-centre GPUs). Each entry also embeds PTX, so newer GPUs run it through
the driver's JIT compiler. The kernels are compiled behind a C ABI; the
headers stay C, and programs using the library are still compiled as C.

Device memory: the library holds at most what was free on the GPU when it
started (Windows would otherwise page GPU memory out to the host and
training would crawl); `NMZ_CUDA_MEMORY=MiB` or `nmz_backend_memory_limit`
change that. Running out is an error with a message ("out of GPU memory:
… Try a smaller batch size"), never a crash, and the device stays usable.
How a backend works and how to add another (HIP, Vulkan, OpenCL):
[DESIGN §11](docs/DESIGN.md#11-device-backends-release-21).

## Using the library from C

```c
#include "neuromz.h"

nmz_net *net;
nmz_dataset *train;
nmz_train_result res;
nmz_train_opts opts = nmz_train_defaults();
int shape[3] = {1, 28, 28};

nmz_dataset_load_idx(&train, "data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte", 0);
nmz_net_create(&net, 3, shape, /*seed*/ 42);
nmz_net_add(net, nmz_layer_conv2d(8, 3, 3, /*stride*/ 1, /*pad*/ 0, NMZ_INIT_AUTO));
nmz_net_add(net, nmz_layer_activation(NMZ_ACT_RELU, 0));
nmz_net_add(net, nmz_layer_maxpool2d(2, 0));
nmz_net_add(net, nmz_layer_flatten());
nmz_net_add_dense(net, 10, NMZ_ACT_SOFTMAX, NMZ_INIT_AUTO);
nmz_net_set_loss(net, NMZ_LOSS_CROSS_ENTROPY);
nmz_net_set_normalization(net, NMZ_NORM_ZSCORE);
opts.optimizer = NMZ_OPT_ADAM;
opts.learning_rate = 0.001f;
opts.max_epochs = 2;
opts.target_loss = 0;
if (nmz_net_train(net, train, &opts, &res) != NMZ_OK)
    fprintf(stderr, "%s\n", nmz_last_error());
nmz_net_save(net, "small_cnn.nmz");
```

Devices: nets compute on the selected backend (the CPU unless told
otherwise); every result the API hands back is on the host.

```c
nmz_device_info di;
nmz_backend_select("auto");               /* "cpu", "cuda", or "auto" */
nmz_backend_info(&di);                    /* e.g. "cuda", "NVIDIA GeForce GTX 1660 Ti", 7.5, 6144 MiB */
printf("%s: %s (%s)\n", di.backend, di.device, di.note);   /* note: why this device */
/* ... train as above; di.mem_used / mem_peak / mem_limit: the library's GPU memory */
```

Threads: a net, dataset, vocabulary or recorder belongs to one thread at a
time; different nets train at once in different threads. A thread can pick
its own device, which wins over the process-wide choice in that thread:

```c
static int worker(void *arg) {             /* one per job, e.g. in a server */
    nmz_backend_select_thread("cuda");     /* this thread only; NULL: back to the process-wide one */
    /* ... build, train and save this job's net ... */
    nmz_backend_select_thread(NULL);
    return 0;
}
```

The CUDA backend gives every thread its own stream, cuBLAS handle and
buffers; the device and its memory cap are shared. Returning non-zero from
`opts.on_batch` or `opts.on_epoch` stops training cleanly after that batch
(`res.stopped`): that is how a server cancels a job. The rules are at the
top of [`include/neuromz.h`](include/neuromz.h) and in
[DESIGN §11.8](docs/DESIGN.md#118-threads).

Models described in words, as the program's `new` takes them:

```c
static void line(const char *text, void *user) { puts(text); }   /* the explanation, line by line */
char *const argv[] = {"2", "8:tanh", "1:sigmoid", "--loss", "mse"};
nmz_model_env env = {0};
env.print = line;
if (nmz_model_new(5, argv, &env, &net) != NMZ_OK) fprintf(stderr, "%s\n", nmz_last_error());
/* presets: nmz_preset_count / nmz_preset_name / nmz_preset_summary;
 * training options as the program spells them: nmz_train_option, nmz_train_describe */
```

All functions are documented in [`include/neuromz.h`](include/neuromz.h).
The beginner API (`nmz_net_create_quick`), forward/backward/update by hand,
trace callbacks and the trace recorder are shown in
[`examples/api_xor.c`](examples/api_xor.c); a CNN on MNIST in
[`examples/api_mnist.c`](examples/api_mnist.c). New layer types implement
the vtable in [`include/neuromz_layer.h`](include/neuromz_layer.h). The
trace events — what a layer reports and under which names — are listed in
[DESIGN §4](docs/DESIGN.md#4-trace-system).

## Performance

One core of an Intel Core i7-7500U (a 2017 laptop, 2.7 GHz), GCC `-O3`.

Convolution is implemented twice: naive loops that mirror the formula, and
im2col (receptive fields copied into a matrix, then contiguous matrix
products; the default). Both pass the same gradient checks and agree to
1e-5. One epoch of the CNN on 10 000 images (`bench_conv`):

| conv implementation | 1 epoch, 10 000 images |
|---|---|
| naive loops | 10.3 s |
| im2col (default) | 5.0 s (2.1× faster; 2.1–2.4× over repeated runs) |

With `-O2` instead of `-O3` im2col was only ~1.1× faster: GCC 12's `-O2` does
not vectorise the inner loops, so `-O3` is now the default ([DESIGN](docs/DESIGN.md) decision 46).

Attention and the feed-forward net are matrix products; `nmz_matmul`
computes them cache-blocked, keeping a 1 × 32 tile of the result in
registers. It gives **bit-identical** results to the three-loop
reference (same sum order; tested), which is also what makes the KV cache
exact. `bench_matmul` (`--target bench`), one core of the i7-7500U:

| product (M × N × K), as in a D = 64 model | naive loops | blocked, 1 thread | speedup | GFLOP/s | 2 threads (OpenMP) |
|---|---|---|---|---|---|
| `x·W_qkv` 4096 × 192 × 64 | 40.5 ms | 5.75 ms | 7.0× | 17.5 | 3.20 ms (31.5 GFLOP/s) |
| `a·W2` 4096 × 64 × 256 | 71.7 ms | 6.80 ms | 10.5× | 19.8 | 5.97 ms (22.5) |
| `x·W1` 4096 × 256 × 64 | 59.9 ms | 8.98 ms | 6.7× | 14.9 | 4.71 ms (28.5) |
| `dW1 = xᵀ·dz` 64 × 256 × 4096 | 488 ms | 13.5 ms | 36× | 9.9 | 7.01 ms (19.1) |
| `dx = dz·W1ᵀ` 4096 × 64 × 256 | 64.5 ms | 6.93 ms | 9.3× | 19.4 | 3.45 ms (38.9) |

(The naive weight-gradient loop reads A by columns; the blocked version
copies Aᵀ once.) A 4 × 8 register tile, the textbook choice, came out at
4–6 GFLOP/s — GCC kept it on the stack — and the plain row loop at 11–12.

`-DNMZ_OPENMP=ON` builds with OpenMP (off by default; the library stays
dependency-free): the row blocks of each product and the (row, head) pairs
of attention are shared among threads, each number is still computed by
one thread in the same order, so results stay bit-identical (all tests
pass in both builds). One training epoch of a 113 k-weight model: 16.2 s
on one thread, 11.2 s on 2 threads; 4 threads (hyper-threads) 14.6 s.

### On the GPU

An Intel Core i5-8500 (6 cores) and an NVIDIA GeForce GTX 1660 Ti (6 GB,
Turing, no tensor cores), Windows 11, MSVC, CUDA 13.5; `bench_device`
([`examples/bench_device.c`](examples/bench_device.c)): the second epoch of
each model, including the evaluation of the training and validation sets
the CLI reports after every epoch.

| model | weights | CPU, 1 thread | CPU, 6 threads (OpenMP) | GPU | GPU vs 1 / 6 threads | GPU memory, peak |
|---|---|---|---|---|---|---|
| MNIST CNN (the CLI's preset), 60 000 images | 27.6 k | 24.3 s | 24.4 s | **2.35 s** | 10× / 10× | 192 MiB |
| Alice LSTM: embed 32 → lstm 64 | 32 k | 6.67 s | 6.68 s | **0.70 s** | 9.5× / 9.5× | 6 MiB |
| Alice Transformer: 2 blocks of 32, context 128 | 32 k | 8.55 s | 4.13 s | **0.54 s** | 16× / 7.6× | 22 MiB |
| Alice Transformer: 4 blocks of 64, context 128 | 213 k | 33.9 s | 13.7 s | **1.04 s** | 33× / 13× | 65 MiB |
| Alice Transformer: 8 blocks of 512, context 256, batch 16 | 25.4 M | ≈ 36 min ¹ | ≈ 8.7 min ¹ | **12.2 s** (8.9 s ¹) | ≈ 240× / 58× ¹ | 3587 MiB |
| XOR (quick mode `2 4 1`), the whole training | 17 | 0.00 s | | 0.15 s | slower | |
| Iris (4 → 16 → 3), 100 epochs, the whole training | 131 | 0.01 s | | 0.42 s | 40× slower | |

¹ updates only (no evaluation), from 3 (1 thread) and 4 (6 threads) of the
epoch's 32 batches; on the GPU the updates are 8.9 s of the 12.2 s epoch.
The largest model is sized to fit comfortably: 3.5 GB of the 6 GB.

Where the GPU does not help: **tiny nets** train *slower* (XOR in 0.15 s
instead of milliseconds, Iris in 0.42 s instead of 0.01 s) — starting the
device and one kernel launch per operation cost more than all of their
arithmetic, which is one reason the CPU stays the default.
**Small models are launch-bound**: of the LSTM's 0.70 s epoch, 0.12 s are
the updates and the rest is evaluating the text step by step, at 10–15 µs
per launch under Windows' display driver. OpenMP helps only the
Transformers (their time is in the blocked product and the attention
heads), not the CNN or the LSTM. The GPU pays off as models grow: 10× at
30 k weights, ≈ 240× at 25 M, where one CPU core needs over half an hour
per epoch.

Training reaches the same quality on both devices (`ctest -L gpu`; the
same seeds; times are whole runs):

| run | CPU | GPU |
|---|---|---|
| XOR, quick mode (172 epochs) | loss 0.0010, 100% | loss 0.0010, 100% |
| Iris, 100 epochs | loss 0.1965, 93.33% | loss 0.1965, 93.33% |
| MNIST CNN, 1 epoch on 20 000 images | 96.56%, 9.3 s | 96.56%, 1.2 s |
| names RNN, 8 epochs | loss 2.1305, 10.3 s | loss 2.1305, 0.70 s |
| Alice LSTM, 3 epochs | loss 1.8555, 19.5 s | loss 1.8555, 2.2 s |
| Alice Transformer (32 k), 3 epochs | loss 2.4199, 24.9 s | loss 2.4199, 1.7 s |

The numbers agree to 4 digits; a GPU sums in another order (cuBLAS, its
own reductions), so they are not bit-identical. With dropout the devices
draw different masks and train along different random paths (the 213 k
Transformer: validation loss 2.4645 on the CPU, 2.4607 on the GPU). Two
runs on the same GPU give identical model files.

## Results in brief

Measured with the CLI's benchmark scripts ([details, samples, attention
patterns and the gradient-survival comparison](https://github.com/mzprog/neuromz-cli#results)); one core
of the same laptop; validation loss in nats per character (lower is
better).

| task | model | result |
|---|---|---|
| MNIST (10 000 test images) | small CNN, 27.6 k weights, 3 epochs, 114 s | **98.8%** test accuracy |
| MNIST | MLP 784 → 128 → 10, 5 epochs, 100 s | 97.0% |
| English names (5163) | LSTM / GRU / Transformer, ~25 k weights | **2.028** / 2.056 / 2.097 |
| Alice in Wonderland | LSTM / GRU / Transformer, ~32 k weights | 1.614 / **1.580** / 1.633 |
| Alice in Wonderland | Transformer, 4 blocks of 64, 213 k weights, 22 min | **1.564** |
| Kalila wa-Dimna (Arabic) | LSTM / GRU / Transformer, ~30 k weights | 1.951 / **1.938** / 1.940 |
| Kalila wa-Dimna | two LSTMs of 96 (130 k) / Transformer (162 k) | **1.863** / 1.883 |

At equal size the Transformer does not beat the gated recurrent models on
these small texts; its advantage is the gradient path (63 characters back
it still receives 0.86% of the last prediction's gradient on Alice, against
7·10⁻⁵ along an LSTM's cell state).

## Tests

`cmake --build build --target test` runs 105 unit tests (~106 000 checks) in
about two seconds, the two C examples as regressions (XOR must converge; the
MNIST CNN must reach 95% after one epoch on 20 000 images, when the data is
present), and in CI a consumer of the installed package:

* **numerical gradient checks** for Dense, every activation (incl. the softmax
  Jacobian), MSE, fused softmax+cross-entropy, a deep mixed network, batch
  size 1, rank-2 samples; **Conv2D** over 7 stride/padding/kernel combinations
  (both implementations), **MaxPool2D** (incl. overlapping windows), Flatten,
  a whole CNN, **Dropout** in evaluation and training mode, and normalization
  (gradients w.r.t. the raw input);
* hand-computed values: dense/SGD/momentum, a 2×2 convolution, max-pool
  winners and gradient routing, **Adam** over three steps;
* **XOR over 20 seeds** (SGD and Adam with schedules), **XOR+AND** with two
  outputs (the case v1 got wrong), Iris-style CSV loading;
* determinism, resuming Adam after save/load (bit-identical), early stop, NaN
  detection, seeded dropout masks;
* model files: round trip, little-endian layout, checksum/truncation, **a real
  format-2 file from phase 1 still loads**, unknown sections are skipped;
* **sequences**: gradient checks for the RNN over 5 steps (incl. input
  gradients), embed → rnn → per-step dense → softmax/CE, a loss on the last
  step only (so earlier gradients can only arrive through h_{t−1}), and the
  RNN with a carried state h₀ ≠ 0; the RNN against hand-computed BPTT; the
  sparse embedding gradient with exact (dense) Adam; gradient clipping;
  learning a repeating text with and without carried state; validation
  splits for text and tabular data; early stopping restores the best epoch;
* **text**: UTF-8 with Arabic and 4-byte characters and every class of
  invalid sequence, vocabulary order/counts, line-per-item normalisation,
  the vocabulary round trip through save/load, deterministic generation
  (seed, greedy top-1, temperature), an Arabic model generating valid UTF-8,
  and **a format-3 file from phase 2 (without the new sections) still loads**;
* **gated layers**: the LSTM and GRU forward against hand-computed values;
  gradient checks over 6 steps with per-step heads, with the loss on the
  last step only, with a carried state (h and c ≠ 0), stacked (lstm→lstm,
  gru→gru, lstm→dropout→gru→dropout→rnn in evaluation mode, lstm→dropout→
  lstm in training mode); **the gradient into c_{t−1} and h_{t−1}** from
  the trace against central differences of the loss in the carried state;
  carrying the state over two halves = one pass over the whole; orthogonal
  initialisation (WᵀW = I for every gate block); forget-gate bias, save and
  load of both layers;
* **phase 4 text**: Arabic normalisation (tashkeel, tatweel, alef forms;
  the vocabulary and the saved flags), batch generation equals one-at-a-time
  generation row by row, a separate validation set, and **a character RNN
  written by the last phase-3 commit still loads and generates exactly the
  same names**;
* **Transformer**: the blocked matrix product bit-identical to the naive
  loops (36 shapes × transposition/accumulation flags); LayerNorm, the
  sinusoidal position table and one attention head against hand-computed
  values; gradient checks of every building block (LayerNorm, learned and
  sinusoidal positions, attention with E = D and E ≠ D, GELU and ReLU
  feed-forward, the block with dropout in evaluation and training mode),
  of whole two-block models (tied and untied output, learned and
  sinusoidal positions, a loss on the last position only); **the causal
  mask** (the output at t is bit-identical when later inputs change, and
  ∂L/∂x_j = 0 exactly for j > t); **weight tying** (one matrix, both
  gradients, save/load keeps the tie); **the KV cache equals recomputing
  the window** for small windows, slides, long primes, items and batches;
  **AdamW and warm-up + cosine against hand-computed steps** (and which
  parameters decay); save/load of a Transformer; the attention trace
  events; and **a model written by the last phase-4 commit still loads and
  generates exactly the same names**;
* **devices**, on every machine: `cpu-generic` (the CPU's ops without its
  fused kernels: exactly what a GPU runs) against the CPU for every layer
  type, and a fake device registered through the public API (its own
  memory, poisoned with NaN, explicit copies) bit-identical to
  cpu-generic for passes, gradient checks, training with the data on the
  device, moving a net mid-training, generation with and without the KV
  cache, traces and a layer written for 2.0; running out of device memory
  and retrying with a smaller batch;
* **the GPU** (`-DNMZ_CUDA=ON`, `ctest -L gpu`, skipped without a GPU):
  every op of the CUDA backend against the CPU on random inputs and odd
  shapes, within tolerances stated and explained in
  [`tests/test_gpu.c`](tests/test_gpu.c) (0 for data movement, 1e-6 to
  1e-5 for arithmetic, 1e-4 for whole models); every layer type and its
  gradient check on the GPU, recurrent layers too wide for the fused
  kernels, traces, the KV cache, a GPU-trained file on the CPU, identical
  files from two GPU runs, running out of memory; the parity runs above
  several GPU nets and a CPU net trained at once in four threads, every
  file identical to the sequential runs (22 tests, ~1 400 checks);
* **threads**: six nets of four kinds trained in six threads give model
  files byte-identical to sequential runs; a thread's device choice does
  not leak into other threads; errors stay per thread; eight threads use
  the backend registry at once; training stopped after three batches
  equals one epoch over those three batches, bit for bit;
* **models in words**: every preset and the layer grammar build the
  models the program builds, errors are messages, data paths go through
  the caller's callback, training options parse as the program's;
* CSV and IDX parsing errors; every API entry point with `NULL`/bad input.
* the training data a model remembers (path and item hashes) through
  save/load, and older files without it.

CI (GitHub Actions) runs them with GCC and Clang on Linux (Release, OpenMP,
and ASan + UBSan), MSVC on Windows (with and without OpenMP) and Apple Clang
on macOS, and compiles the CUDA backend with GCC and MSVC (CUDA 13.5); those
builds' tests run there without a GPU (the GPU tests report "skipped"). Benchmarks and dataset downloads are not part of CI; the data
scripts have a manual workflow (`data.yml`).

## Layout

```
include/neuromz.h        public API            src/       the library (C11)
include/neuromz_layer.h  layer-author API      src/cuda/  the CUDA backend (one .cu, C ABI)
include/neuromz_backend.h backend-author API   tests/     unit tests, GPU tests, fixtures, package consumer
docs/DESIGN.md           design + decisions    examples/  example data, C examples, benchmarks
cmake/                   package config        scripts/   MNIST / text download (sh + PowerShell)
legacy/                  v1 (2017–2020)        data/      downloaded datasets (gitignored)
```

Dataset sources and licences: [`examples/DATASETS.md`](examples/DATASETS.md).
Model files (`.nmz`) use a portable little-endian format with a magic
number, version, optional sections and CRC-32 ([DESIGN §7](docs/DESIGN.md)).

## Roadmap

Done: dense networks (phase 1); convolution, pooling, dropout, Adam,
schedules, normalization and MNIST (phase 2); text, embeddings, a vanilla
RNN with truncated BPTT, clipping, validation/early stopping and generation
(phase 3); LSTM, GRU, gate traces, gradient survival, Arabic prose (phase
4); a decoder-only Transformer with AdamW, warm-up, a blocked matrix
product, a KV cache and attention traces (phase 5); CMake, native Windows,
CI, the first public release and the split into repositories (v2.0.0);
device backends with CUDA (v2.1.0); threads and models described in
words (v2.2.0), with
[`neuromz-protocol`](https://github.com/mzprog/neuromz-protocol) and
[`neuromz-server`](https://github.com/mzprog/neuromz-server) in their own
repositories and the program's remote mode — over HTTP at first (protocol
1.0.0, server 0.1.0), then over a binary protocol of its own (protocol
2.0.0, server 0.2.0, program 2.3.0). Next: `neuromz-php` (a PHP client of
the protocol, for web applications and a web viewer of the trace events);
in this repository, more backends (HIP, Vulkan) on the same table of ops.

## License

MIT — see [`LICENSE`](LICENSE). Copyright (c) 2017–2026 Mohamad Zbib.
