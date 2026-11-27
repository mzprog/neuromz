# Changelog

All notable changes to the NeuroMZ library. The command-line program has
its own changelog in [neuromz-cli](https://github.com/mzprog/neuromz-cli);
both use the same version numbers. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [2.1.0] — 2026-11-27

Device backends, and a CUDA backend that trains every model type on an
NVIDIA GPU. Design: [`docs/DESIGN.md`](docs/DESIGN.md) §11 and decisions
127–157. On the CPU, everything computes exactly as in 2.0.1, bit for bit.

### Added
- **The backend layer** (`include/neuromz_backend.h`): a table of memory
  functions and about fifty ops (GEMM with transpose flags, element-wise
  ops, activations, softmax and the fused cross-entropy, reductions,
  im2col/col2im, pooling, dropout, embeddings, LayerNorm, attention,
  recurrent cells, the optimizer updates, clipping, the training loop's
  gathers), plus optional fused layer kernels. Every layer computes
  through it; a new backend (HIP, Vulkan, OpenCL) needs no change to the
  engine, the layers or the file format (DESIGN §11.6).
  `nmz_backend_register` adds one from outside the library.
- **The CUDA backend** (`-DNMZ_CUDA=ON`, off by default): cuBLAS for the
  matrix products (also strided batched for attention), the library's own
  kernels for everything else, fused sequence kernels for RNN/LSTM/GRU;
  the model, gradients, optimizer state and the training data stay on the
  GPU; per batch only the loss comes back. Deterministic: two runs give
  identical model files. `NMZ_CUDA_ARCHITECTURES` (default: the building
  machine's GPU, else 7.5; every entry with PTX).
- Device selection: `nmz_backend_select("auto" | "cpu" | "cuda")`,
  `nmz_backend_info` (backend, device name, compute capability, memory
  total / free / held / peak / limit, a note on why this device),
  `nmz_backend_query`, `nmz_backend_count`, `nmz_backend_name`,
  `nmz_backend_memory_limit` and `NMZ_CUDA_MEMORY`; `nmz_tensor_ptr` for
  layer authors. The CPU stays the default.
- A CUDA build runs on machines without an NVIDIA GPU or driver: the
  runtime is linked statically, cuBLAS is loaded on first use, and
  "auto" falls back to the CPU with the reason.
- Running out of GPU memory is an `NMZ_ERR_NOMEM` with a message (what was
  needed, what is held, "try a smaller batch size"); the net frees its
  device buffers, so a retry with a smaller batch works in the same
  process. The memory cap keeps Windows from paging GPU memory to the host.
- Tests: `cpu-generic` and a fake device check the device code paths on
  every machine (9 new unit tests); with CUDA, `ctest -L gpu` runs every op
  against the CPU with stated tolerances, gradient checks of every layer
  type on the GPU, and training parity (XOR, Iris, MNIST CNN, names RNN,
  Alice LSTM, a Transformer); skipped without a GPU.
- `examples/bench_device.c` and the README's CPU / OpenMP / GPU table:
  10× (MNIST CNN) to ≈ 240× (a 25M-parameter Transformer) on a GTX 1660 Ti
  against one CPU core; tiny nets are slower on the GPU.
- CI: the CUDA backend compiled on Linux (GCC) and Windows (MSVC) with
  CUDA 13.5, and those builds' tests run without a GPU.

### Changed
- `nmz_tensor` has one more field (`dev`), `nmz_ctx` one more
  (`backend`), `nmz_layer_ops` one more optional entry (`buffers`):
  source-compatible (`NMZ_TENSOR_EMPTY` and designated initialisers keep
  working), but code built against 2.0 must be recompiled.
- Layers without the new `NMZ_LAYER_BACKEND` flag (written for 2.0, reading
  `->data`) still work on a device: the engine runs them on the host.
- Faster on the CPU too: im2col epoch 4.32 → 4.17 s, naive convolution
  13.2 → 10.6 s, a char-LSTM epoch 6.80 → 6.53 s, a char-GPT epoch 8.64 →
  8.45 s (MSVC; plain pointers into the kernels optimise better).

## [2.0.1] — 2026-11-27

### Fixed
- The Windows data scripts under Windows PowerShell 5.1: `get_text.ps1`
  wrote the Arabic words of a regex as literal letters, which 5.1 reads in
  the ANSI code page, so `kalila.txt` came out different and the script
  stopped with a checksum error; and both scripts hashed with
  `Get-FileHash`, which does not load when 5.1 is started from PowerShell
  7. They now use `\uXXXX` escapes and .NET hashing, CMake prefers `pwsh`
  when it is installed, and CI checks that the scripts stay ASCII. The
  manual `data` workflow verified identical files on Linux, PowerShell 7
  and Windows PowerShell 5.1.
- `.gitattributes`: LF line endings in every checkout (Git for Windows
  converted the example data to CRLF by default).

## [2.0.0] — 2026-11-27

The first public release of v2, a rewrite of the 2017 v1 (kept in
`legacy/`, tag `v1-legacy`). The design and every decision are in
[`docs/DESIGN.md`](docs/DESIGN.md); decisions 111–124 are this release's.

### The library (phases 1–5)
- Dense layers and activations, MSE and fused softmax + cross-entropy,
  SGD with momentum; trace events for every number of a pass; numerical
  gradient checks; portable model files (phase 1).
- Conv2D (naive and im2col), MaxPool2D, Flatten, Dropout, Adam, step and
  cosine schedules, input normalisation, IDX/MNIST loading (phase 2).
- UTF-8 text, code-point vocabularies, Embedding, RNN with BPTT inside the
  layer, truncated BPTT with a carried state, gradient clipping, validation
  and early stopping, text generation (phase 3).
- LSTM and GRU, orthogonal initialisation, stacking, gate traces, gradient
  survival, Arabic normalisation, batch generation (phase 4).
- A decoder-only Transformer (position encodings, LayerNorm, causal
  multi-head attention, GELU feed-forward, pre-LN blocks, weight tying),
  a cache-blocked matrix product, AdamW, warm-up + cosine, KV-cache
  generation, optional OpenMP (phase 5).

### Added in 2.0.0
- CMake build (≥ 3.21) for GCC, Clang and MSVC, replacing the Makefile;
  targets `test`, `sanitize`, `bench`, `data`, `data-text`; options
  `NMZ_OPENMP`, `NMZ_SANITIZE`, `NMZ_WERROR`.
- `cmake --install` with a CMake package: `find_package(neuromz)` and the
  target `neuromz::neuromz`; usable through FetchContent as well.
- A model remembers its training data: `nmz_net_set_data_source`,
  `nmz_net_data_path`, `nmz_net_item_count`, `nmz_net_knows_item` (a path
  and the 64-bit hashes of line-per-item data, saved in a new optional
  `data_source` section that older readers skip).
- UTF-8 file names on Windows (the library opens files with `_wfopen`).
- PowerShell versions of the data scripts; both script families also
  verify the SHA-256 of the files they produce.
- `examples/api_mnist.c`, a CNN on MNIST from C, run by CTest as a
  regression when the data is present; `examples/api_xor.c` is one too.
- CI on GitHub Actions: Linux GCC and Clang (Release, OpenMP, ASan +
  UBSan), Windows MSVC (with and without OpenMP), macOS Apple Clang, and a
  check that the installed package works.
- MIT licence.

### Changed in 2.0.0
- The `neuromz` program moved to its own repository,
  [neuromz-cli](https://github.com/mzprog/neuromz-cli), with its history,
  its smoke test, its walkthroughs and the CLI-driven benchmarks.
- Floating-point contraction is off (`-ffp-contract=off`) so that the
  blocked matrix product stays bit-identical to the naive loops on CPUs
  with FMA (Apple Clang on arm64 contracted by default).
- `bench_conv` uses C11 `timespec_get`; unsigned negations are written as
  `0u - x` (MSVC C4146); two Clang warnings in the tests.

[2.1.0]: https://github.com/mzprog/neuromz/releases/tag/v2.1.0
[2.0.1]: https://github.com/mzprog/neuromz/releases/tag/v2.0.1
[2.0.0]: https://github.com/mzprog/neuromz/releases/tag/v2.0.0
