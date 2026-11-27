# Changelog

All notable changes to the NeuroMZ library. The command-line program has
its own changelog in [neuromz-cli](https://github.com/mzprog/neuromz-cli);
both use the same version numbers. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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

[2.0.0]: https://github.com/mzprog/neuromz/releases/tag/v2.0.0
