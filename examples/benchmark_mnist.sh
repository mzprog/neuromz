#!/bin/sh
# benchmark_mnist.sh — train both MNIST presets on all 60 000 training images
# and test on the 10 000 test images. Needs `make data`. Slow: a few minutes.
# Usage: sh examples/benchmark_mnist.sh [path/to/neuromz]
set -e
B=${1:-./build/neuromz}
OUT=build/bench
mkdir -p "$OUT"
[ -f data/train-images-idx3-ubyte ] || { echo "MNIST missing: run make data" >&2; exit 1; }

for p in mnist-cnn mnist-mlp; do
    echo "== $p"
    "$B" new --preset "$p" -o "$OUT/$p.nmz" | tail -1
    start=$(date +%s)
    "$B" -m "$OUT/$p.nmz" train mnist -o "$OUT/$p.nmz" | grep -E "epoch|Finished|Converged"
    end=$(date +%s)
    "$B" -m "$OUT/$p.nmz" test mnist | sed -n 2p
    echo "   wall time for training: $((end - start)) s"
done

echo "== conv implementation: naive loops vs im2col (10 000 images, 1 epoch)"
./build/bench_conv 10000
