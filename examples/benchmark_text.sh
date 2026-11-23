#!/bin/sh
# benchmark_text.sh — character RNNs on the example texts: validation loss,
# perplexity, training time and generated samples. The book needs
# `make data-text`. Slow (a few minutes). Usage: sh examples/benchmark_text.sh [neuromz]
set -e
B=${1:-./build/neuromz}
OUT=build/bench
mkdir -p "$OUT"

run() { # run NAME PRESET DATA GEN-ARGS...
    name=$1; preset=$2; data=$3; shift 3
    echo "== $name ($data)"
    "$B" new --preset "$preset" --data "$data" -o "$OUT/$name.nmz" | grep -E "vocabulary|Created"
    start=$(date +%s)
    "$B" -m "$OUT/$name.nmz" train "$data" -o "$OUT/$name.nmz" | grep -E "Early stop|Finished|validation \\(|epoch +1 "
    end=$(date +%s)
    echo "   wall time for training: $((end - start)) s"
    "$B" -m "$OUT/$name.nmz" generate "$@"
    echo
}

run names-en names examples/names_en.txt --count 10 --seed 1 --data examples/names_en.txt
run names-ar names examples/names_ar.txt --count 10 --seed 1 --data examples/names_ar.txt
if [ -f data/alice.txt ]; then
    run alice char-rnn data/alice.txt --length 300 --prime "Alice " --temperature 0.7 --seed 1
else
    echo "== alice: skipped (run make data-text)"
fi
