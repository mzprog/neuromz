#!/bin/sh
# benchmark_transformer.sh — phase 5: a small decoder-only Transformer against
# the LSTM and the GRU of phase 4 with about the same number of weights
# (char-gpt 32 043 / lstm:64 32 107 / gru:75 32 400 on Alice), on Alice and on
# Kalila wa-Dimna; then the best Transformer that trains in about 20 minutes
# on one laptop core; the matrix-product speedups; the KV cache; the
# attention patterns of the trained models; the gradient-survival comparison.
# The books need `make data-text`. Slow: about two hours on a laptop CPU.
# Usage: sh examples/benchmark_transformer.sh [neuromz] [matmul] [names] [alice] [kalila] [big]
set -e
B=${1:-./build/neuromz}
[ $# -gt 0 ] && shift
OUT=build/bench
mkdir -p "$OUT"
WHICH=${*:-matmul names alice kalila big}
GPT="text embed:32 pos:learned:128 block:heads=4 block:heads=4 ln vocab:tied"

# run NAME DATA NEW-ARGS... -- GEN-ARGS...   (create, train, report, sample)
run() {
    name=$1; data=$2; shift 2
    n=0
    for a in "$@"; do [ "$a" = "--" ] && break; n=$((n + 1)); done
    echo "== $name"
    i=0; newargs=""; genargs=""
    for a in "$@"; do
        if [ $i -lt $n ]; then newargs="$newargs $a"; elif [ $i -gt $n ]; then genargs="$genargs $a"; fi
        i=$((i + 1))
    done
    # shellcheck disable=SC2086
    "$B" new $newargs --data "$data" -o "$OUT/$name.nmz" | grep -E "Created|Transformer:"
    start=$(date +%s)
    "$B" -m "$OUT/$name.nmz" train "$data" -o "$OUT/$name.nmz" --quiet | grep -E "Early stop|Finished|validation \\("
    end=$(date +%s)
    echo "   wall time for training: $((end - start)) s"
    # shellcheck disable=SC2086
    "$B" -m "$OUT/$name.nmz" generate $genargs
    echo
}

for w in $WHICH; do
    case $w in
    matmul)
        echo "== matrix product: blocked vs naive (bit-identical results)"
        make -s build/bench_matmul && ./build/bench_matmul 5
        if make -s OMP=1 build/omp/bench_matmul 2>/dev/null; then
            echo "== the same with OpenMP, 2 threads (make OMP=1)"
            OMP_NUM_THREADS=2 ./build/omp/bench_matmul 5
        fi
        ;;
    names)
        d=examples/names_en.txt
        run names_en-lstm5 "$d" --preset names-lstm -- --count 10 --seed 1 --data "$d"
        run names_en-gpt "$d" --preset names-gpt -- --count 10 --seed 1 --data "$d"
        d=examples/names_ar.txt
        run names_ar-gpt "$d" --preset names-gpt -- --count 10 --seed 1 --data "$d"
        ;;
    alice)
        if [ ! -f data/alice.txt ]; then echo "== alice: skipped (run make data-text)"; continue; fi
        G="--length 300 --prime Alice --temperature 0.7 --seed 1"
        # shellcheck disable=SC2086
        run alice-lstm5 data/alice.txt --preset char-lstm -- $G
        # shellcheck disable=SC2086
        run alice-gru5 data/alice.txt text embed:32 gru:75 vocab --lr 0.01 --batch 16 --epochs 30 -- $G
        # shellcheck disable=SC2086
        run alice-gpt data/alice.txt --preset char-gpt -- $G
        echo "== KV cache: 20 x 300 characters from the trained char-gpt (same text both ways)"
        t0=$(date +%s%N)
        "$B" -m "$OUT/alice-gpt.nmz" generate --length 300 --count 20 --seed 1 > "$OUT/kv_cache.txt"
        t1=$(date +%s%N)
        "$B" -m "$OUT/alice-gpt.nmz" generate --length 300 --count 20 --seed 1 --no-cache > "$OUT/kv_full.txt"
        t2=$(date +%s%N)
        cmp -s "$OUT/kv_cache.txt" "$OUT/kv_full.txt" && same=identical || same=DIFFERENT
        echo "   with the cache: $(( (t1 - t0) / 1000000 )) ms, recomputing the window: $(( (t2 - t1) / 1000000 )) ms, text $same"
        echo "== gradient survival, Alice (65 characters from the validation part)"
        "$B" compare "$OUT/alice-lstm5.nmz" "$OUT/alice-gru5.nmz" "$OUT/alice-gpt.nmz" --data data/alice.txt \
            --from 140000 --chars 65
        echo "== attention patterns, char-gpt on Alice"
        "$B" -m "$OUT/alice-gpt.nmz" show attention --data data/alice.txt --from 130000 --chars 8000 --layer 1 --head 1
        ;;
    kalila)
        if [ ! -f data/kalila.txt ]; then echo "== kalila: skipped (run make data-text)"; continue; fi
        G="--length 300 --prime قال --temperature 0.7 --seed 1"
        # shellcheck disable=SC2086
        run kalila-lstm5 data/kalila.txt --preset char-lstm -- $G
        # shellcheck disable=SC2086
        run kalila-gru5 data/kalila.txt text embed:32 gru:75 vocab --lr 0.01 --batch 16 --epochs 30 -- $G
        # shellcheck disable=SC2086
        run kalila-gpt data/kalila.txt $GPT --lr 0.01 --batch 8 --epochs 30 -- $G
        # shellcheck disable=SC2086
        run kalila-arabic-gpt data/kalila.txt --preset arabic-gpt -- $G
        echo "== gradient survival, Kalila wa-Dimna (65 characters from the validation part)"
        "$B" compare "$OUT/kalila-lstm5.nmz" "$OUT/kalila-gru5.nmz" "$OUT/kalila-gpt.nmz" --data data/kalila.txt \
            --from 230000 --chars 65
        echo "== attention patterns, arabic-gpt on Kalila"
        "$B" -m "$OUT/kalila-arabic-gpt.nmz" show attention --data data/kalila.txt --from 225000 --chars 8000
        ;;
    big)
        if [ ! -f data/alice.txt ]; then echo "== big: skipped (run make data-text)"; continue; fi
        # the best of three larger configurations tried within ~30 min of one core (README): D = 64,
        # 4 blocks, dropout 0.2 (213 k weights); D = 96 with dropout 0.1 / 0.3 overfitted or lagged
        BB="block:heads=4:dropout=0.2"
        # shellcheck disable=SC2086
        run alice-gpt-big data/alice.txt text embed:64 pos:learned:128 $BB $BB $BB $BB ln vocab:tied \
            --lr 0.005 --batch 8 --epochs 30 -- --length 300 --prime Alice --temperature 0.7 --seed 1
        "$B" -m "$OUT/alice-gpt-big.nmz" show attention --data data/alice.txt --from 130000 --chars 8000
        if [ -f "$OUT/alice-gru5.nmz" ]; then
            echo "== gradient survival, Alice: GRU vs the big Transformer"
            "$B" compare "$OUT/alice-gru5.nmz" "$OUT/alice-gpt-big.nmz" --data data/alice.txt --from 140000 --chars 65
        fi
        ;;
    esac
done
