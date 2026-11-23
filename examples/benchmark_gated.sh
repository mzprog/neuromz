#!/bin/sh
# benchmark_gated.sh — RNN vs LSTM vs GRU with about the same number of
# weights (rnn:128, lstm:64, gru:75: within ±3% of each other) on the four
# example texts: validation loss, perplexity, training time, samples, and the
# gradient-survival comparison of the trained models. Learning rates are the
# best of a small sweep per architecture (README, DESIGN decision 86). The
# books need `make data-text`. Slow: about an hour on a laptop CPU.
# Usage: sh examples/benchmark_gated.sh [neuromz] [names] [alice] [kalila]
set -e
B=${1:-./build/neuromz}
[ $# -gt 0 ] && shift
OUT=build/bench
mkdir -p "$OUT"
WHICH=${*:-names alice kalila}
NL='
'

# run NAME DATA NEW-ARGS... -- GEN-ARGS...   (create, train, report)
run() {
    name=$1; data=$2; shift 2
    n=0
    for a in "$@"; do [ "$a" = "--" ] && break; n=$((n + 1)); done
    echo "== $name"
    # the arguments before -- go to `new`, the rest to `generate`
    i=0; newargs=""; genargs=""
    for a in "$@"; do
        if [ $i -lt $n ]; then newargs="$newargs $a"; elif [ $i -gt $n ]; then genargs="$genargs $a"; fi
        i=$((i + 1))
    done
    # shellcheck disable=SC2086
    "$B" new $newargs --data "$data" -o "$OUT/$name.nmz" | grep -E "Created"
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
    names)
        for f in en ar; do
            d=examples/names_$f.txt
            run "names_$f-rnn" "$d" --preset names -- --count 10 --seed 1 --data "$d"
            run "names_$f-lstm" "$d" --preset names-lstm -- --count 10 --seed 1 --data "$d"
            run "names_$f-gru" "$d" text embed:16 gru:75 vocab --lr 0.01 --epochs 40 -- --count 10 --seed 1 --data "$d"
        done
        echo "== gradient survival, English names"
        "$B" compare "$OUT/names_en-rnn.nmz" "$OUT/names_en-lstm.nmz" "$OUT/names_en-gru.nmz" \
            --text "${NL}Maximilian${NL}Christopher"
        ;;
    alice)
        if [ ! -f data/alice.txt ]; then echo "== alice: skipped (run make data-text)"; continue; fi
        run alice-rnn data/alice.txt --preset char-rnn -- --length 300 --prime Alice --temperature 0.7 --seed 1
        run alice-lstm data/alice.txt --preset char-lstm -- --length 300 --prime Alice --temperature 0.7 --seed 1
        run alice-gru data/alice.txt text embed:32 gru:75 vocab --lr 0.01 --batch 16 --epochs 30 \
            -- --length 300 --prime Alice --temperature 0.7 --seed 1
        echo "== gradient survival, Alice (65 characters from the validation part)"
        "$B" compare "$OUT/alice-rnn.nmz" "$OUT/alice-lstm.nmz" "$OUT/alice-gru.nmz" --data data/alice.txt \
            --from 140000 --chars 65
        "$B" -m "$OUT/alice-lstm.nmz" show gates --data data/alice.txt --from 130000 --chars 14000
        ;;
    kalila)
        if [ ! -f data/kalila.txt ]; then echo "== kalila: skipped (run make data-text)"; continue; fi
        G="--length 300 --prime قال --temperature 0.7 --seed 1"
        # shellcheck disable=SC2086
        run kalila-rnn data/kalila.txt text embed:32 rnn:128 vocab --lr 0.004 --batch 16 --epochs 30 -- $G
        # shellcheck disable=SC2086
        run kalila-lstm data/kalila.txt --preset char-lstm -- $G
        # shellcheck disable=SC2086
        run kalila-gru data/kalila.txt text embed:32 gru:75 vocab --lr 0.01 --batch 16 --epochs 30 -- $G
        # shellcheck disable=SC2086
        run kalila-lstm-strip data/kalila.txt --preset char-lstm --strip-diacritics -- $G
        # shellcheck disable=SC2086
        run kalila-stacked data/kalila.txt --preset arabic-text -- $G
        echo "== gradient survival, Kalila wa-Dimna (65 characters from the validation part)"
        "$B" compare "$OUT/kalila-rnn.nmz" "$OUT/kalila-lstm.nmz" "$OUT/kalila-gru.nmz" --data data/kalila.txt \
            --from 230000 --chars 65
        ;;
    esac
done
