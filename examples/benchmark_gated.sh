#!/bin/sh
# benchmark_gated.sh — RNN vs LSTM vs GRU with about the same number of
# weights (rnn:128, lstm:64, gru:75: within ±3% of each other) on the four
# example texts: validation loss, perplexity, training time, samples, and the
# gradient-survival comparison of the trained models. The books need
# `make data-text`. Slow (about 40 minutes on a laptop CPU, one thread).
# Usage: sh examples/benchmark_gated.sh [neuromz] [names|alice|kalila]...
set -e
B=${1:-./build/neuromz}
[ $# -gt 0 ] && shift
OUT=build/bench
mkdir -p "$OUT"
WHICH=${*:-names alice kalila}

# run NAME DATA NEW-ARGS... -- GEN-ARGS...   (create, train, report)
run() {
    name=$1; data=$2; shift 2
    newargs=""
    while [ "$1" != "--" ]; do newargs="$newargs $1"; shift; done
    shift
    echo "== $name"
    # shellcheck disable=SC2086
    "$B" new $newargs --data "$data" -o "$OUT/$name.nmz" | grep -E "Created"
    start=$(date +%s)
    "$B" -m "$OUT/$name.nmz" train "$data" -o "$OUT/$name.nmz" --quiet | grep -E "Early stop|Finished|validation \\("
    end=$(date +%s)
    echo "   wall time for training: $((end - start)) s"
    "$B" -m "$OUT/$name.nmz" generate "$@"
    echo
}

for w in $WHICH; do
    case $w in
    names)
        for f in en ar; do
            d=examples/names_$f.txt
            run "names_$f-rnn" "$d" --preset names -- --count 10 --seed 1 --data "$d"
            run "names_$f-lstm" "$d" --preset names-lstm -- --count 10 --seed 1 --data "$d"
            run "names_$f-gru" "$d" --preset names-lstm text embed:16 gru:75 vocab -- --count 10 --seed 1 --data "$d"
        done
        echo "== gradient survival, English names"
        "$B" compare "$OUT/names_en-rnn.nmz" "$OUT/names_en-lstm.nmz" "$OUT/names_en-gru.nmz" \
            --text "$(printf '\nMaximilian\nChristopher')"
        ;;
    alice)
        if [ ! -f data/alice.txt ]; then echo "== alice: skipped (run make data-text)"; continue; fi
        G="--length 300 --prime Alice --temperature 0.7 --seed 1"
        run alice-rnn data/alice.txt --preset char-rnn -- $G
        run alice-lstm data/alice.txt --preset char-lstm -- $G
        run alice-gru data/alice.txt --preset char-lstm text embed:32 gru:75 vocab -- $G
        echo "== gradient survival, Alice (65 characters from the validation part)"
        "$B" compare "$OUT/alice-rnn.nmz" "$OUT/alice-lstm.nmz" "$OUT/alice-gru.nmz" --data data/alice.txt \
            --from 140000 --chars 65
        ;;
    kalila)
        if [ ! -f data/kalila.txt ]; then echo "== kalila: skipped (run make data-text)"; continue; fi
        G="--length 300 --prime قال --temperature 0.7 --seed 1"
        K="--preset char-lstm --data data/kalila.txt"
        run kalila-rnn data/kalila.txt $K text embed:32 rnn:128 vocab --lr 0.004 -- $G
        run kalila-lstm data/kalila.txt $K -- $G
        run kalila-gru data/kalila.txt $K text embed:32 gru:75 vocab -- $G
        run kalila-lstm-strip data/kalila.txt $K --strip-diacritics -- $G
        echo "== gradient survival, Kalila wa-Dimna (65 characters from the validation part)"
        "$B" compare "$OUT/kalila-rnn.nmz" "$OUT/kalila-lstm.nmz" "$OUT/kalila-gru.nmz" --data data/kalila.txt \
            --from 230000 --chars 65
        ;;
    esac
done
