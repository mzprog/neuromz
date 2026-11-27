#!/bin/sh
# cli_smoke.sh — drive the neuromz CLI through normal use and abuse.
# Usage: sh tests/cli_smoke.sh path/to/neuromz
# Every command must exit with the expected status (0 ok, 1 handled error),
# never crash, and (under `make sanitize`) never trigger ASan/UBSan.
B=${1:-./build/neuromz}
T=$(mktemp -d "${TMPDIR:-/tmp}/nmz_smoke.XXXXXX") || exit 1
trap 'rm -rf "$T"' EXIT
export ASAN_OPTIONS="${ASAN_OPTIONS:+$ASAN_OPTIONS:}exitcode=99"
export UBSAN_OPTIONS="${UBSAN_OPTIONS:+$UBSAN_OPTIONS:}exitcode=99"
fails=0
count=0

check() { # check WANT_RC PATTERN-or-"" CMD...
    want=$1; pat=$2; shift 2
    count=$((count + 1))
    "$@" >"$T/out" 2>"$T/err"
    rc=$?
    if [ "$rc" -ne "$want" ] || grep -q "Sanitizer\|runtime error" "$T/err" ||
        { [ -n "$pat" ] && ! grep -q -- "$pat" "$T/out" "$T/err"; }; then
        echo "FAIL (rc=$rc, want $want${pat:+, pattern '$pat'}): $*"
        head -20 "$T/out" "$T/err"
        fails=$((fails + 1))
    else
        echo "ok   $*"
    fi
}

stdin_check() { # stdin_check WANT_RC PATTERN INPUT-FILE
    want=$1; pat=$2; in=$3
    check "$want" "$pat" sh -c "\"$B\" < \"$in\""
}

# --- basics
check 0 "Commands" "$B" --help
check 0 "neuromz 2" "$B" --version
check 0 "trace (X1" "$B" help trace
check 1 "unknown command" "$B" frobnicate

# --- nothing loaded: every command fails cleanly
for c in predict test train trace step save gradcheck set reinit; do
    check 1 "no network loaded" "$B" $c 1 2
done
check 1 "no network loaded" "$B" show
check 1 "no network loaded" "$B" show filters
check 0 "no network" "$B" clear

# --- quick mode, training, prediction, persistence
check 0 "\[default\]" "$B" new 2 4 1 -o "$T/xor.nmz"
check 0 "Converged" "$B" -m "$T/xor.nmz" train examples/xor.csv -o "$T/xor.nmz"
check 0 "output" "$B" -m "$T/xor.nmz" predict 1 0
check 0 "accuracy 100.0%" "$B" -m "$T/xor.nmz" test examples/xor.csv
check 0 "dense 2->4" "$B" -m "$T/xor.nmz" show weights
check 0 "WEIGHT UPDATE" "$B" -m "$T/xor.nmz" trace 1 1 --target 0 --json "$T/t.jsonl"
check 0 '"phase":"backward"' cat "$T/t.jsonl"
check 0 "Weights updated" "$B" -m "$T/xor.nmz" step 1 1 -t 0 --lr 0.1 --momentum 0
check 0 "PASS" "$B" -m "$T/xor.nmz" gradcheck 0.3 0.7 --target 1
check 0 "lr 0.25" "$B" -m "$T/xor.nmz" set --lr 0.25 --epochs 10 -o "$T/xor.nmz"
check 0 "lr 0.25" "$B" -m "$T/xor.nmz" show
check 0 "seed 9" "$B" -m "$T/xor.nmz" reinit 9
check 0 "--ascii" "$B" --help
check 0 "sum_i x_i\*W_ij" "$B" --ascii -m "$T/xor.nmz" trace 1 0 -t 1

# --- advanced mode and classification
check 0 "\[yours\]" "$B" new 4 6:relu:he 5:leaky_relu=0.1 3:softmax --loss ce --seed 3 --lr 0.02 -o "$T/iris.nmz"
check 0 "accuracy" "$B" -m "$T/iris.nmz" train examples/iris_train.csv --epochs 300 --every 100 -o "$T/iris.nmz"
check 0 "confusion" "$B" -m "$T/iris.nmz" test examples/iris_test.csv
check 0 "predicted class" "$B" -m "$T/iris.nmz" predict 6.3 2.9 5.6 1.8
check 0 "softmax + cross-entropy" "$B" -m "$T/iris.nmz" trace 5.1 3.5 1.4 0.2 --target setosa
check 0 "2 hidden" sh -c "\"$B\" new 2 3 3 1 && echo 2 hidden"
check 0 "\[yours\]" "$B" new 2 6 2:sigmoid --loss mse -o "$T/xa.nmz"
check 0 "Converged" "$B" -m "$T/xa.nmz" train examples/xor_and.csv

# --- image layers on a tiny generated dataset (1x4x4 images, 2 classes)
awk 'BEGIN { srand(7); for (i = 0; i < 40; i++) { c = i % 2; s = "";
    for (p = 0; p < 16; p++) { x = int(rand() * 60); if ((c == 0 && p % 4 == 1) || (c == 1 && int(p / 4) == 2)) x += 180;
        s = s x ","; } print s (c ? "bar" : "stripe"); } }' >"$T/img.csv"
check 0 "conv 1x4x4->2x4x4" "$B" new 1x4x4 conv:2:3:relu:pad=same pool:2 dropout:0.2 2:softmax --normalize minmax \
    --optimizer adam --schedule cosine --seed 5 -o "$T/img.nmz"
check 0 "flatten 2x2x2->8" "$B" -m "$T/img.nmz" show
check 0 "accuracy" "$B" -m "$T/img.nmz" train "$T/img.csv" --limit 30 --epochs 20 -o "$T/img.nmz"
check 0 "Adam state stored" "$B" -m "$T/img.nmz" show
check 0 "input patch x" "$B" -m "$T/img.nmz" trace --sample 2 --data "$T/img.csv" --at 1,2,3
check 0 "Adam step" "$B" -m "$T/img.nmz" trace --sample 2 --data "$T/img.csv"
check 0 "winner" "$B" -m "$T/img.nmz" step --sample 1 --data "$T/img.csv" --full
check 0 "PASS" "$B" -m "$T/img.nmz" gradcheck --sample 3 --data "$T/img.csv"
check 0 "conv2d: 2 filters" "$B" -m "$T/img.nmz" show filters
check 0 "prediction" "$B" -m "$T/img.nmz" show sample 4 --data "$T/img.csv"
check 0 "predicted class" "$B" -m "$T/img.nmz" predict --sample 4 --data "$T/img.csv"
check 0 "step schedule" "$B" -m "$T/img.nmz" set --optimizer sgd --schedule step --step-size 2 --gamma 0.5
check 0 "Saved" "$B" -m "$T/img.nmz" save "$T/img2.nmz" --no-optimizer
check 0 "accuracy" "$B" -m "$T/img.nmz" test "$T/img.csv" --limit 10
check 0 "\[preset\]" "$B" new --preset mnist-cnn
check 0 "\[preset\]" "$B" new --preset mnist-mlp
check 1 "unknown preset" "$B" new --preset imagenet
check 1 "do not also give sizes" "$B" new --preset mnist-cnn 2 3
check 1 "image input shape" "$B" new 16 conv:2:3 2
check 1 "must be a dense layer" "$B" new 1x4x4 conv:2:3
check 1 "conv needs filters" "$B" new 1x4x4 conv:2 2
check 1 "pool needs a window" "$B" new 1x4x4 pool 2
check 1 "dropout needs a rate" "$B" new 4 dropout:1.0 2
check 1 "larger than" "$B" new 1x4x4 conv:2:7 2
check 1 "pad=same needs" "$B" new 1x4x4 conv:2:2:pad=same 2
check 1 "--at needs" "$B" -m "$T/img.nmz" trace --sample 1 --data "$T/img.csv" --at 0,1
check 1 "out of range" "$B" -m "$T/img.nmz" trace --sample 99 --data "$T/img.csv"
check 1 "make data" env NEUROMZ_DATA="$T/none" "$B" -m "$T/img.nmz" test mnist
check 1 "unknown optimizer" "$B" new 2 4 1 --optimizer rmsprop
check 1 "unknown normalization" "$B" new 2 4 1 --normalize max

# --- text models: character RNN, Arabic, generation, unrolled trace
check 0 "\[preset\]" "$B" new --preset names --epochs 3 -o "$T/n.nmz"
check 0 "validation" "$B" -m "$T/n.nmz" train examples/names_en.txt --limit 300 -o "$T/n.nmz"
check 0 " 5\. " "$B" -m "$T/n.nmz" generate --count 5 --data examples/names_en.txt
"$B" -m "$T/n.nmz" generate --seed 7 --count 3 >"$T/g1" 2>&1
"$B" -m "$T/n.nmz" generate --seed 7 --count 3 >"$T/g2" 2>&1
check 0 "" cmp "$T/g1" "$T/g2"                     # same seed, same names
check 0 "Mar" "$B" -m "$T/n.nmz" generate --prime Mar --count 2 --temperature 0.5 --top-k 3
check 0 "U+000A" "$B" -m "$T/n.nmz" show vocab
check 0 "perplexity" "$B" -m "$T/n.nmz" test examples/names_en.txt --limit 50
check 0 "UNROLLED" "$B" -m "$T/n.nmz" trace --text "Anna"
check 0 "VANISHING" "$B" -m "$T/n.nmz" trace --text "Christopher" --unit 3
check 0 "Weights updated" "$B" -m "$T/n.nmz" step --sample 2 --data examples/names_en.txt
check 0 "PASS" "$B" -m "$T/n.nmz" gradcheck --text "Eve"
check 0 "\[preset\]" "$B" new --preset names --data examples/names_ar.txt --epochs 2 -o "$T/ar.nmz"
check 0 "validation" "$B" -m "$T/ar.nmz" train examples/names_ar.txt -o "$T/ar.nmz"
check 0 " 3\. " "$B" -m "$T/ar.nmz" generate --count 3
check 0 "م ح م" "$B" -m "$T/ar.nmz" trace --text "محمد"
check 0 "U+0645" "$B" --ascii -m "$T/ar.nmz" trace --text "محمد"
check 0 "rnn Tx4->Tx6" "$B" new text embed:4 rnn:6 vocab --data examples/names_en.txt --seq-len 8 --no-carry
check 1 "not in the model's vocabulary" "$B" -m "$T/ar.nmz" trace --text "abc"
check 1 "not in the model's vocabulary" "$B" -m "$T/ar.nmz" train examples/names_en.txt
check 1 "at least 2 characters" "$B" -m "$T/n.nmz" trace --text "A"
check 1 "needs --data" "$B" new text embed:8 rnn:8 vocab
check 1 "starts with an embedding" "$B" new text rnn:8 vocab --data examples/names_en.txt
check 1 "need the .text. input" "$B" new 4 rnn:8 3
check 1 "text model" "$B" -m "$T/xor.nmz" generate
printf 'ab\377cd\nabce\nabcf\n' >"$T/bad.txt"
check 0 "U+FFFD" sh -c "\"$B\" new text embed:2 rnn:3 vocab --data \"$T/bad.txt\" --seq-len 2 -o \"$T/b.nmz\" && \"$B\" -m \"$T/b.nmz\" show vocab"

# --- gated layers (phase 4): LSTM, GRU, stacks, gate views, compare
check 0 "\[preset\]" "$B" new --preset names-lstm --epochs 2 -o "$T/nl.nmz"
check 0 "validation" "$B" -m "$T/nl.nmz" train examples/names_en.txt --limit 200 -o "$T/nl.nmz"
check 0 "forget gate" "$B" -m "$T/nl.nmz" trace --text "Anna"
check 0 "FORGET GATE f of every cell" "$B" -m "$T/nl.nmz" trace --text "Christopher" --unit 2
check 0 "along the cell state" "$B" -m "$T/nl.nmz" trace --text "Christopher"
check 0 "PASS" "$B" -m "$T/nl.nmz" gradcheck --text "Eve"
check 0 "Weights updated" "$B" -m "$T/nl.nmz" step --text "Bob"
check 0 "mean forget gate f per cell" "$B" -m "$T/nl.nmz" show gates --data examples/names_en.txt --chars 500
check 0 "longest memory" "$B" -m "$T/nl.nmz" show gates --text "AnnaMaria"
# v2.0.0: the model remembers its training data (decision 111)
check 0 "of the training data" "$B" -m "$T/nl.nmz" show gates
check 0 " 3\. " "$B" -m "$T/nl.nmz" generate --count 3 --new-only
check 0 "the 8 largest" "$B" -m "$T/nl.nmz" trace --text "Anna"
check 0 "val accuracy" "$B" -m "$T/nl.nmz" train examples/names_en.txt --limit 100 --epochs 1
check 0 "" sh -c "\"$B\" -m \"$T/nl.nmz\" train examples/names_en.txt --limit 50 --epochs 1 | grep 'adam lr' | grep -v 'sgd lr'"
check 1 "give --data FILE" "$B" -m tests/data/v3_phase4_names_en.nmz generate --new-only   # a phase-4 file
check 0 " 3\. " "$B" -m tests/data/v3_phase4_names_en.nmz generate --new-only --count 3 --data examples/names_en.txt
check 0 " 4\. " "$B" -m "$T/nl.nmz" generate --count 4 --seed 3
check 0 "gru Tx8->Tx6" "$B" new text embed:8 gru:6 vocab --data examples/names_en.txt --epochs 2 -o "$T/ng.nmz"
check 0 "validation" "$B" -m "$T/ng.nmz" train examples/names_en.txt --limit 100 -o "$T/ng.nmz"
check 0 "UPDATE GATE z" "$B" -m "$T/ng.nmz" trace --text "Anna"
check 0 "GRADIENT SURVIVAL" "$B" compare "$T/n.nmz" "$T/nl.nmz" "$T/ng.nmz" --text "Maximilian"
check 0 "lstm c" "$B" compare "$T/nl.nmz" --data examples/names_en.txt --from 100 --chars 30 --full
check 0 "sigmoid(2.5) = 0.92" "$B" new text embed:4 lstm:6:forget_bias=2.5:xavier dropout:0.2 lstm:5 vocab --data examples/names_ar.txt -o "$T/st.nmz"
check 0 "layer 3 (lstm)" "$B" -m "$T/st.nmz" trace --text "سارة" --layer 2
check 0 "Validating on" "$B" -m "$T/st.nmz" train examples/names_ar.txt --epochs 1 --val-file examples/names_ar.txt
check 0 "orthogonal" "$B" new text embed:4 rnn:6:orthogonal vocab --data examples/names_en.txt
check 0 "strip-marks" "$B" new text embed:4 lstm:4 vocab --data examples/names_ar.txt --strip-diacritics -o "$T/sd.nmz"
check 0 "tashkeel" "$B" -m "$T/sd.nmz" show vocab
check 0 " 2\. " "$B" -m "$T/sd.nmz" generate --count 2 --prime "مُ"
check 1 "fewer recurrent layers" "$B" -m "$T/st.nmz" trace --text "سارة" --layer 3
check 1 "no gated layer" "$B" -m "$T/n.nmz" show gates --text "Anna"
check 1 "not in the vocabulary" "$B" compare "$T/nl.nmz" "$T/st.nmz" --text "Anna"
check 1 "usage: compare" "$B" compare
check 1 "orthogonal init is for recurrent" "$B" new text embed:4:orthogonal vocab --data examples/names_en.txt
check 1 "forget_bias" "$B" new text embed:4 lstm:4:forget_bias=x vocab --data examples/names_en.txt
check 1 "gru needs a number" "$B" new text embed:4 gru vocab --data examples/names_en.txt
check 1 "text models only" "$B" new 2 4 1 --strip-diacritics
check 1 "cannot open\|cannot read" "$B" -m "$T/nl.nmz" train examples/names_en.txt --val-file "$T/missing.txt"

# --- Transformers (phase 5): specs, trace, show attention/params, KV-cache generation, compare
NL='
'
check 0 "Transformer: context 16" "$B" new text embed:8 pos:learned:16 block:heads=2 block:heads=2:relu:dropout=0.1 ln vocab:tied --data examples/names_en.txt -o "$T/g.nmz"
check 0 "weight decay 0.05" "$B" -m "$T/g.nmz" train examples/names_en.txt --epochs 1 --weight-decay 0.05 --warmup 3 -o "$T/g.nmz"
check 0 "ATTENTION AT ONE POSITION" "$B" -m "$T/g.nmz" trace --text "${NL}Anna${NL}Bo"
check 0 "head 2 of 2" "$B" -m "$T/g.nmz" trace --text "Anna" --layer 2 --head 2 --pos 1 --full
check 0 "THE RESIDUAL STREAM" "$B" -m "$T/g.nmz" trace --text "Anna" --ascii
check 0 "AdamW" "$B" -m "$T/g.nmz" step --text "Anna" -o "$T/g2.nmz"
check 1 "context window is 16" "$B" -m "$T/g.nmz" trace --text "abcdefghijklmnopqrstuvwxyz"
check 1 "positions 0..2" "$B" -m "$T/g.nmz" trace --text "Anna" --pos 9
check 1 "fewer attention layers" "$B" -m "$T/g.nmz" trace --text "Anna" --layer 3
check 1 "head is 1-based" "$B" -m "$T/g.nmz" trace --text "Anna" --head 0
check 1 "Transformer models" "$B" -m "$T/nl.nmz" trace --text "Anna" --pos 1
check 0 "WHAT EACH HEAD ATTENDS TO" "$B" -m "$T/g.nmz" show attention --data examples/names_en.txt --chars 300
check 0 "layer 2, head 2" "$B" -m "$T/g.nmz" show attention --text "AnnaMariaBob" --layer 2 --head 2 --full
check 1 "has 2 heads" "$B" -m "$T/g.nmz" show attention --text "Anna" --head 3
check 1 "needs a Transformer" "$B" -m "$T/nl.nmz" show attention --text "Anna"
check 0 "feed-forward" "$B" -m "$T/g.nmz" show params
check 0 "reuses the embedding" "$B" -m "$T/g.nmz" show params
check 0 "   3. " "$B" -m "$T/g.nmz" generate --count 3 --seed 2
check 0 "   3. " "$B" -m "$T/g.nmz" generate --count 3 --seed 2 --no-cache --context 8
check 1 "longer than the model" "$B" -m "$T/g.nmz" generate --context 99
check 0 "gpt x" "$B" compare "$T/nl.nmz" "$T/g.nmz" --data examples/names_en.txt --from 100 --chars 16
check 1 "sees at most 16" "$B" compare "$T/g.nmz" --data examples/names_en.txt --chars 40
check 0 "Transformer: context 8" "$B" new text embed:8 pos:sin:8 attn:heads=2 ffn:16:relu ln vocab --data examples/names_ar.txt -o "$T/ga.nmz"
check 0 "validation" "$B" -m "$T/ga.nmz" train examples/names_ar.txt --epochs 1 -o "$T/ga.nmz"
check 0 "ATTENTION HEATMAP" "$B" -m "$T/ga.nmz" trace --text "سارة"
check 0 "WHAT EACH HEAD" "$B" -m "$T/ga.nmz" show attention --data examples/names_ar.txt --chars 100 --ascii
check 0 "\[preset\]" "$B" new --preset names-gpt --epochs 1 -o "$T/ngp.nmz"
check 1 "carry does not apply" "$B" new text embed:8 pos:learned:8 block vocab:tied --data examples/names_en.txt --carry
check 1 "need the .text. input" "$B" new 4 block 2
check 1 "heads needs a positive" "$B" new text embed:8 block:heads=x vocab --data examples/names_en.txt
check 1 "do not split into 4 heads" "$B" new text embed:10 block:heads=4 vocab --data examples/names_en.txt
check 1 "pos needs a kind" "$B" new text embed:8 pos:foo:8 vocab --data examples/names_en.txt
check 1 "longer than the context" "$B" new text embed:8 pos:learned:8 block vocab:tied --data examples/names_en.txt --seq-len 20
check 1 "weight-decay needs" "$B" -m "$T/g.nmz" train examples/names_en.txt --weight-decay -1
check 1 "warmup needs" "$B" -m "$T/g.nmz" train examples/names_en.txt --warmup x
check 1 "see .help new." "$B" new text embed:8 block:foo=1 vocab --data examples/names_en.txt

if [ -f data/alice.txt ]; then
    check 0 "\[preset\]" "$B" new --preset char-rnn --epochs 1 -o "$T/c.nmz"
    check 0 "validation" "$B" -m "$T/c.nmz" train data/alice.txt --limit 60 -o "$T/c.nmz"
    check 0 "" "$B" -m "$T/c.nmz" generate --length 40 --prime "Alice"
    check 0 "seed 2" "$B" -m "$T/c.nmz" generate --length 20 --count 2 --seed 1
    check 0 "\[preset\]" "$B" new --preset char-lstm --epochs 1 -o "$T/cl.nmz"
    check 0 "validation" "$B" -m "$T/cl.nmz" train data/alice.txt --limit 40 -o "$T/cl.nmz"
    check 0 "GRADIENT SURVIVAL" "$B" compare "$T/c.nmz" "$T/cl.nmz" --data data/alice.txt --from 1000
    check 0 "\[preset\]" "$B" new --preset char-gpt --epochs 1 -o "$T/cg.nmz"
    check 0 "validation" "$B" -m "$T/cg.nmz" train data/alice.txt --limit 40 -o "$T/cg.nmz"
    check 0 "gpt x" "$B" compare "$T/cl.nmz" "$T/cg.nmz" --data data/alice.txt --from 1000
    check 0 "" "$B" -m "$T/cg.nmz" generate --length 300 --prime "Alice" --seed 4
else
    echo "skip char-rnn checks (run make data-text)"
fi
if [ -f data/kalila.txt ]; then
    check 0 "\[preset\]" "$B" new --preset arabic-text --epochs 1 --strip-diacritics -o "$T/k.nmz"
    check 0 "validation" "$B" -m "$T/k.nmz" train data/kalila.txt --limit 20 -o "$T/k.nmz"
    check 0 "" "$B" -m "$T/k.nmz" generate --length 30 --prime "قال"
    check 0 "\[preset\]" "$B" new --preset arabic-gpt --epochs 1 -o "$T/kg.nmz"
    check 0 "validation" "$B" -m "$T/kg.nmz" train data/kalila.txt --limit 20 -o "$T/kg.nmz"
    check 0 "WHAT EACH HEAD" "$B" -m "$T/kg.nmz" show attention --data data/kalila.txt --chars 500
    check 0 "ATTENTION AT ONE POSITION" "$B" -m "$T/kg.nmz" trace --text "قال دمنة"
else
    echo "skip Arabic text checks (run make data-text)"
fi

# --- MNIST (only when `make data` has been run)
if [ -f data/train-images-idx3-ubyte ] && [ -f data/t10k-labels-idx1-ubyte ]; then
    check 0 "\[preset\]" "$B" new --preset mnist-cnn --epochs 1 -o "$T/m.nmz"
    check 0 "200 samples" "$B" -m "$T/m.nmz" train mnist --limit 200 -o "$T/m.nmz"
    check 0 "accuracy" "$B" -m "$T/m.nmz" test mnist --limit 100
    check 0 "prediction" "$B" -m "$T/m.nmz" show sample 3
    check 0 "where this filter responds most" "$B" -m "$T/m.nmz" trace --sample 1
else
    echo "skip MNIST checks (run make data)"
fi

# --- bad input never crashes
check 1 "usage" "$B" new
check 1 "usage" "$B" new 2
check 1 "positive integer" "$B" new 2 0 1
check 1 "positive integer" "$B" new 2 x 1
check 1 "is not an activation" "$B" new 2 4:swish 1
check 1 "is not an activation" "$B" new 2 4:relu:magic 1
check 1 "only leaky_relu" "$B" new 2 4:tanh=3 1
check 1 "softmax" "$B" new 2 4 1 --loss ce
check 1 "needs a value" "$B" new 2 4 1 --lr
check 1 "positive number" "$B" new 2 4 1 --lr -3
check 1 "momentum" "$B" new 2 4 1 --momentum 1.5
check 1 "unknown option" "$B" new 2 4 1 --bogus
check 1 "output" "$B" -m "$T/xor.nmz" train examples/xor_and.csv
check 1 "cannot open" "$B" -m "$T/xor.nmz" train "$T/missing.csv"
check 1 "expected 2 input" "$B" -m "$T/xor.nmz" predict 1
check 1 "too many" "$B" -m "$T/xor.nmz" predict 1 2 3
check 1 "not a number" "$B" -m "$T/xor.nmz" predict 1 abc
check 1 "target is needed" "$B" -m "$T/xor.nmz" trace 1 0
check 1 "expected 1 target" "$B" -m "$T/xor.nmz" trace 1 0 --target
check 1 "cannot open" "$B" -m "$T/nope.nmz" show
printf 'garbage that is not a model' >"$T/bad.nmz"
check 1 "not a NeuroMZ model" "$B" load "$T/bad.nmz"
head -c 100 "$T/xor.nmz" >"$T/trunc.nmz"
check 1 "end of file" "$B" load "$T/trunc.nmz"
check 2 "needs a file" "$B" show -m
printf '1,2,3\n4,5\n' >"$T/ragged.csv"
check 1 "expected 3 columns" "$B" -m "$T/xor.nmz" train "$T/ragged.csv"

# --- the interactive shell, fed from stdin
printf 'help\nnew 2 3 1\nshow\nsave %s/s.nmz\nclear\nload %s/s.nmz\npredict 1 1\nquit\n' "$T" "$T" >"$T/s1"
stdin_check 0 "Loaded" "$T/s1"
printf 'predict 1 2\nnew 2 2 1\n"unterminated\nnosuchcmd\n\n   \n# comment only\nshow' >"$T/s2"
stdin_check 1 "unterminated quote" "$T/s2"     # errors → exit 1, EOF without newline handled
head -c 300000 /dev/zero | tr '\0' 'a' >"$T/long"
stdin_check 1 "unknown command" "$T/long"      # one 300 KB token
head -c 20000 /dev/urandom >"$T/random"
check 1 "" sh -c "\"$B\" < \"$T/random\""       # binary garbage
: >"$T/empty"
stdin_check 0 "" "$T/empty"                     # immediate EOF

echo
echo "$count CLI checks, $fails failure(s)"
[ "$fails" -eq 0 ]
