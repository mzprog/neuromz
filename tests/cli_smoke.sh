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
check 0 "trace X1" "$B" help trace
check 1 "unknown command" "$B" frobnicate

# --- nothing loaded: every command fails cleanly
for c in show predict test train trace step save gradcheck set reinit; do
    check 1 "no network loaded" "$B" $c 1 2
done
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

# --- bad input never crashes
check 1 "usage" "$B" new
check 1 "usage" "$B" new 2
check 1 "positive integer" "$B" new 2 0 1
check 1 "positive integer" "$B" new 2 x 1
check 1 "unknown activation" "$B" new 2 4:swish 1
check 1 "unknown init" "$B" new 2 4:relu:magic 1
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
