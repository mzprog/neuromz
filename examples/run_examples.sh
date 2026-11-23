#!/bin/sh
# run_examples.sh — train the bundled examples with the CLI and report results.
# Usage: sh examples/run_examples.sh [path/to/neuromz]
set -e
B=${1:-./build/neuromz}
OUT=build/examples
mkdir -p "$OUT"

echo "== XOR, quick mode (2 -> 4 -> 1)"
"$B" new 2 4 1 -o "$OUT/xor.nmz" >/dev/null
"$B" -m "$OUT/xor.nmz" train examples/xor.csv --quiet -o "$OUT/xor.nmz"
for x in "0 0" "0 1" "1 0" "1 1"; do printf '  %s -> ' "$x"; "$B" -m "$OUT/xor.nmz" predict $x | sed 's/output: *//'; done

echo
echo "== XOR and AND at once: two sigmoid outputs (multi-label, so --loss mse)"
"$B" new 2 4 2:sigmoid --loss mse -o "$OUT/xor_and.nmz" >/dev/null
"$B" -m "$OUT/xor_and.nmz" train examples/xor_and.csv --quiet -o "$OUT/xor_and.nmz"
for x in "0 0" "0 1" "1 0" "1 1"; do printf '  %s -> ' "$x"; "$B" -m "$OUT/xor_and.nmz" predict $x | sed 's/output: *//'; done

echo
echo "== Iris, quick mode (4 -> 8 -> 3, softmax + cross-entropy), 120 train / 30 test"
"$B" new 4 8 3 -o "$OUT/iris.nmz" >/dev/null
"$B" -m "$OUT/iris.nmz" train examples/iris_train.csv --quiet -o "$OUT/iris.nmz"
"$B" -m "$OUT/iris.nmz" test examples/iris_test.csv

echo
echo "== Iris, advanced mode (relu/He, leaky relu, mini-batch 16)"
"$B" new 4 16:relu:he 8:leaky_relu=0.05 3:softmax --loss ce --lr 0.01 --batch 16 --seed 7 -o "$OUT/iris_adv.nmz" >/dev/null
"$B" -m "$OUT/iris_adv.nmz" train examples/iris_train.csv --quiet -o "$OUT/iris_adv.nmz"
"$B" -m "$OUT/iris_adv.nmz" test examples/iris_test.csv
