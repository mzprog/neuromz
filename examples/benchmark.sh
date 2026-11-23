#!/bin/sh
# benchmark.sh — convergence over many seeds; produces the README table.
# Usage: sh examples/benchmark.sh [path/to/neuromz] [N_SEEDS]
B=${1:-./build/neuromz}
N=${2:-20}
T=$(mktemp -d "${TMPDIR:-/tmp}/nmz_bench.XXXXXX") || exit 1
trap 'rm -rf "$T"' EXIT

# run NAME TRAIN_CSV TEST_CSV(or -) NEW_ARGS...
run() {
    name=$1; train=$2; test=$3; shift 3
    conv=0; epochs=""; accsum=0; accn=0
    seed=1
    while [ "$seed" -le "$N" ]; do
        "$B" new "$@" --seed "$seed" -o "$T/m.nmz" >/dev/null || exit 1
        line=$("$B" -m "$T/m.nmz" train "$train" --quiet -o "$T/m.nmz" | grep -E '^(Converged|Stopped)')
        case "$line" in
        Converged*)
            conv=$((conv + 1))
            epochs="$epochs $(echo "$line" | sed 's/Converged after \([0-9]*\).*/\1/')" ;;
        esac
        if [ "$test" != "-" ]; then
            acc=$("$B" -m "$T/m.nmz" test "$test" | sed -n 's/.*accuracy *\([0-9.]*\)%.*/\1/p')
            accsum=$(echo "$accsum + $acc" | bc -l 2>/dev/null || echo "$accsum")
            accn=$((accn + 1))
        fi
        seed=$((seed + 1))
    done
    med=$(echo "$epochs" | tr ' ' '\n' | sed '/^$/d' | sort -n | awk '{a[NR]=$1} END {if (NR) print a[int((NR+1)/2)]; else print "-"}')
    if [ "$accn" -gt 0 ] && command -v bc >/dev/null; then
        mean=$(echo "scale=1; $accsum / $accn" | bc -l)
        printf '| %-44s | %2d/%d | %6s | %5s%% |\n' "$name" "$conv" "$N" "$med" "$mean"
    else
        printf '| %-44s | %2d/%d | %6s | %6s |\n' "$name" "$conv" "$N" "$med" "-"
    fi
}

echo "| example (seeds 1..$N) | converged | median epochs | mean test accuracy |"
echo "|---|---|---|---|"
run "XOR, quick \`new 2 4 1\`" examples/xor.csv - 2 4 1
run "XOR, minimal \`new 2 2 1\`" examples/xor.csv - 2 2 1
run "XOR+AND, \`new 2 4 2:sigmoid --loss mse\`" examples/xor_and.csv - 2 4 2:sigmoid --loss mse
run "Iris, quick \`new 4 8 3\`" examples/iris_train.csv examples/iris_test.csv 4 8 3
run "Iris, \`new 4 16:relu:he 8:leaky_relu=0.05 3:softmax\` lr 0.01 batch 16" examples/iris_train.csv examples/iris_test.csv \
    4 16:relu:he 8:leaky_relu=0.05 3:softmax --lr 0.01 --batch 16
