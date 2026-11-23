#!/bin/sh
# get_mnist.sh — download MNIST into data/ (gitignored) and verify checksums.
# Usage: sh scripts/get_mnist.sh [DIR]      (default: data)
#
# The original host (yann.lecun.com) now redirects/refuses automated
# downloads, so two public mirrors are tried in order. The MD5 sums of the
# .gz files are the ones published by torchvision (an independent source);
# after verification the files are decompressed to the raw IDX format that
# NeuroMZ reads.
set -eu
DIR=${1:-data}
MIRRORS="https://storage.googleapis.com/cvdf-datasets/mnist https://ossci-datasets.s3.amazonaws.com/mnist"

mkdir -p "$DIR"

md5_of() {
    if command -v md5sum >/dev/null 2>&1; then md5sum "$1" | cut -d' ' -f1
    else md5 -q "$1"; fi # macOS
}

fetch() { # fetch URL OUT
    if command -v curl >/dev/null 2>&1; then curl -fsSL --retry 2 -o "$2" "$1"
    else wget -q -O "$2" "$1"; fi
}

while read -r name md5; do
    [ -n "$name" ] || continue
    raw="$DIR/${name%.gz}"
    if [ -f "$raw" ]; then
        echo "ok      $raw (already present)"
        continue
    fi
    gz="$DIR/$name"
    if [ ! -f "$gz" ] || [ "$(md5_of "$gz")" != "$md5" ]; then
        rm -f "$gz"
        for m in $MIRRORS; do
            echo "fetch   $m/$name"
            if fetch "$m/$name" "$gz.part"; then mv "$gz.part" "$gz"; break; fi
            rm -f "$gz.part"
        done
    fi
    [ -f "$gz" ] || { echo "error: could not download $name from any mirror" >&2; exit 1; }
    got=$(md5_of "$gz")
    if [ "$got" != "$md5" ]; then
        echo "error: checksum mismatch for $name: got $got, expected $md5" >&2
        rm -f "$gz"
        exit 1
    fi
    echo "verify  $name md5 $got"
    gunzip -c "$gz" >"$raw.part" && mv "$raw.part" "$raw"
    rm -f "$gz"
    echo "ok      $raw"
done <<'EOF'
train-images-idx3-ubyte.gz f68b3c2dcbeaaa9fbdd348bbdeb94873
train-labels-idx1-ubyte.gz d53e105ee54ea40749a09fcbcd1e9432
t10k-images-idx3-ubyte.gz 9fb629c4189551a2d022fa330f9573f3
t10k-labels-idx1-ubyte.gz ec29112dd5afa0611ce80d1b7f02629c
EOF
echo "MNIST is ready in $DIR/"
