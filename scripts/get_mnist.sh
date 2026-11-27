#!/bin/sh
# get_mnist.sh — download MNIST into data/ (gitignored) and verify checksums.
# Usage: sh scripts/get_mnist.sh [DIR]      (default: data)
#
# The original host (yann.lecun.com) now redirects/refuses automated
# downloads, so two public mirrors are tried in order. The MD5 sums of the
# .gz files are the ones published by torchvision (an independent source);
# after verification the files are decompressed to the raw IDX format that
# NeuroMZ reads, and those are checked against a SHA-256 too. On Windows use
# get_mnist.ps1 (same sums) or `cmake --build build --target data`.
set -eu
DIR=${1:-data}
MIRRORS="https://storage.googleapis.com/cvdf-datasets/mnist https://ossci-datasets.s3.amazonaws.com/mnist"

mkdir -p "$DIR"

sha_of() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
    else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

md5_of() {
    if command -v md5sum >/dev/null 2>&1; then md5sum "$1" | cut -d' ' -f1
    else md5 -q "$1"; fi # macOS
}

fetch() { # fetch URL OUT
    if command -v curl >/dev/null 2>&1; then curl -fsSL --retry 2 -o "$2" "$1"
    else wget -q -O "$2" "$1"; fi
}

while read -r name md5 sha; do
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
    gunzip -c "$gz" >"$raw.part"
    got=$(sha_of "$raw.part")
    if [ "$got" != "$sha" ]; then
        echo "error: decompressed $name has sha256 $got, expected $sha" >&2
        rm -f "$raw.part"
        exit 1
    fi
    mv "$raw.part" "$raw"
    rm -f "$gz"
    echo "ok      $raw"
done <<'EOF'
train-images-idx3-ubyte.gz f68b3c2dcbeaaa9fbdd348bbdeb94873 ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db
train-labels-idx1-ubyte.gz d53e105ee54ea40749a09fcbcd1e9432 65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5
t10k-images-idx3-ubyte.gz 9fb629c4189551a2d022fa330f9573f3 0fa7898d509279e482958e8ce81c8e77db3f2f8254e26661ceb7762c4d494ce7
t10k-labels-idx1-ubyte.gz ec29112dd5afa0611ce80d1b7f02629c ff7bcfd416de33731a308c3f266cc351222c34898ecbeaf847f06e48f7ec33f2
EOF
echo "MNIST is ready in $DIR/"
