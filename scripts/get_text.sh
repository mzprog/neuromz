#!/bin/sh
# get_text.sh — download a public-domain book for the character RNN into
# data/ (gitignored) and verify its checksum.
# Usage: sh scripts/get_text.sh [DIR]      (default: data)
#
# Lewis Carroll, "Alice's Adventures in Wonderland" (1865), Project Gutenberg
# eBook #11 — public domain in the USA. www.gutenberg.org refuses automated
# downloads, so two official Gutenberg mirrors are tried. The SHA-256 below
# is of the file as served on 2026-11-23 (identical on both mirrors). Only
# the book text between the "*** START" and "*** END" lines is kept, i.e.
# without the Project Gutenberg header/footer and license text.
set -eu
DIR=${1:-data}
NAME=11-0.txt
SHA256=a3a27f8edbf7fcd9b8ba8435494440e24952deaa3e2f2d65192d4cb7ca403754
MIRRORS="https://gutenberg.pglaf.org/1/11 https://mirrors.xmission.com/gutenberg/1/11"
OUT="$DIR/alice.txt"

mkdir -p "$DIR"
if [ -f "$OUT" ]; then
    echo "ok      $OUT (already present)"
    exit 0
fi

sha_of() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
    else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

raw="$DIR/$NAME.part"
for m in $MIRRORS; do
    echo "fetch   $m/$NAME"
    if command -v curl >/dev/null 2>&1; then curl -fsSL --retry 2 -o "$raw" "$m/$NAME" && break
    else wget -q -O "$raw" "$m/$NAME" && break; fi
done
[ -f "$raw" ] || { echo "error: could not download $NAME from any mirror" >&2; exit 1; }
got=$(sha_of "$raw")
if [ "$got" != "$SHA256" ]; then
    echo "error: checksum mismatch for $NAME: got $got, expected $SHA256" >&2
    echo "       (the mirror may have updated the file; check it and update SHA256 in $0)" >&2
    rm -f "$raw"
    exit 1
fi
echo "verify  $NAME sha256 $got"
# keep the text between the START and END markers
awk '/^\*\*\* START OF/ { on = 1; next } /^\*\*\* END OF/ { on = 0 } on' "$raw" >"$OUT.part"
mv "$OUT.part" "$OUT"
rm -f "$raw"
echo "ok      $OUT ($(wc -c <"$OUT") bytes)"
