#!/bin/sh
# get_text.sh — download the public-domain texts for the character models
# into data/ (gitignored) and verify their checksums.
# Usage: sh scripts/get_text.sh [DIR]      (default: data)
#
# 1. data/alice.txt — Lewis Carroll, "Alice's Adventures in Wonderland"
#    (1865), Project Gutenberg eBook #11, public domain in the USA.
#    www.gutenberg.org refuses automated downloads, so two official
#    Gutenberg mirrors are tried. The SHA-256 is of the file as served on
#    2026-11-23 (identical on both mirrors). Only the book text between the
#    "*** START" and "*** END" lines is kept (no Gutenberg header/license).
# 2. data/kalila.txt — Ibn al-Muqaffaʿ, "Kalila wa-Dimna" (Arabic, 8th
#    century, public domain), the 19 chapters as transcribed on Arabic
#    Wikisource (CC BY-SA 4.0). Each chapter is fetched at a fixed page
#    revision (oldid), which never changes, so per-revision SHA-256 sums
#    are stable. The wiki markup (header template, links, image, category)
#    is removed by scripts/strip_wikitext.awk; one paragraph per line.
# Both prepared files are checked against a SHA-256 as well. On Windows use
# get_text.ps1 (the same steps and sums) or `cmake --build build --target data-text`.
set -eu
DIR=${1:-data}
HERE=$(dirname "$0")
mkdir -p "$DIR"

sha_of() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
    else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

fetch() { # fetch URL OUT
    if command -v curl >/dev/null 2>&1; then curl -fsSL --retry 2 -A "neuromz-get-text/1.0" -o "$2" "$1"
    else wget -q -U "neuromz-get-text/1.0" -O "$2" "$1"; fi
}

# the prepared file must be byte-identical everywhere (get_text.ps1 checks
# the same sums), so every result in the README is reproducible
check_out() { # check_out FILE SHA256
    got=$(sha_of "$1")
    [ "$got" = "$2" ] && return 0
    echo "error: $1 has sha256 $got, expected $2 (different awk or text tools?)" >&2
    rm -f "$1"
    return 1
}

get_alice() {
    NAME=11-0.txt
    SHA256=a3a27f8edbf7fcd9b8ba8435494440e24952deaa3e2f2d65192d4cb7ca403754
    MIRRORS="https://gutenberg.pglaf.org/1/11 https://mirrors.xmission.com/gutenberg/1/11"
    OUT="$DIR/alice.txt"
    if [ -f "$OUT" ]; then
        echo "ok      $OUT (already present)"
        return 0
    fi
    raw="$DIR/$NAME.part"
    rm -f "$raw"
    for m in $MIRRORS; do
        echo "fetch   $m/$NAME"
        fetch "$m/$NAME" "$raw" && break
        rm -f "$raw"
    done
    [ -f "$raw" ] || { echo "error: could not download $NAME from any mirror" >&2; return 1; }
    got=$(sha_of "$raw")
    if [ "$got" != "$SHA256" ]; then
        echo "error: checksum mismatch for $NAME: got $got, expected $SHA256" >&2
        echo "       (the mirror may have updated the file; check it and update SHA256 in $0)" >&2
        rm -f "$raw"
        return 1
    fi
    echo "verify  $NAME sha256 $got"
    # keep the text between the START and END markers
    awk '/^\*\*\* START OF/ { on = 1; next } /^\*\*\* END OF/ { on = 0 } on' "$raw" >"$OUT.part"
    check_out "$OUT.part" 2c9eddc277e75b19ce8e5e884e1ee57f2341fa798be0e30057ca82d32366e4af || { rm -f "$raw"; return 1; }
    mv "$OUT.part" "$OUT"
    rm -f "$raw"
    echo "ok      $OUT ($(wc -c <"$OUT") bytes)"
}

# Kalila wa-Dimna: page revision, SHA-256 of its raw wikitext — in the
# book's order (the index page https://ar.wikisource.org/wiki/كليلة_ودمنة)
KALILA_REVS="
488570 e6c20d524aeb5a28e77dc7547664ded0809c1b283387cedb8fae6141c0b61dbc
245550 7ec8e336e98828ed2fb1d1c1258fa22fa17fecf3dfb2daf2128e45c0d00901ad
245551 c0439c96172f72eb44c86e067555ba44943dedb44bcf63bb534cd2f73911d22b
141243 3c1b1f9009952a7cc63e9799958ae6f5059b1c0f73d8bd8d301482e885b37394
272368 1bb1ab2f940e799d3bc6315d190f61fbae6f0120047833d1b062e37feb4dd8dc
290965 107dc47453f5c0138842c2561239580bb84a3948923e07d0c17a9523264c80ee
245541 45f5d526726e83adb150f02d652826f35fd6716e0accf00b0fc09bc9134bfd86
285510 1fa9b2daf0e4a1f7b670a93fc6b4026fd1b4ab49be128b60cc8024e60d66b882
231360 7e91090670ce60ea93db7e2e67078b8f77a642a12e272c6f463173ea57f6ee89
251292 0e8e4373dbaa3f121abc25fbbc8621740f90c19877998811e1b306948b617230
231362 20846ca4f50efd4114ccbc1229799d10de61064a845a6bb9f6a097141c6498fe
231363 b7a5aa9ac135254b406ceaea168fe0e44775ef2c9e48b031ec9a160874cc31cc
302554 210617ce591afcf6f766301250addd178d2ba8f95e515507c684f19cdb3275d2
231372 201a4ffa9f68905f7a4df9e05f3f4ec5e954b71be39df804b8340c7e27a630bc
374244 ab8688bd587efb310d54c0f084a90ef1f10581811077a2641e2c12d7e88ea43a
231371 2f8506a25a6211259a8758d44f001a839dd65497746451f62e1214e03600ecff
381514 a91887cd095491c699bb0bbfcf495458d8f0cf5545adac74ff62b9c642235460
231368 39e5c7eb6650429cc7ed7424bfa03ac812a11da4e44ad990d173610d32b97497
231369 c0479eb75d3b07b4fe2fb9782e84d622d0d4f26b58b230179125f037449318fb
"

get_kalila() {
    OUT="$DIR/kalila.txt"
    if [ -f "$OUT" ]; then
        echo "ok      $OUT (already present)"
        return 0
    fi
    tmp="$DIR/kalila.part.d"
    rm -rf "$tmp"
    mkdir -p "$tmp"
    n=0
    echo "fetch   19 chapters of Kalila wa-Dimna from ar.wikisource.org (fixed revisions)"
    echo "$KALILA_REVS" | while read -r rev sum; do
        [ -n "$rev" ] || continue
        n=$((n + 1))
        f="$tmp/$(printf %02d "$n").txt"
        if ! fetch "https://ar.wikisource.org/w/index.php?oldid=$rev&action=raw" "$f"; then
            echo "error: could not download revision $rev" >&2
            exit 1
        fi
        got=$(sha_of "$f")
        if [ "$got" != "$sum" ]; then
            echo "error: checksum mismatch for revision $rev: got $got, expected $sum" >&2
            exit 1
        fi
    done || { rm -rf "$tmp"; return 1; }
    echo "verify  19 revisions sha256 ok"
    cat "$tmp"/*.txt | LC_ALL=C awk -f "$HERE/strip_wikitext.awk" >"$OUT.part"
    check_out "$OUT.part" 709e5b8fc92a2e3393d57382a046bba51713d4d511bddf9e62cfbb8def0ae90a || { rm -rf "$tmp"; return 1; }
    mv "$OUT.part" "$OUT"
    rm -rf "$tmp"
    echo "ok      $OUT ($(wc -c <"$OUT") bytes)"
}

rc=0
get_alice || rc=1
get_kalila || rc=1
exit $rc
