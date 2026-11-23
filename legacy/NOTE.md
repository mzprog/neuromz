# Legacy NeuroMZ (v1, 2017–2020)

These are the original v1 sources, kept read-only for reference. The git tag
`v1-legacy` points at the last v1 commit.

They do **not** build as-is (`src/main.c:238` calls `showDet()` with too few
arguments since 2020-08-06) and contain known math and memory bugs, all of
which v2 was designed to fix. See `docs/DESIGN.md` § "What v2 fixes from v1".
