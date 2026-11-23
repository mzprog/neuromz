/* text.c — UTF-8 decoding, character vocabularies and text datasets.
 *
 * Characters are Unicode code points, never bytes: "محمد" is 4 tokens
 * (8 bytes of UTF-8), "é" is one. Invalid UTF-8 never aborts a load; each
 * offending byte becomes U+FFFD and is counted, so the caller can report it. */
#include "internal.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ UTF-8 */

nmz_status nmz_utf8_decode(const char *str, size_t n, uint32_t **out, size_t *out_n, size_t *invalid) {
    const unsigned char *s = (const unsigned char *)str;
    uint32_t *cps;
    size_t i = 0, k = 0, bad = 0;
    if (!out || !out_n || (!str && n)) return nmz__fail(NMZ_ERR_ARG, "NULL argument");
    cps = malloc((n ? n : 1) * sizeof *cps); /* at most one code point per byte */
    if (!cps) return nmz__fail(NMZ_ERR_NOMEM, "out of memory decoding %zu bytes", n);
    while (i < n) {
        unsigned char c = s[i];
        uint32_t cp, min;
        int len, j, ok = 1;
        if (c < 0x80) {
            cps[k++] = c;
            i++;
            continue;
        } else if ((c & 0xE0) == 0xC0) {
            len = 2;
            cp = c & 0x1F;
            min = 0x80;
        } else if ((c & 0xF0) == 0xE0) {
            len = 3;
            cp = c & 0x0F;
            min = 0x800;
        } else if ((c & 0xF8) == 0xF0) {
            len = 4;
            cp = c & 0x07;
            min = 0x10000;
        } else {
            len = 0; /* stray continuation byte or 0xF8..0xFF */
            cp = 0;
            min = 0;
            ok = 0;
        }
        for (j = 1; ok && j < len; j++) {
            if (i + (size_t)j >= n || (s[i + j] & 0xC0) != 0x80) ok = 0; /* truncated */
            else cp = (cp << 6) | (s[i + j] & 0x3F);
        }
        if (ok && (cp < min || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF))) ok = 0; /* overlong/surrogate */
        if (ok) {
            cps[k++] = cp;
            i += (size_t)len;
        } else {
            cps[k++] = NMZ_REPLACEMENT_CHAR; /* replace one byte and resynchronise */
            bad++;
            i++;
        }
    }
    *out = cps;
    *out_n = k;
    if (invalid) *invalid = bad;
    return NMZ_OK;
}

int nmz_utf8_encode(uint32_t cp, char out[5]) {
    unsigned char *o = (unsigned char *)out;
    int n;
    if (cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) {
        out[0] = '\0';
        return 0;
    }
    if (cp < 0x80) {
        o[0] = (unsigned char)cp;
        n = 1;
    } else if (cp < 0x800) {
        o[0] = (unsigned char)(0xC0 | (cp >> 6));
        o[1] = (unsigned char)(0x80 | (cp & 0x3F));
        n = 2;
    } else if (cp < 0x10000) {
        o[0] = (unsigned char)(0xE0 | (cp >> 12));
        o[1] = (unsigned char)(0x80 | ((cp >> 6) & 0x3F));
        o[2] = (unsigned char)(0x80 | (cp & 0x3F));
        n = 3;
    } else {
        o[0] = (unsigned char)(0xF0 | (cp >> 18));
        o[1] = (unsigned char)(0x80 | ((cp >> 12) & 0x3F));
        o[2] = (unsigned char)(0x80 | ((cp >> 6) & 0x3F));
        o[3] = (unsigned char)(0x80 | (cp & 0x3F));
        n = 4;
    }
    out[n] = '\0';
    return n;
}

/* ------------------------------------------------------------- vocabulary */

void nmz_vocab_free(nmz_vocab *v) {
    if (!v) return;
    free(v->codepoints);
    free(v->counts);
    free(v);
}

nmz_status nmz_vocab_create(nmz_vocab **out, int size, const uint32_t *codepoints, const uint64_t *counts,
                            int items) {
    nmz_vocab *v;
    int i;
    if (!out || size < 1 || size > NMZ_MAX_VOCAB || !codepoints) return nmz__fail(NMZ_ERR_ARG, "invalid vocabulary");
    *out = NULL;
    for (i = 0; i < size; i++) {
        if (codepoints[i] > 0x10FFFF) return nmz__fail(NMZ_ERR_ARG, "invalid code point in vocabulary");
        if (i && codepoints[i] <= codepoints[i - 1])
            return nmz__fail(NMZ_ERR_ARG, "vocabulary code points must be strictly increasing");
    }
    v = calloc(1, sizeof *v);
    if (!v) return nmz__fail(NMZ_ERR_NOMEM, "out of memory");
    v->codepoints = malloc((size_t)size * sizeof *v->codepoints);
    v->counts = calloc((size_t)size, sizeof *v->counts);
    if (!v->codepoints || !v->counts) {
        nmz_vocab_free(v);
        return nmz__fail(NMZ_ERR_NOMEM, "out of memory");
    }
    memcpy(v->codepoints, codepoints, (size_t)size * sizeof *codepoints);
    if (counts) memcpy(v->counts, counts, (size_t)size * sizeof *counts);
    v->size = size;
    v->items = items != 0;
    *out = v;
    return NMZ_OK;
}

nmz_status nmz_vocab_copy(nmz_vocab **out, const nmz_vocab *v) {
    nmz_status s;
    if (!v) return nmz__fail(NMZ_ERR_ARG, "vocabulary is NULL");
    s = nmz_vocab_create(out, v->size, v->codepoints, v->counts, v->items);
    if (s == NMZ_OK) (*out)->normalize = v->normalize;
    return s;
}

static int cmp_u32(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return x < y ? -1 : x > y;
}

nmz_status nmz_vocab_build(nmz_vocab **out, const uint32_t *cps, size_t n) {
    uint32_t *sorted, *uniq;
    uint64_t *counts;
    size_t i, u = 0;
    nmz_status s;
    if (!out || (!cps && n)) return nmz__fail(NMZ_ERR_ARG, "NULL argument");
    if (n == 0) return nmz__fail(NMZ_ERR_FORMAT, "the text is empty");
    sorted = malloc(n * sizeof *sorted);
    if (!sorted) return nmz__fail(NMZ_ERR_NOMEM, "out of memory");
    memcpy(sorted, cps, n * sizeof *cps);
    qsort(sorted, n, sizeof *sorted, cmp_u32);
    uniq = malloc(n * sizeof *uniq);
    counts = calloc(n, sizeof *counts);
    if (!uniq || !counts) {
        free(sorted);
        free(uniq);
        free(counts);
        return nmz__fail(NMZ_ERR_NOMEM, "out of memory");
    }
    for (i = 0; i < n; i++) {
        if (i == 0 || sorted[i] != sorted[i - 1]) uniq[u++] = sorted[i];
        counts[u - 1]++;
    }
    free(sorted);
    if (u > NMZ_MAX_VOCAB) {
        free(uniq);
        free(counts);
        return nmz__fail(NMZ_ERR_FORMAT, "the text has %zu distinct characters (limit %d)", u, NMZ_MAX_VOCAB);
    }
    s = nmz_vocab_create(out, (int)u, uniq, counts, 0);
    free(uniq);
    free(counts);
    return s;
}

int nmz_vocab_index(const nmz_vocab *v, uint32_t cp) {
    int lo = 0, hi;
    if (!v) return -1;
    hi = v->size - 1;
    while (lo <= hi) { /* binary search: code points are sorted */
        int mid = (lo + hi) / 2;
        if (v->codepoints[mid] == cp) return mid;
        if (v->codepoints[mid] < cp) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}

/* ----------------------------------------------------------- text dataset */

nmz_text_opts nmz_text_defaults(void) {
    nmz_text_opts o;
    o.seq_len = 64;
    o.items = 0;
    o.vocab = NULL;
    o.normalize = 0;
    return o;
}

static nmz_status read_file(const char *path, char **buf, size_t *len) {
    FILE *f = fopen(path, "rb");
    size_t cap = 65536, n = 0, got;
    char *b;
    if (!f) return nmz__fail(NMZ_ERR_IO, "cannot open '%s': %s", path, strerror(errno));
    b = malloc(cap);
    if (!b) {
        fclose(f);
        return nmz__fail(NMZ_ERR_NOMEM, "out of memory");
    }
    while ((got = fread(b + n, 1, cap - n, f)) > 0) {
        n += got;
        if (n == cap) {
            char *p = realloc(b, cap *= 2);
            if (!p) {
                free(b);
                fclose(f);
                return nmz__fail(NMZ_ERR_NOMEM, "out of memory reading '%s'", path);
            }
            b = p;
        }
    }
    if (ferror(f)) {
        free(b);
        fclose(f);
        return nmz__fail(NMZ_ERR_IO, "error reading '%s'", path);
    }
    fclose(f);
    *buf = b;
    *len = n;
    return NMZ_OK;
}

/* BOM, CR LF / CR → LF; in items mode: drop empty lines, trim trailing
 * spaces, start with '\n' and end every item with '\n'. In place. */
size_t nmz_text_normalize(uint32_t *c, size_t n, unsigned flags) {
    size_t i, k = 0;
    if (!c) return 0;
    for (i = 0; i < n; i++) {
        uint32_t ch = c[i];
        if ((flags & NMZ_TEXT_STRIP_DIACRITICS) && ((ch >= 0x064B && ch <= 0x0652) || ch == 0x0670 || ch == 0x0640))
            continue;
        if ((flags & NMZ_TEXT_UNIFY_ALEF) && (ch == 0x0623 || ch == 0x0625 || ch == 0x0622 || ch == 0x0671))
            ch = 0x0627;
        c[k++] = ch;
    }
    return k;
}

static size_t normalise(uint32_t *c, size_t n, int items) {
    size_t i = 0, k = 0;
    while (i < n && c[i] == 0xFEFF) i++; /* BOM(s) */
    if (items) c[k++] = '\n'; /* safe: k <= i + 1 and slot 0 is read first */
    for (; i < n; i++) {
        uint32_t ch = c[i];
        if (ch == '\r') {
            if (i + 1 < n && c[i + 1] == '\n') continue;
            ch = '\n';
        }
        if (items && ch == '\n') {
            while (k > 0 && (c[k - 1] == ' ' || c[k - 1] == '\t')) k--;
            if (k > 0 && c[k - 1] == '\n') continue; /* empty line */
        }
        c[k++] = ch;
    }
    if (items) {
        while (k > 0 && (c[k - 1] == ' ' || c[k - 1] == '\t')) k--;
        if (k == 0 || c[k - 1] != '\n') c[k++] = '\n';
    }
    return k;
}

nmz_status nmz_dataset_load_text(nmz_dataset **out, const char *path, const nmz_text_opts *opts) {
    nmz_text_opts o = opts ? *opts : nmz_text_defaults();
    char *bytes = NULL;
    size_t nbytes = 0, ncp = 0, bad = 0, i, n_seq, t, removed = 0;
    uint32_t *cps = NULL;
    int *ids = NULL;
    nmz_dataset *ds = NULL;
    nmz_status s;
    if (!out || !path) return nmz__fail(NMZ_ERR_ARG, "NULL argument");
    *out = NULL;
    if (o.seq_len < 1 || o.seq_len > 100000) return nmz__fail(NMZ_ERR_ARG, "sequence length must be 1..100000");
    if ((s = read_file(path, &bytes, &nbytes)) != NMZ_OK) return s;
    /* one extra slot: items mode prepends '\n' */
    s = nmz_utf8_decode(bytes, nbytes, &cps, &ncp, &bad);
    free(bytes);
    if (s != NMZ_OK) return s;
    {
        uint32_t *bigger = realloc(cps, (ncp + 2) * sizeof *cps);
        if (!bigger) {
            free(cps);
            return nmz__fail(NMZ_ERR_NOMEM, "out of memory");
        }
        cps = bigger;
        if (o.items) { /* shift by one so normalise() can write the leading '\n' */
            memmove(cps + 1, cps, ncp * sizeof *cps);
            cps[0] = 0xFEFF; /* skipped like a BOM */
            ncp++;
        }
    }
    {
        unsigned flags = o.vocab ? o.vocab->normalize : o.normalize;
        size_t before = ncp;
        ncp = nmz_text_normalize(cps, ncp, flags);
        removed = before - ncp;
    }
    ncp = normalise(cps, ncp, o.items);
    if (ncp < (size_t)o.seq_len + 1) {
        s = nmz__fail(NMZ_ERR_FORMAT, "%s: %zu characters are not enough for one sequence of %d (+1)", path, ncp,
                      o.seq_len);
        goto done;
    }
    ds = calloc(1, sizeof *ds);
    if (!ds) {
        s = nmz__fail(NMZ_ERR_NOMEM, "out of memory");
        goto done;
    }
    if (o.vocab) {
        if ((s = nmz_vocab_copy(&ds->vocab, o.vocab)) != NMZ_OK) goto done;
    } else {
        if ((s = nmz_vocab_build(&ds->vocab, cps, ncp)) != NMZ_OK) goto done;
        ds->vocab->normalize = o.normalize & NMZ_TEXT_NORMALIZE_ALL;
    }
    ds->vocab->items = o.items;
    ids = malloc(ncp * sizeof *ids);
    if (!ids) {
        s = nmz__fail(NMZ_ERR_NOMEM, "out of memory");
        goto done;
    }
    for (i = 0; i < ncp; i++) {
        ids[i] = nmz_vocab_index(ds->vocab, cps[i]);
        if (ids[i] < 0) {
            char u[5];
            nmz_utf8_encode(cps[i], u);
            s = nmz__fail(NMZ_ERR_FORMAT, "%s: character '%s' (U+%04X) is not in the model's vocabulary", path, u,
                          (unsigned)cps[i]);
            goto done;
        }
    }
    if (o.vocab) { /* counts describe this text */
        memset(ds->vocab->counts, 0, (size_t)ds->vocab->size * sizeof *ds->vocab->counts);
        for (i = 0; i < ncp; i++) ds->vocab->counts[ids[i]]++;
    }
    n_seq = (ncp - 1) / (size_t)o.seq_len;
    if (n_seq > 100000000u) {
        s = nmz__fail(NMZ_ERR_FORMAT, "%s: too many sequences", path);
        goto done;
    }
    ds->n_samples = (int)n_seq;
    ds->n_inputs = o.seq_len;
    ds->n_outputs = ds->vocab->size;
    ds->time_steps = o.seq_len;
    ds->y_index = 1;
    ds->sample_rank = 1;
    ds->sample_shape[0] = o.seq_len;
    ds->n_tokens = ncp;
    ds->invalid_utf8 = bad;
    ds->n_removed = removed;
    ds->x = malloc(n_seq * (size_t)o.seq_len * sizeof(float));
    ds->y = malloc(n_seq * (size_t)o.seq_len * sizeof(float));
    if (!ds->x || !ds->y) {
        s = nmz__fail(NMZ_ERR_NOMEM, "out of memory");
        goto done;
    }
    for (i = 0; i < n_seq; i++)
        for (t = 0; t < (size_t)o.seq_len; t++) {
            ds->x[i * (size_t)o.seq_len + t] = (float)ids[i * (size_t)o.seq_len + t];
            ds->y[i * (size_t)o.seq_len + t] = (float)ids[i * (size_t)o.seq_len + t + 1];
        }
    s = NMZ_OK;
done:
    free(cps);
    free(ids);
    if (s == NMZ_OK)
        *out = ds;
    else
        nmz_dataset_free(ds);
    return s;
}
