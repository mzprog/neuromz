/* error.c — status strings, the thread-local last-error message, and the
 * library's only operating-system difference: file names (UTF-8 on every
 * system; on Windows the C runtime's narrow functions use the ANSI code
 * page, so names are converted to UTF-16 for _wfopen — decision 113). */
#include "internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(_WIN32)
#include <wchar.h>
#endif

#if defined(_MSC_VER)
#define NMZ_THREAD_LOCAL __declspec(thread)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#define NMZ_THREAD_LOCAL _Thread_local
#else
#define NMZ_THREAD_LOCAL
#endif

static NMZ_THREAD_LOCAL char g_last_error[512];

const char *nmz_status_str(nmz_status s) {
    switch (s) {
    case NMZ_OK: return "NMZ_OK";
    case NMZ_ERR_NOMEM: return "NMZ_ERR_NOMEM";
    case NMZ_ERR_ARG: return "NMZ_ERR_ARG";
    case NMZ_ERR_SHAPE: return "NMZ_ERR_SHAPE";
    case NMZ_ERR_STATE: return "NMZ_ERR_STATE";
    case NMZ_ERR_IO: return "NMZ_ERR_IO";
    case NMZ_ERR_FORMAT: return "NMZ_ERR_FORMAT";
    case NMZ_ERR_UNSUPPORTED: return "NMZ_ERR_UNSUPPORTED";
    case NMZ_ERR_NUMERIC: return "NMZ_ERR_NUMERIC";
    }
    return "NMZ_ERR_UNKNOWN";
}

const char *nmz_last_error(void) { return g_last_error; }

nmz_status nmz__fail(nmz_status s, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_last_error, sizeof g_last_error, fmt, ap);
    va_end(ap);
    return s;
}

#if defined(_WIN32)
/* UTF-8 → UTF-16 with the library's own decoder (no <windows.h>) */
static wchar_t *widen(const char *s) {
    uint32_t *cps = NULL;
    size_t n = 0, i, k = 0;
    wchar_t *w;
    if (nmz_utf8_decode(s, strlen(s), &cps, &n, NULL) != NMZ_OK) return NULL;
    w = malloc((2 * n + 1) * sizeof *w);
    for (i = 0; w && i < n; i++) {
        uint32_t c = cps[i];
        if (c >= 0x10000u) { /* surrogate pair */
            c -= 0x10000u;
            w[k++] = (wchar_t)(0xD800u + (c >> 10));
            w[k++] = (wchar_t)(0xDC00u + (c & 0x3FFu));
        } else {
            w[k++] = (wchar_t)c;
        }
    }
    if (w) w[k] = 0;
    free(cps);
    return w;
}
#endif

FILE *nmz__fopen(const char *path, const char *mode) {
#if defined(_WIN32)
    wchar_t *wp = widen(path), *wm = widen(mode);
    FILE *f = wp && wm ? _wfopen(wp, wm) : NULL;
    free(wp);
    free(wm);
    return f;
#else
    return fopen(path, mode);
#endif
}

int nmz__remove(const char *path) {
#if defined(_WIN32)
    wchar_t *wp = widen(path);
    int r = wp ? _wremove(wp) : -1;
    free(wp);
    return r;
#else
    return remove(path);
#endif
}

int nmz__rename(const char *from, const char *to) {
#if defined(_WIN32)
    wchar_t *a = widen(from), *b = widen(to);
    int r = a && b ? _wrename(a, b) : -1;
    free(a);
    free(b);
    return r;
#else
    return rename(from, to);
#endif
}

char *nmz__strdup(const char *s) {
    size_t n;
    char *d;
    if (!s) return NULL;
    n = strlen(s) + 1;
    d = malloc(n);
    if (d) memcpy(d, s, n);
    return d;
}

nmz_status nmz__shape_size(int rank, const int *shape, size_t *out) {
    size_t n = 1;
    int i;
    if (rank < 0 || rank > NMZ_MAX_RANK)
        return nmz__fail(NMZ_ERR_SHAPE, "rank %d is outside 0..%d", rank, NMZ_MAX_RANK);
    if (rank > 0 && !shape) return nmz__fail(NMZ_ERR_ARG, "shape is NULL");
    for (i = 0; i < rank; i++) {
        if (shape[i] <= 0 || shape[i] > NMZ_MAX_DIM)
            return nmz__fail(NMZ_ERR_SHAPE, "dimension %d has invalid size %d", i, shape[i]);
        if (n > NMZ_MAX_ELEMS / (size_t)shape[i])
            return nmz__fail(NMZ_ERR_SHAPE, "tensor too large");
        n *= (size_t)shape[i];
    }
    *out = n;
    return NMZ_OK;
}

int nmz__shape_equal(int rank_a, const int *a, int rank_b, const int *b) {
    int i;
    if (rank_a != rank_b) return 0;
    for (i = 0; i < rank_a; i++)
        if (a[i] != b[i]) return 0;
    return 1;
}
