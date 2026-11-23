/* dataset.c — in-memory dataset and a small CSV loader.
 *
 * CSV rules: one sample per line; fields separated by `delimiter`; blank
 * lines and lines starting with '#' are ignored; surrounding spaces and
 * double quotes are stripped; the last `n_targets` columns are targets.
 * A non-numeric label column is one-hot encoded (classes numbered in order of
 * first appearance). Errors report "file:line". */
#include "internal.h"

#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define NMZ_MAX_CLASSES_CSV 10000

nmz_csv_opts nmz_csv_defaults(void) {
    nmz_csv_opts o;
    o.n_targets = 1;
    o.n_inputs = 0;
    o.header = -1;
    o.categorical = -1;
    o.delimiter = ',';
    return o;
}

void nmz_dataset_free(nmz_dataset *ds) {
    int i;
    if (!ds) return;
    free(ds->x);
    free(ds->y);
    for (i = 0; i < ds->n_classes; i++) free(ds->class_names ? ds->class_names[i] : NULL);
    free(ds->class_names);
    nmz_vocab_free(ds->vocab);
    free(ds);
}

nmz_status nmz_dataset_create(nmz_dataset **out, int n_samples, int n_inputs, int n_outputs,
                              const float *x, const float *y) {
    nmz_dataset *ds;
    size_t nx, ny;
    if (!out) return nmz__fail(NMZ_ERR_ARG, "out is NULL");
    *out = NULL;
    if (n_samples < 1 || n_inputs < 1 || n_outputs < 1 || !x)
        return nmz__fail(NMZ_ERR_ARG, "dataset needs samples, inputs, outputs and data");
    nx = (size_t)n_samples * (size_t)n_inputs;
    ny = (size_t)n_samples * (size_t)n_outputs;
    ds = calloc(1, sizeof *ds);
    if (!ds) return nmz__fail(NMZ_ERR_NOMEM, "out of memory");
    ds->x = malloc(nx * sizeof(float));
    ds->y = calloc(ny, sizeof(float));
    if (!ds->x || !ds->y) {
        nmz_dataset_free(ds);
        return nmz__fail(NMZ_ERR_NOMEM, "out of memory");
    }
    memcpy(ds->x, x, nx * sizeof(float));
    if (y) memcpy(ds->y, y, ny * sizeof(float));
    ds->n_samples = n_samples;
    ds->n_inputs = n_inputs;
    ds->n_outputs = n_outputs;
    *out = ds;
    return NMZ_OK;
}

/* ------------------------------------------------------------------- CSV */

typedef struct row {
    char **f;
    int n;
    int line;
} row;

static char *trim(char *s) {
    char *e;
    while (*s == ' ' || *s == '\t') s++;
    e = s + strlen(s);
    while (e > s && (e[-1] == ' ' || e[-1] == '\t' || e[-1] == '\r')) *--e = '\0';
    if (e - s >= 2 && s[0] == '"' && e[-1] == '"') {
        e[-1] = '\0';
        s++;
    }
    return s;
}

static int parse_number(const char *s, float *out) {
    char *end;
    double v;
    if (!*s) return 0;
    errno = 0;
    v = strtod(s, &end);
    if (*end != '\0' || errno == ERANGE || !isfinite(v) || fabs(v) > 3.0e38) return 0;
    *out = (float)v;
    return 1;
}

static nmz_status read_all(const char *path, char **out) {
    FILE *f = fopen(path, "rb");
    char *buf = NULL;
    size_t len = 0, cap = 0, n;
    if (!f) return nmz__fail(NMZ_ERR_IO, "cannot open '%s': %s", path, strerror(errno));
    for (;;) {
        if (cap - len < 4097) {
            size_t nc = cap ? cap * 2 : 8192;
            char *p = realloc(buf, nc);
            if (!p) {
                free(buf);
                fclose(f);
                return nmz__fail(NMZ_ERR_NOMEM, "out of memory reading '%s'", path);
            }
            buf = p;
            cap = nc;
        }
        n = fread(buf + len, 1, cap - len - 1, f);
        len += n;
        if (n == 0) break;
    }
    if (ferror(f)) {
        free(buf);
        fclose(f);
        return nmz__fail(NMZ_ERR_IO, "error reading '%s'", path);
    }
    fclose(f);
    buf[len] = '\0';
    if (strlen(buf) != len) {
        free(buf);
        return nmz__fail(NMZ_ERR_FORMAT, "'%s' contains NUL bytes (not a text CSV file?)", path);
    }
    *out = buf;
    return NMZ_OK;
}

static void free_rows(row *rows, int n) {
    int i;
    for (i = 0; i < n; i++) free(rows[i].f);
    free(rows);
}

/* Split the buffer into rows of trimmed fields (pointers into buf). */
static nmz_status split(char *buf, char delim, row **out_rows, int *out_n) {
    row *rows = NULL;
    int n = 0, cap = 0, line = 0;
    char *p = buf;
    while (*p) {
        char *eol = strchr(p, '\n'), *s, *fs;
        row r;
        int fcap = 8;
        line++;
        if (eol) *eol = '\0';
        s = trim(p);
        p = eol ? eol + 1 : p + strlen(p);
        if (*s == '\0' || *s == '#') continue;
        r.n = 0;
        r.line = line;
        r.f = malloc((size_t)fcap * sizeof *r.f);
        if (!r.f) goto oom;
        for (fs = s;;) {
            char *d = strchr(fs, delim);
            if (d) *d = '\0';
            if (r.n == fcap) {
                char **q = realloc(r.f, (size_t)(fcap *= 2) * sizeof *q);
                if (!q) {
                    free(r.f);
                    goto oom;
                }
                r.f = q;
            }
            r.f[r.n++] = trim(fs);
            if (!d) break;
            fs = d + 1;
        }
        if (n == cap) {
            row *q = realloc(rows, (size_t)(cap = cap ? cap * 2 : 64) * sizeof *q);
            if (!q) {
                free(r.f);
                goto oom;
            }
            rows = q;
        }
        rows[n++] = r;
    }
    *out_rows = rows;
    *out_n = n;
    return NMZ_OK;
oom:
    free_rows(rows, n);
    return nmz__fail(NMZ_ERR_NOMEM, "out of memory");
}

static int find_class(char **names, int n, const char *s) {
    int i;
    for (i = 0; i < n; i++)
        if (strcmp(names[i], s) == 0) return i;
    return -1;
}

nmz_status nmz_dataset_load_csv(nmz_dataset **out, const char *path, const nmz_csv_opts *opts) {
    nmz_csv_opts o = opts ? *opts : nmz_csv_defaults();
    char *buf = NULL;
    row *rows = NULL;
    int n_rows = 0, first = 0, n_cols, n_in, i, k, categorical;
    int *labels = NULL;
    nmz_dataset *ds = NULL;
    nmz_status s;
    float tmp;

    if (!out || !path) return nmz__fail(NMZ_ERR_ARG, "NULL argument");
    *out = NULL;
    if (o.delimiter == '\0' || o.delimiter == '\n' || o.delimiter == '"')
        return nmz__fail(NMZ_ERR_ARG, "invalid CSV delimiter");
    if ((s = read_all(path, &buf)) != NMZ_OK) return s;
    if ((s = split(buf, o.delimiter, &rows, &n_rows)) != NMZ_OK) goto done;
    if (n_rows == 0) {
        s = nmz__fail(NMZ_ERR_FORMAT, "%s: no data rows", path);
        goto done;
    }
    n_cols = rows[0].n;
    if (o.n_inputs > 0) {
        if (o.n_inputs >= n_cols) {
            s = nmz__fail(NMZ_ERR_FORMAT, "%s:%d: %d column(s), but %d input column(s) plus at least one target are needed",
                          path, rows[0].line, n_cols, o.n_inputs);
            goto done;
        }
        o.n_targets = n_cols - o.n_inputs;
    }
    if (o.n_targets < 1 || o.n_targets >= n_cols) {
        s = nmz__fail(NMZ_ERR_FORMAT, "%s:%d: %d column(s) cannot hold %d target column(s) plus at least one input",
                      path, rows[0].line, n_cols, o.n_targets);
        goto done;
    }
    n_in = n_cols - o.n_targets;

    /* header: explicit, or auto = some input field of row 1 is not a number */
    if (o.header == 1) {
        first = 1;
    } else if (o.header == -1) {
        for (k = 0; k < n_in; k++)
            if (!parse_number(rows[0].f[k], &tmp)) first = 1;
    }
    if (first >= n_rows) {
        s = nmz__fail(NMZ_ERR_FORMAT, "%s: only a header, no data rows", path);
        goto done;
    }
    for (i = first; i < n_rows; i++)
        if (rows[i].n != n_cols) {
            s = nmz__fail(NMZ_ERR_FORMAT, "%s:%d: expected %d columns, found %d", path, rows[i].line, n_cols,
                          rows[i].n);
            goto done;
        }

    categorical = 0;
    if (o.categorical != 0 && o.n_targets == 1) {
        categorical = o.categorical == 1;
        if (o.categorical == -1)
            for (i = first; i < n_rows; i++)
                if (!parse_number(rows[i].f[n_in], &tmp)) categorical = 1;
    } else if (o.categorical == 1) {
        s = nmz__fail(NMZ_ERR_ARG, "categorical labels need exactly one target column");
        goto done;
    }

    ds = calloc(1, sizeof *ds);
    if (!ds) {
        s = nmz__fail(NMZ_ERR_NOMEM, "out of memory");
        goto done;
    }
    ds->n_samples = n_rows - first;
    ds->n_inputs = n_in;
    ds->x = malloc((size_t)ds->n_samples * (size_t)n_in * sizeof(float));
    labels = categorical ? malloc((size_t)ds->n_samples * sizeof *labels) : NULL;
    if (!ds->x || (categorical && !labels)) {
        s = nmz__fail(NMZ_ERR_NOMEM, "out of memory");
        goto done;
    }
    if (categorical) {
        ds->class_names = calloc(NMZ_MAX_CLASSES_CSV, sizeof *ds->class_names);
        if (!ds->class_names) {
            s = nmz__fail(NMZ_ERR_NOMEM, "out of memory");
            goto done;
        }
    }
    for (i = first; i < n_rows; i++) {
        int r = i - first;
        for (k = 0; k < n_in; k++)
            if (!parse_number(rows[i].f[k], &ds->x[(size_t)r * n_in + k])) {
                s = nmz__fail(NMZ_ERR_FORMAT, "%s:%d: column %d: '%s' is not a number", path, rows[i].line,
                              k + 1, rows[i].f[k]);
                goto done;
            }
        if (categorical) {
            const char *lab = rows[i].f[n_in];
            int c = find_class(ds->class_names, ds->n_classes, lab);
            if (c < 0) {
                if (ds->n_classes == NMZ_MAX_CLASSES_CSV) {
                    s = nmz__fail(NMZ_ERR_FORMAT, "%s: more than %d classes", path, NMZ_MAX_CLASSES_CSV);
                    goto done;
                }
                if (!*lab) {
                    s = nmz__fail(NMZ_ERR_FORMAT, "%s:%d: empty class label", path, rows[i].line);
                    goto done;
                }
                ds->class_names[ds->n_classes] = nmz__strdup(lab);
                if (!ds->class_names[ds->n_classes]) {
                    s = nmz__fail(NMZ_ERR_NOMEM, "out of memory");
                    goto done;
                }
                c = ds->n_classes++;
            }
            labels[r] = c;
        }
    }
    ds->n_outputs = categorical ? ds->n_classes : o.n_targets;
    if (categorical && ds->n_classes < 2) {
        s = nmz__fail(NMZ_ERR_FORMAT, "%s: the label column has only one class", path);
        goto done;
    }
    ds->y = calloc((size_t)ds->n_samples * (size_t)ds->n_outputs, sizeof(float));
    if (!ds->y) {
        s = nmz__fail(NMZ_ERR_NOMEM, "out of memory");
        goto done;
    }
    for (i = first; i < n_rows; i++) {
        int r = i - first;
        if (categorical) {
            ds->y[(size_t)r * ds->n_outputs + labels[r]] = 1.0f; /* one-hot */
        } else {
            for (k = 0; k < o.n_targets; k++)
                if (!parse_number(rows[i].f[n_in + k], &ds->y[(size_t)r * ds->n_outputs + k])) {
                    s = nmz__fail(NMZ_ERR_FORMAT, "%s:%d: target column %d: '%s' is not a number", path,
                                  rows[i].line, n_in + k + 1, rows[i].f[n_in + k]);
                    goto done;
                }
        }
    }
    s = NMZ_OK;
done:
    if (s == NMZ_OK) {
        *out = ds;
    } else {
        nmz_dataset_free(ds);
    }
    free(labels);
    free_rows(rows, n_rows);
    free(buf);
    return s;
}

/* ------------------------------------------------------------------- IDX */

nmz_status nmz_dataset_truncate(nmz_dataset *ds, int n) {
    if (!ds || n < 1) return nmz__fail(NMZ_ERR_ARG, "limit must be >= 1");
    if (n < ds->n_samples) ds->n_samples = n;
    return NMZ_OK;
}

static int read_be32(FILE *f, uint32_t *v) {
    unsigned char b[4];
    if (fread(b, 1, 4, f) != 4) return 0;
    *v = (uint32_t)b[0] << 24 | (uint32_t)b[1] << 16 | (uint32_t)b[2] << 8 | b[3];
    return 1;
}

static nmz_status idx_open(const char *path, uint32_t magic, int ndims, uint32_t *dims, FILE **out) {
    FILE *f = fopen(path, "rb");
    uint32_t m = 0;
    int i;
    if (!f) return nmz__fail(NMZ_ERR_IO, "cannot open '%s': %s", path, strerror(errno));
    if (!read_be32(f, &m) || m != magic) {
        fclose(f);
        return nmz__fail(NMZ_ERR_FORMAT, "%s: not an IDX file of the expected kind (magic 0x%08x, wanted 0x%08x)",
                         path, m, magic);
    }
    for (i = 0; i < ndims; i++)
        if (!read_be32(f, &dims[i]) || dims[i] == 0 || dims[i] > 100000000u) {
            fclose(f);
            return nmz__fail(NMZ_ERR_FORMAT, "%s: corrupt IDX header", path);
        }
    *out = f;
    return NMZ_OK;
}

nmz_status nmz_dataset_load_idx(nmz_dataset **out, const char *images_path, const char *labels_path, int limit) {
    FILE *fi = NULL, *fl = NULL;
    uint32_t di[3] = {0, 0, 0}, dl[1] = {0};
    nmz_dataset *ds = NULL;
    unsigned char *pixels = NULL, *labels = NULL;
    size_t pix, n, i, k;
    int max_label = 0;
    nmz_status s;
    if (!out || !images_path || !labels_path) return nmz__fail(NMZ_ERR_ARG, "NULL argument");
    *out = NULL;
    if ((s = idx_open(images_path, 0x00000803u, 3, di, &fi)) != NMZ_OK) return s;
    if ((s = idx_open(labels_path, 0x00000801u, 1, dl, &fl)) != NMZ_OK) goto done;
    if (di[0] != dl[0]) {
        s = nmz__fail(NMZ_ERR_FORMAT, "%s has %u images but %s has %u labels", images_path, di[0], labels_path, dl[0]);
        goto done;
    }
    if (di[1] > 4096 || di[2] > 4096) {
        s = nmz__fail(NMZ_ERR_FORMAT, "%s: images of %ux%u are too large", images_path, di[1], di[2]);
        goto done;
    }
    n = di[0];
    if (limit > 0 && (size_t)limit < n) n = (size_t)limit;
    pix = (size_t)di[1] * di[2];
    labels = malloc(dl[0]); /* all labels: the class count must not depend on the limit */
    pixels = malloc(pix);
    ds = calloc(1, sizeof *ds);
    if (!labels || !pixels || !ds) {
        s = nmz__fail(NMZ_ERR_NOMEM, "out of memory");
        goto done;
    }
    if (fread(labels, 1, dl[0], fl) != dl[0]) {
        s = nmz__fail(NMZ_ERR_FORMAT, "%s: truncated (expected %u labels)", labels_path, dl[0]);
        goto done;
    }
    for (i = 0; i < dl[0]; i++)
        if (labels[i] > max_label) max_label = labels[i];
    ds->n_samples = (int)n;
    ds->n_inputs = (int)pix;
    ds->n_classes = max_label + 1 < 2 ? 2 : max_label + 1;
    ds->n_outputs = ds->n_classes;
    ds->sample_rank = 3;
    ds->sample_shape[0] = 1;
    ds->sample_shape[1] = (int)di[1];
    ds->sample_shape[2] = (int)di[2];
    ds->x = malloc(n * pix * sizeof(float));
    ds->y = calloc(n * (size_t)ds->n_outputs, sizeof(float));
    ds->class_names = calloc((size_t)ds->n_classes, sizeof *ds->class_names);
    if (!ds->x || !ds->y || !ds->class_names) {
        s = nmz__fail(NMZ_ERR_NOMEM, "out of memory (%zu images)", n);
        goto done;
    }
    for (i = 0; i < (size_t)ds->n_classes; i++) {
        char name[8];
        snprintf(name, sizeof name, "%u", (unsigned)i);
        if (!(ds->class_names[i] = nmz__strdup(name))) {
            s = nmz__fail(NMZ_ERR_NOMEM, "out of memory");
            goto done;
        }
    }
    for (i = 0; i < n; i++) {
        if (fread(pixels, 1, pix, fi) != pix) {
            s = nmz__fail(NMZ_ERR_FORMAT, "%s: truncated at image %zu", images_path, i);
            goto done;
        }
        for (k = 0; k < pix; k++) ds->x[i * pix + k] = (float)pixels[k];
        ds->y[i * (size_t)ds->n_outputs + labels[i]] = 1.0f;
    }
    s = NMZ_OK;
done:
    if (fi) fclose(fi);
    if (fl) fclose(fl);
    free(pixels);
    free(labels);
    if (s == NMZ_OK)
        *out = ds;
    else
        nmz_dataset_free(ds);
    return s;
}
