/* commands.c — the neuromz commands (except `new`, see cmd_new.c). Every
 * command validates its input and works (or fails with a message) whether
 * or not a network is loaded. */
#include "cli.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "neuromz_layer.h" /* layer shapes for `show` */

#define fail cli_fail
#define lib_fail cli_lib_fail
#define need_net cli_need_net

void cli_state_free(cli_state *st) {
    nmz_net_free(st->net);
    st->net = NULL;
    cli_set_path(st, NULL);
}

static int argmax(const float *v, int n) {
    int i, b = 0;
    for (i = 1; i < n; i++)
        if (v[i] > v[b]) b = i;
    return b;
}

static int parse_header_opt(const char *a, int *header) {
    if (!strcmp(a, "--header")) *header = 1;
    else if (!strcmp(a, "--no-header")) *header = 0;
    else return 0;
    return 1;
}

/* --limit N */
static int parse_limit(int argc, char **argv, int *i, int *limit) {
    long n;
    const char *v = cli_opt_value(argc, argv, i);
    if (!v) return 0;
    if (!cli_parse_int(v, 1, 1000000000L, &n)) return !fail("--limit needs a positive integer, got '%s'", v);
    *limit = (int)n;
    return 1;
}

/* ------------------------------------------------------------------ show */

static void show_filters(cli_state *st) {
    int li, found = 0;
    for (li = 0; li < nmz_net_layer_count(st->net); li++) {
        nmz_layer *l = nmz_net_layer(st->net, li);
        const nmz_tensor *W;
        int F, C, kh, kw, f, c, y, per_row = 8;
        if (strcmp(nmz_layer_type(l), "conv2d")) continue;
        found = 1;
        W = nmz_layer_param_value(l, 0);
        F = W->shape[0];
        C = W->shape[1];
        kh = W->shape[2];
        kw = W->shape[3];
        printf("[layer %d] conv2d: %d filters of %dx%d over %d channel(s)   (each filter scaled to its own min..max: "
               "' ' lowest, '@' highest)\n", li, F, kh, kw, C);
        for (c = 0; c < 1; c++) {
            int f0;
            if (C > 1) printf("  input channel 1 of %d (each filter has one %dx%d kernel per input channel)\n", C, kh, kw);
            for (f0 = 0; f0 < F; f0 += per_row) {
                int f1 = f0 + per_row < F ? f0 + per_row : F;
                printf("  ");
                for (f = f0; f < f1; f++) printf(" f%-*d", kw * 2 + 3, f + 1);
                printf("\n");
                for (y = 0; y < kh; y++) {
                    printf("  ");
                    for (f = f0; f < f1; f++) {
                        const float *k = W->data + ((size_t)f * C + (size_t)c) * kh * kw;
                        double lo = k[0], hi = k[0];
                        int i, x;
                        for (i = 1; i < kh * kw; i++) {
                            if (k[i] < lo) lo = k[i];
                            if (k[i] > hi) hi = k[i];
                        }
                        putchar('|');
                        for (x = 0; x < kw; x++) {
                            char ch = heat_char(k[y * kw + x], lo, hi);
                            putchar(ch);
                            putchar(ch);
                        }
                        printf("|  ");
                    }
                    printf("\n");
                }
            }
        }
        for (f = 0; f < F && f < 4; f++) {
            const float *k = W->data + (size_t)f * C * kh * kw;
            printf("  filter %d, channel 1, row by row:", f + 1);
            for (y = 0; y < kh * kw; y++) printf("%s%6.3f%s", y % kw ? " " : "  [", (double)k[y], y % kw == kw - 1 ? "]" : "");
            printf("\n");
        }
        if (F > 4) printf("  (use `show weights` for every value)\n");
    }
    if (!found) printf("This network has no conv2d layers (`show weights` prints dense weights).\n");
}

static int show_sample(cli_state *st, long idx, const char *spec, int header) {
    nmz_dataset *ds = NULL;
    const float *x;
    int i;
    if (cli_load_data(st, spec, 1, (int)idx, header, 0, &ds)) return 1;
    if (idx < 1 || idx > ds->n_samples) {
        fprintf(stderr, "error: sample %ld is out of range (1..%d)\n", idx, ds->n_samples);
        nmz_dataset_free(ds);
        return 1;
    }
    x = ds->x + (size_t)(idx - 1) * ds->n_inputs;
    printf("Sample %ld of %s", idx, spec);
    if (ds->n_classes)
        printf(", label %s\n", ds->class_names[argmax(ds->y + (size_t)(idx - 1) * ds->n_outputs, ds->n_outputs)]);
    else
        printf("\n");
    if (ds->sample_rank == 3) {
        printf("  +");
        for (i = 0; i < ds->sample_shape[2] * 2; i++) putchar('-');
        printf("+\n");
        {
            /* raw pixel values are 0..255: use a fixed scale so faint strokes stay faint */
            int y, xx, w = ds->sample_shape[2];
            for (y = 0; y < ds->sample_shape[1]; y++) {
                printf("  |");
                for (xx = 0; xx < w; xx++) {
                    char ch = heat_char(x[y * w + xx], 0.0, 255.0);
                    putchar(ch);
                    putchar(ch);
                }
                printf("|\n");
            }
        }
        printf("  +");
        for (i = 0; i < ds->sample_shape[2] * 2; i++) putchar('-');
        printf("+\n");
    } else {
        printf("  inputs:");
        for (i = 0; i < ds->n_inputs && i < 32; i++) printf(" %g", (double)x[i]);
        printf(ds->n_inputs > 32 ? " ...\n" : "\n");
    }
    if (st->net) {
        int n = nmz_net_output_size(st->net), k, best, ncls = nmz_net_class_count(st->net);
        float *y = malloc(sizeof(float) * (size_t)n);
        if (y && nmz_net_predict(st->net, x, y) == NMZ_OK) {
            best = argmax(y, n);
            printf("  prediction: %s (%.1f%%)\n", ncls == n ? nmz_net_class_name(st->net, best) : "", 100.0 * y[best]);
            for (k = 0; k < n && n <= 20; k++) {
                int bar = (int)(y[k] * 40 + 0.5), b;
                printf("  %6s %6.2f%% ", ncls == n ? nmz_net_class_name(st->net, k) : "", 100.0 * y[k]);
                for (b = 0; b < bar; b++) putchar('#');
                printf("%s\n", k == best ? "  <" : "");
            }
        } else if (y) {
            lib_fail("predict");
        }
        free(y);
    }
    nmz_dataset_free(ds);
    return 0;
}

static void print_shape(const nmz_layer *l) {
    int i;
    for (i = 0; i < l->in_rank; i++) {
        if (l->in_shape[i] == NMZ_ANY_LENGTH) printf(i ? "xT" : "T");
        else printf(i ? "x%d" : "%d", l->in_shape[i]);
    }
}

static int cmd_show(cli_state *st, int argc, char **argv) {
    int i, j, weights = 0, filters = 0, n_cls, fitted = 0, groups = 0, header = -1;
    long sample = 0;
    const char *data = "mnist:test";
    char desc[160];
    nmz_norm norm;
    if (argc > 1 && !strcmp(argv[1], "gates")) return show_gates(st, argc, argv);
    if (argc > 1 && !strcmp(argv[1], "attention")) return show_attention(st, argc, argv);
    if (argc > 1 && !strcmp(argv[1], "params")) {
        if (argc > 2) return fail("show params takes no options, got '%s'", argv[2]);
        return need_net(st) ? show_params(st) : 1;
    }
    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "weights") || !strcmp(argv[i], "--weights"))
            weights = 1;
        else if (!strcmp(argv[i], "filters"))
            filters = 1;
        else if (!strcmp(argv[i], "vocab"))
            filters = 2;
        else if (!strcmp(argv[i], "sample")) {
            if (i + 1 >= argc || !cli_parse_int(argv[i + 1], 1, 1000000000L, &sample))
                return fail("usage: show sample N [--data mnist:test|FILE]%s", NULL);
            i++;
        } else if (!strcmp(argv[i], "--data")) {
            if (!(data = cli_opt_value(argc, argv, &i))) return 1;
        } else if (parse_header_opt(argv[i], &header)) {
        } else
            return fail("show takes `weights`, `filters`, `vocab`, `params`, `gates [--text T | --data FILE]`, "
                        "`attention [--text T | --data FILE]` or `sample N [--data SPEC]`, got '%s'", argv[i]);
    }
    if (sample) return show_sample(st, sample, data, header);
    if (!need_net(st)) return 1;
    if (filters == 2) {
        const nmz_vocab *v = nmz_net_vocab(st->net);
        uint64_t total = 0, mx = 1;
        int k;
        if (!v) return fail("this model has no vocabulary (it is not a text model)%s", NULL);
        for (k = 0; k < v->size; k++) {
            total += v->counts[k];
            if (v->counts[k] > mx) mx = v->counts[k];
        }
        printf("Vocabulary: %d characters (Unicode code points), %llu occurrences in the training text%s\n", v->size,
               (unsigned long long)total, v->items ? "; ⏎ = end of an item" : "");
        if (v->normalize)
            printf("Text normalization: %s%s%s (applied to every text the model reads)\n",
                   v->normalize & NMZ_TEXT_STRIP_DIACRITICS ? "tashkeel and tatweel removed" : "",
                   v->normalize == NMZ_TEXT_NORMALIZE_ALL ? ", " : "",
                   v->normalize & NMZ_TEXT_UNIFY_ALEF ? "alef forms written as bare alef" : "");
        printf("  %4s  %-6s  %-8s %10s  %6s\n", "id", "char", "code", "count", "share");
        for (k = 0; k < v->size; k++) {
            char lbl[32];
            int bar = (int)(30.0 * (double)v->counts[k] / (double)mx + 0.5), b;
            cli_char_label(v->codepoints[k], lbl, sizeof lbl);
            printf("  %4d  %s", k, lbl);
            {
                int w = 0, c;
                for (c = 0; lbl[c]; c++) w += ((unsigned char)lbl[c] & 0xC0) != 0x80; /* characters, not bytes */
                for (; w < 6; w++) putchar(' ');
            }
            printf("  U+%04X   %10llu  %5.1f%%  ", (unsigned)v->codepoints[k],
                   (unsigned long long)v->counts[k], total ? 100.0 * (double)v->counts[k] / (double)total : 0.0);
            for (b = 0; b < bar; b++) putchar('#');
            putchar('\n');
        }
        return 0;
    }
    if (filters) {
        show_filters(st);
        return 0;
    }
    printf("Network %s%s\n", st->path ? st->path : "(not saved yet)", st->dirty && st->path ? "  [unsaved changes]" : "");
    if (nmz_net_layer_count(st->net) > 0) {
        printf("  input        ");
        if (nmz_net_is_sequence(st->net)) {
            const nmz_vocab *v = nmz_net_vocab(st->net);
            printf("text: T character ids per sequence (any T)%s\n",
                   v && v->items ? ", one item per line" : "");
            if (v) printf("  vocabulary   %d characters (`show vocab` lists them)%s\n", v->size,
                          v->normalize ? ", Arabic text normalization on" : "");
        } else {
            print_shape(nmz_net_layer(st->net, 0));
            printf(" (%d value(s) per sample)\n", nmz_net_input_size(st->net));
        }
    }
    norm = nmz_net_normalization(st->net, &fitted, &groups, NULL, NULL);
    if (norm != NMZ_NORM_NONE)
        printf("  normalize    %s, %s\n", nmz_norm_name(norm),
               fitted ? (groups == 1 ? "fitted (1 group)" : "fitted") : "not fitted yet (fitted at the first train)");
    for (i = 0; i < nmz_net_layer_count(st->net); i++) {
        nmz_layer *l = nmz_net_layer(st->net, i);
        size_t count = 0;
        for (j = 0; j < nmz_layer_param_count(l); j++) count += nmz_layer_param_value(l, j)->size;
        nmz_layer_describe(l, desc, sizeof desc);
        if (count)
            printf("  [layer %d]  %-58s %zu params\n", i, desc, count);
        else
            printf("  [layer %d]  %s\n", i, desc);
    }
    printf("  loss         %s\n", nmz_loss_name(nmz_net_loss_kind(st->net)));
    printf("  parameters   %zu\n", nmz_net_param_count(st->net));
    printf("  steps        %ld backward pass(es) so far\n", nmz_net_step(st->net));
    {
        long t = 0;
        int k = nmz_net_optimizer_state(st->net, &t);
        if (k == NMZ_OPT_ADAM)
            printf("  optimizer    Adam state stored (t = %ld): training continues where it stopped\n", t);
        else if (k == NMZ_OPT_SGD)
            printf("  optimizer    SGD velocity stored: training continues where it stopped\n");
    }
    printf("  training     ");
    cli_print_train_config(nmz_net_train_config(st->net));
    n_cls = nmz_net_class_count(st->net);
    if (n_cls) {
        printf("  classes     ");
        for (i = 0; i < n_cls; i++) printf(" %d=%s", i + 1, nmz_net_class_name(st->net, i));
        printf("\n");
    }
    if (weights) {
        for (i = 0; i < nmz_net_layer_count(st->net); i++) {
            nmz_layer *l = nmz_net_layer(st->net, i);
            for (j = 0; j < nmz_layer_param_count(l); j++) {
                char name[64];
                const nmz_tensor *v = nmz_layer_param_value(l, j);
                snprintf(name, sizeof name, "[layer %d] %s", i, nmz_layer_param_name(l, j));
                if (v->rank <= 2)
                    print_matrix(name, v, 0);
                else {
                    /* conv kernels [F,C,KH,KW]: one KH x KW matrix per filter and channel */
                    nmz_tensor view = *v;
                    view.rank = 2;
                    view.shape[0] = v->shape[0] * v->shape[1] * v->shape[2];
                    view.shape[1] = v->shape[3];
                    printf("  %s  (%d filters x %d channels, rows grouped by filter/channel, %d rows each)\n", name,
                           v->shape[0], v->shape[1], v->shape[2]);
                    print_matrix(NULL, &view, 0);
                }
            }
        }
    }
    return 0;
}

/* -------------------------------------------------------------- generate */

/* does `item` occur as a whole line of the file? (small files only) */
static int line_in_file(const char *path, const char *item) {
    FILE *f = cli_fopen(path, "rb");
    char *line;
    int found = 0;
    if (!f) return 0;
    while (!found && (line = cli_read_line(f)) != NULL) {
        size_t n = strlen(line);
        while (n && (line[n - 1] == ' ' || line[n - 1] == '\t')) line[--n] = '\0';
        found = !strcmp(line, item) || (!strncmp(line, "\xef\xbb\xbf", 3) && !strcmp(line + 3, item));
        free(line);
    }
    fclose(f);
    return found;
}

/* one training item as the model sees it: trailing blanks and a UTF-8 BOM
 * removed, the model's text normalisation applied (malloc'd, or NULL) */
static char *item_key(const char *line, unsigned normalize, int first) {
    size_t n = strlen(line), m = 0, i;
    uint32_t *cps = NULL;
    char *out;
    if (first && !strncmp(line, "\xef\xbb\xbf", 3)) line += 3, n -= 3;
    while (n && (line[n - 1] == ' ' || line[n - 1] == '\t' || line[n - 1] == '\r')) n--;
    if (!normalize) {
        if (!(out = malloc(n + 1))) return NULL;
        memcpy(out, line, n);
        out[n] = '\0';
        return out;
    }
    if (nmz_utf8_decode(line, n, &cps, &m, NULL) != NMZ_OK) return NULL;
    m = nmz_text_normalize(cps, m, normalize);
    if ((out = malloc(4 * m + 1)) != NULL) {
        size_t k = 0;
        for (i = 0; i < m; i++) {
            char b[5];
            int len = nmz_utf8_encode(cps[i], b);
            memcpy(out + k, b, (size_t)len);
            k += (size_t)len;
        }
        out[k] = '\0';
    }
    free(cps);
    return out;
}

/* After training on a text file, the model remembers where it came from and,
 * for line-per-item data, which items it saw (decision 111), so `generate
 * --new-only` and `show gates` work without --data. */
static void remember_data_source(cli_state *st, const char *spec) {
    const nmz_vocab *v = nmz_net_vocab(st->net);
    char *path, **items = NULL, *line;
    int n = 0, cap = 0, ok = 1;
    FILE *f;
    if (!v || !nmz_net_is_sequence(st->net) || !cli_file_exists(spec)) return;
    path = cli_abspath(spec);
    if (v->items && (f = cli_fopen(spec, "rb")) != NULL) {
        while (ok && (line = cli_read_line(f)) != NULL) {
            char *key = item_key(line, v->normalize, n == 0 && cap == 0);
            free(line);
            if (!key) {
                ok = 0;
            } else if (!*key) {
                free(key);
            } else {
                if (n == cap) {
                    char **g = realloc(items, (size_t)(cap = cap ? 2 * cap : 256) * sizeof *g);
                    if (!g) {
                        free(key);
                        ok = 0;
                        break;
                    }
                    items = g;
                }
                items[n++] = key;
            }
        }
        fclose(f);
    }
    if (!ok || nmz_net_set_data_source(st->net, path, (const char *const *)items, ok ? n : 0) != NMZ_OK)
        printf("  (note: could not record the training data in the model)\n");
    while (n > 0) free(items[--n]);
    free(items);
    free(path);
}

/* is prime + out one of the training items? */
static int item_known(const nmz_net *net, const char *data, const char *prime, const char *out) {
    size_t a = prime ? strlen(prime) : 0, b = strlen(out);
    char *full = malloc(a + b + 1);
    int known;
    if (!full) return 0;
    memcpy(full, prime ? prime : "", a);
    memcpy(full + a, out, b + 1);
    known = data ? line_in_file(data, full) : nmz_net_knows_item(net, full) > 0;
    free(full);
    return known;
}

static int cmd_generate(cli_state *st, int argc, char **argv) {
    nmz_gen_opts g = nmz_gen_defaults();
    const nmz_vocab *v;
    const char *data = NULL;
    long count = 0, n;
    int i, length_set = 0, new_only = 0, made = 0, attempts = 0;
    double d;
    uint64_t seed = 1;
    char **seen = NULL;
    if (!need_net(st)) return 1;
    v = nmz_net_vocab(st->net);
    if (!v || !nmz_net_is_sequence(st->net)) return fail("generate needs a text model (see `new --preset names`)%s", NULL);
    for (i = 1; i < argc; i++) {
        const char *a = argv[i], *val;
        if (!strcmp(a, "--new-only")) {
            new_only = 1;
            continue;
        }
        if (!strcmp(a, "--no-cache")) {
            g.no_cache = 1;
            continue;
        }
        if (!(val = cli_opt_value(argc, argv, &i))) return 1;
        if (!strcmp(a, "--prime")) g.prime = val;
        else if (!strcmp(a, "--length")) {
            if (!cli_parse_int(val, 1, 1000000, &n)) return fail("--length needs a positive integer, got '%s'", val);
            g.length = (int)n;
            length_set = 1;
        } else if (!strcmp(a, "--temperature")) {
            if (!cli_parse_double(val, &d) || d <= 0) return fail("--temperature must be > 0, got '%s'", val);
            g.temperature = (float)d;
        } else if (!strcmp(a, "--top-k")) {
            if (!cli_parse_int(val, 0, 1000000, &n)) return fail("--top-k needs an integer >= 0, got '%s'", val);
            g.top_k = (int)n;
        } else if (!strcmp(a, "--seed")) {
            if (!cli_parse_u64(val, &seed)) return fail("--seed needs a non-negative integer, got '%s'", val);
        } else if (!strcmp(a, "--count")) {
            if (!cli_parse_int(val, 1, 10000, &count)) return fail("--count needs a positive integer, got '%s'", val);
        } else if (!strcmp(a, "--data")) {
            data = val;
        } else if (!strcmp(a, "--context")) {
            if (!cli_parse_int(val, 1, 100000, &n)) return fail("--context needs a positive integer, got '%s'", val);
            g.context = (int)n;
        } else {
            return fail("unknown option '%s' for generate (see `help generate`)", a);
        }
    }
    /* which items count as "known": --data FILE, else the items stored with
     * the model, else the file it was trained on (decision 111) */
    if (!data && nmz_net_knows_item(st->net, "") < 0 && nmz_net_data_path(st->net) &&
        cli_file_exists(nmz_net_data_path(st->net)))
        data = nmz_net_data_path(st->net);
    if (new_only && !data && nmz_net_knows_item(st->net, "") < 0)
        return fail("--new-only: this model does not record its training items (trained before v2.0.0, or not on "
                    "a file); give --data FILE%s", NULL);
    if (!v->items) { /* running text: one passage per seed, all generated at once */
        char **outs;
        if (!count) count = 1;
        if (!length_set) g.length = 300;
        g.seed = seed;
        outs = calloc((size_t)count, sizeof *outs);
        if (!outs) return fail("out of memory%s", NULL);
        if (nmz_net_generate_batch(st->net, &g, (int)count, outs, NULL) != NMZ_OK) {
            free(outs);
            return lib_fail("generate");
        }
        for (i = 0; i < count; i++) {
            if (count > 1) printf("--- seed %llu\n", (unsigned long long)(seed + (uint64_t)i));
            printf("%s%s\n", g.prime ? g.prime : "", outs[i]);
            free(outs[i]);
        }
        free(outs);
        return 0;
    }
    /* items (names): stop at the end-of-item token, keep distinct ones. Rows
     * are generated in batches (one forward pass per step for the batch) and
     * judged in seed order, so the result is the same as one item at a time
     * with seeds seed, seed+1, ... */
    if (!count) count = 10;
    if (!length_set) g.length = 64;
    g.stop_at_end = 1;
    seen = calloc((size_t)count, sizeof *seen);
    if (!seen) return fail("out of memory%s", NULL);
    while (made < count && attempts < count * 50) {
        int need = (int)count - made, rows = need, k, r; /* repeated rounds replace duplicates */
        char **outs;
        if (rows > 1024) rows = 1024;
        if (rows > count * 50 - attempts) rows = (int)count * 50 - attempts;
        outs = calloc((size_t)rows, sizeof *outs);
        g.seed = seed + (uint64_t)attempts;
        if (!outs || nmz_net_generate_batch(st->net, &g, rows, outs, NULL) != NMZ_OK) {
            for (k = 0; k < made; k++) free(seen[k]);
            free(seen);
            free(outs);
            return outs ? lib_fail("generate") : fail("out of memory%s", NULL);
        }
        for (r = 0; r < rows; r++) {
            char *out = outs[r];
            size_t len;
            int dup = 0, known;
            outs[r] = NULL;
            if (made >= count) { /* enough: the rest of the batch is not used */
                free(out);
                continue;
            }
            attempts++;
            len = strlen(out);
            if (len && out[len - 1] == '\n') out[--len] = '\0';
            for (k = 0; k < made && !dup; k++) dup = !strcmp(seen[k], out);
            known = !len ? 0 : item_known(st->net, data, g.prime, out);
            if (!len || dup || (new_only && known)) {
                free(out);
                continue;
            }
            seen[made++] = out;
            printf("  %2d. %s%s%s\n", made, g.prime ? g.prime : "", out, known ? "   (in the training data)" : "");
        }
        free(outs);
    }
    if (made < count) printf("(only %d distinct item(s) after %d attempts)\n", made, attempts);
    for (i = 0; i < made; i++) free(seen[i]);
    free(seen);
    return 0;
}

/* ----------------------------------------------------------------- train */

typedef struct progress {
    int every, quiet, tty, max_epochs, ce, has_val;
    int drawn; /* characters of the live line now on screen (0 = none) */
    int width; /* terminal columns */
    double t0, last_draw;
} progress;

static void print_acc(double acc) {
    if (isnan(acc))
        printf("accuracy n/a");
    else
        printf("accuracy %5.1f%%", 100.0 * acc);
}

/* Erase the live line: overwrite exactly what was drawn, then return to
 * column 0. The line is never wider than the terminal (a wrapped line
 * cannot be erased with \r), so this always leaves a clean row. */
static void clear_line(progress *p) {
    if (p->tty && p->drawn > 0) printf("\r%*s\r", p->drawn, "");
    p->drawn = 0;
}

static int on_batch(const nmz_batch_info *b, void *user) {
    progress *p = user;
    double now;
    char line[256];
    int filled, i, n, bar = 30;
    if (p->quiet || !p->tty) return 0; /* the live line is for terminals only */
    now = cli_now();
    if (now - p->last_draw < 0.1 && b->batch != b->n_batches) return 0;
    p->last_draw = now;
    if (p->width < 70) bar = p->width > 50 ? p->width - 50 : 0;
    filled = (int)((double)bar * b->batch / b->n_batches);
    n = snprintf(line, sizeof line, "  epoch %d/%d [", b->epoch, p->max_epochs);
    for (i = 0; i < bar && n < (int)sizeof line - 1; i++) line[n++] = i < filled ? '#' : '.';
    if (n < (int)sizeof line)
        snprintf(line + n, sizeof line - (size_t)n, "] %3d%%  batch loss %-9.4g %6.1fs", 100 * b->batch / b->n_batches,
                 b->loss, now - p->t0);
    n = (int)strlen(line);
    if (n > p->width - 1) n = p->width - 1; /* ASCII only, so bytes = columns */
    line[n] = '\0';
    printf("\r%s", line);
    if (p->drawn > n) printf("%*s\r%s", p->drawn - n, "", line); /* a shorter line: blank the old tail */
    p->drawn = n;
    fflush(stdout);
    return 0;
}

static int on_epoch(const nmz_epoch_info *info, void *user) {
    progress *p = user;
    if (p->quiet) return 0;
    clear_line(p);
    if (info->epoch == 1 || info->epoch % p->every == 0 || info->epoch == p->max_epochs || p->has_val) {
        printf("  epoch %6d   loss %-10.5g", info->epoch, info->loss);
        if (p->ce) printf(" (ppl %6.2f)", exp(info->loss));
        if (p->has_val) {
            printf("  val %-9.5g", info->val_loss);
            if (p->ce) printf(" (ppl %6.2f)", exp(info->val_loss));
            printf(info->improved ? " *" : "  ");
        }
        printf("  %s ", p->has_val ? "val" : "train");
        print_acc(p->has_val ? info->val_accuracy : info->accuracy);
        if (info->clipped_batches) printf("  clipped %d", info->clipped_batches);
        printf("   lr %-8.3g %6.1fs\n", (double)info->learning_rate, cli_now() - p->t0);
        fflush(stdout);
    }
    return 0;
}

static int cmd_train(cli_state *st, int argc, char **argv) {
    nmz_train_opts o;
    nmz_train_result res;
    nmz_dataset *ds = NULL, *vds = NULL;
    progress pr;
    const char *spec = NULL, *val_file = NULL;
    unsigned mask = 0;
    int i, header = -1, limit = 0, fitted_before = 0, fitted_after = 0;
    long every = 0;
    nmz_status s;
    memset(&pr, 0, sizeof pr);
    if (!need_net(st)) return 1;
    o = *nmz_net_train_config(st->net);
    for (i = 1; i < argc; i++) {
        int r;
        if (cli_is_option(argv[i])) {
            r = cli_train_option(argc, argv, &i, &o, &mask);
            if (r < 0) return 1;
            if (r) continue;
            if (parse_header_opt(argv[i], &header)) continue;
            if (!strcmp(argv[i], "--quiet")) {
                pr.quiet = 1;
            } else if (!strcmp(argv[i], "--limit")) {
                if (!parse_limit(argc, argv, &i, &limit)) return 1;
            } else if (!strcmp(argv[i], "--val-file")) {
                if (!(val_file = cli_opt_value(argc, argv, &i))) return 1;
            } else if (!strcmp(argv[i], "--every")) {
                const char *v = cli_opt_value(argc, argv, &i);
                if (!v || !cli_parse_int(v, 1, 1000000000L, &every))
                    return v ? fail("--every needs a positive integer, got '%s'", v) : 1;
            } else {
                return fail("unknown option '%s' for train (see `help train`)", argv[i]);
            }
        } else if (!spec) {
            spec = argv[i];
        } else {
            return fail("train takes one dataset; unexpected '%s'", argv[i]);
        }
    }
    if (!spec) return fail("usage: train DATA [options]   DATA = FILE.csv, mnist, or an IDX images file%s", NULL);
    if (cli_load_data(st, spec, 0, limit, header, o.seq_len, &ds)) return 1;
    if (val_file && cli_load_data(st, val_file, 1, 0, header, o.seq_len, &vds)) {
        nmz_dataset_free(ds);
        return 1;
    }
    pr.max_epochs = o.max_epochs;
    pr.every = every ? (int)every : (o.max_epochs >= 20 ? o.max_epochs / 20 : 1);
    pr.tty = cli_stdout_is_tty();
    pr.width = pr.tty ? cli_term_width() : 80;
    pr.t0 = cli_now();
    pr.ce = nmz_net_loss_kind(st->net) == NMZ_LOSS_CROSS_ENTROPY;
    pr.has_val = o.val_fraction > 0 || vds;
    o.on_epoch = on_epoch;
    o.on_batch = on_batch;
    o.user = &pr;
    nmz_net_normalization(st->net, &fitted_before, NULL, NULL, NULL);
    printf("Training on %s: %d samples%s, %d input(s), %d output(s)\n", spec, ds->n_samples,
           limit ? " (--limit)" : "", ds->n_inputs, ds->n_outputs);
    if (vds)
        printf("Validating on %s: %d samples (all of %s is used for training)\n", val_file, vds->n_samples, spec);
    printf("  ");
    cli_print_train_config(&o);
    if (o.batch_size > ds->n_samples)
        printf("  (batch %d > %d samples: using the full dataset as one batch)\n", o.batch_size, ds->n_samples);
    if (!pr.quiet)
        printf("  (each epoch line: the whole %s re-evaluated after the epoch's updates; the live bar shows the "
               "last batch)\n",
               pr.has_val ? "training set (loss) and validation set (val)" : "training set");
    s = nmz_net_train_val(st->net, ds, vds, &o, &res);
    clear_line(&pr);
    st->dirty = 1;
    if (s == NMZ_OK) remember_data_source(st, spec);
    nmz_net_normalization(st->net, &fitted_after, NULL, NULL, NULL);
    if (!fitted_before && fitted_after)
        printf("  (input normalization was fitted on this data and is now part of the model)\n");
    if (s != NMZ_OK) {
        nmz_dataset_free(ds);
        nmz_dataset_free(vds);
        return lib_fail("training stopped");
    }
    if (res.converged)
        printf("Converged after %d epoch(s): loss %.6g <= target %g, ", res.epochs, res.loss, o.target_loss);
    else if (res.best_epoch && res.best_epoch < res.epochs)
        printf("Early stop after %d epoch(s): kept the weights of epoch %d (best validation loss), loss %.6g, ",
               res.epochs, res.best_epoch, res.loss);
    else if (o.target_loss <= 0)
        printf("Finished %d epoch(s): loss %.6g, ", res.epochs, res.loss);
    else
        printf("Stopped at the epoch limit (%d): loss %.6g > target %g, ", res.epochs, res.loss, o.target_loss);
    print_acc(res.accuracy);
    printf(" on the training set%s, %.1fs\n",
           res.best_epoch && res.best_epoch < res.epochs ? " (re-evaluated with the kept weights)" : "",
           cli_now() - pr.t0);
    if (vds || o.val_fraction > 0) {
        if (vds)
            printf("  validation (%s): loss %.6g", val_file, res.val_loss);
        else
            printf("  validation (last %g%% of the data): loss %.6g", 100.0 * o.val_fraction, res.val_loss);
        if (pr.ce) printf(", perplexity %.3f", exp(res.val_loss));
        if (!isnan(res.val_accuracy)) printf(", accuracy %.1f%%", 100.0 * res.val_accuracy);
        printf("\n");
    }
    if (!res.converged && o.target_loss > 0)
        printf("  hint: try more epochs (--epochs), another learning rate (--lr), more hidden units, or a looser "
               "--target-loss\n");
    nmz_dataset_free(ds);
    nmz_dataset_free(vds);
    return 0;
}

/* ------------------------------------------------------------ test/predict */

static int cmd_test(cli_state *st, int argc, char **argv) {
    nmz_dataset *ds = NULL;
    const char *spec = NULL;
    double loss, acc, t0;
    int i, header = -1, n_out, n_cls, limit = 0;
    float *y = NULL;
    if (!need_net(st)) return 1;
    for (i = 1; i < argc; i++) {
        if (parse_header_opt(argv[i], &header)) continue;
        if (!strcmp(argv[i], "--limit")) {
            if (!parse_limit(argc, argv, &i, &limit)) return 1;
            continue;
        }
        if (cli_is_option(argv[i])) return fail("unknown option '%s' for test", argv[i]);
        if (spec) return fail("test takes one dataset; unexpected '%s'", argv[i]);
        spec = argv[i];
    }
    if (!spec) return fail("usage: test DATA [--limit N]   DATA = FILE.csv, mnist (= test set), or an IDX file%s", NULL);
    if (cli_load_data(st, spec, 1, limit, header, 0, &ds)) return 1;
    t0 = cli_now();
    if (nmz_net_evaluate(st->net, ds, &loss, &acc) != NMZ_OK) {
        nmz_dataset_free(ds);
        return lib_fail("test");
    }
    printf("Tested on %s: %d samples (%.1fs)\n  loss %.6g   ", spec, ds->n_samples, cli_now() - t0, loss);
    if (nmz_net_loss_kind(st->net) == NMZ_LOSS_CROSS_ENTROPY && ds->time_steps)
        printf("perplexity %.3f (per character)   ", exp(loss));
    print_acc(acc);
    printf(ds->time_steps ? " (next character)\n" : "\n");
    n_out = ds->n_outputs;
    n_cls = nmz_net_class_count(st->net);
    if (n_out > 1 && n_out <= 20 && (y = malloc(sizeof(float) * (size_t)n_out)) != NULL) {
        int *cm = calloc((size_t)(n_out * n_out), sizeof *cm), r, c;
        if (cm) {
            for (r = 0; r < ds->n_samples; r++) {
                if (nmz_net_predict(st->net, ds->x + (size_t)r * ds->n_inputs, y) != NMZ_OK) break;
                cm[argmax(ds->y + (size_t)r * n_out, n_out) * n_out + argmax(y, n_out)]++;
            }
            printf("  confusion matrix (rows = true class, columns = predicted)\n  %14s", "");
            for (c = 0; c < n_out; c++) printf(" %*.10s", n_out > 8 ? 6 : 10, n_cls == n_out ? nmz_net_class_name(st->net, c) : "");
            printf("\n");
            for (r = 0; r < n_out; r++) {
                char lbl[32];
                snprintf(lbl, sizeof lbl, "%d", r + 1);
                printf("  %14.14s", n_cls == n_out ? nmz_net_class_name(st->net, r) : lbl);
                for (c = 0; c < n_out; c++) printf(" %*d", n_out > 8 ? 6 : 10, cm[r * n_out + c]);
                printf("\n");
            }
            free(cm);
        }
    }
    free(y);
    nmz_dataset_free(ds);
    return 0;
}

/* read exactly n numbers from argv[*i...] (stops at an option) */
static int read_numbers(int argc, char **argv, int *i, float *out, int n, const char *what) {
    int k = 0;
    double d;
    while (*i < argc && !cli_is_option(argv[*i]) && strcmp(argv[*i], "-t")) {
        if (k >= n) return fail("too many %s values", what);
        if (!cli_parse_double(argv[*i], &d)) return fail("'%s' is not a number", argv[*i]);
        out[k++] = (float)d;
        (*i)++;
    }
    if (k != n) {
        fprintf(stderr, "error: expected %d %s value(s), got %d%s\n", n, what, k,
                n > 16 ? " (for images use --sample N [--data mnist:test])" : "");
        return 1;
    }
    return 0;
}

/* --------------------------------------------- one sample: x (and t) */

typedef struct sample_args {
    float *x, *t;
    int have_t;
    long sample; /* --sample N (1-based), 0 = values on the command line */
    const char *data;
    const char *json;
    int full;
    print_opts po;
    nmz_train_opts o;
    unsigned mask;
} sample_args;

static void sample_free(sample_args *a) {
    free(a->x);
    free(a->t);
    a->x = a->t = NULL;
}

/* x1 .. xn | --sample N [--data SPEC]   [--target T.. | CLASS]  [--json F] [--full] [--at F,Y,X] [train opts] */
static int parse_sample(cli_state *st, int argc, char **argv, sample_args *a, int allow_train_opts, int need_target) {
    int n_in = nmz_net_input_size(st->net), n_out = nmz_net_output_size(st->net), i = 1, header = -1;
    memset(a, 0, sizeof *a);
    a->o = *nmz_net_train_config(st->net);
    a->data = "mnist:test";
    a->x = malloc(sizeof(float) * (size_t)n_in);
    a->t = calloc((size_t)n_out, sizeof(float));
    if (!a->x || !a->t) return fail("out of memory%s", NULL);
    if (i < argc && !cli_is_option(argv[i]) && strcmp(argv[i], "-t"))
        if (read_numbers(argc, argv, &i, a->x, n_in, "input")) return 1;
    while (i < argc) {
        const char *s = argv[i];
        if (!strcmp(s, "--target") || !strcmp(s, "-t")) {
            int c, n_cls = nmz_net_class_count(st->net);
            i++;
            if (i < argc && n_cls == n_out && n_cls > 0) { /* a class name → one-hot */
                for (c = 0; c < n_cls; c++)
                    if (!strcmp(argv[i], nmz_net_class_name(st->net, c))) break;
                if (c < n_cls && !(n_out == 1)) {
                    memset(a->t, 0, sizeof(float) * (size_t)n_out);
                    a->t[c] = 1.0f;
                    i++;
                    a->have_t = 1;
                    continue;
                }
            }
            if (read_numbers(argc, argv, &i, a->t, n_out, "target")) return 1;
            a->have_t = 1;
        } else if (!strcmp(s, "--sample")) {
            const char *v = cli_opt_value(argc, argv, &i);
            if (!v || !cli_parse_int(v, 1, 1000000000L, &a->sample))
                return v ? fail("--sample needs a positive integer (1-based), got '%s'", v) : 1;
            i++;
        } else if (!strcmp(s, "--data")) {
            if (!(a->data = cli_opt_value(argc, argv, &i))) return 1;
            i++;
        } else if (!strcmp(s, "--json")) {
            if (!(a->json = cli_opt_value(argc, argv, &i))) return 1;
            i++;
        } else if (!strcmp(s, "--full")) {
            a->full = 1;
            i++;
        } else if (!strcmp(s, "--at")) {
            const char *v = cli_opt_value(argc, argv, &i);
            int c = 0, y = 0, x = 0;
            char extra;
            if (!v || sscanf(v, "%d,%d,%d%c", &c, &y, &x, &extra) != 3 || c < 1 || y < 1 || x < 1)
                return v ? fail("--at needs CHANNEL,ROW,COL (1-based), e.g. --at 1,14,14, got '%s'", v) : 1;
            a->po.at_set = 1;
            a->po.at_c = c - 1;
            a->po.at_y = y - 1;
            a->po.at_x = x - 1;
            i++;
        } else if (parse_header_opt(s, &header)) {
            i++;
        } else if (allow_train_opts && cli_is_option(s)) {
            int r = cli_train_option(argc, argv, &i, &a->o, &a->mask);
            if (r <= 0) return r < 0 ? 1 : fail("unexpected '%s' (see `help trace`)", s);
            i++;
        } else {
            return fail("unexpected '%s' (see `help trace`)", s);
        }
    }
    if (a->sample) {
        nmz_dataset *ds = NULL;
        if (cli_load_data(st, a->data, 1, (int)a->sample, header, 0, &ds)) return 1;
        if (a->sample > ds->n_samples) {
            fprintf(stderr, "error: sample %ld is out of range (1..%d)\n", a->sample, ds->n_samples);
            nmz_dataset_free(ds);
            return 1;
        }
        memcpy(a->x, ds->x + (size_t)(a->sample - 1) * n_in, sizeof(float) * (size_t)n_in);
        if (!a->have_t) {
            memcpy(a->t, ds->y + (size_t)(a->sample - 1) * n_out, sizeof(float) * (size_t)n_out);
            a->have_t = 1;
        }
        nmz_dataset_free(ds);
    }
    if (need_target && !a->have_t) {
        fprintf(stderr, "error: a target is needed to compute the loss and gradients: add --target with %d value(s)%s\n",
                n_out, nmz_net_class_count(st->net) == n_out ? " or a class name" : "");
        return 1;
    }
    a->po.full = a->full;
    a->po.max_rows = a->full ? 0 : 12;
    a->po.max_terms = a->full ? 0 : 6;
    return 0;
}

static int cmd_predict(cli_state *st, int argc, char **argv) {
    sample_args a;
    int n_out, k, n_cls, rc = 1;
    float *y = NULL;
    if (!need_net(st)) return 1;
    if (parse_sample(st, argc, argv, &a, 0, 0)) goto out;
    n_out = nmz_net_output_size(st->net);
    y = malloc(sizeof(float) * (size_t)n_out);
    if (!y || nmz_net_predict(st->net, a.x, y) != NMZ_OK) {
        lib_fail("predict");
        goto out;
    }
    n_cls = nmz_net_class_count(st->net);
    printf("output:");
    for (k = 0; k < n_out; k++) {
        if (n_cls == n_out)
            printf("  %s %.4f", nmz_net_class_name(st->net, k), (double)y[k]);
        else
            printf("  %.6g", (double)y[k]);
    }
    printf("\n");
    if (n_out > 1 && nmz_net_loss_kind(st->net) == NMZ_LOSS_CROSS_ENTROPY) {
        int b = argmax(y, n_out);
        if (n_cls == n_out)
            printf("predicted class: %s (%.1f%%)", nmz_net_class_name(st->net, b), 100.0 * y[b]);
        else
            printf("predicted class: %d (%.1f%%)", b + 1, 100.0 * y[b]);
        if (a.sample && a.have_t && n_cls == n_out)
            printf("   true class: %s", nmz_net_class_name(st->net, argmax(a.t, n_out)));
        printf("\n");
    }
    rc = 0;
out:
    free(y);
    sample_free(&a);
    return rc;
}

/* --------------------------------------------------------- trace / step */

static int write_json(const nmz_trace_recorder *rec, const char *path) {
    FILE *f = cli_fopen(path, "w");
    size_t i;
    if (!f) return fail("cannot write '%s'", path);
    for (i = 0; i < nmz_trace_recorder_count(rec); i++) nmz_trace_event_write_json(nmz_trace_recorder_get(rec, i), f);
    if (fclose(f) != 0) return fail("cannot write '%s'", path);
    printf("\n(%zu trace events written to %s, one JSON object per line)\n", nmz_trace_recorder_count(rec), path);
    return 0;
}

static int sample_tensors(cli_state *st, const sample_args *a, nmz_tensor *x, nmz_tensor *t) {
    int shape[NMZ_MAX_RANK], i;
    nmz_layer *l0 = nmz_net_layer(st->net, 0);
    shape[0] = 1;
    for (i = 0; i < l0->in_rank; i++) shape[i + 1] = l0->in_shape[i];
    if (nmz_tensor_init(x, l0->in_rank + 1, shape) != NMZ_OK) return 1;
    memcpy(x->data, a->x, x->size * sizeof(float));
    shape[1] = nmz_net_output_size(st->net);
    if (nmz_tensor_init(t, 2, shape) != NMZ_OK) return 1;
    memcpy(t->data, a->t, t->size * sizeof(float));
    return 0;
}

/* ------------------------------------------------------ trace on a sequence */

typedef struct seq_args {
    int *ids, *tgt, T, unit, full, layer, pos, head, attn_n;
    long sample;
    const char *data, *json, *text;
    print_opts po;
    nmz_train_opts o;
    unsigned mask;
} seq_args;

static int parse_seq_args(cli_state *st, int argc, char **argv, seq_args *a) {
    const nmz_vocab *v = nmz_net_vocab(st->net);
    int i;
    long n;
    memset(a, 0, sizeof *a);
    a->o = *nmz_net_train_config(st->net);
    a->unit = -1;
    a->layer = -1;
    a->pos = -1;
    a->head = -1;
    a->attn_n = 1;
    for (i = 1; i < argc; i++) {
        const char *s = argv[i], *val;
        if (!strcmp(s, "--full")) {
            a->full = 1;
            continue;
        }
        if (!strcmp(s, "--pos") || !strcmp(s, "--head")) {
            if (!(val = cli_opt_value(argc, argv, &i))) return 1;
            if (!net_has_attention(st->net)) return fail("%s applies to Transformer models (attention)", s);
            if (!cli_parse_int(val, 0, 1000000, &n)) {
                fprintf(stderr, "error: %s needs an integer, got '%s'\n", s, val);
                return 1;
            }
            if (!strcmp(s, "--pos")) a->pos = (int)n; /* 0-based, like the trace's position numbers */
            else if (n < 1) return fail("--head is 1-based, got '%s'", val);
            else a->head = (int)n - 1;
            continue;
        }
        if (!strcmp(s, "--text") || !strcmp(s, "--sample") || !strcmp(s, "--data") || !strcmp(s, "--json") ||
            !strcmp(s, "--unit") || !strcmp(s, "--layer")) {
            if (!(val = cli_opt_value(argc, argv, &i))) return 1;
            if (!strcmp(s, "--text")) a->text = val;
            else if (!strcmp(s, "--data")) a->data = val;
            else if (!strcmp(s, "--json")) a->json = val;
            else if (!strcmp(s, "--sample")) {
                if (!cli_parse_int(val, 1, 1000000000L, &a->sample)) return fail("--sample needs a positive integer, got '%s'", val);
            } else if (!strcmp(s, "--layer")) {
                if (!cli_parse_int(val, 1, 1000, &n)) return fail("--layer needs a positive integer (1-based), got '%s'", val);
                if (net_has_attention(st->net)) {
                    if (seq_attention_layer(st->net, (int)n) < 0)
                        return fail("--layer %s: the model has fewer attention layers (blocks)", val);
                    a->attn_n = (int)n;
                } else if ((a->layer = seq_recurrent_layer(st->net, (int)n)) < 0)
                    return fail("--layer %s: the model has fewer recurrent layers (rnn/lstm/gru)", val);
            } else {
                if (!cli_parse_int(val, 1, 100000, &n)) return fail("--unit needs a positive integer (1-based), got '%s'", val);
                a->unit = (int)n - 1;
            }
            continue;
        }
        if (cli_is_option(s)) {
            int r = cli_train_option(argc, argv, &i, &a->o, &a->mask);
            if (r < 0) return 1;
            if (r) continue;
        }
        return fail("unexpected '%s' (for a text model: trace --text \"some text\" or --sample N --data FILE)", s);
    }
    if (a->text) {
        uint32_t *cp = NULL;
        size_t m = 0, bad = 0, k;
        if (nmz_utf8_decode(a->text, strlen(a->text), &cp, &m, &bad) != NMZ_OK) return lib_fail("--text");
        m = nmz_text_normalize(cp, m, v->normalize); /* the model's normalisation (e.g. no tashkeel) */
        if (m < 2 || m > 10000) {
            free(cp);
            return fail("--text needs at least 2 characters (the net predicts each next one)%s", NULL);
        }
        a->T = (int)m - 1;
        a->ids = malloc(m * sizeof *a->ids);
        a->tgt = malloc(m * sizeof *a->tgt);
        if (!a->ids || !a->tgt) {
            free(cp);
            return fail("out of memory%s", NULL);
        }
        for (k = 0; k < m; k++) {
            int id = nmz_vocab_index(v, cp[k]);
            if (id < 0) {
                char lbl[24];
                fprintf(stderr, "error: '%s' (U+%04X) is not in the model's vocabulary (`show vocab` lists it)\n",
                        cli_char_label(cp[k], lbl, sizeof lbl), (unsigned)cp[k]);
                free(cp);
                return 1;
            }
            if (k + 1 < m) a->ids[k] = id;
            if (k > 0) a->tgt[k - 1] = id;
        }
        free(cp);
    } else if (a->sample) {
        nmz_dataset *ds = NULL;
        int t;
        if (!a->data) return fail("--sample N needs --data FILE (the text to take sequence N from)%s", NULL);
        if (cli_load_data(st, a->data, 0, (int)a->sample, -1, a->o.seq_len, &ds)) return 1;
        if (a->sample > ds->n_samples) {
            fprintf(stderr, "error: sequence %ld is out of range (1..%d)\n", a->sample, ds->n_samples);
            nmz_dataset_free(ds);
            return 1;
        }
        a->T = ds->time_steps;
        a->ids = malloc((size_t)a->T * sizeof *a->ids);
        a->tgt = malloc((size_t)a->T * sizeof *a->tgt);
        if (!a->ids || !a->tgt) {
            nmz_dataset_free(ds);
            return fail("out of memory%s", NULL);
        }
        for (t = 0; t < a->T; t++) {
            a->ids[t] = (int)ds->x[(size_t)(a->sample - 1) * a->T + t];
            a->tgt[t] = (int)ds->y[(size_t)(a->sample - 1) * a->T + t];
        }
        nmz_dataset_free(ds);
    } else {
        return fail("give a sequence: trace --text \"some text\" (or --sample N --data FILE)%s", NULL);
    }
    if (a->pos >= a->T) {
        fprintf(stderr, "error: --pos %d: the text has positions 0..%d\n", a->pos, a->T - 1);
        return 1;
    }
    {
        int ctx = nmz_net_context_length(st->net);
        if (ctx > 0 && a->T > ctx) {
            fprintf(stderr, "error: the text has %d positions but the model's context window is %d characters (use a "
                            "shorter --text)\n", a->T, ctx);
            return 1;
        }
    }
    a->po.full = a->full;
    a->po.max_rows = a->full ? 0 : 24;
    a->po.max_terms = a->full ? 0 : 6;
    a->po.at_set = a->unit >= 0;
    return 0;
}

static int seq_tensors(const seq_args *a, int V, int last_only, nmz_tensor *x, nmz_tensor *t) {
    int xs[2], ts[3], k;
    xs[0] = ts[0] = 1;
    xs[1] = ts[1] = a->T;
    ts[2] = V;
    if (nmz_tensor_reshape(x, 2, xs) != NMZ_OK || nmz_tensor_reshape(t, 3, ts) != NMZ_OK) return 1;
    nmz_tensor_zero(t);
    for (k = 0; k < a->T; k++) {
        x->data[k] = (float)a->ids[k];
        if (!last_only || k == a->T - 1) t->data[(size_t)k * V + a->tgt[k]] = 1.0f;
    }
    return 0;
}

static int run_trace_seq(cli_state *st, int argc, char **argv, int apply) {
    seq_args a;
    nmz_trace_recorder *rec = NULL;
    nmz_tensor x = NMZ_TENSOR_EMPTY, t = NMZ_TENSOR_EMPTY;
    const nmz_vocab *v = nmz_net_vocab(st->net);
    seq_survival sv;
    double L = 0, *grad_e = NULL;
    int rc = 1, V = nmz_net_output_size(st->net);
    memset(&sv, 0, sizeof sv);
    if (parse_seq_args(st, argc, argv, &a)) goto out;
    if (!v || v->size != V) {
        fail("this sequence model has no matching vocabulary%s", NULL);
        goto out;
    }
    if (nmz_trace_recorder_create(&rec) != NMZ_OK) goto lib;
    /* 1. the loss of the last step alone, followed back through time */
    if (seq_tensors(&a, V, 1, &x, &t)) goto lib;
    nmz_net_set_trace(st->net, nmz_trace_recorder_fn, rec);
    if (nmz_net_forward_mode(st->net, &x, 0, NULL) != NMZ_OK || nmz_net_backward(st->net, &t, NULL) != NMZ_OK) goto lib;
    if (seq_survival_collect(rec, a.layer, a.T, &sv)) seq_survival_free(&sv); /* (no recurrent layer: no view) */
    if (net_has_attention(st->net)) { /* |dL/dx_t| at the embedding output */
        size_t i;
        free(grad_e);
        grad_e = calloc((size_t)a.T, sizeof *grad_e);
        for (i = 0; grad_e && i < nmz_trace_recorder_count(rec); i++) {
            const nmz_trace_event *ev = nmz_trace_recorder_get(rec, i);
            const nmz_tensor *de = nmz_trace_find(ev, "dL/de");
            int k, d;
            if (ev->phase != NMZ_TRACE_BACKWARD || strcmp(ev->layer_type, "embedding") || !de || de->rank != 3) continue;
            for (k = 0; k < a.T; k++) {
                double sq = 0;
                for (d = 0; d < de->shape[2]; d++) sq += (double)de->data[(size_t)k * de->shape[2] + d] * de->data[(size_t)k * de->shape[2] + d];
                grad_e[k] = sqrt(sq);
            }
        }
    }
    nmz_trace_recorder_clear(rec);
    /* 2. the real pass: every step's loss */
    if (seq_tensors(&a, V, 0, &x, &t)) goto lib;
    if (nmz_net_forward_mode(st->net, &x, apply, NULL) != NMZ_OK || nmz_net_backward(st->net, &t, &L) != NMZ_OK) goto lib;
    if (a.o.clip_norm > 0 && nmz_net_clip_gradients(st->net, a.o.clip_norm, NULL, NULL) != NMZ_OK) goto lib;
    if (apply ? nmz_net_optimizer_step(st->net, &a.o, a.o.learning_rate) != NMZ_OK
              : nmz_net_optimizer_preview(st->net, &a.o, a.o.learning_rate) != NMZ_OK)
        goto lib;
    nmz_net_set_trace(st->net, NULL, NULL);
    if (nmz_trace_recorder_status(rec) != NMZ_OK) {
        fail("out of memory while recording the trace%s", NULL);
        goto out;
    }
    if (net_has_attention(st->net)) {
        printf("%s of one sequence (%d positions) through a Transformer: attention step by step\n",
               apply ? "STEP" : "TRACE", a.T);
        print_attn_trace(rec, v, &a.po, a.pos, a.attn_n, a.head, grad_e, st->net);
        print_update_section(rec, &a.po, apply);
    } else {
        printf("%s of one sequence (%d steps): every value of the forward and backward pass, unrolled through time\n",
               apply ? "STEP" : "TRACE", a.T);
        print_seq_trace(rec, v, &a.po, apply, a.unit, a.layer, &sv);
    }
    printf("\n");
    print_rule();
    if (apply) {
        printf("Weights updated (training step %ld). Loss before the step: %.6g\n", nmz_net_step(st->net), L);
        st->dirty = 1;
    } else {
        printf("Loss %.6g (perplexity %.3f). Nothing was changed; `step` with the same arguments applies the update.\n", L,
               exp(L));
    }
    if (a.json && write_json(rec, a.json)) goto out;
    rc = 0;
    goto out;
lib:
    nmz_net_set_trace(st->net, NULL, NULL);
    lib_fail(apply ? "step" : "trace");
out:
    nmz_trace_recorder_free(rec);
    nmz_tensor_free(&x);
    nmz_tensor_free(&t);
    seq_survival_free(&sv);
    free(grad_e);
    free(a.ids);
    free(a.tgt);
    return rc;
}

static int run_trace(cli_state *st, int argc, char **argv, int apply) {
    sample_args a;
    nmz_trace_recorder *rec = NULL;
    nmz_tensor x = NMZ_TENSOR_EMPTY, t = NMZ_TENSOR_EMPTY;
    int rc = 1;
    double L;
    if (!need_net(st)) return 1;
    if (nmz_net_layer_count(st->net) == 0) return fail("the network has no layers%s", NULL);
    if (nmz_net_is_sequence(st->net)) return run_trace_seq(st, argc, argv, apply);
    if (parse_sample(st, argc, argv, &a, 1, 1)) goto out;
    if (sample_tensors(st, &a, &x, &t)) goto lib;
    if (nmz_trace_recorder_create(&rec) != NMZ_OK) goto lib;
    nmz_net_set_trace(st->net, nmz_trace_recorder_fn, rec);
    /* trace = evaluation-mode pass + preview (changes nothing); step = a real training step */
    if (nmz_net_forward_mode(st->net, &x, apply, NULL) != NMZ_OK || nmz_net_backward(st->net, &t, &L) != NMZ_OK)
        goto lib;
    if (a.o.clip_norm > 0 && nmz_net_clip_gradients(st->net, a.o.clip_norm, NULL, NULL) != NMZ_OK) goto lib;
    if (apply ? nmz_net_optimizer_step(st->net, &a.o, a.o.learning_rate) != NMZ_OK
              : nmz_net_optimizer_preview(st->net, &a.o, a.o.learning_rate) != NMZ_OK)
        goto lib;
    nmz_net_set_trace(st->net, NULL, NULL);
    if (nmz_trace_recorder_status(rec) != NMZ_OK) {
        fail("out of memory while recording the trace%s", NULL);
        goto out;
    }
    printf("%s for one sample%s: every value of the forward and backward pass\n", apply ? "STEP" : "TRACE",
           a.sample ? "" : "");
    if (a.sample) printf("(sample %ld of %s)\n", a.sample, a.data);
    print_trace_all(rec, &a.po, apply);
    printf("\n");
    print_rule();
    if (apply) {
        printf("Weights updated (training step %ld). Loss before the step: %.6g\n", nmz_net_step(st->net), L);
        st->dirty = 1;
    } else {
        printf("Loss %.6g. Nothing was changed; run `step` with the same arguments to apply this update.\n", L);
    }
    if (a.json && write_json(rec, a.json)) goto out;
    rc = 0;
    goto out;
lib:
    nmz_net_set_trace(st->net, NULL, NULL);
    lib_fail(apply ? "step" : "trace");
out:
    nmz_trace_recorder_free(rec);
    nmz_tensor_free(&x);
    nmz_tensor_free(&t);
    sample_free(&a);
    return rc;
}

static int cmd_trace_fwd(cli_state *st, int argc, char **argv) { return run_trace(st, argc, argv, 0); }
static int cmd_step_fwd(cli_state *st, int argc, char **argv) { return run_trace(st, argc, argv, 1); }

static int cmd_gradcheck(cli_state *st, int argc, char **argv) {
    sample_args a;
    nmz_tensor x = NMZ_TENSOR_EMPTY, t = NMZ_TENSOR_EMPTY;
    nmz_gradcheck_result r;
    int rc = 1;
    if (!need_net(st)) return 1;
    memset(&a, 0, sizeof a);
    if (nmz_net_is_sequence(st->net)) {
        seq_args sa;
        int ok = !parse_seq_args(st, argc, argv, &sa) && !seq_tensors(&sa, nmz_net_output_size(st->net), 0, &x, &t);
        free(sa.ids);
        free(sa.tgt);
        if (!ok) goto out;
    } else {
        if (parse_sample(st, argc, argv, &a, 0, 1)) goto out;
        if (sample_tensors(st, &a, &x, &t)) goto out;
    }
    if (nmz_net_gradcheck_mode(st->net, &x, &t, 1e-3, 0, 0, 0, &r) != NMZ_OK) {
        lib_fail("gradcheck");
        goto out;
    }
    printf("Numerical gradient check: %zu gradients compared with (L(w+e) - L(w-e)) / 2e, e = 0.001\n", r.checked);
    printf("  max |backprop - numeric| = %.3g\n", r.max_abs_err);
    printf("  largest mismatch: %s of %s %d, index %zu: backprop %.6g vs numeric %.6g\n", r.worst_param,
           r.worst_layer < 0 ? "input" : "layer", r.worst_layer < 0 ? 0 : r.worst_layer, r.worst_index,
           r.worst_analytic, r.worst_numeric);
    if (r.kinks)
        printf("  skipped %zu check(s) where a relu changed side or a max-pool winner changed within +-e:\n"
               "  the loss has a kink there, so (L(w+e) - L(w-e)) / 2e is not a derivative\n", r.kinks);
    if (r.failures == 0)
        printf("PASS: backpropagation agrees with the numerical derivative\n");
    else
        printf("FAIL: %zu gradient(s) differ\n", r.failures);
    rc = r.failures ? 1 : 0;
out:
    nmz_tensor_free(&x);
    nmz_tensor_free(&t);
    sample_free(&a);
    return rc;
}

/* ------------------------------------------------------------- save/load */

static int cmd_save(cli_state *st, int argc, char **argv) {
    const char *path = NULL;
    unsigned flags = 0;
    int i;
    if (!need_net(st)) return 1;
    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--no-optimizer"))
            flags |= NMZ_SAVE_NO_OPTIMIZER;
        else if (!path && !cli_is_option(argv[i]))
            path = argv[i];
        else
            return fail("usage: save [FILE] [--no-optimizer]%s", NULL);
    }
    if (!path) path = st->path;
    if (!path) return fail("no file name yet: use `save FILE`%s", NULL);
    if (nmz_net_save_ex(st->net, path, flags) != NMZ_OK) return lib_fail("save");
    if (path != st->path) cli_set_path(st, path);
    st->dirty = 0;
    printf("Saved to %s\n", st->path);
    return 0;
}

static int cmd_load(cli_state *st, int argc, char **argv) {
    nmz_net *net = NULL;
    if (argc != 2) return fail("usage: load FILE%s", NULL);
    if (nmz_net_load(&net, argv[1]) != NMZ_OK) return lib_fail("load");
    cli_replace_net(st, net, argv[1]);
    if (st->oneshot) return 0; /* -m FILE: stay quiet */
    printf("Loaded %s: %d input(s) %s %d output(s), %zu parameters\n", argv[1], nmz_net_input_size(net),
           sym("→", "->"), nmz_net_output_size(net), nmz_net_param_count(net));
    return 0;
}

static int cmd_clear(cli_state *st, int argc, char **argv) {
    (void)argv;
    if (argc != 1) return fail("clear takes no arguments%s", NULL);
    if (!st->net) {
        printf("(no network loaded)\n");
        return 0;
    }
    cli_replace_net(st, NULL, NULL);
    st->dirty = 0;
    printf("Network cleared.\n");
    return 0;
}

static int cmd_set(cli_state *st, int argc, char **argv) {
    nmz_train_opts o;
    unsigned mask = 0;
    int i;
    if (!need_net(st)) return 1;
    o = *nmz_net_train_config(st->net);
    if (argc < 2) return fail("usage: set OPTION VALUE ... (see `help set`)%s", NULL);
    for (i = 1; i < argc; i++) {
        int r;
        if (!strcmp(argv[i], "--normalize")) {
            nmz_norm n;
            const char *v = cli_opt_value(argc, argv, &i);
            if (!v || nmz_norm_parse(v, &n) != NMZ_OK) return v ? lib_fail("--normalize") : 1;
            if (nmz_net_set_normalization(st->net, n) != NMZ_OK) return lib_fail("--normalize");
            printf("Normalization: %s%s\n", nmz_norm_name(n), n ? " (fitted at the next train)" : "");
            continue;
        }
        r = cli_train_option(argc, argv, &i, &o, &mask);
        if (r < 0) return 1;
        if (r == 0) return fail("unknown setting '%s' (see `help set`)", argv[i]);
    }
    *nmz_net_train_config(st->net) = o;
    st->dirty = 1;
    printf("Training config: ");
    cli_print_train_config(&o);
    return 0;
}

static int cmd_reinit(cli_state *st, int argc, char **argv) {
    uint64_t seed = 42u;
    if (!need_net(st)) return 1;
    if (argc > 2 || (argc == 2 && !cli_parse_u64(argv[1], &seed))) return fail("usage: reinit [SEED]%s", NULL);
    if (nmz_net_reinit(st->net, seed) != NMZ_OK) return lib_fail("reinit");
    st->dirty = 1;
    printf("New random weights (seed %llu); training steps and optimizer state reset.\n", (unsigned long long)seed);
    return 0;
}

/* ------------------------------------------------------------------ help */

typedef struct cmd {
    const char *name;
    int (*fn)(cli_state *, int, char **);
    const char *usage;
    const char *help;
} cmd;

static int cmd_help(cli_state *st, int argc, char **argv);
static int cmd_quit(cli_state *st, int argc, char **argv) {
    (void)argc;
    (void)argv;
    st->quit = 1;
    return 0;
}

static const cmd k_cmds[] = {
    {"new", cmd_new, "new INPUT LAYER... [options]  |  new --preset NAME",
     "Create a network. INPUT is a size (784) or an image shape (1x28x28, or 28x28).\n"
     "  Layers (each one argument):\n"
     "    N[:ACT][:INIT]                     dense layer, e.g. 64:relu, 10:softmax, 4:tanh:xavier\n"
     "    conv:F:K[:ACT][:INIT][:stride=S][:pad=P|same]   e.g. conv:8:3:relu, conv:16:5:relu:pad=same\n"
     "    pool:K[:stride=S]                  max pooling, e.g. pool:2\n"
     "    flatten | dropout:RATE             (a flatten is inserted automatically before dense layers)\n"
     "  activations: linear sigmoid tanh relu leaky_relu[=ALPHA] softmax    inits: auto xavier he\n"
     "  Text models: INPUT `text` with --data FILE (builds the character vocabulary), then\n"
     "    embed:D  rnn:H  lstm:H[:forget_bias=F]  gru:H  (inits: auto = orthogonal, xavier, he)  vocab\n"
     "    e.g. new text embed:32 lstm:64 dropout:0.2 lstm:64 vocab --data FILE   (stacked)\n"
     "    --items/--no-items, --strip-diacritics (Arabic tashkeel + tatweel), --unify-alef (أ إ آ -> ا)\n"
     "  Transformers (decoder-only, GPT-style), text input:\n"
     "    pos:learned:CTX | pos:sin:CTX     position encoding; CTX = the context window\n"
     "    block[:heads=H][:head_dim=K][:ff=F][:gelu|relu][:dropout=R]   pre-LN block (H=4, K=D/H, F=4D)\n"
     "    attn[:heads=H][:head_dim=K]  ffn[:F][:gelu|relu]  ln          the parts on their own\n"
     "    vocab:tied                        output = the embedding matrix, transposed (weight tying)\n"
     "    e.g. new text embed:64 pos:learned:128 block:heads=4 block:heads=4 ln vocab:tied --data FILE\n"
     "    defaults: AdamW (weight decay 0.1 on weight matrices), warm-up + cosine per batch, clip 1\n"
     "  Quick mode: new 2 4 1   (defaults chosen and explained).\n"
     "  Presets: mnist-cnn mnist-mlp names char-rnn names-lstm char-lstm arabic-text\n"
     "           names-gpt char-gpt arabic-gpt\n"
     "  options: --loss mse|ce --normalize none|minmax|zscore --seed N\n"
     "           --optimizer sgd|adam|adamw --lr X --momentum X --beta1 X --beta2 X --eps X\n"
     "           --weight-decay X --decay-all --warmup N (batches)\n"
     "           --schedule constant|step|cosine|warmup-cosine --step-size N --gamma X --min-lr X\n"
     "           --batch N --epochs N --target-loss X --no-shuffle"},
    {"show", cmd_show, "show [weights | filters | vocab | params | gates (--text T | --data FILE) | attention (--text T | --data FILE) | sample N [--data SPEC]]",
     "Describe the network; `weights` prints every parameter, `filters` draws the conv kernels as\n"
     "  ASCII heatmaps, `sample N` draws sample N (default from mnist:test) with the prediction,\n"
     "  `vocab` lists a text model's characters. `gates` runs a text (--text, or --data FILE with\n"
     "  --from K --chars N, default 20000 characters) through an LSTM/GRU model and summarises its\n"
     "  gates: means, how often they are near 0 or 1, the forget/update gate per unit, memory half-lives.\n"
     "  `attention` (Transformers): a heatmap of query x key positions for --layer N --head H (1-based,\n"
     "  default 1 1; --full: all positions), and for every head what it mostly attends to (previous\n"
     "  character, itself, start of the word, ...) - measured over the text (default 4000 characters of\n"
     "  --data), in windows of the context length. `params`: parameters per layer and per component."},
    {"train", cmd_train, "train DATA [options]",
     "Train. DATA is a CSV file (first INPUTS columns are inputs, the rest targets; a text label column\n"
     "  is one-hot encoded), `mnist` (the training set, after `make data`) or an IDX images file.\n"
     "  Options override the stored config for this run only (use `set` to change it):\n"
     "  --limit N (first N samples) --epochs N --batch N --lr X --optimizer sgd|adam --schedule ...\n"
     "  --every N --quiet --header/--no-header. A live progress line is shown on terminals.\n"
     "  Validation: --val F (the last fraction F of DATA) or --val-file FILE (a separate set; then all\n"
     "  of DATA trains). Text models: --seq-len N --carry/--no-carry --clip X --patience N.\n"
     "  AdamW: --weight-decay X --decay-all; per-batch schedule: --schedule warmup-cosine --warmup N."},
    {"test", cmd_test, "test DATA [--limit N]",
     "Loss and accuracy on a dataset (`mnist` = the 10 000 test images), with a confusion matrix."},
    {"predict", cmd_predict, "predict X1 ... Xn  |  predict --sample N [--data SPEC]",
     "Run one sample forward and print the outputs."},
    {"trace", cmd_trace_fwd, "trace (X1 ... Xn --target T... | --sample N [--data SPEC] | --text TEXT) [--at F,Y,X] [--unit J] [--json FILE] [--full]",
     "Run ONE sample forward and backward and print every value with its formula; conv layers show\n"
     "  the receptive-field patch, the kernel, the products and the sum at one output position\n"
     "  (--at FILTER,ROW,COL, 1-based; default: filter 1 at the centre), max-pool shows the window and\n"
     "  its winner. Nothing is changed (evaluation mode; the update is only previewed).\n"
     "  --target may be a class name; with --sample the dataset label is the target.\n"
     "  Text models: trace --text \"Alice was\" (or --sample N --data FILE) shows the recurrent layer\n"
     "  unrolled through time for unit --unit J: rnn h_t = tanh(...); lstm every gate i f g o, c_t and\n"
     "  h_t with bars; gru z, r, h~ and h_t. Then heatmaps over time (all hidden units, or the gates of\n"
     "  one cell and the forget/update gate of every cell), backpropagation through time and how the\n"
     "  gradient of the last character vanishes (along h, and along the LSTM cell state c).\n"
     "  Stacked models: --layer N shows the N-th recurrent layer.\n"
     "  Transformers: one query position (--pos P, 0-based; default the last) in one layer and head\n"
     "  (--layer N, --head H; default layer 1 and its most decisive head): q, every k, the raw scores,\n"
     "  the causal mask, the softmax weights as bars with the characters, why the winner wins, the\n"
     "  weighted sum of the values; then all heads at that position, a heatmap, what each head\n"
     "  attends to, the residual stream, the prediction and the direct gradient path."},
    {"step", cmd_step_fwd, "step (same arguments as trace) [--lr X] [--optimizer ...]",
     "Like trace, but in training mode and the weight update is applied: one step you can follow."},
    {"generate", cmd_generate,
     "generate [--prime TEXT] [--length N] [--temperature X] [--top-k K] [--seed N] [--count N] [--data FILE] [--new-only] [--context N] [--no-cache]",
     "Sample text from a character model. Line-per-item models (names) print --count distinct items\n"
     "  (default 10), each ended by the end-of-item token; with --data FILE items that occur in that\n"
     "  file are marked, --new-only skips them. Running-text models print --length characters (300).\n"
     "  --temperature < 1 is safer/more repetitive, > 1 wilder; --top-k K samples among the K most\n"
     "  likely characters only; the same --seed gives the same text.\n"
     "  Transformers use a KV cache (one new position per step); past the context window the newest half\n"
     "  is kept and re-encoded. --context N uses a smaller window; --no-cache recomputes the whole window\n"
     "  at every step (same text, slower: the reference the cache is tested against)."},
    {"compare", cmd_compare, "compare MODEL.nmz [MODEL.nmz ...] (--text TEXT | --data FILE [--from K] [--chars N]) [--layer N] [--full]",
     "Gradient survival side by side: for each trained text model (e.g. an rnn, an lstm and a gru\n"
     "  trained on the same data), how much of the LAST character's loss gradient reaches the step k\n"
     "  characters back - along h, along an LSTM's cell state c, and the product of the forget (LSTM)\n"
     "  or update (GRU) gates. With --data the text is --chars characters (default 65) from offset --from.\n"
     "  Transformers are compared at the input of their first block: attention reaches every earlier\n"
     "  position directly (the text must fit the context window)."},
    {"gradcheck", cmd_gradcheck, "gradcheck (X1 ... Xn --target T... | --sample N [--data SPEC] | --text TEXT)",
     "Compare every backprop gradient with a numerical derivative (L(w+e)-L(w-e))/2e."},
    {"save", cmd_save, "save [FILE] [--no-optimizer]",
     "Save (portable .nmz). Optimizer state is included so training can resume; --no-optimizer omits it."},
    {"load", cmd_load, "load FILE", "Load a .nmz model (format 2 or 3)."},
    {"set", cmd_set, "set OPTION VALUE ...",
     "Change the stored training configuration (--lr, --optimizer, --schedule, --batch, --epochs, ...)\n"
     "  or the input normalization (--normalize none|minmax|zscore; refitted at the next train)."},
    {"reinit", cmd_reinit, "reinit [SEED]", "Start over with new random weights (default seed 42)."},
    {"clear", cmd_clear, "clear", "Unload the current network."},
    {"help", cmd_help, "help [COMMAND]", "Show help."},
    {"quit", cmd_quit, "quit", "Leave the shell (also: exit, Ctrl-D)."},
    {"exit", cmd_quit, "exit", NULL},
};

void cli_print_help(const char *topic) {
    size_t i;
    if (topic) {
        for (i = 0; i < sizeof k_cmds / sizeof k_cmds[0]; i++)
            if (!strcmp(k_cmds[i].name, topic) && k_cmds[i].help) {
                printf("usage: %s\n%s\n", k_cmds[i].usage, k_cmds[i].help);
                return;
            }
        printf("No help for '%s'.\n", topic);
        return;
    }
    printf("NeuroMZ %s — an educational neural-network engine\n\n", NMZ_VERSION_STRING);
    printf("Commands:\n");
    for (i = 0; i < sizeof k_cmds / sizeof k_cmds[0]; i++)
        if (k_cmds[i].help) printf("  %s\n", k_cmds[i].usage);
    printf("\nOne-shot use from the shell:  neuromz [-m MODEL] COMMAND ARGS... [-o MODEL]\n"
           "  -m FILE  load a model before the command     -o FILE  save the model after it\n"
           "  --ascii  plain ASCII formulas (no Greek letters)\n"
           "Without a command, neuromz starts an interactive shell. `help COMMAND` for details.\n");
}

static int cmd_help(cli_state *st, int argc, char **argv) {
    (void)st;
    if (argc > 2) return fail("usage: help [COMMAND]%s", NULL);
    cli_print_help(argc == 2 ? argv[1] : NULL);
    return 0;
}

int cli_run(cli_state *st, int argc, char **argv) {
    size_t i;
    if (argc < 1) return 0;
    for (i = 0; i < sizeof k_cmds / sizeof k_cmds[0]; i++)
        if (!strcmp(k_cmds[i].name, argv[0])) return k_cmds[i].fn(st, argc, argv);
    fprintf(stderr, "error: unknown command '%s' (type `help`)\n", argv[0]);
    return 1;
}
