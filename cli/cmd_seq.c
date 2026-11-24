/* cmd_seq.c — commands for recurrent text models:
 *
 *   show gates  run a text through the model and summarise what the gates
 *               do: mean of every gate, the share of values near 0 / near 1,
 *               the mean forget (LSTM) or update (GRU) gate per unit, how
 *               many units stay mostly open, and the memory "half-life"
 *               of the most open ones.
 *   compare     the centrepiece of phase 4: for 2+ trained models and the
 *               same text, how much of the LAST character's loss gradient
 *               survives k steps back — side by side, per step, along h
 *               and (for an LSTM) along the cell state c; for a Transformer
 *               at the input of its first block (attention's direct path).
 *   show attention   heatmap of one head and what every head attends to,
 *               over any text (phase 5)
 *   show params      parameters per layer and per component */
#include "cli.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "neuromz_layer.h"

int net_has_attention(const nmz_net *net) { return seq_attention_layer(net, 1) >= 0; }

int seq_attention_layer(const nmz_net *net, int n) {
    int i;
    for (i = 0; net && i < nmz_net_layer_count(net); i++)
        if (nmz_layer_has_attention(nmz_net_layer((nmz_net *)net, i)) && --n == 0) return i;
    return -1;
}

int seq_recurrent_layer(const nmz_net *net, int n) {
    int i;
    for (i = 0; i < nmz_net_layer_count(net); i++)
        if (nmz_layer_is_recurrent(nmz_net_layer((nmz_net *)net, i)) && --n == 0) return i;
    return -1;
}

/* text → ids with the model's vocabulary and normalisation; NULL on error
 * (message printed). `what` names the source in messages. */
static int *encode(const nmz_vocab *v, const char *text, size_t len, int *n, const char *what) {
    uint32_t *cp = NULL;
    size_t m = 0, bad = 0, k;
    int *ids;
    if (nmz_utf8_decode(text, len, &cp, &m, &bad) != NMZ_OK) {
        cli_lib_fail("text");
        return NULL;
    }
    m = nmz_text_normalize(cp, m, v->normalize);
    ids = malloc((m ? m : 1) * sizeof *ids);
    if (!ids) {
        free(cp);
        cli_fail("out of memory%s", NULL);
        return NULL;
    }
    for (k = 0; k < m; k++) {
        ids[k] = nmz_vocab_index(v, cp[k]);
        if (ids[k] < 0) {
            char lbl[24];
            fprintf(stderr, "error: '%s' (U+%04X) is not in the vocabulary of %s (`show vocab` lists it)\n",
                    cli_char_label(cp[k], lbl, sizeof lbl), (unsigned)cp[k], what);
            free(cp);
            free(ids);
            return NULL;
        }
    }
    free(cp);
    *n = (int)m;
    return ids;
}

/* the whole file (for --data) */
static char *read_all(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    char *buf = NULL;
    size_t cap = 0, n = 0;
    if (!f) {
        fprintf(stderr, "error: cannot open '%s'\n", path);
        return NULL;
    }
    for (;;) {
        size_t got;
        if (n + 65536 + 1 > cap) {
            char *p = realloc(buf, cap = (n + 65536 + 1) * 2);
            if (!p) {
                free(buf);
                fclose(f);
                cli_fail("out of memory%s", NULL);
                return NULL;
            }
            buf = p;
        }
        got = fread(buf + n, 1, 65536, f);
        n += got;
        if (got < 65536) break;
    }
    fclose(f);
    buf[n] = '\0';
    *len = n;
    return buf;
}

/* --text TEXT, or --data FILE [--from K] [--chars N] (characters, not bytes) */
typedef struct text_src {
    const char *text, *data;
    long from, chars;
    char *owned; /* file contents */
    const char *s;
    size_t len;
} text_src;

static int text_src_option(int argc, char **argv, int *i, text_src *ts) {
    const char *a = argv[*i], *v;
    if (strcmp(a, "--text") && strcmp(a, "--data") && strcmp(a, "--from") && strcmp(a, "--chars")) return 0;
    if (!(v = cli_opt_value(argc, argv, i))) return -1;
    if (!strcmp(a, "--text")) ts->text = v;
    else if (!strcmp(a, "--data")) ts->data = v;
    else if (!strcmp(a, "--from")) {
        if (!cli_parse_int(v, 0, 2000000000L, &ts->from)) return cli_fail("--from needs a character offset >= 0, got '%s'", v), -1;
    } else if (!cli_parse_int(v, 2, 10000000L, &ts->chars)) {
        return cli_fail("--chars needs a number >= 2, got '%s'", v), -1;
    }
    return 1;
}

/* resolve to UTF-8 bytes: the text, or chars [from, from+chars) of the file */
static int text_src_load(text_src *ts, long default_chars) {
    size_t i, start = 0, end;
    long c = 0;
    if (ts->text) {
        ts->s = ts->text;
        ts->len = strlen(ts->text);
        return 0;
    }
    if (!ts->data) return cli_fail("give a text: --text \"...\" or --data FILE [--from K] [--chars N]%s", NULL);
    if (!(ts->owned = read_all(ts->data, &ts->len))) return 1;
    if (!ts->chars) ts->chars = default_chars;
    /* character offsets → byte offsets (UTF-8 lead bytes start characters) */
    for (i = 0; i < ts->len; i++)
        if (((unsigned char)ts->owned[i] & 0xC0) != 0x80 && c++ == ts->from) break;
    start = i;
    c = 0;
    for (end = start; end < ts->len; end++)
        if (((unsigned char)ts->owned[end] & 0xC0) != 0x80 && c++ == ts->chars) break;
    if (start >= ts->len) return cli_fail("--from is beyond the end of %s", ts->data);
    ts->s = ts->owned + start;
    ts->len = end - start;
    return 0;
}

/* ---------------------------------------------------------- show gates */

typedef struct gate_acc {
    int layer, lstm, H;
    long steps;
    double *unit_sum; /* [4 × H]: per gate block of `gates`, per unit */
    long hist[4][4];  /* per gate block: values < 0.1, 0.1-0.5, 0.5-0.9, > 0.9 */
} gate_acc;

typedef struct gate_stats {
    gate_acc *acc;
    int n;
} gate_stats;

/* gate k of a layer in the "gates" tensor: lstm i f g o → 0 1 2 3; gru z r h~ */
static void gate_trace(const nmz_trace_event *ev, void *user) {
    gate_stats *gs = user;
    const nmz_tensor *g;
    int a, T, G, t, k, j;
    if (ev->phase != NMZ_TRACE_FORWARD || ev->time_step != -1) return;
    if (strcmp(ev->layer_type, "lstm") && strcmp(ev->layer_type, "gru")) return;
    g = nmz_trace_find(ev, "gates");
    if (!g || g->rank != 3) return;
    for (a = 0; a < gs->n && gs->acc[a].layer != ev->layer; a++) {
    }
    if (a == gs->n) return;
    T = g->shape[1];
    G = g->shape[2];
    for (t = 0; t < T; t++)
        for (k = 0; k < (gs->acc[a].lstm ? 4 : 3); k++) {
            if (k == 2) continue; /* g (lstm) and h~ (gru) are candidates, not gates */
            for (j = 0; j < gs->acc[a].H; j++) {
                double v = g->data[(size_t)t * G + (size_t)k * gs->acc[a].H + j];
                int bin = v < 0.1 ? 0 : v < 0.5 ? 1 : v < 0.9 ? 2 : 3;
                gs->acc[a].unit_sum[(size_t)k * gs->acc[a].H + j] += v;
                gs->acc[a].hist[k][bin]++;
            }
        }
    gs->acc[a].steps += T;
}

static void print_gate_block(const gate_acc *a, const nmz_net *net) {
    static const char *const lnames[4] = {"i  input gate ", "f  forget gate", "", "o  output gate"};
    static const char *const gnames[3] = {"z  update gate", "r  reset gate ", ""};
    int H = a->H, k, j, keep = a->lstm ? 1 : 0, open = 0, closed = 0, top[5], nt = 0;
    long total = a->steps * H;
    char desc[160];
    nmz_layer_describe(nmz_net_layer((nmz_net *)net, a->layer), desc, sizeof desc);
    printf("\nlayer %d: %s\n", a->layer, desc);
    printf("  %-16s %7s   %8s %8s %8s %8s    (share of all %ld values)\n", "gate", "mean", "< 0.1", "0.1-0.5",
           "0.5-0.9", "> 0.9", total);
    for (k = 0; k < (a->lstm ? 4 : 3); k++) {
        double sum = 0;
        if (k == 2) continue;
        for (j = 0; j < H; j++) sum += a->unit_sum[(size_t)k * H + j];
        printf("  %-16s %7.3f   %7.1f%% %7.1f%% %7.1f%% %7.1f%%\n", a->lstm ? lnames[k] : gnames[k], sum / (double)total,
               100.0 * a->hist[k][0] / total, 100.0 * a->hist[k][1] / total, 100.0 * a->hist[k][2] / total,
               100.0 * a->hist[k][3] / total);
    }
    /* per-unit mean of the keep gate (f, or z) */
    {
        long bins[10] = {0}, mx = 1;
        double *m = malloc((size_t)H * sizeof *m);
        if (!m) return;
        for (j = 0; j < H; j++) {
            int b;
            m[j] = a->unit_sum[(size_t)keep * H + j] / (double)a->steps;
            b = (int)(m[j] * 10);
            if (b > 9) b = 9;
            if (b < 0) b = 0;
            if (++bins[b] > mx) mx = bins[b];
            open += m[j] > 0.9;
            closed += m[j] < 0.1;
        }
        printf("\n  mean %s per %s (%d %ss): how much of its state each one keeps per step\n",
               a->lstm ? "forget gate f" : "update gate z", a->lstm ? "cell" : "unit", H, a->lstm ? "cell" : "unit");
        for (k = 0; k < 10; k++) {
            int n = (int)(40.0 * bins[k] / mx + 0.5), q;
            printf("    %.1f-%.1f %4ld ", k / 10.0, (k + 1) / 10.0, bins[k]);
            for (q = 0; q < n; q++) fputs(sym("█", "#"), stdout);
            if (k == 0 && bins[k]) printf("  <- reset at almost every step");
            if (k == 9 && bins[k]) printf("  <- long memory");
            printf("\n");
        }
        printf("  mostly open (mean > 0.9): %d of %d; mostly closed (mean < 0.1): %d of %d\n", open, H, closed, H);
        /* the five most open */
        for (k = 0; k < 5 && k < H; k++) {
            int best = -1;
            for (j = 0; j < H; j++) {
                int used = 0, q;
                for (q = 0; q < nt; q++) used |= top[q] == j;
                if (!used && (best < 0 || m[j] > m[best])) best = j;
            }
            top[nt++] = best;
        }
        printf("  longest memory (a state that is only multiplied by %s halves after ln 0.5 / ln %s steps):\n",
               a->lstm ? "f" : "z", a->lstm ? "f" : "z");
        for (k = 0; k < nt; k++) {
            double f = m[top[k]], half = f > 0 && f < 1 ? log(0.5) / log(f) : INFINITY;
            printf("    %s %4d   mean %s = %.3f   half-life ~ %s%.1f steps\n", a->lstm ? "cell" : "unit", top[k] + 1,
                   a->lstm ? "f" : "z", f, isinf(half) ? ">" : "", isinf(half) ? 1e6 : half);
        }
        free(m);
    }
}

int show_gates(cli_state *st, int argc, char **argv) {
    text_src ts;
    gate_stats gs;
    const nmz_vocab *v;
    nmz_tensor x = NMZ_TENSOR_EMPTY;
    int *ids = NULL, n = 0, i, rc = 1, chunk = 256, pos;
    double t0;
    memset(&ts, 0, sizeof ts);
    memset(&gs, 0, sizeof gs);
    if (!cli_need_net(st)) return 1;
    v = nmz_net_vocab(st->net);
    if (!v || !nmz_net_is_sequence(st->net)) return cli_fail("show gates needs a text model with LSTM or GRU layers%s", NULL);
    for (i = 1; i < argc; i++) {
        int r = text_src_option(argc, argv, &i, &ts);
        if (r < 0) return 1;
        if (r == 0 && strcmp(argv[i], "gates") != 0)
            return cli_fail("show gates: unexpected '%s' (use --text TEXT or --data FILE [--from K] [--chars N])", argv[i]);
    }
    for (i = 0; i < nmz_net_layer_count(st->net); i++) {
        const char *ty = nmz_layer_type(nmz_net_layer(st->net, i));
        if (!strcmp(ty, "lstm") || !strcmp(ty, "gru")) gs.n++;
    }
    if (!gs.n) return cli_fail("this model has no gated layer (lstm or gru); a vanilla rnn has no gates%s", NULL);
    if (text_src_load(&ts, 20000)) goto out;
    if (!(ids = encode(v, ts.s, ts.len, &n, "the model"))) goto out;
    if (n < 1) {
        cli_fail("the text is empty%s", NULL);
        goto out;
    }
    gs.acc = calloc((size_t)gs.n, sizeof *gs.acc);
    if (!gs.acc) goto oom;
    for (i = 0, gs.n = 0; i < nmz_net_layer_count(st->net); i++) {
        nmz_layer *l = nmz_net_layer(st->net, i);
        const char *ty = nmz_layer_type(l);
        if (strcmp(ty, "lstm") && strcmp(ty, "gru")) continue;
        gs.acc[gs.n].layer = i;
        gs.acc[gs.n].lstm = !strcmp(ty, "lstm");
        gs.acc[gs.n].H = nmz_layer_output_size(l);
        gs.acc[gs.n].unit_sum = calloc((size_t)4 * gs.acc[gs.n].H, sizeof(double));
        if (!gs.acc[gs.n++].unit_sum) goto oom;
    }
    t0 = cli_now();
    nmz_net_reset_state(st->net);
    nmz_net_set_carry(st->net, 1); /* one stream: the state runs through the chunks */
    nmz_net_set_trace(st->net, gate_trace, &gs);
    for (pos = 0; pos < n; pos += chunk) {
        int len = n - pos < chunk ? n - pos : chunk, shape[2], k;
        shape[0] = 1;
        shape[1] = len;
        if (nmz_tensor_reshape(&x, 2, shape) != NMZ_OK) goto oom;
        for (k = 0; k < len; k++) x.data[k] = (float)ids[pos + k];
        if (nmz_net_forward_mode(st->net, &x, 0, NULL) != NMZ_OK) {
            nmz_net_set_trace(st->net, NULL, NULL);
            nmz_net_set_carry(st->net, 0);
            nmz_net_reset_state(st->net);
            cli_lib_fail("show gates");
            goto out;
        }
    }
    nmz_net_set_trace(st->net, NULL, NULL);
    nmz_net_set_carry(st->net, 0);
    nmz_net_reset_state(st->net);
    printf("Gate activity over %d characters%s%s (evaluation mode, state carried through the text, %.1fs)\n", n,
           ts.data ? " of " : "", ts.data ? ts.data : "", cli_now() - t0);
    printf("A gate value is a fraction: 0 = closed (blocks), 1 = open (passes everything).\n");
    for (i = 0; i < gs.n; i++) print_gate_block(&gs.acc[i], st->net);
    rc = 0;
    goto out;
oom:
    cli_fail("out of memory%s", NULL);
out:
    if (gs.acc)
        for (i = 0; i < gs.n; i++) free(gs.acc[i].unit_sum);
    free(gs.acc);
    free(ids);
    free(ts.owned);
    nmz_tensor_free(&x);
    return rc;
}

/* ------------------------------------------------------------- compare */

typedef struct cmp_model {
    const char *path;
    nmz_net *net;
    seq_survival sv;
    double loss;
    char desc[96];
    char letter;
} cmp_model;

/* the trace of a last-step-only backward pass of one model */
static int survival_of(cmp_model *m, const int *ids, int T, int layer_n) {
    const nmz_vocab *v = nmz_net_vocab(m->net);
    nmz_trace_recorder *rec = NULL;
    nmz_tensor x = NMZ_TENSOR_EMPTY, t = NMZ_TENSOR_EMPTY;
    int xs[2], ts[3], k, V = v->size, layer = seq_recurrent_layer(m->net, layer_n), rc = 1,
                                       ctx = nmz_net_context_length(m->net);
    if (layer < 0 && !net_has_attention(m->net)) {
        fprintf(stderr, "error: %s has no recurrent layer number %d\n", m->path, layer_n);
        return 1;
    }
    if (ctx > 0 && T > ctx) {
        fprintf(stderr, "error: the text has %d steps but %s sees at most %d positions (a shorter --text, or --chars %d with --data)\n", T, m->path,
                ctx, ctx + 1);
        return 1;
    }
    xs[0] = ts[0] = 1;
    xs[1] = ts[1] = T;
    ts[2] = V;
    if (nmz_tensor_init(&x, 2, xs) != NMZ_OK || nmz_tensor_init(&t, 3, ts) != NMZ_OK ||
        nmz_trace_recorder_create(&rec) != NMZ_OK)
        goto lib;
    for (k = 0; k < T; k++) x.data[k] = (float)ids[k];
    t.data[(size_t)(T - 1) * V + ids[T]] = 1.0f; /* the loss of the LAST prediction only */
    nmz_net_reset_state(m->net);
    nmz_net_set_trace(m->net, nmz_trace_recorder_fn, rec);
    if (nmz_net_forward_mode(m->net, &x, 0, NULL) != NMZ_OK || nmz_net_backward(m->net, &t, &m->loss) != NMZ_OK) {
        nmz_net_set_trace(m->net, NULL, NULL);
        goto lib;
    }
    nmz_net_set_trace(m->net, NULL, NULL);
    m->loss *= T; /* the mean over T rows, of which only the last has a target */
    if (nmz_trace_recorder_status(rec) != NMZ_OK || seq_survival_collect(rec, layer, T, &m->sv)) {
        cli_fail("out of memory while recording the trace%s", NULL);
        goto out;
    }
    rc = 0;
    goto out;
lib:
    cli_lib_fail(m->path);
out:
    nmz_trace_recorder_free(rec);
    nmz_tensor_free(&x);
    nmz_tensor_free(&t);
    return rc;
}

/* print s left-aligned in w display columns (UTF-8 aware) */
static void put_pad(const char *s, int w, int right) {
    int n = 0, q;
    for (q = 0; s[q]; q++) n += ((unsigned char)s[q] & 0xC0) != 0x80;
    if (right) printf("%*s%s", n < w ? w - n : 0, "", s);
    else printf("%s%*s", s, n < w ? w - n : 0, "");
}

static const char *base_name(const char *p) {
    const char *b = strrchr(p, '/');
    return b ? b + 1 : p;
}

/* one series = one path of one model */
typedef struct series {
    const cmp_model *m;
    const double *v; /* norms (or the product of gates) per step */
    char label[24];
    int is_prod;
} series;

static double rel(const series *s, int t, int T) {
    if (s->is_prod) return s->v[t];
    return s->v[T - 1] > 0 ? s->v[t] / s->v[T - 1] : 0;
}

/* sparkline over steps back 0 … T−1 on a log axis 1 … lo */
static void spark(const series *s, int T, double lo) {
    static const char *const blocks[8] = {"▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"};
    static const char ascii[] = " .:-=+*#";
    int k;
    for (k = 0; k < T; k++) {
        double r = rel(s, T - 1 - k, T), f = r > 0 ? 1.0 - log10(r) / log10(lo) : 0;
        int b = (int)(f * 8);
        if (f <= 0 || r <= 0) {
            fputs(" ", stdout);
            continue;
        }
        if (b > 7) b = 7;
        if (b < 0) b = 0;
        if (print_is_ascii())
            putchar(ascii[b]);
        else
            fputs(blocks[b], stdout);
    }
}

static int steps_until(const series *s, int T, double frac) {
    int k;
    for (k = 1; k < T; k++)
        if (rel(s, T - 1 - k, T) < frac) return k;
    return -1;
}

int cmd_compare(cli_state *st, int argc, char **argv) {
    cmp_model ms[6];
    series se[18];
    text_src ts;
    int nm = 0, ns = 0, i, k, rc = 1, T = 0, *ids0 = NULL, n0 = 0, full = 0;
    long layer_n = 1;
    double lo = 1e-6;
    (void)st;
    memset(ms, 0, sizeof ms);
    memset(se, 0, sizeof se);
    memset(&ts, 0, sizeof ts);
    for (i = 1; i < argc; i++) {
        int r = text_src_option(argc, argv, &i, &ts);
        if (r < 0) goto out;
        if (r) continue;
        if (!strcmp(argv[i], "--full")) {
            full = 1;
        } else if (!strcmp(argv[i], "--layer")) {
            const char *v = cli_opt_value(argc, argv, &i);
            if (!v || !cli_parse_int(v, 1, 1000, &layer_n)) {
                if (v) cli_fail("--layer needs a positive integer, got '%s'", v);
                goto out;
            }
        } else if (cli_is_option(argv[i])) {
            cli_fail("unknown option '%s' for compare (see `help compare`)", argv[i]);
            goto out;
        } else if (nm < 6) {
            ms[nm].path = argv[i];
            ms[nm].letter = (char)('A' + nm);
            nm++;
        } else {
            cli_fail("compare takes at most 6 models%s", NULL);
            goto out;
        }
    }
    if (nm < 1) {
        cli_fail("usage: compare MODEL.nmz [MODEL.nmz ...] --text \"...\" | --data FILE [--from K] [--chars N]%s", NULL);
        goto out;
    }
    if (text_src_load(&ts, 65)) goto out;
    for (i = 0; i < nm; i++) {
        int *ids, n, li;
        const nmz_vocab *v;
        if (nmz_net_load(&ms[i].net, ms[i].path) != NMZ_OK) {
            cli_lib_fail(ms[i].path);
            goto out;
        }
        v = nmz_net_vocab(ms[i].net);
        if (!v || !nmz_net_is_sequence(ms[i].net)) {
            fprintf(stderr, "error: %s is not a text model\n", ms[i].path);
            goto out;
        }
        if (!(ids = encode(v, ts.s, ts.len, &n, ms[i].path))) goto out;
        if (n < 3) {
            free(ids);
            cli_fail("the text needs at least 3 characters%s", NULL);
            goto out;
        }
        if (i == 0) {
            n0 = n;
            ids0 = ids; /* kept for printing (model A's vocabulary) */
        } else if (n != n0) {
            free(ids);
            fprintf(stderr, "error: %s normalises the text to %d characters, %s to %d: compare models with the "
                            "same text settings\n", ms[0].path, n0, ms[i].path, n);
            goto out;
        }
        T = n - 1;
        k = survival_of(&ms[i], ids, T, (int)layer_n);
        if (i) free(ids);
        if (k) goto out;
        li = ms[i].sv.layer;
        if (!strcmp(ms[i].sv.kind, "gpt")) {
            int nb = 0;
            while (seq_attention_layer(ms[i].net, nb + 1) >= 0) nb++;
            snprintf(ms[i].desc, sizeof ms[i].desc, "transformer, %d block%s, %zu parameters", nb, nb == 1 ? "" : "s",
                     nmz_net_param_count(ms[i].net));
        } else {
            nmz_layer *l = nmz_net_layer(ms[i].net, li);
            int stacked = seq_recurrent_layer(ms[i].net, 2) >= 0;
            snprintf(ms[i].desc, sizeof ms[i].desc, "%s %d%s, %zu parameters", nmz_layer_type(l),
                     nmz_layer_output_size(l), stacked ? " (stacked)" : "", nmz_net_param_count(ms[i].net));
        }
    }
    /* the series: h of every model; c and Π f of an LSTM; Π z of a GRU */
    for (i = 0; i < nm; i++) {
        series *s = &se[ns++];
        s->m = &ms[i];
        s->v = ms[i].sv.h;
        if (!strcmp(ms[i].sv.kind, "gpt"))
            snprintf(s->label, sizeof s->label, "%c gpt x", ms[i].letter);
        else
            snprintf(s->label, sizeof s->label, "%c %s h", ms[i].letter, ms[i].sv.kind);
        if (ms[i].sv.c) {
            s = &se[ns++];
            s->m = &ms[i];
            s->v = ms[i].sv.c;
            snprintf(s->label, sizeof s->label, "%c lstm c", ms[i].letter);
        }
        if (ms[i].sv.gate_prod) {
            s = &se[ns++];
            s->m = &ms[i];
            s->v = ms[i].sv.gate_prod;
            s->is_prod = 1;
            snprintf(s->label, sizeof s->label, "%c %s", ms[i].letter,
                     ms[i].sv.c ? sym("Π f", "prod f") : sym("Π z", "prod z"));
        }
    }
    printf("GRADIENT SURVIVAL   how much of the LAST character's loss gradient reaches each earlier step\n\n");
    printf("text (%d steps; the models predict the last character from all before it):\n  ", T);
    fwrite(ts.s, 1, ts.len, stdout);
    printf("\n\nmodels (recurrent layer %ld of each):\n", layer_n);
    for (i = 0; i < nm; i++)
        printf("  %c  %-20s %-36s loss of the last prediction %.3f\n", ms[i].letter, base_name(ms[i].path), ms[i].desc,
               ms[i].loss);
    printf("\n|dL/dh| (and |dL/dc|) k steps back, relative to the last step: 1 = all of it, 0.01 = 1%%.\n");
    for (i = 0; i < nm; i++)
        if (!strcmp(ms[i].sv.kind, "gpt")) {
            printf("gpt x: for a Transformer, |dL/dx_t| at the input of its first block (character + position\n"
                   "vector): there is no chain of steps, attention reaches every earlier position directly.\n");
            break;
        }
    printf("%s f / %s z: the mean product of the forget / update gates over those k steps - what the\n"
           "cell-state (LSTM) or direct (GRU) path alone would pass back.\n\n",
           sym("Π", "prod"), sym("Π", "prod"));
    printf("  %4s  %-5s", "back", "char");
    for (k = 0; k < ns; k++) {
        putchar(' ');
        put_pad(se[k].label, 10, 1);
    }
    printf("\n");
    for (k = 0; k < T; k++) {
        int t = T - 1 - k, j;
        char lbl[24];
        if (!full && k > 12 && k != T - 1 && k % (T > 100 ? 16 : 8) != 0) continue;
        printf("  %4d  ", k);
        {
            int w = 0, q;
            const nmz_vocab *v = nmz_net_vocab(ms[0].net);
            cli_char_label(v->codepoints[ids0[t]], lbl, sizeof lbl);
            for (q = 0; lbl[q]; q++) w += ((unsigned char)lbl[q] & 0xC0) != 0x80;
            printf("%s%*s", lbl, w < 5 ? 5 - w : 0, "");
        }
        for (j = 0; j < ns; j++) {
            double r = rel(&se[j], t, T);
            char b[24];
            if (r >= 0.01 || r == 0)
                snprintf(b, sizeof b, "%.3f", r);
            else
                snprintf(b, sizeof b, "%.1e", r);
            printf(" %10s", b);
        }
        printf("\n");
    }
    if (!full && T > 14) printf("  (every %dth step after 12; --full prints all)\n", T > 100 ? 16 : 8);
    printf("\nthe same on a log scale, one column per step back (0 at the left; '%s' = 1, empty = below %.0e):\n",
           sym("█", "#"), lo);
    for (k = 0; k < ns; k++) {
        printf("  ");
        put_pad(se[k].label, 10, 0);
        putchar(' ');
        spark(&se[k], T, lo);
        printf("\n");
    }
    {
        int k0 = T - 1 > 32 ? 16 : (T - 1) / 2;
        printf("\nsummary\n");
        printf("  %-10s %12s %13s %13s %13s %13s\n", "", "x per step", "x/step k>=", "left at", "left at",
               "< 1% after");
        printf("  %-10s %12s %13d %13s %13s %13s\n", "", "(all)", k0, "k=10", "k=end", "");
        for (k = 0; k < ns; k++) {
            double left, per = seq_survival_rate(se[k].v, T, &left), far, a = rel(&se[k], T - 1 - k0, T);
            int u = steps_until(&se[k], T, 0.01);
            char b1[24], b2[24], b3[24], b4[24], b5[24];
            if (se[k].is_prod) {
                per = T > 1 && se[k].v[0] > 0 ? pow(se[k].v[0], 1.0 / (T - 1)) : NAN;
                left = se[k].v[0];
            }
            far = a > 0 && rel(&se[k], 0, T) > 0 && T - 1 > k0 ? pow(rel(&se[k], 0, T) / a, 1.0 / (T - 1 - k0)) : NAN;
            snprintf(b1, sizeof b1, "x%.3f", per);
            snprintf(b5, sizeof b5, "x%.3f", far);
            snprintf(b2, sizeof b2, "%.2g", T > 10 ? rel(&se[k], T - 1 - 10, T) : NAN);
            snprintf(b3, sizeof b3, "%.2g", left);
            if (u < 0)
                snprintf(b4, sizeof b4, "never (>%d)", T - 1);
            else
                snprintf(b4, sizeof b4, "%d steps", u);
            printf("  ");
            put_pad(se[k].label, 10, 0);
            printf(" %12s %13s %13s %13s %13s\n", b1, b5, b2, b3, b4);
        }
        for (i = 0; i < nm; i++)
            if (ms[i].sv.n_keep)
                printf("  %c: %d of %d %ss keep at least half along the %s path over all %d steps (%s >= 0.5)\n",
                       ms[i].letter, ms[i].sv.n_keep[0], ms[i].sv.units, ms[i].sv.c ? "cell" : "unit",
                       ms[i].sv.c ? "cell-state" : "direct", T - 1,
                       ms[i].sv.c ? sym("Π f", "prod f") : sym("Π z", "prod z"));
    }
    printf("\n  (\"x per step\": the geometric mean of the per-step factor, over all steps and from the\n"
           "  k in the header on; \"left at k\": the relative size k steps back. A vanilla RNN multiplies the\n"
           "  gradient by W_hh%s and tanh' at every step, so it shrinks geometrically; an LSTM's cell state only\n"
           "  multiplies it by f, and a GRU's direct path by z: units whose gate stays near 1 carry it far back.)\n",
           sym("ᵀ", "^T"));
    for (i = 0; i < nm; i++)
        if (!strcmp(ms[i].sv.kind, "gpt")) {
            printf("  (A Transformer has no per-step factor: the last position's query reaches position t in one\n"
                   "  attention step, so what arrives at t is set by how strongly the layers attend there, not by\n"
                   "  the distance. The \"x per step\" columns are only a summary of that profile.)\n");
            break;
        }
    rc = 0;
out:
    for (i = 0; i < 6; i++) {
        seq_survival_free(&ms[i].sv);
        nmz_net_free(ms[i].net);
    }
    free(ids0);
    free(ts.owned);
    return rc;
}

/* ------------------------------------------------------- show attention */

typedef struct attn_cb {
    attn_pats *ap;
    const int *ids; /* the current window */
    const nmz_vocab *v;
    int want_layer, want_head, grabbed, oom;
    nmz_tensor w;   /* the first window's weights of the chosen layer */
} attn_cb;

static void attn_trace_cb(const nmz_trace_event *ev, void *user) {
    attn_cb *c = user;
    if (ev->phase != NMZ_TRACE_FORWARD || strcmp(ev->layer_type, "attention")) return;
    if (attn_pats_add(c->ap, ev, c->ids, c->v)) c->oom = 1;
    if (!c->grabbed && ev->layer == c->want_layer) {
        const nmz_tensor *w = nmz_trace_find(ev, "weights");
        if (w && nmz_tensor_copy(&c->w, w) != NMZ_OK) c->oom = 1;
        c->grabbed = 1;
    }
}

int show_attention(cli_state *st, int argc, char **argv) {
    text_src ts;
    attn_cb cb;
    nmz_tensor x = NMZ_TENSOR_EMPTY;
    const nmz_vocab *v;
    int i, n = 0, *ids = NULL, rc = 1, ctx, W, s0, full = 0, H = 0, from;
    long layer_n = 1, head_n = 1;
    double t0 = cli_now();
    memset(&ts, 0, sizeof ts);
    memset(&cb, 0, sizeof cb);
    if (!cli_need_net(st)) return 1;
    v = nmz_net_vocab(st->net);
    if (!v || !net_has_attention(st->net))
        return cli_fail("show attention needs a Transformer text model (e.g. new --preset char-gpt)%s", NULL);
    for (i = 2; i < argc; i++) {
        int r = text_src_option(argc, argv, &i, &ts);
        if (r < 0) return 1;
        if (r) continue;
        if (!strcmp(argv[i], "--full")) {
            full = 1;
        } else if (!strcmp(argv[i], "--layer") || !strcmp(argv[i], "--head")) {
            const char *val = cli_opt_value(argc, argv, &i);
            long *dst = !strcmp(argv[i - 1], "--layer") ? &layer_n : &head_n;
            if (!val || !cli_parse_int(val, 1, 100000, dst))
                return val ? cli_fail("--layer/--head need a positive integer (1-based), got '%s'", val) : 1;
        } else {
            return cli_fail("show attention takes --text T | --data FILE [--from K] [--chars N], --layer N, --head H, "
                            "--full; got '%s'", argv[i]);
        }
    }
    cb.want_layer = seq_attention_layer(st->net, (int)layer_n);
    if (cb.want_layer < 0) return cli_fail("--layer: the model has fewer attention layers (blocks)%s", NULL);
    if (text_src_load(&ts, 4000)) goto out;
    if (!(ids = encode(v, ts.s, ts.len, &n, "the model"))) goto out;
    if (n < 3) {
        cli_fail("the text needs at least 3 characters%s", NULL);
        goto out;
    }
    if (!(cb.ap = attn_pats_new())) goto oom;
    cb.v = v;
    ctx = nmz_net_context_length(st->net);
    W = ctx > 0 ? ctx : 256;
    nmz_net_set_carry(st->net, 0);
    nmz_net_set_trace(st->net, attn_trace_cb, &cb);
    for (s0 = 0; s0 < n; s0 += W) { /* windows of the context length, each from position 0 */
        int len = n - s0 < W ? n - s0 : W, shape[2];
        if (len < 3) break;
        shape[0] = 1;
        shape[1] = len;
        if (nmz_tensor_reshape(&x, 2, shape) != NMZ_OK) goto oom;
        for (i = 0; i < len; i++) x.data[i] = (float)ids[s0 + i];
        cb.ids = ids + s0;
        if (nmz_net_forward(st->net, &x, NULL) != NMZ_OK) {
            nmz_net_set_trace(st->net, NULL, NULL);
            cli_lib_fail("show attention");
            goto out;
        }
        if (cb.oom) goto oom;
    }
    nmz_net_set_trace(st->net, NULL, NULL);
    H = cb.w.rank == 4 ? cb.w.shape[1] : 0;
    if (head_n > H) {
        fprintf(stderr, "error: --head %ld: layer %ld has %d heads\n", head_n, layer_n, H);
        goto out;
    }
    printf("Attention over %d characters%s%s (%d window%s of up to %d positions, evaluation mode, %.1fs)\n", n,
           ts.data ? " of " : "", ts.data ? ts.data : "", (n + W - 1) / W, (n + W - 1) / W == 1 ? "" : "s", W,
           cli_now() - t0);
    printf("\nHEATMAP   layer %ld, head %ld, the first window\n", layer_n, head_n);
    {
        int T = cb.w.shape[2];
        from = !full && T > 64 ? 0 : 0;
        attn_heatmap(&cb.w, 0, (int)head_n - 1, ids, v, from, !full && T > 64 ? 64 : T);
        if (!full && T > 64) printf("  (the first 64 of %d positions; --full shows all)\n", T);
    }
    printf("\nWHAT EACH HEAD ATTENDS TO   measured over the whole text\n");
    attn_pats_print(cb.ap, st->net);
    rc = 0;
    goto out;
oom:
    nmz_net_set_trace(st->net, NULL, NULL);
    cli_fail("out of memory%s", NULL);
out:
    attn_pats_free(cb.ap);
    nmz_tensor_free(&cb.w);
    nmz_tensor_free(&x);
    free(ids);
    free(ts.owned);
    return rc;
}

/* ---------------------------------------------------------- show params */

static const char *component(const nmz_layer *l, const char *pname) {
    const char *t = nmz_layer_type(l);
    if (!strcmp(t, "embedding")) return "embedding";
    if (!strcmp(t, "posenc")) return "positions";
    if (!strcmp(t, "layernorm") || !strncmp(pname, "ln", 2)) return "layernorm";
    if (!strcmp(t, "attention") || !strncmp(pname, "attn.", 5)) return "attention";
    if (!strcmp(t, "ffn") || !strncmp(pname, "ffn.", 4)) return "feed-forward";
    if (!strcmp(t, "dense") || !strcmp(t, "tied_output")) return "output";
    if (nmz_layer_is_recurrent(l)) return "recurrent";
    if (!strcmp(t, "conv2d")) return "convolution";
    return "other";
}

int show_params(cli_state *st) {
    static const char *const comps[] = {"embedding", "positions", "attention", "feed-forward", "layernorm", "output",
                                        "recurrent", "convolution", "other"};
    size_t total = nmz_net_param_count(st->net), sums[9] = {0};
    const nmz_train_opts *o = nmz_net_train_config(st->net);
    int i, j, c, tied = -1;
    printf("Parameters of %s: %zu in total\n\n", st->path ? st->path : "the network", total);
    printf("  %-5s %-12s %-12s %-14s %10s %7s  %s\n", "layer", "type", "name", "shape", "count", "share",
           "AdamW decay");
    for (i = 0; i < nmz_net_layer_count(st->net); i++) {
        nmz_layer *l = nmz_net_layer(st->net, i);
        if (!strcmp(nmz_layer_type(l), "tied_output")) tied = i;
        for (j = 0; j < nmz_layer_param_count(l); j++) {
            const nmz_tensor *t = nmz_layer_param_value(l, j);
            const char *name = nmz_layer_param_name(l, j), *comp = component(l, name);
            char shape[48];
            int d, w = 0;
            for (d = 0; d < t->rank; d++) w += snprintf(shape + w, sizeof shape - (size_t)w, d ? "x%d" : "%d", t->shape[d]);
            for (c = 0; c < 9; c++)
                if (!strcmp(comps[c], comp)) sums[c] += t->size;
            printf("  %-5d %-12s %-12s %-14s %10zu %6.1f%%  %s\n", i, nmz_layer_type(l), name, shape, t->size,
                   100.0 * (double)t->size / (double)(total ? total : 1),
                   nmz_layer_param_decays(l, j, o->decay_all) ? "yes" : "no");
        }
    }
    printf("\n  by component:\n");
    for (c = 0; c < 9; c++)
        if (sums[c]) {
            int bar = (int)(40.0 * (double)sums[c] / (double)(total ? total : 1) + 0.5), b;
            printf("  %-13s %10zu %6.1f%%  ", comps[c], sums[c], 100.0 * (double)sums[c] / (double)(total ? total : 1));
            for (b = 0; b < bar; b++) fputs(sym("█", "#"), stdout);
            printf("\n");
        }
    if (tied >= 0)
        printf("\n  The output (layer %d) reuses the embedding matrix: its %d x %d weights are counted once, under\n"
               "  embedding; without tying the output would add another %zu.\n",
               tied, nmz_layer_param_value(nmz_net_layer(st->net, 0), 0)->shape[0],
               nmz_layer_param_value(nmz_net_layer(st->net, 0), 0)->shape[1],
               nmz_layer_param_value(nmz_net_layer(st->net, 0), 0)->size);
    if (nmz_net_context_length(st->net) > 0)
        printf("  Context window: %d positions. \"AdamW decay\" = whether weight decay applies (%s).\n",
               nmz_net_context_length(st->net), o->decay_all ? "--decay-all: every parameter" : "weight matrices only");
    return 0;
}
