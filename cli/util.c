/* util.c — helpers shared by the CLI commands: errors, number parsing,
 * training options, and dataset resolution (CSV, IDX, "mnist"). */
#include "cli.h"

#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

int cli_fail(const char *fmt, const char *arg) {
    fputs("error: ", stderr);
    fprintf(stderr, fmt, arg ? arg : "");
    fputc('\n', stderr);
    return 1;
}

int cli_lib_fail(const char *what) {
    fprintf(stderr, "error: %s: %s\n", what, nmz_last_error());
    return 1;
}

int cli_need_net(cli_state *st) {
    if (st->net) return 1;
    fprintf(stderr, "error: no network loaded. Create one with `new 2 4 1` (or `new --preset mnist-cnn`) "
                    "or open one with `load FILE`.\n");
    return 0;
}

int cli_parse_int(const char *s, long lo, long hi, long *out) {
    char *end;
    long v;
    if (!s || !*s) return 0;
    errno = 0;
    v = strtol(s, &end, 10);
    if (*end || errno || v < lo || v > hi) return 0;
    *out = v;
    return 1;
}

int cli_parse_u64(const char *s, uint64_t *out) {
    char *end;
    unsigned long long v;
    if (!s || !*s || *s == '-') return 0;
    errno = 0;
    v = strtoull(s, &end, 10);
    if (*end || errno) return 0;
    *out = (uint64_t)v;
    return 1;
}

int cli_parse_double(const char *s, double *out) {
    char *end;
    double v;
    if (!s || !*s) return 0;
    errno = 0;
    v = strtod(s, &end);
    if (*end || errno == ERANGE || !isfinite(v)) return 0;
    *out = v;
    return 1;
}

int cli_is_option(const char *s) { return s[0] == '-' && s[1] == '-'; }

const char *cli_opt_value(int argc, char **argv, int *i) {
    if (*i + 1 >= argc) {
        cli_fail("option %s needs a value", argv[*i]);
        return NULL;
    }
    return argv[++*i];
}

char *cli_strdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *d = malloc(n);
    if (d) memcpy(d, s, n);
    return d;
}

void cli_set_path(cli_state *st, const char *path) {
    char *p = path ? cli_strdup(path) : NULL;
    free(st->path);
    st->path = p;
}

void cli_replace_net(cli_state *st, nmz_net *net, const char *path) {
    if (st->net && st->dirty && st->interactive)
        printf("(the previous network had unsaved changes; it was discarded)\n");
    nmz_net_free(st->net);
    st->net = net;
    cli_set_path(st, path);
    st->dirty = path == NULL;
}

void cli_print_train_config(const nmz_train_opts *o) {
    if (o->optimizer == NMZ_OPT_ADAM)
        printf("adam lr %g (beta1 %g, beta2 %g, eps %g)", (double)o->learning_rate, (double)o->beta1,
               (double)o->beta2, (double)o->epsilon);
    else
        printf("sgd lr %g, momentum %g", (double)o->learning_rate, (double)o->momentum);
    if (o->schedule == NMZ_SCHED_STEP)
        printf(", step schedule (x%g every %d epochs)", (double)o->gamma, o->step_size);
    else if (o->schedule == NMZ_SCHED_COSINE)
        printf(", cosine schedule to %g", (double)o->min_lr);
    printf(", batch %d, max epochs %d, target loss %g, shuffle %s\n", o->batch_size, o->max_epochs, o->target_loss,
           o->shuffle ? "on" : "off");
}

/* returns 1 if consumed, 0 if not a training option, -1 on a bad value */
int cli_train_option(int argc, char **argv, int *i, nmz_train_opts *o, unsigned *mask) {
    const char *a = argv[*i], *v;
    double d;
    long n;
    static const char *const valued[] = {"--lr",    "--momentum",  "--batch", "--epochs", "--target-loss",
                                         "--optimizer", "--beta1", "--beta2", "--eps",    "--schedule",
                                         "--step-size", "--gamma", "--min-lr", NULL};
    int k;
    if (!strcmp(a, "--shuffle") || !strcmp(a, "--no-shuffle")) {
        o->shuffle = !strcmp(a, "--shuffle");
        *mask |= CLI_SET_SHUFFLE;
        return 1;
    }
    for (k = 0; valued[k] && strcmp(a, valued[k]); k++) {
    }
    if (!valued[k]) return 0;
    if (!(v = cli_opt_value(argc, argv, i))) return -1;
    switch (k) {
    case 0:
        if (!cli_parse_double(v, &d) || d <= 0) return -cli_fail("--lr needs a positive number, got '%s'", v);
        o->learning_rate = (float)d;
        *mask |= CLI_SET_LR;
        break;
    case 1:
        if (!cli_parse_double(v, &d) || d < 0 || d >= 1) return -cli_fail("--momentum must be in [0, 1), got '%s'", v);
        o->momentum = (float)d;
        *mask |= CLI_SET_MOMENTUM;
        break;
    case 2:
        if (!cli_parse_int(v, 1, 1000000000L, &n)) return -cli_fail("--batch needs a positive integer, got '%s'", v);
        o->batch_size = (int)n;
        *mask |= CLI_SET_BATCH;
        break;
    case 3:
        if (!cli_parse_int(v, 1, 1000000000L, &n)) return -cli_fail("--epochs needs a positive integer, got '%s'", v);
        o->max_epochs = (int)n;
        *mask |= CLI_SET_EPOCHS;
        break;
    case 4:
        if (!cli_parse_double(v, &d) || d < 0) return -cli_fail("--target-loss needs a number >= 0, got '%s'", v);
        o->target_loss = d;
        *mask |= CLI_SET_TARGET;
        break;
    case 5:
        if (nmz_optimizer_parse(v, &o->optimizer) != NMZ_OK) return -cli_lib_fail("--optimizer");
        *mask |= CLI_SET_OPTIMIZER;
        break;
    case 6:
    case 7:
        if (!cli_parse_double(v, &d) || d < 0 || d >= 1) return -cli_fail("Adam betas must be in [0, 1), got '%s'", v);
        if (k == 6) o->beta1 = (float)d;
        else o->beta2 = (float)d;
        *mask |= CLI_SET_BETAS;
        break;
    case 8:
        if (!cli_parse_double(v, &d) || d <= 0) return -cli_fail("--eps needs a positive number, got '%s'", v);
        o->epsilon = (float)d;
        *mask |= CLI_SET_BETAS;
        break;
    case 9:
        if (nmz_schedule_parse(v, &o->schedule) != NMZ_OK) return -cli_lib_fail("--schedule");
        *mask |= CLI_SET_SCHEDULE;
        break;
    case 10:
        if (!cli_parse_int(v, 1, 1000000000L, &n)) return -cli_fail("--step-size needs a positive integer, got '%s'", v);
        o->step_size = (int)n;
        *mask |= CLI_SET_SCHEDULE;
        break;
    case 11:
        if (!cli_parse_double(v, &d) || d <= 0 || d > 1) return -cli_fail("--gamma must be in (0, 1], got '%s'", v);
        o->gamma = (float)d;
        *mask |= CLI_SET_SCHEDULE;
        break;
    default:
        if (!cli_parse_double(v, &d) || d < 0) return -cli_fail("--min-lr needs a number >= 0, got '%s'", v);
        o->min_lr = (float)d;
        *mask |= CLI_SET_SCHEDULE;
        break;
    }
    return 1;
}

/* ------------------------------------------------------------------ data */

static int ends_with(const char *s, const char *suffix) {
    size_t a = strlen(s), b = strlen(suffix);
    return a >= b && !strcmp(s + a - b, suffix);
}

static int file_exists(const char *p) {
    FILE *f = fopen(p, "rb");
    if (f) fclose(f);
    return f != NULL;
}

/* "mnist", "mnist:train", "mnist:test", an IDX image file (labels found by
 * name), or a CSV file. */
int cli_load_data(cli_state *st, const char *spec, int want_test, int limit, int header, nmz_dataset **out) {
    nmz_status s;
    *out = NULL;
    if (!strcmp(spec, "mnist") || !strcmp(spec, "mnist:train") || !strcmp(spec, "mnist:test")) {
        const char *dir = getenv("NEUROMZ_DATA");
        int test = !strcmp(spec, "mnist:test") || (!strcmp(spec, "mnist") && want_test);
        char img[1024], lab[1024];
        if (!dir || !*dir) dir = "data";
        snprintf(img, sizeof img, "%s/%s-images-idx3-ubyte", dir, test ? "t10k" : "train");
        snprintf(lab, sizeof lab, "%s/%s-labels-idx1-ubyte", dir, test ? "t10k" : "train");
        if (!file_exists(img) || !file_exists(lab)) {
            fprintf(stderr, "error: MNIST not found in %s/ — run `make data` (downloads ~11 MB, verifies checksums)\n", dir);
            return 1;
        }
        s = nmz_dataset_load_idx(out, img, lab, limit);
    } else if (ends_with(spec, "idx3-ubyte")) {
        char *lab = cli_strdup(spec), *p;
        if (!lab) return cli_fail("out of memory%s", NULL);
        p = strstr(lab, "images-idx3-ubyte");
        if (!p) {
            free(lab);
            return cli_fail("cannot guess the labels file for '%s' (expected a name like train-images-idx3-ubyte)", spec);
        }
        memcpy(p, "labels-idx1-ubyte", 17);
        s = nmz_dataset_load_idx(out, spec, lab, limit);
        free(lab);
    } else {
        nmz_csv_opts co = nmz_csv_defaults();
        co.n_inputs = st->net ? nmz_net_input_size(st->net) : 0;
        co.header = header;
        s = nmz_dataset_load_csv(out, spec, &co);
        if (s == NMZ_OK && limit > 0) nmz_dataset_truncate(*out, limit);
    }
    if (s != NMZ_OK) return cli_lib_fail("cannot read data");
    if (st->net && ((*out)->n_inputs != nmz_net_input_size(st->net) ||
                    (*out)->n_outputs != nmz_net_output_size(st->net))) {
        fprintf(stderr, "error: %s has %d input(s) and %d target value(s) per sample%s, but the network has %d input(s) "
                        "and %d output(s)\n",
                spec, (*out)->n_inputs, (*out)->n_outputs, (*out)->n_classes ? " (one-hot classes)" : "",
                nmz_net_input_size(st->net), nmz_net_output_size(st->net));
        nmz_dataset_free(*out);
        *out = NULL;
        return 1;
    }
    return 0;
}

const char *cli_act_reason(nmz_act a) {
    switch (a) {
    case NMZ_ACT_LINEAR: return "no squashing: outputs any real number (regression)";
    case NMZ_ACT_SIGMOID: return "squashes into (0,1)";
    case NMZ_ACT_TANH: return "squashes into (-1,1), zero-centred";
    case NMZ_ACT_RELU: return "max(0,z): cheap and does not saturate for z > 0 (units can die)";
    case NMZ_ACT_LEAKY_RELU: return "like relu with a small slope for z < 0, so units cannot die";
    case NMZ_ACT_SOFTMAX: return "probabilities over classes that sum to 1";
    case NMZ_ACT_COUNT: break;
    }
    return "";
}

const char *cli_init_reason(int init) {
    if (init == NMZ_INIT_HE) return "He: Var(W) = 2/fan_in, makes up for relu zeroing half of the signal";
    return "Glorot/Xavier: Var(W) = 2/(fan_in+fan_out), keeps tanh/sigmoid layers in their useful range";
}
