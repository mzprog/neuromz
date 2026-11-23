/* commands.c — the neuromz commands. Every command validates its input and
 * works (or fails with a message) whether or not a network is loaded. */
#include "cli.h"

#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "neuromz_layer.h" /* nmz_param for `show weights` */

#define DEFAULT_SEED 42u

/* ------------------------------------------------------------- utilities */

static int fail(const char *fmt, const char *arg) {
    fputs("error: ", stderr);
    fprintf(stderr, fmt, arg ? arg : "");
    fputc('\n', stderr);
    return 1;
}

static int lib_fail(const char *what) {
    fprintf(stderr, "error: %s: %s\n", what, nmz_last_error());
    return 1;
}

static int need_net(cli_state *st) {
    if (st->net) return 1;
    fprintf(stderr, "error: no network loaded. Create one with `new 2 4 1` or open one with `load FILE`.\n");
    return 0;
}

static int parse_int(const char *s, long lo, long hi, long *out) {
    char *end;
    long v;
    if (!s || !*s) return 0;
    errno = 0;
    v = strtol(s, &end, 10);
    if (*end || errno || v < lo || v > hi) return 0;
    *out = v;
    return 1;
}

static int parse_u64(const char *s, uint64_t *out) {
    char *end;
    unsigned long long v;
    if (!s || !*s || *s == '-') return 0;
    errno = 0;
    v = strtoull(s, &end, 10);
    if (*end || errno) return 0;
    *out = (uint64_t)v;
    return 1;
}

static int parse_double(const char *s, double *out) {
    char *end;
    double v;
    if (!s || !*s) return 0;
    errno = 0;
    v = strtod(s, &end);
    if (*end || errno == ERANGE || !isfinite(v)) return 0;
    *out = v;
    return 1;
}

static int is_option(const char *s) { return s[0] == '-' && s[1] == '-'; }

/* fetch the value of option argv[*i] */
static const char *opt_value(int argc, char **argv, int *i) {
    if (*i + 1 >= argc) {
        fail("option %s needs a value", argv[*i]);
        return NULL;
    }
    return argv[++*i];
}

static char *xstrdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *d = malloc(n);
    if (d) memcpy(d, s, n);
    return d;
}

static void set_path(cli_state *st, const char *path) {
    char *p = path ? xstrdup(path) : NULL;
    free(st->path);
    st->path = p;
}

void cli_state_free(cli_state *st) {
    nmz_net_free(st->net);
    st->net = NULL;
    set_path(st, NULL);
}

static void replace_net(cli_state *st, nmz_net *net, const char *path) {
    if (st->net && st->dirty && st->interactive)
        printf("(the previous network had unsaved changes; it was discarded)\n");
    nmz_net_free(st->net);
    st->net = net;
    set_path(st, path);
    st->dirty = path == NULL;
}

static const char *act_reason(nmz_act a) {
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

static const char *init_reason(int init) {
    if (init == NMZ_INIT_HE) return "He: Var(W) = 2/fan_in, makes up for relu zeroing half of the signal";
    return "Glorot/Xavier: Var(W) = 2/(fan_in+fan_out), keeps tanh/sigmoid layers in their useful range";
}

static void print_train_config(const nmz_train_opts *o) {
    printf("lr %g, momentum %g, batch %d, max epochs %d, target loss %g, shuffle %s\n", (double)o->learning_rate,
           (double)o->momentum, o->batch_size, o->max_epochs, o->target_loss, o->shuffle ? "on" : "off");
}

/* Parse --lr/--momentum/--batch/--epochs/--target-loss/--shuffle/--no-shuffle.
 * Returns 1 if argv[*i] was one of them, -1 on a bad value, 0 otherwise. */
static int train_option(int argc, char **argv, int *i, nmz_train_opts *o, unsigned *set_mask) {
    const char *a = argv[*i], *v;
    double d;
    long n;
    if (!strcmp(a, "--shuffle") || !strcmp(a, "--no-shuffle")) {
        o->shuffle = !strcmp(a, "--shuffle");
        *set_mask |= 32;
        return 1;
    }
    if (strcmp(a, "--lr") && strcmp(a, "--momentum") && strcmp(a, "--batch") && strcmp(a, "--epochs") &&
        strcmp(a, "--target-loss"))
        return 0;
    if (!(v = opt_value(argc, argv, i))) return -1;
    if (!strcmp(a, "--lr")) {
        if (!parse_double(v, &d) || d <= 0) return -fail("--lr needs a positive number, got '%s'", v);
        o->learning_rate = (float)d;
        *set_mask |= 1;
    } else if (!strcmp(a, "--momentum")) {
        if (!parse_double(v, &d) || d < 0 || d >= 1) return -fail("--momentum must be in [0, 1), got '%s'", v);
        o->momentum = (float)d;
        *set_mask |= 2;
    } else if (!strcmp(a, "--batch")) {
        if (!parse_int(v, 1, 1000000000L, &n)) return -fail("--batch needs a positive integer, got '%s'", v);
        o->batch_size = (int)n;
        *set_mask |= 4;
    } else if (!strcmp(a, "--epochs")) {
        if (!parse_int(v, 1, 1000000000L, &n)) return -fail("--epochs needs a positive integer, got '%s'", v);
        o->max_epochs = (int)n;
        *set_mask |= 8;
    } else {
        if (!parse_double(v, &d) || d < 0) return -fail("--target-loss needs a number >= 0, got '%s'", v);
        o->target_loss = d;
        *set_mask |= 16;
    }
    return 1;
}

/* ------------------------------------------------------------------- new */

typedef struct layer_spec {
    int units;
    nmz_act act;
    int act_set;
    float alpha;
    nmz_init init;
    int init_set;
} layer_spec;

/* SIZE[:ACT[:INIT]], ACT may be leaky_relu=ALPHA */
static int parse_spec(const char *s, layer_spec *sp) {
    char *copy = xstrdup(s), *p1, *p2, *eq;
    long n;
    int ok = 0;
    memset(sp, 0, sizeof *sp);
    if (!copy) return 0;
    p1 = strchr(copy, ':');
    if (p1) *p1++ = '\0';
    p2 = p1 ? strchr(p1, ':') : NULL;
    if (p2) *p2++ = '\0';
    if (!parse_int(copy, 1, 1000000, &n)) {
        fail("layer size must be a positive integer, got '%s'", copy);
        goto out;
    }
    sp->units = (int)n;
    if (p1 && *p1) {
        double alpha;
        eq = strchr(p1, '=');
        if (eq) *eq++ = '\0';
        if (nmz_act_parse(p1, &sp->act) != NMZ_OK) {
            lib_fail("layer spec");
            goto out;
        }
        if (eq) {
            if (sp->act != NMZ_ACT_LEAKY_RELU || !parse_double(eq, &alpha) || alpha <= 0 || alpha >= 1) {
                fail("only leaky_relu takes a parameter, in (0,1): '%s'", s);
                goto out;
            }
            sp->alpha = (float)alpha;
        }
        sp->act_set = 1;
    }
    if (p2 && *p2) {
        if (nmz_init_parse(p2, &sp->init) != NMZ_OK) {
            lib_fail("layer spec");
            goto out;
        }
        sp->init_set = sp->init != NMZ_INIT_AUTO;
    }
    ok = 1;
out:
    free(copy);
    return ok;
}

static void explain(const char *what, const char *value, int chosen, const char *why) {
    printf("  %-16s %-14s %-9s %s\n", what, value, chosen ? "[yours]" : "[default]", why);
}

/* For a value the user set: show it next to what quick mode would have used. */
static const char *yours(char *buf, size_t n, int chosen, const char *dflt, const char *why) {
    if (!chosen) return why;
    snprintf(buf, n, "(quick mode would use %s)", dflt);
    return buf;
}

static int cmd_new(cli_state *st, int argc, char **argv) {
    layer_spec *specs = NULL;
    int n_specs = 0, i, inputs = 0, loss_set = 0, rc = 1, n_hidden;
    long n;
    uint64_t seed = DEFAULT_SEED;
    int seed_set = 0;
    unsigned mask = 0;
    nmz_loss loss = NMZ_LOSS_MSE;
    nmz_quick q, qt;
    nmz_train_opts o;
    nmz_net *net = NULL;
    int *sizes = NULL;
    char buf[64];

    specs = calloc((size_t)argc, sizeof *specs);
    sizes = calloc((size_t)argc, sizeof *sizes);
    if (!specs || !sizes) {
        fail("out of memory%s", NULL);
        goto out;
    }
    memset(&o, 0, sizeof o);
    for (i = 1; i < argc; i++) {
        const char *a = argv[i];
        int r;
        if (is_option(a)) {
            r = train_option(argc, argv, &i, &o, &mask);
            if (r < 0) goto out;
            if (r) continue;
            if (!strcmp(a, "--loss")) {
                const char *v = opt_value(argc, argv, &i);
                if (!v || nmz_loss_parse(v, &loss) != NMZ_OK) {
                    if (v) lib_fail("--loss");
                    goto out;
                }
                loss_set = 1;
            } else if (!strcmp(a, "--seed")) {
                const char *v = opt_value(argc, argv, &i);
                if (!v || !parse_u64(v, &seed)) {
                    if (v) fail("--seed needs a non-negative integer, got '%s'", v);
                    goto out;
                }
                seed_set = 1;
            } else {
                fail("unknown option '%s' for new (see `help new`)", a);
                goto out;
            }
        } else if (!inputs) {
            if (!parse_int(a, 1, 1000000, &n)) {
                fail("the first number is the input size and must be a positive integer, got '%s'", a);
                goto out;
            }
            inputs = (int)n;
        } else {
            if (!parse_spec(a, &specs[n_specs])) goto out;
            n_specs++;
        }
    }
    if (!inputs || n_specs == 0) {
        fail("usage: new INPUTS [HIDDEN...] OUTPUTS [options]   e.g. `new 2 4 1` (see `help new`)%s", NULL);
        goto out;
    }
    sizes[0] = inputs;
    for (i = 0; i < n_specs; i++) sizes[i + 1] = specs[i].units;
    if (nmz_quick_defaults(n_specs + 1, sizes, &q) != NMZ_OK) {
        lib_fail("new");
        goto out;
    }
    n_hidden = n_specs - 1;
    for (i = 0; i < n_specs; i++)
        if (!specs[i].act_set) specs[i].act = i < n_hidden ? q.hidden_act : q.output_act;
    if (!loss_set) loss = specs[n_specs - 1].act == NMZ_ACT_SOFTMAX ? NMZ_LOSS_CROSS_ENTROPY : NMZ_LOSS_MSE;
    nmz_quick_train_defaults(loss, &qt);
    {
        nmz_train_opts user = o;
        o = qt.train;
        if (mask & 1) o.learning_rate = user.learning_rate;
        if (mask & 2) o.momentum = user.momentum;
        if (mask & 4) o.batch_size = user.batch_size;
        if (mask & 8) o.max_epochs = user.max_epochs;
        if (mask & 16) o.target_loss = user.target_loss;
        if (mask & 32) o.shuffle = user.shuffle;
    }

    if (nmz_net_create(&net, 1, &inputs, seed) != NMZ_OK) {
        lib_fail("new");
        goto out;
    }
    for (i = 0; i < n_specs; i++) {
        if (nmz_net_add(net, nmz_layer_dense(specs[i].units, specs[i].init)) != NMZ_OK ||
            (specs[i].act != NMZ_ACT_LINEAR &&
             nmz_net_add(net, nmz_layer_activation(specs[i].act, specs[i].alpha)) != NMZ_OK)) {
            lib_fail("new");
            goto out;
        }
    }
    if (nmz_net_set_loss(net, loss) != NMZ_OK || nmz_net_build(net) != NMZ_OK) {
        lib_fail("new");
        goto out;
    }
    *nmz_net_train_config(net) = o;

    /* explain every choice, one line each */
    printf("Created a network:");
    for (i = 0; i <= n_specs; i++) {
        if (i) printf(" %s", sym("→", "->"));
        printf(" %d", sizes[i]);
    }
    printf("   (%zu parameters)\n", nmz_net_param_count(net));
    {
        int li = 0;
        for (i = 0; i < n_specs; i++) {
            char what[48], val[48];
            nmz_layer *d = nmz_net_layer(net, li);
            int is_out = i == n_specs - 1;
            snprintf(what, sizeof what, "%s %d->%d", is_out ? "output" : "hidden", sizes[i], sizes[i + 1]);
            snprintf(val, sizeof val, "%s", nmz_act_name(specs[i].act));
            explain(what, val, specs[i].act_set,
                    specs[i].act_set ? act_reason(specs[i].act)
                                     : (is_out ? q.why_output_act : q.why_hidden_act));
            explain("  init", nmz_init_name((nmz_init)nmz_layer_dense_init(d)), specs[i].init_set,
                    init_reason(nmz_layer_dense_init(d)));
            li += specs[i].act == NMZ_ACT_LINEAR ? 1 : 2;
        }
    }
    explain("loss", nmz_loss_name(loss), loss_set,
            loss_set ? (loss == NMZ_LOSS_MSE ? "L = 1/2 sum (y-t)^2, averaged over the batch"
                                             : "L = -sum t*log(p), fused with softmax: delta = p - t")
                     : (loss == q.loss ? q.why_loss
                                       : (loss == NMZ_LOSS_MSE ? "the output is not softmax, so squared error"
                                                               : "softmax output, so cross-entropy")));
    {
        char d[32], why[96];
        snprintf(buf, sizeof buf, "%g", (double)o.learning_rate);
        snprintf(d, sizeof d, "%g", (double)qt.train.learning_rate);
        explain("learning rate", buf, (mask & 1) != 0, yours(why, sizeof why, mask & 1, d, qt.why_learning_rate));
        snprintf(buf, sizeof buf, "%g", (double)o.momentum);
        snprintf(d, sizeof d, "%g", (double)qt.train.momentum);
        explain("momentum", buf, (mask & 2) != 0, yours(why, sizeof why, mask & 2, d, qt.why_momentum));
        snprintf(buf, sizeof buf, "%d", o.batch_size);
        snprintf(d, sizeof d, "%d", qt.train.batch_size);
        explain("batch size", buf, (mask & 4) != 0, yours(why, sizeof why, mask & 4, d, qt.why_batch));
        snprintf(buf, sizeof buf, "%d", o.max_epochs);
        snprintf(d, sizeof d, "%d", qt.train.max_epochs);
        explain("max epochs", buf, (mask & 8) != 0, yours(why, sizeof why, mask & 8, d, qt.why_epochs));
        snprintf(buf, sizeof buf, "%g", o.target_loss);
        snprintf(d, sizeof d, "%g", qt.train.target_loss);
        explain("target loss", buf, (mask & 16) != 0, yours(why, sizeof why, mask & 16, d, qt.why_target_loss));
        explain("shuffle", o.shuffle ? "on" : "off", (mask & 32) != 0,
                yours(why, sizeof why, mask & 32, "on", "visit samples in a new random order every epoch"));
    }
    snprintf(buf, sizeof buf, "%llu", (unsigned long long)seed);
    explain("seed", buf, seed_set, "same seed => same initial weights and same shuffling, every run");
    replace_net(st, net, NULL);
    net = NULL;
    if (st->oneshot && !st->output) printf("(not saved: add -o FILE to keep it)\n");
    rc = 0;
out:
    nmz_net_free(net);
    free(specs);
    free(sizes);
    return rc;
}

/* ------------------------------------------------------------------ show */

static int cmd_show(cli_state *st, int argc, char **argv) {
    int i, j, weights = 0, n_cls;
    char desc[128];
    if (!need_net(st)) return 1;
    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "weights") || !strcmp(argv[i], "--weights"))
            weights = 1;
        else
            return fail("show takes no argument except `weights`, got '%s'", argv[i]);
    }
    printf("Network %s%s\n", st->path ? st->path : "(not saved yet)", st->dirty && st->path ? "  [unsaved changes]" : "");
    printf("  input        %d value(s) per sample\n", nmz_net_input_size(st->net));
    for (i = 0; i < nmz_net_layer_count(st->net); i++) {
        nmz_layer *l = nmz_net_layer(st->net, i);
        size_t count = 0;
        for (j = 0; j < nmz_layer_param_count(l); j++) count += nmz_layer_param_value(l, j)->size;
        nmz_layer_describe(l, desc, sizeof desc);
        if (count)
            printf("  [layer %d]  %-34s %zu params\n", i, desc, count);
        else
            printf("  [layer %d]  %s\n", i, desc);
    }
    printf("  loss         %s\n", nmz_loss_name(nmz_net_loss_kind(st->net)));
    printf("  parameters   %zu\n", nmz_net_param_count(st->net));
    printf("  steps        %ld backward pass(es) so far\n", nmz_net_step(st->net));
    printf("  training     ");
    print_train_config(nmz_net_train_config(st->net));
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
                snprintf(name, sizeof name, "[layer %d] %s", i, nmz_layer_param_name(l, j));
                print_matrix(name, nmz_layer_param_value(l, j), 0);
            }
        }
    }
    return 0;
}

/* ----------------------------------------------------------------- train */

typedef struct progress {
    int every;
    int max_epochs;
    int quiet;
} progress;

static void print_acc(double acc) {
    if (isnan(acc))
        printf("accuracy n/a");
    else
        printf("accuracy %5.1f%%", 100.0 * acc);
}

static int on_epoch(const nmz_epoch_info *info, void *user) {
    progress *p = user;
    if (p->quiet) return 0;
    if (info->epoch == 1 || info->epoch % p->every == 0) {
        printf("  epoch %6d   loss %-12.6g ", info->epoch, info->loss);
        print_acc(info->accuracy);
        printf("\n");
        fflush(stdout);
    }
    return 0;
}

static int load_csv_for_net(cli_state *st, const char *path, nmz_dataset **ds, int header) {
    nmz_csv_opts co = nmz_csv_defaults();
    co.n_inputs = nmz_net_input_size(st->net);
    co.header = header;
    if (nmz_dataset_load_csv(ds, path, &co) != NMZ_OK) return lib_fail("cannot read data");
    if ((*ds)->n_outputs != nmz_net_output_size(st->net)) {
        fprintf(stderr, "error: %s has %d target value(s) per row%s but the network has %d output(s)\n", path,
                (*ds)->n_outputs, (*ds)->n_classes ? " (one-hot classes)" : "", nmz_net_output_size(st->net));
        nmz_dataset_free(*ds);
        *ds = NULL;
        return 1;
    }
    return 0;
}

static int parse_header_opt(const char *a, int *header) {
    if (!strcmp(a, "--header")) *header = 1;
    else if (!strcmp(a, "--no-header")) *header = 0;
    else return 0;
    return 1;
}

static int cmd_train(cli_state *st, int argc, char **argv) {
    nmz_train_opts o;
    nmz_train_result res;
    nmz_dataset *ds = NULL;
    progress pr = {0, 0, 0};
    const char *path = NULL;
    unsigned mask = 0;
    int i, header = -1;
    long every = 0;
    if (!need_net(st)) return 1;
    o = *nmz_net_train_config(st->net);
    for (i = 1; i < argc; i++) {
        int r;
        if (is_option(argv[i])) {
            r = train_option(argc, argv, &i, &o, &mask);
            if (r < 0) return 1;
            if (r) continue;
            if (parse_header_opt(argv[i], &header)) continue;
            if (!strcmp(argv[i], "--quiet")) {
                pr.quiet = 1;
            } else if (!strcmp(argv[i], "--every")) {
                const char *v = opt_value(argc, argv, &i);
                if (!v || !parse_int(v, 1, 1000000000L, &every)) return v ? fail("--every needs a positive integer, got '%s'", v) : 1;
            } else {
                return fail("unknown option '%s' for train (see `help train`)", argv[i]);
            }
        } else if (!path) {
            path = argv[i];
        } else {
            return fail("train takes one CSV file; unexpected '%s'", argv[i]);
        }
    }
    if (!path) return fail("usage: train FILE.csv [options]   (see `help train`)%s", NULL);
    if (load_csv_for_net(st, path, &ds, header)) return 1;
    pr.max_epochs = o.max_epochs;
    pr.every = every ? (int)every : (o.max_epochs >= 20 ? o.max_epochs / 20 : 1);
    o.on_epoch = on_epoch;
    o.user = &pr;
    printf("Training on %s: %d samples, %d input(s), %d output(s)\n", path, ds->n_samples, ds->n_inputs,
           ds->n_outputs);
    printf("  ");
    print_train_config(&o);
    if (o.batch_size > ds->n_samples) printf("  (batch %d > %d samples: using the full dataset as one batch)\n", o.batch_size, ds->n_samples);
    if (nmz_net_train(st->net, ds, &o, &res) != NMZ_OK) {
        nmz_dataset_free(ds);
        st->dirty = 1;
        return lib_fail("training stopped");
    }
    st->dirty = 1;
    if (!pr.quiet && res.epochs % pr.every != 0 && res.epochs != 1) {
        printf("  epoch %6d   loss %-12.6g ", res.epochs, res.loss);
        print_acc(res.accuracy);
        printf("\n");
    }
    if (res.converged)
        printf("Converged after %d epoch(s): loss %.6g <= target %g, ", res.epochs, res.loss, o.target_loss);
    else
        printf("Stopped at the epoch limit (%d): loss %.6g > target %g, ", res.epochs, res.loss, o.target_loss);
    print_acc(res.accuracy);
    printf("\n");
    if (!res.converged)
        printf("  hint: try more epochs (--epochs), another learning rate (--lr), more hidden units, or a looser --target-loss\n");
    nmz_dataset_free(ds);
    return 0;
}

/* ------------------------------------------------------------ test/predict */

static int argmax(const float *v, int n) {
    int i, b = 0;
    for (i = 1; i < n; i++)
        if (v[i] > v[b]) b = i;
    return b;
}

static int cmd_test(cli_state *st, int argc, char **argv) {
    nmz_dataset *ds = NULL;
    const char *path = NULL;
    double loss, acc;
    int i, header = -1, n_out, n_cls;
    float *y = NULL;
    if (!need_net(st)) return 1;
    for (i = 1; i < argc; i++) {
        if (parse_header_opt(argv[i], &header)) continue;
        if (is_option(argv[i])) return fail("unknown option '%s' for test", argv[i]);
        if (path) return fail("test takes one CSV file; unexpected '%s'", argv[i]);
        path = argv[i];
    }
    if (!path) return fail("usage: test FILE.csv%s", NULL);
    if (load_csv_for_net(st, path, &ds, header)) return 1;
    if (nmz_net_evaluate(st->net, ds, &loss, &acc) != NMZ_OK) {
        nmz_dataset_free(ds);
        return lib_fail("test");
    }
    printf("Tested on %s: %d samples\n  loss %.6g   ", path, ds->n_samples, loss);
    print_acc(acc);
    printf("\n");
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
            for (c = 0; c < n_out; c++) printf(" %10.10s", n_cls == n_out ? nmz_net_class_name(st->net, c) : "");
            printf("\n");
            for (r = 0; r < n_out; r++) {
                char lbl[32];
                snprintf(lbl, sizeof lbl, "%d", r + 1);
                printf("  %14.14s", n_cls == n_out ? nmz_net_class_name(st->net, r) : lbl);
                for (c = 0; c < n_out; c++) printf(" %10d", cm[r * n_out + c]);
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
    while (*i < argc && !is_option(argv[*i]) && strcmp(argv[*i], "-t")) {
        if (k >= n) return fail("too many %s values", what);
        if (!parse_double(argv[*i], &d)) return fail("'%s' is not a number", argv[*i]);
        out[k++] = (float)d;
        (*i)++;
    }
    if (k != n) {
        fprintf(stderr, "error: expected %d %s value(s), got %d\n", n, what, k);
        return 1;
    }
    return 0;
}

static int cmd_predict(cli_state *st, int argc, char **argv) {
    int n_in, n_out, i = 1, k, n_cls;
    float *x, *y;
    int rc = 1;
    if (!need_net(st)) return 1;
    n_in = nmz_net_input_size(st->net);
    n_out = nmz_net_output_size(st->net);
    x = malloc(sizeof(float) * (size_t)n_in);
    y = malloc(sizeof(float) * (size_t)n_out);
    if (!x || !y) {
        fail("out of memory%s", NULL);
        goto out;
    }
    if (read_numbers(argc, argv, &i, x, n_in, "input")) goto out;
    if (i < argc) {
        fail("unexpected '%s' (predict takes only the input values)", argv[i]);
        goto out;
    }
    if (nmz_net_predict(st->net, x, y) != NMZ_OK) {
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
            printf("predicted class: %s (%.1f%%)\n", nmz_net_class_name(st->net, b), 100.0 * y[b]);
        else
            printf("predicted class: %d (%.1f%%)\n", b + 1, 100.0 * y[b]);
    }
    rc = 0;
out:
    free(x);
    free(y);
    return rc;
}

/* --------------------------------------------------------- trace / step */

typedef struct sample_args {
    float *x, *t;
    const char *json;
    int full;
    nmz_train_opts o;
    unsigned mask;
} sample_args;

/* x1 .. xn (--target|-t) t1 .. tm | CLASSNAME  [--json F] [--full] [--lr] [--momentum] */
static int parse_sample(cli_state *st, int argc, char **argv, sample_args *a, int allow_train_opts) {
    int n_in = nmz_net_input_size(st->net), n_out = nmz_net_output_size(st->net), i = 1, have_t = 0;
    memset(a, 0, sizeof *a);
    a->o = *nmz_net_train_config(st->net);
    a->x = malloc(sizeof(float) * (size_t)n_in);
    a->t = calloc((size_t)n_out, sizeof(float));
    if (!a->x || !a->t) return fail("out of memory%s", NULL);
    if (read_numbers(argc, argv, &i, a->x, n_in, "input")) return 1;
    while (i < argc) {
        const char *s = argv[i];
        if (!strcmp(s, "--target") || !strcmp(s, "-t")) {
            int c, n_cls = nmz_net_class_count(st->net);
            i++;
            /* a class name instead of numbers → one-hot */
            if (i < argc && n_cls == n_out && n_cls > 0) {
                for (c = 0; c < n_cls; c++)
                    if (!strcmp(argv[i], nmz_net_class_name(st->net, c))) break;
                if (c < n_cls) {
                    a->t[c] = 1.0f;
                    i++;
                    have_t = 1;
                    continue;
                }
            }
            if (read_numbers(argc, argv, &i, a->t, n_out, "target")) return 1;
            have_t = 1;
        } else if (!strcmp(s, "--json")) {
            if (!(a->json = opt_value(argc, argv, &i))) return 1;
            i++;
        } else if (!strcmp(s, "--full")) {
            a->full = 1;
            i++;
        } else if (allow_train_opts && (!strcmp(s, "--lr") || !strcmp(s, "--momentum"))) {
            if (train_option(argc, argv, &i, &a->o, &a->mask) < 0) return 1;
            i++;
        } else {
            return fail("unexpected '%s' (see `help trace`)", s);
        }
    }
    if (!have_t) {
        fprintf(stderr, "error: a target is needed to compute the loss and gradients: add --target with %d value(s)%s\n",
                n_out, nmz_net_class_count(st->net) == n_out ? " or a class name" : "");
        return 1;
    }
    return 0;
}

static int write_json(const nmz_trace_recorder *rec, const char *path) {
    FILE *f = fopen(path, "w");
    size_t i;
    if (!f) return fail("cannot write '%s'", path);
    for (i = 0; i < nmz_trace_recorder_count(rec); i++) nmz_trace_event_write_json(nmz_trace_recorder_get(rec, i), f);
    if (fclose(f) != 0) return fail("cannot write '%s'", path);
    printf("\n(%zu trace events written to %s, one JSON object per line)\n", nmz_trace_recorder_count(rec), path);
    return 0;
}

static int run_trace(cli_state *st, int argc, char **argv, int apply) {
    sample_args a;
    nmz_trace_recorder *rec = NULL;
    nmz_tensor x = NMZ_TENSOR_EMPTY, t = NMZ_TENSOR_EMPTY;
    int shape[2], rc = 1;
    double L;
    print_opts po;
    if (!need_net(st)) return 1;
    if (parse_sample(st, argc, argv, &a, 1)) goto out;
    shape[0] = 1;
    shape[1] = nmz_net_input_size(st->net);
    if (nmz_tensor_init(&x, 2, shape) != NMZ_OK) goto lib;
    memcpy(x.data, a.x, x.size * sizeof(float));
    shape[1] = nmz_net_output_size(st->net);
    if (nmz_tensor_init(&t, 2, shape) != NMZ_OK) goto lib;
    memcpy(t.data, a.t, t.size * sizeof(float));
    if (nmz_trace_recorder_create(&rec) != NMZ_OK) goto lib;
    nmz_net_set_trace(st->net, nmz_trace_recorder_fn, rec);
    if (nmz_net_forward(st->net, &x, NULL) != NMZ_OK || nmz_net_backward(st->net, &t, &L) != NMZ_OK) goto lib;
    if (apply ? nmz_net_update(st->net, a.o.learning_rate, a.o.momentum) != NMZ_OK
              : nmz_net_preview_update(st->net, a.o.learning_rate, a.o.momentum) != NMZ_OK)
        goto lib;
    nmz_net_set_trace(st->net, NULL, NULL);
    if (nmz_trace_recorder_status(rec) != NMZ_OK) {
        fail("out of memory while recording the trace%s", NULL);
        goto out;
    }
    po.max_rows = a.full ? 0 : 12;
    po.max_terms = a.full ? 0 : 6;
    printf("%s for one sample: every value of the forward and backward pass\n", apply ? "STEP" : "TRACE");
    print_trace_all(rec, &po, apply);
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
    free(a.x);
    free(a.t);
    return rc;
}

static int cmd_gradcheck(cli_state *st, int argc, char **argv) {
    sample_args a;
    nmz_tensor x = NMZ_TENSOR_EMPTY, t = NMZ_TENSOR_EMPTY;
    nmz_gradcheck_result r;
    int shape[2], rc = 1;
    if (!need_net(st)) return 1;
    if (parse_sample(st, argc, argv, &a, 0)) goto out;
    shape[0] = 1;
    shape[1] = nmz_net_input_size(st->net);
    if (nmz_tensor_init(&x, 2, shape) != NMZ_OK) goto out;
    memcpy(x.data, a.x, x.size * sizeof(float));
    shape[1] = nmz_net_output_size(st->net);
    if (nmz_tensor_init(&t, 2, shape) != NMZ_OK) goto out;
    memcpy(t.data, a.t, t.size * sizeof(float));
    if (nmz_net_gradcheck(st->net, &x, &t, 1e-3, 0, 0, &r) != NMZ_OK) {
        lib_fail("gradcheck");
        goto out;
    }
    printf("Numerical gradient check: %zu gradients compared with (L(w+e) - L(w-e)) / 2e, e = 0.001\n", r.checked);
    printf("  max |backprop - numeric| = %.3g\n", r.max_abs_err);
    printf("  largest mismatch: %s of %s %d, index %zu: backprop %.6g vs numeric %.6g\n", r.worst_param,
           r.worst_layer < 0 ? "input" : "layer", r.worst_layer < 0 ? 0 : r.worst_layer, r.worst_index,
           r.worst_analytic, r.worst_numeric);
    if (r.failures == 0)
        printf("PASS: backpropagation agrees with the numerical derivative\n");
    else
        printf("FAIL: %zu gradient(s) differ (a kink of relu at exactly 0 can also cause this)\n", r.failures);
    rc = r.failures ? 1 : 0;
out:
    nmz_tensor_free(&x);
    nmz_tensor_free(&t);
    free(a.x);
    free(a.t);
    return rc;
}

/* ------------------------------------------------------------- save/load */

static int cmd_save(cli_state *st, int argc, char **argv) {
    const char *path;
    if (!need_net(st)) return 1;
    if (argc > 2) return fail("usage: save [FILE]%s", NULL);
    path = argc == 2 ? argv[1] : st->path;
    if (!path) return fail("no file name yet: use `save FILE`%s", NULL);
    if (nmz_net_save(st->net, path) != NMZ_OK) return lib_fail("save");
    if (path != st->path) set_path(st, path);
    st->dirty = 0;
    printf("Saved to %s\n", st->path);
    return 0;
}

static int cmd_load(cli_state *st, int argc, char **argv) {
    nmz_net *net = NULL;
    if (argc != 2) return fail("usage: load FILE%s", NULL);
    if (nmz_net_load(&net, argv[1]) != NMZ_OK) return lib_fail("load");
    replace_net(st, net, argv[1]);
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
    replace_net(st, NULL, NULL);
    st->dirty = 0;
    printf("Network cleared.\n");
    return 0;
}

static int cmd_set(cli_state *st, int argc, char **argv) {
    nmz_train_opts o;
    unsigned mask = 0;
    int i = 1;
    if (!need_net(st)) return 1;
    o = *nmz_net_train_config(st->net);
    if (argc < 2) return fail("usage: set --lr X | --momentum X | --batch N | --epochs N | --target-loss X | --shuffle | --no-shuffle%s", NULL);
    for (i = 1; i < argc; i++) {
        int r = train_option(argc, argv, &i, &o, &mask);
        if (r < 0) return 1;
        if (r == 0) return fail("unknown setting '%s' (see `help set`)", argv[i]);
    }
    *nmz_net_train_config(st->net) = o;
    st->dirty = 1;
    printf("Training config: ");
    print_train_config(&o);
    return 0;
}

static int cmd_reinit(cli_state *st, int argc, char **argv) {
    uint64_t seed = DEFAULT_SEED;
    if (!need_net(st)) return 1;
    if (argc > 2 || (argc == 2 && !parse_u64(argv[1], &seed))) return fail("usage: reinit [SEED]%s", NULL);
    if (nmz_net_reinit(st->net, seed) != NMZ_OK) return lib_fail("reinit");
    st->dirty = 1;
    printf("New random weights (seed %llu); training steps reset to 0.\n", (unsigned long long)seed);
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
static int cmd_trace_fwd(cli_state *st, int argc, char **argv);
static int cmd_step_fwd(cli_state *st, int argc, char **argv);
static int cmd_quit(cli_state *st, int argc, char **argv) {
    (void)argc;
    (void)argv;
    st->quit = 1;
    return 0;
}

static const cmd k_cmds[] = {
    {"new", cmd_new, "new INPUTS [LAYER...] OUTPUTS [options]",
     "Create a network. Each LAYER is SIZE[:ACTIVATION[:INIT]].\n"
     "  Quick mode:    new 2 4 1          (defaults are chosen and explained)\n"
     "  Advanced mode: new 4 16:relu:he 8:leaky_relu=0.05 3:softmax --loss ce --seed 7\n"
     "  activations: linear sigmoid tanh relu leaky_relu[=ALPHA] softmax\n"
     "  inits: auto xavier he      losses (--loss): mse, cross_entropy (ce; needs softmax output)\n"
     "  options: --loss L --seed N --lr X --momentum X --batch N --epochs N --target-loss X --no-shuffle"},
    {"show", cmd_show, "show [weights]", "Describe the network: layers, parameters, loss, training config, classes."},
    {"train", cmd_train, "train FILE.csv [options]",
     "Train on a CSV file. The first INPUTS columns are inputs, the rest are targets;\n"
     "  a text label column (e.g. iris species) is one-hot encoded automatically.\n"
     "  Options override the stored config for this run only (use `set` to change it):\n"
     "  --epochs N --batch N --lr X --momentum X --target-loss X --no-shuffle --every N --quiet\n"
     "  --header / --no-header (default: auto-detect)"},
    {"test", cmd_test, "test FILE.csv", "Loss and accuracy on a CSV file (plus a confusion matrix for classes)."},
    {"predict", cmd_predict, "predict X1 ... Xn", "Run one sample forward and print the outputs."},
    {"trace", cmd_trace_fwd, "trace X1 ... Xn --target T1 ... Tm [--json FILE] [--full]",
     "Run ONE sample forward and backward and print every value with its formula:\n"
     "  z, a, the loss, every delta and gradient, and the weight update it WOULD make.\n"
     "  Nothing is changed. --target may be a class name. --json writes the raw events (JSON lines)."},
    {"step", cmd_step_fwd, "step X1 ... Xn --target T1 ... Tm [--lr X] [--momentum X] [--json FILE] [--full]",
     "Like trace, but the weight update is applied: one training step you can follow by hand."},
    {"gradcheck", cmd_gradcheck, "gradcheck X1 ... Xn --target T1 ... Tm",
     "Compare every backprop gradient with a numerical derivative (L(w+e)-L(w-e))/2e."},
    {"save", cmd_save, "save [FILE]", "Save to FILE (or to the file it was loaded from). Portable .nmz format."},
    {"load", cmd_load, "load FILE", "Load a .nmz model."},
    {"set", cmd_set, "set --lr X | --momentum X | --batch N | --epochs N | --target-loss X | --[no-]shuffle",
     "Change the training configuration stored with the model."},
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
        if (k_cmds[i].help) printf("  %-60s\n", k_cmds[i].usage);
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

static int cmd_trace_fwd(cli_state *st, int argc, char **argv) { return run_trace(st, argc, argv, 0); }
static int cmd_step_fwd(cli_state *st, int argc, char **argv) { return run_trace(st, argc, argv, 1); }

int cli_run(cli_state *st, int argc, char **argv) {
    size_t i;
    if (argc < 1) return 0;
    for (i = 0; i < sizeof k_cmds / sizeof k_cmds[0]; i++)
        if (!strcmp(k_cmds[i].name, argv[0])) return k_cmds[i].fn(st, argc, argv);
    fprintf(stderr, "error: unknown command '%s' (type `help`)\n", argv[0]);
    return 1;
}
