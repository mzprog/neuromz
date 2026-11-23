/* cmd_new.c — `new`: build a network from an input shape and layer specs,
 * or from a preset, and explain every choice in one line.
 *
 *   input:   N  |  CxHxW  |  HxW (= 1xHxW)
 *   dense:   N[:ACT][:INIT]            or dense:N[:ACT][:INIT]
 *   conv:    conv:FILTERS:K[:ACT][:INIT][:stride=S][:pad=P|same]   K = 3 or 3x5
 *   pool:    pool:K[:stride=S]          (also maxpool:K)
 *   other:   flatten | dropout:RATE
 * A flatten is inserted automatically before a dense layer that follows
 * conv/pool layers. The last layer must be dense (the output). */
#include "cli.h"

#include <stdlib.h>
#include <string.h>

#include "neuromz_layer.h" /* layer shapes */

#define DEFAULT_SEED 42u

typedef enum { LS_DENSE, LS_CONV, LS_POOL, LS_FLATTEN, LS_DROPOUT } ls_kind;

typedef struct lspec {
    ls_kind kind;
    int n; /* units, filters or pool size */
    int kh, kw, stride, pad;
    float rate, alpha;
    nmz_act act;
    int act_set;
    nmz_init init;
    int init_set;
    const char *why; /* preset explanation */
} lspec;

/* ------------------------------------------------------------- presets */

typedef struct preset {
    const char *name, *summary, *input, *why_input;
    const char *layers[12][2]; /* spec, why */
    nmz_norm norm;
    const char *why_norm;
    nmz_optimizer opt;
    float lr;
    const char *why_opt, *why_lr;
    int batch;
    const char *why_batch;
    int epochs;
    const char *why_epochs;
    nmz_schedule sched;
    const char *why_sched;
    double target;
    const char *why_target;
} preset;

static const preset k_presets[] = {
    {"mnist-cnn", "small convolutional network for MNIST digits", "1x28x28",
     "MNIST digits: 1 channel (grey levels 0..255), 28x28 pixels",
     {{"conv:8:3:relu", "8 filters of 3x3: small kernels find strokes and edges; relu + He init"},
      {"pool:2", "2x2 max-pool keeps the strongest response and halves the map (26 -> 13)"},
      {"conv:16:3:relu", "16 filters combine edges into parts of digits (13 -> 11)"},
      {"pool:2", "halves again (11 -> 5): 16x5x5 = 400 features"},
      {"64:relu", "64 hidden units combine the 400 features (flatten is inserted automatically)"},
      {"10:softmax", "10 digits -> class probabilities"},
      {NULL, NULL}},
     NMZ_NORM_ZSCORE, "pixels 0..255 -> mean 0, std 1 (fitted on the training set, stored in the model)",
     NMZ_OPT_ADAM, 0.001f, "Adam adapts the step per weight: conv and dense weights get very different gradients",
     "0.001 is Adam's standard step size", 32, "32 images per update: ~1900 updates per epoch", 3,
     "3 passes: about a minute each on a laptop CPU (naive convolution), enough for > 98%", NMZ_SCHED_COSINE,
     "cosine: large steps first, a smooth decay to 0 by the last epoch for fine-tuning", 0.0,
     "0 = always run all epochs (test accuracy, not training loss, is the goal)"},
    {"mnist-mlp", "fully connected baseline for MNIST", "784", "MNIST digits as 784 numbers (28x28 pixels, row by row)",
     {{"128:relu", "128 hidden units with relu (He init): a classic MLP baseline"},
      {"10:softmax", "10 digits -> class probabilities"},
      {NULL, NULL}},
     NMZ_NORM_ZSCORE, "pixels 0..255 -> mean 0, std 1 per pixel position (fitted on the training set)", NMZ_OPT_ADAM,
     0.001f, "Adam converges in a few epochs without tuning", "0.001 is Adam's standard step size", 32,
     "32 images per update", 5, "5 passes, ~25 s each", NMZ_SCHED_COSINE, "cosine decay to 0 over the 5 epochs",
     0.0, "0 = always run all epochs"},
};

static const preset *find_preset(const char *name) {
    size_t i;
    for (i = 0; i < sizeof k_presets / sizeof k_presets[0]; i++)
        if (!strcmp(k_presets[i].name, name)) return &k_presets[i];
    return NULL;
}

/* -------------------------------------------------------------- parsing */

static int parse_input(const char *s, int *rank, int *shape) {
    long a, b, c;
    char *copy = cli_strdup(s), *p1, *p2;
    int ok = 0;
    if (!copy) return 0;
    p1 = strchr(copy, 'x');
    if (p1) *p1++ = '\0';
    p2 = p1 ? strchr(p1, 'x') : NULL;
    if (p2) *p2++ = '\0';
    if (!cli_parse_int(copy, 1, 1000000, &a)) goto out;
    if (!p1) {
        *rank = 1;
        shape[0] = (int)a;
    } else if (!cli_parse_int(p1, 1, 100000, &b)) {
        goto out;
    } else if (!p2) {
        *rank = 3; /* HxW = 1 channel */
        shape[0] = 1;
        shape[1] = (int)a;
        shape[2] = (int)b;
    } else if (!cli_parse_int(p2, 1, 100000, &c)) {
        goto out;
    } else {
        *rank = 3;
        shape[0] = (int)a;
        shape[1] = (int)b;
        shape[2] = (int)c;
    }
    ok = 1;
out:
    free(copy);
    return ok;
}

/* act[=alpha] | init | key=value modifiers after the fixed fields */
static int parse_modifier(const char *tok, lspec *sp, const char *whole) {
    char buf[64], *eq;
    double d;
    long n;
    if (strlen(tok) >= sizeof buf) return cli_fail("layer spec part too long: '%s'", tok), 0;
    strcpy(buf, tok);
    eq = strchr(buf, '=');
    if (eq) *eq++ = '\0';
    if (!strcmp(buf, "stride") && eq) {
        if (sp->kind != LS_CONV && sp->kind != LS_POOL) return cli_fail("stride= only applies to conv/pool: '%s'", whole), 0;
        if (!cli_parse_int(eq, 1, 1000, &n)) return cli_fail("stride must be a positive integer in '%s'", whole), 0;
        sp->stride = (int)n;
        return 1;
    }
    if (!strcmp(buf, "pad") && eq) {
        if (sp->kind != LS_CONV) return cli_fail("pad= only applies to conv layers: '%s'", whole), 0;
        if (!strcmp(eq, "same")) sp->pad = NMZ_PAD_SAME;
        else if (cli_parse_int(eq, 0, 1000, &n)) sp->pad = (int)n;
        else return cli_fail("pad must be a number >= 0 or 'same' in '%s'", whole), 0;
        return 1;
    }
    if (sp->kind == LS_DENSE || sp->kind == LS_CONV) {
        nmz_act a;
        nmz_init in;
        if (nmz_act_parse(buf, &a) == NMZ_OK) {
            sp->act = a;
            sp->act_set = 1;
            if (eq) {
                if (a != NMZ_ACT_LEAKY_RELU || !cli_parse_double(eq, &d) || d <= 0 || d >= 1)
                    return cli_fail("only leaky_relu takes a parameter, in (0,1): '%s'", whole), 0;
                sp->alpha = (float)d;
            }
            return 1;
        }
        if (!eq && nmz_init_parse(buf, &in) == NMZ_OK) {
            sp->init = in;
            sp->init_set = in != NMZ_INIT_AUTO;
            return 1;
        }
        if (sp->kind == LS_CONV && sp->act_set)
            return cli_fail("unknown init or option in '%s' (inits: auto xavier he; options: stride= pad=)", whole), 0;
        fprintf(stderr, "error: '%s' in '%s' is not an activation, init or option\n"
                        "       (activations: linear sigmoid tanh relu leaky_relu softmax; inits: auto xavier he)\n",
                tok, whole);
        return 0;
    }
    return cli_fail("unexpected '%s' (see `help new`)", whole), 0;
}

static int parse_kernel(const char *s, int *kh, int *kw) {
    long a, b;
    char buf[32], *x;
    if (strlen(s) >= sizeof buf) return 0;
    strcpy(buf, s);
    x = strchr(buf, 'x');
    if (x) *x++ = '\0';
    if (!cli_parse_int(buf, 1, 1000, &a)) return 0;
    b = a;
    if (x && !cli_parse_int(x, 1, 1000, &b)) return 0;
    *kh = (int)a;
    *kw = (int)b;
    return 1;
}

static int parse_layer(const char *s, lspec *sp) {
    char *copy = cli_strdup(s), *tok[16];
    int nt = 0, i, first = 1, ok = 0;
    long n;
    double d;
    memset(sp, 0, sizeof *sp);
    sp->stride = 0;
    if (!copy) return cli_fail("out of memory%s", NULL), 0;
    tok[nt++] = copy;
    for (i = 0; copy[i] && nt < 16; i++)
        if (copy[i] == ':') {
            copy[i] = '\0';
            tok[nt++] = copy + i + 1;
        }
    if (cli_parse_int(tok[0], 1, 1000000, &n)) { /* N[:act][:init] */
        sp->kind = LS_DENSE;
        sp->n = (int)n;
    } else if (!strcmp(tok[0], "dense")) {
        if (nt < 2 || !cli_parse_int(tok[1], 1, 1000000, &n)) {
            cli_fail("dense needs a size: dense:N, got '%s'", s);
            goto out;
        }
        sp->kind = LS_DENSE;
        sp->n = (int)n;
        first = 2;
    } else if (!strcmp(tok[0], "conv")) {
        if (nt < 3 || !cli_parse_int(tok[1], 1, 100000, &n) || !parse_kernel(tok[2], &sp->kh, &sp->kw)) {
            cli_fail("conv needs filters and a kernel size: conv:FILTERS:K (e.g. conv:8:3 or conv:8:3x5), got '%s'", s);
            goto out;
        }
        sp->kind = LS_CONV;
        sp->n = (int)n;
        sp->stride = 1;
        first = 3;
    } else if (!strcmp(tok[0], "pool") || !strcmp(tok[0], "maxpool")) {
        if (nt < 2 || !cli_parse_int(tok[1], 1, 1000, &n)) {
            cli_fail("pool needs a window size: pool:K (e.g. pool:2), got '%s'", s);
            goto out;
        }
        sp->kind = LS_POOL;
        sp->n = (int)n;
        first = 2;
    } else if (!strcmp(tok[0], "flatten")) {
        sp->kind = LS_FLATTEN;
    } else if (!strcmp(tok[0], "dropout")) {
        if (nt < 2 || !cli_parse_double(tok[1], &d) || d < 0 || d >= 1) {
            cli_fail("dropout needs a rate in [0, 1): dropout:0.25, got '%s'", s);
            goto out;
        }
        sp->kind = LS_DROPOUT;
        sp->rate = (float)d;
        first = 2;
    } else {
        fprintf(stderr, "error: layer spec must be a size (e.g. 64:relu) or a positive integer, or start with "
                        "dense:, conv:, pool:, flatten or dropout: — got '%s'\n", s);
        goto out;
    }
    for (i = first; i < nt; i++) {
        if (!*tok[i]) continue;
        if (!parse_modifier(tok[i], sp, s)) goto out;
    }
    ok = 1;
out:
    free(copy);
    return ok;
}

/* ----------------------------------------------------------- explaining */

static void explain(const char *what, const char *value, const char *tag, const char *why) {
    printf("  %-24s %-15s %-9s %s\n", what, value, tag, why ? why : "");
}

static const char *tag_of(int set, int from_preset) { return set ? "[yours]" : from_preset ? "[preset]" : "[default]"; }

static void shape_str(char *b, size_t n, const nmz_layer *l, int out) {
    int r = out ? l->out_rank : l->in_rank, i, w = 0;
    const int *s = out ? l->out_shape : l->in_shape;
    for (i = 0; i < r && w >= 0 && (size_t)w < n; i++) w += snprintf(b + w, n - (size_t)w, i ? "x%d" : "%d", s[i]);
}

/* ------------------------------------------------------------------ new */

int cmd_new(cli_state *st, int argc, char **argv) {
    lspec *specs = NULL;
    int n_specs = 0, i, rank = 0, shape[NMZ_MAX_RANK], loss_set = 0, rc = 1, has_conv = 0, last_dense = -1;
    int norm_set = 0, seed_set = 0, n_dense = 0, *dense_sizes = NULL;
    uint64_t seed = DEFAULT_SEED;
    unsigned mask = 0;
    nmz_loss loss = NMZ_LOSS_MSE;
    nmz_norm norm = NMZ_NORM_NONE;
    nmz_quick q, qt;
    nmz_train_opts o, user;
    nmz_net *net = NULL;
    const preset *pre = NULL;
    const char *input_text = NULL;
    char buf[160], what[200];
    const nmz_layer *prev;

    specs = calloc((size_t)argc + 16, sizeof *specs);
    dense_sizes = calloc((size_t)argc + 16, sizeof *dense_sizes);
    if (!specs || !dense_sizes) {
        cli_fail("out of memory%s", NULL);
        goto out;
    }
    memset(&user, 0, sizeof user);
    user = nmz_train_defaults();
    for (i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (cli_is_option(a)) {
            int r = cli_train_option(argc, argv, &i, &user, &mask);
            const char *v;
            if (r < 0) goto out;
            if (r) continue;
            if (!strcmp(a, "--preset")) {
                if (!(v = cli_opt_value(argc, argv, &i))) goto out;
                if (!(pre = find_preset(v))) {
                    cli_fail("unknown preset '%s' (available: mnist-cnn, mnist-mlp)", v);
                    goto out;
                }
            } else if (!strcmp(a, "--loss")) {
                if (!(v = cli_opt_value(argc, argv, &i))) goto out;
                if (nmz_loss_parse(v, &loss) != NMZ_OK) {
                    cli_lib_fail("--loss");
                    goto out;
                }
                loss_set = 1;
            } else if (!strcmp(a, "--normalize")) {
                if (!(v = cli_opt_value(argc, argv, &i))) goto out;
                if (nmz_norm_parse(v, &norm) != NMZ_OK) {
                    cli_lib_fail("--normalize");
                    goto out;
                }
                norm_set = 1;
            } else if (!strcmp(a, "--seed")) {
                if (!(v = cli_opt_value(argc, argv, &i))) goto out;
                if (!cli_parse_u64(v, &seed)) {
                    cli_fail("--seed needs a non-negative integer, got '%s'", v);
                    goto out;
                }
                seed_set = 1;
            } else {
                cli_fail("unknown option '%s' for new (see `help new`)", a);
                goto out;
            }
        } else if (!input_text) {
            input_text = a;
        } else {
            if (!parse_layer(a, &specs[n_specs])) goto out;
            n_specs++;
        }
    }
    if (pre) {
        if (input_text) {
            cli_fail("--preset defines the layers; do not also give sizes (got '%s')", input_text);
            goto out;
        }
        input_text = pre->input;
        for (i = 0; pre->layers[i][0]; i++) {
            if (!parse_layer(pre->layers[i][0], &specs[n_specs])) goto out;
            specs[n_specs].why = pre->layers[i][1];
            specs[n_specs].act_set = specs[n_specs].init_set = 0;
            n_specs++;
        }
    }
    if (!input_text || n_specs == 0) {
        cli_fail("usage: new INPUT LAYER... [options]   e.g. `new 2 4 1`, `new 1x28x28 conv:8:3 pool:2 10:softmax`, "
                 "or `new --preset mnist-cnn` (see `help new`)%s", NULL);
        goto out;
    }
    if (!parse_input(input_text, &rank, shape)) {
        cli_fail("the input must be a size like 784 or a shape like 1x28x28, got '%s'", input_text);
        goto out;
    }
    for (i = 0; i < n_specs; i++) {
        if (specs[i].kind == LS_CONV || specs[i].kind == LS_POOL) has_conv = 1;
        if (specs[i].kind == LS_DENSE) {
            last_dense = i;
            dense_sizes[++n_dense] = specs[i].n;
        }
    }
    if (last_dense != n_specs - 1) {
        cli_fail("the last layer must be a dense layer (the output), e.g. 10:softmax%s", NULL);
        goto out;
    }
    if (has_conv && rank != 3) {
        cli_fail("conv/pool layers need an image input shape like 1x28x28 (got '%s')", input_text);
        goto out;
    }

    /* defaults */
    dense_sizes[0] = 1;
    for (i = 0; i < rank; i++) dense_sizes[0] *= shape[i];
    if (nmz_quick_defaults(n_dense + 1, dense_sizes, &q) != NMZ_OK) {
        cli_lib_fail("new");
        goto out;
    }
    for (i = 0; i < n_specs; i++) {
        if (specs[i].act_set) continue;
        if (specs[i].kind == LS_CONV) specs[i].act = NMZ_ACT_RELU;
        if (specs[i].kind == LS_DENSE)
            specs[i].act = i == last_dense ? q.output_act : (has_conv ? NMZ_ACT_RELU : q.hidden_act);
    }
    if (!loss_set) loss = specs[last_dense].act == NMZ_ACT_SOFTMAX ? NMZ_LOSS_CROSS_ENTROPY : NMZ_LOSS_MSE;
    nmz_quick_train_defaults(loss, &qt);
    o = qt.train;
    if (pre) {
        o.optimizer = pre->opt;
        o.learning_rate = pre->lr;
        o.batch_size = pre->batch;
        o.max_epochs = pre->epochs;
        o.schedule = pre->sched;
        o.min_lr = 0.0f;
        o.target_loss = pre->target;
        if (!norm_set) norm = pre->norm;
    } else if (has_conv) {
        o.optimizer = NMZ_OPT_ADAM;
        o.learning_rate = 0.001f;
        o.max_epochs = 10;
        if (!norm_set) norm = NMZ_NORM_ZSCORE;
    }
    if (mask & CLI_SET_LR) o.learning_rate = user.learning_rate;
    if (mask & CLI_SET_MOMENTUM) o.momentum = user.momentum;
    if (mask & CLI_SET_BATCH) o.batch_size = user.batch_size;
    if (mask & CLI_SET_EPOCHS) o.max_epochs = user.max_epochs;
    if (mask & CLI_SET_TARGET) o.target_loss = user.target_loss;
    if (mask & CLI_SET_SHUFFLE) o.shuffle = user.shuffle;
    if (mask & CLI_SET_OPTIMIZER) {
        o.optimizer = user.optimizer;
        if (!(mask & CLI_SET_LR)) o.learning_rate = o.optimizer == NMZ_OPT_ADAM ? 0.001f : qt.train.learning_rate;
    }
    if (mask & CLI_SET_BETAS) {
        o.beta1 = user.beta1;
        o.beta2 = user.beta2;
        o.epsilon = user.epsilon;
    }
    if (mask & CLI_SET_SCHEDULE) {
        o.schedule = user.schedule;
        o.step_size = user.step_size;
        o.gamma = user.gamma;
        o.min_lr = user.min_lr;
    }

    /* build, explaining as we go */
    if (nmz_net_create(&net, rank, shape, seed) != NMZ_OK) {
        cli_lib_fail("new");
        goto out;
    }
    if (pre) printf("Preset %s: %s\n", pre->name, pre->summary);
    if (pre)
        printf("Input %s   %s\n", input_text, pre->why_input);
    else if (rank == 3)
        printf("Input %s   %d channel(s) of %dx%d per sample\n", input_text, shape[0], shape[1], shape[2]);
    else
        printf("Input %s   %d value(s) per sample\n", input_text, shape[0]);
    for (i = 0; i < n_specs; i++) {
        lspec *sp = &specs[i];
        nmz_layer *l = NULL;
        int is_out = i == last_dense, before = nmz_net_layer_count(net);
        prev = before ? nmz_net_layer(net, before - 1) : NULL;
        if (sp->kind == LS_DENSE && prev && prev->out_rank > 1) {
            if (nmz_net_add(net, nmz_layer_flatten()) != NMZ_OK) goto libfail;
            prev = nmz_net_layer(net, nmz_net_layer_count(net) - 1);
            shape_str(buf, sizeof buf, prev, 0);
            snprintf(what, sizeof what, "flatten %s->%d", buf, prev->out_shape[0]);
            explain(what, "", "[auto]", "dense layers need a flat vector: the maps are laid out in one row");
        }
        switch (sp->kind) {
        case LS_DENSE: l = nmz_layer_dense(sp->n, sp->init); break;
        case LS_CONV: l = nmz_layer_conv2d(sp->n, sp->kh, sp->kw, sp->stride, sp->pad, sp->init); break;
        case LS_POOL: l = nmz_layer_maxpool2d(sp->n, sp->stride); break;
        case LS_FLATTEN: l = nmz_layer_flatten(); break;
        case LS_DROPOUT: l = nmz_layer_dropout(sp->rate); break;
        }
        if (nmz_net_add(net, l) != NMZ_OK) goto libfail;
        l = nmz_net_layer(net, nmz_net_layer_count(net) - 1);
        if (sp->kind != LS_FLATTEN && sp->kind != LS_DROPOUT && sp->kind != LS_POOL &&
            sp->act != NMZ_ACT_LINEAR && nmz_net_add(net, nmz_layer_activation(sp->act, sp->alpha)) != NMZ_OK)
            goto libfail;
        {
            char in[48], outs[48], val[64];
            const char *why = sp->why;
            shape_str(in, sizeof in, l, 0);
            shape_str(outs, sizeof outs, l, 1);
            switch (sp->kind) {
            case LS_DENSE:
                snprintf(what, sizeof what, "%s %s->%s", is_out ? "output" : "dense", in, outs);
                snprintf(val, sizeof val, "%s", nmz_act_name(sp->act));
                if (!why) why = sp->act_set ? cli_act_reason(sp->act)
                                            : (is_out ? q.why_output_act
                                                      : (has_conv ? "relu, like the conv layers (He init)" : q.why_hidden_act));
                break;
            case LS_CONV:
                snprintf(what, sizeof what, "conv %s->%s", in, outs);
                snprintf(val, sizeof val, "%dx %dx%d %s", sp->n, sp->kh, sp->kw, nmz_act_name(sp->act));
                if (!why) why = sp->act_set ? cli_act_reason(sp->act)
                                            : "relu after a convolution is the standard choice; pairs with He init";
                break;
            case LS_POOL:
                snprintf(what, sizeof what, "maxpool %s->%s", in, outs);
                snprintf(val, sizeof val, "%dx%d", sp->n, sp->n);
                if (!why) why = "keeps the largest value of each window; halves the map for a 2x2 window";
                break;
            case LS_FLATTEN:
                snprintf(what, sizeof what, "flatten %s->%s", in, outs);
                val[0] = '\0';
                if (!why) why = "lay the maps out as one vector";
                break;
            case LS_DROPOUT:
                snprintf(what, sizeof what, "dropout");
                snprintf(val, sizeof val, "%g", (double)sp->rate);
                if (!why) why = "randomly zero this fraction during training: discourages co-adapted units";
                break;
            }
            explain(what, val, tag_of(sp->act_set, sp->why != NULL), why);
        }
    }
    if (nmz_net_set_loss(net, loss) != NMZ_OK || nmz_net_set_normalization(net, norm) != NMZ_OK ||
        nmz_net_build(net) != NMZ_OK)
        goto libfail;
    *nmz_net_train_config(net) = o;
    {
        /* report the inits that were actually used */
        int li;
        for (li = 0; li < nmz_net_layer_count(net); li++) {
            nmz_layer *l = nmz_net_layer(net, li);
            int init = nmz_layer_init_used(l);
            if (init < 0) continue;
            snprintf(what, sizeof what, "  init layer %d (%s)", li, nmz_layer_type(l));
            explain(what, nmz_init_name((nmz_init)init), "[auto]", cli_init_reason(init));
        }
    }
    explain("loss", nmz_loss_name(loss), tag_of(loss_set, 0),
            loss_set ? (loss == NMZ_LOSS_MSE ? "L = 1/2 sum (y-t)^2, averaged over the batch"
                                             : "L = -sum t*log(p), fused with softmax: delta = p - t")
                     : loss == q.loss ? q.why_loss
                     : (loss == NMZ_LOSS_CROSS_ENTROPY ? "softmax output: cross-entropy, gradient p - t"
                                                       : "the output is not softmax, so squared error"));
    explain("normalize", nmz_norm_name(norm), tag_of(norm_set, pre != NULL),
            pre && !norm_set ? pre->why_norm
                             : norm == NMZ_NORM_ZSCORE ? "inputs -> mean 0, std 1 (fitted at the first train, stored in the model)"
                             : norm == NMZ_NORM_MINMAX ? "inputs -> [0, 1] (fitted at the first train, stored in the model)"
                                                       : "inputs are used as they are");
    explain("optimizer", nmz_optimizer_name(o.optimizer), tag_of(mask & CLI_SET_OPTIMIZER, pre != NULL),
            pre && !(mask & CLI_SET_OPTIMIZER) ? pre->why_opt
            : o.optimizer == NMZ_OPT_ADAM
                ? (has_conv ? "Adam: per-weight step sizes, robust for conv nets" : "Adam: per-weight step sizes from running gradient averages")
                : "gradient descent with momentum: the textbook method");
    snprintf(buf, sizeof buf, "%g", (double)o.learning_rate);
    explain("learning rate", buf, tag_of(mask & CLI_SET_LR, pre != NULL),
            pre && !(mask & CLI_SET_LR) ? pre->why_lr
            : o.optimizer == NMZ_OPT_ADAM ? "0.001 is Adam's standard step size" : qt.why_learning_rate);
    if (o.optimizer == NMZ_OPT_SGD) {
        snprintf(buf, sizeof buf, "%g", (double)o.momentum);
        explain("momentum", buf, tag_of(mask & CLI_SET_MOMENTUM, 0), qt.why_momentum);
    } else {
        snprintf(buf, sizeof buf, "%g/%g/%g", (double)o.beta1, (double)o.beta2, (double)o.epsilon);
        explain("beta1/beta2/eps", buf, tag_of(mask & CLI_SET_BETAS, 0),
                "averaging rates of the gradient (m) and squared gradient (v), and a guard against /0");
    }
    snprintf(buf, sizeof buf, "%s", nmz_schedule_name(o.schedule));
    explain("schedule", buf, tag_of(mask & CLI_SET_SCHEDULE, pre != NULL),
            pre && !(mask & CLI_SET_SCHEDULE) ? pre->why_sched
            : o.schedule == NMZ_SCHED_COSINE ? "cosine decay of the learning rate over the epochs"
            : o.schedule == NMZ_SCHED_STEP   ? "multiply the learning rate by gamma every step-size epochs"
                                             : "the same learning rate in every epoch");
    snprintf(buf, sizeof buf, "%d", o.batch_size);
    explain("batch size", buf, tag_of(mask & CLI_SET_BATCH, pre != NULL),
            pre && !(mask & CLI_SET_BATCH) ? pre->why_batch : qt.why_batch);
    snprintf(buf, sizeof buf, "%d", o.max_epochs);
    explain("max epochs", buf, tag_of(mask & CLI_SET_EPOCHS, pre != NULL),
            pre && !(mask & CLI_SET_EPOCHS) ? pre->why_epochs
            : has_conv ? "at most 10 passes; use --limit while experimenting" : qt.why_epochs);
    snprintf(buf, sizeof buf, "%g", o.target_loss);
    explain("target loss", buf, tag_of(mask & CLI_SET_TARGET, pre != NULL),
            pre && !(mask & CLI_SET_TARGET) ? pre->why_target : qt.why_target_loss);
    explain("shuffle", o.shuffle ? "on" : "off", tag_of(mask & CLI_SET_SHUFFLE, 0),
            "visit samples in a new random order every epoch");
    snprintf(buf, sizeof buf, "%llu", (unsigned long long)seed);
    explain("seed", buf, tag_of(seed_set, 0), "same seed => same initial weights and same shuffling, every run");
    printf("Created a network with %zu parameters.\n", nmz_net_param_count(net));
    cli_replace_net(st, net, NULL);
    net = NULL;
    if (st->oneshot && !st->output) printf("(not saved: add -o FILE to keep it)\n");
    rc = 0;
    goto out;
libfail:
    cli_lib_fail("new");
out:
    nmz_net_free(net);
    free(specs);
    free(dense_sizes);
    return rc;
}
