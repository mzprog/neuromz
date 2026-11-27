/* test_model.c — models described in words (nmz_model_new, decision 159):
 * presets and layer specs build the expected nets, every choice is
 * explained, errors are messages (never output), and the training-option
 * parser and description. The program's own test compares its `new`
 * output with what it printed before the move, byte for byte. */
#include "test.h"

#include <string.h>

typedef struct lines {
    int n;
    char last[1024];
    int saw_seed, saw_created, saw_extra;
} lines;

static void collect(const char *line, void *user) {
    lines *l = (lines *)user;
    l->n++;
    snprintf(l->last, sizeof l->last, "%s", line);
    if (strstr(line, "  seed ")) l->saw_seed = 1;
    if (strstr(line, "Created a network with")) l->saw_created = 1;
}

static void extra(void *user) { ((lines *)user)->saw_extra = 1; }

static nmz_status build(int argc, char **argv, lines *l, nmz_net **net) {
    nmz_model_env env;
    memset(&env, 0, sizeof env);
    memset(l, 0, sizeof *l);
    env.print = collect;
    env.user = l;
    env.after_settings = extra;
    return nmz_model_new(argc, argv, &env, net);
}

static void test_presets_and_specs(void) {
    char *xor_args[] = {"2", "4", "1"};
    char *cnn[] = {"--preset", "mnist-cnn", "--lr", "0.002"};
    char *text[] = {"text", "embed:8", "lstm:16", "vocab", "--data", "examples/names_en.txt"};
    lines l;
    nmz_net *net = NULL;
    int i, found = 0;
    CHECK_OK(build(3, xor_args, &l, &net));
    CHECK(net && nmz_net_param_count(net) == 17);
    CHECK(l.saw_seed && l.saw_created && l.saw_extra && l.n > 8);
    nmz_net_free(net);
    net = NULL;
    CHECK_OK(build(4, cnn, &l, &net));
    CHECK(net && nmz_net_param_count(net) == 27562);
    CHECK(net && fabs(nmz_net_train_config(net)->learning_rate - 0.002f) < 1e-9);
    nmz_net_free(net);
    net = NULL;
    CHECK_OK(build(6, text, &l, &net));
    CHECK(net && nmz_net_vocab(net) && nmz_net_is_sequence(net));
    nmz_net_free(net);
    for (i = 0; i < nmz_preset_count(); i++) found += !strcmp(nmz_preset_name(i), "char-gpt");
    CHECK(found == 1 && nmz_preset_count() == 10);
    CHECK(nmz_preset_name(nmz_preset_count()) == NULL);
}

static void test_errors_are_messages(void) {
    char *bad1[] = {"2", "x", "1"};
    char *bad2[] = {"--preset", "nope"};
    char *bad3[] = {"text", "lstm:4", "vocab"};
    lines l;
    nmz_net *net = NULL;
    CHECK_STATUS(build(3, bad1, &l, &net), NMZ_ERR_ARG);
    CHECK(net == NULL && strstr(nmz_last_error(), "layer spec") != NULL);
    CHECK_STATUS(build(2, bad2, &l, &net), NMZ_ERR_ARG);
    CHECK(strstr(nmz_last_error(), "unknown preset 'nope'") != NULL && l.n == 0);
    CHECK(build(3, bad3, &l, &net) != NMZ_OK && strstr(nmz_last_error(), "--data") != NULL);
    CHECK(net == NULL);
}

static int no_data(const char *name, char *path, size_t n, char *err, size_t errn, void *user) {
    (void)path;
    (void)n;
    (void)user;
    snprintf(err, errn, "no data set named '%s' on this server", name);
    return 1;
}

static void test_data_path_callback(void) {
    char *text[] = {"--preset", "names", "--data", "names-en"};
    nmz_model_env env;
    nmz_net *net = NULL;
    memset(&env, 0, sizeof env);
    env.data_path = no_data;
    CHECK(nmz_model_new(4, text, &env, &net) != NMZ_OK);
    CHECK(strstr(nmz_last_error(), "no data set named 'names-en'") != NULL);
}

static void test_train_options(void) {
    char *args[] = {"--lr", "0.05", "--optimizer", "adamw", "--epochs", "7", "--no-shuffle", "--frob", "--batch", "zero"};
    nmz_train_opts o = nmz_train_defaults();
    unsigned mask = 0;
    int i = 0;
    char line[600];
    CHECK(nmz_train_option(10, args, &i, &o, &mask) == 1 && i == 1);
    i = 2;
    CHECK(nmz_train_option(10, args, &i, &o, &mask) == 1);
    i = 4;
    CHECK(nmz_train_option(10, args, &i, &o, &mask) == 1);
    i = 6;
    CHECK(nmz_train_option(10, args, &i, &o, &mask) == 1);
    i = 7;
    CHECK(nmz_train_option(10, args, &i, &o, &mask) == 0);
    i = 8;
    CHECK(nmz_train_option(10, args, &i, &o, &mask) == -1);
    CHECK(strstr(nmz_last_error(), "--batch needs a positive integer, got 'zero'") != NULL);
    CHECK(o.optimizer == NMZ_OPT_ADAMW && o.max_epochs == 7 && !o.shuffle && fabs(o.learning_rate - 0.05f) < 1e-9);
    CHECK(mask == (NMZ_SET_LR | NMZ_SET_OPTIMIZER | NMZ_SET_EPOCHS | NMZ_SET_SHUFFLE));
    nmz_train_describe(&o, line, sizeof line);
    CHECK(strncmp(line, "adamw lr 0.05", 13) == 0 && strstr(line, "max epochs 7") && strstr(line, "shuffle off"));
}

const test_case model_tests[] = {{"model/presets_and_specs", test_presets_and_specs},
                                 {"model/errors_are_messages", test_errors_are_messages},
                                 {"model/data_path_callback", test_data_path_callback},
                                 {"model/train_options", test_train_options},
                                 {NULL, NULL}};
