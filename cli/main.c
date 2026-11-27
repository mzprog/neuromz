/* main.c — entry point of `neuromz`.
 *
 *   neuromz                          interactive shell (reads stdin; prompt only on a terminal)
 *   neuromz [-m FILE] CMD ARGS [-o FILE]   one command, optionally loading/saving a model
 *
 * Operating-system calls live in platform.c; the library stays plain C11. */
#include "cli.h"

#include <stdlib.h>
#include <string.h>

static int repl(cli_state *st) {
    int errors = 0;
    if (st->interactive) {
        printf("NeuroMZ %s — type `help`, or start with `new 2 4 1`. Ctrl-D to quit.\n", NMZ_VERSION_STRING);
    }
    while (!st->quit) {
        char *line, **tok = NULL;
        const char *err = NULL;
        int n;
        if (st->interactive) {
            printf("neuromz%s%s%s> ", st->path ? "[" : "", st->path ? st->path : "", st->path ? "]" : "");
            fflush(stdout);
        }
        line = cli_read_line(stdin);
        if (!line) { /* EOF (Ctrl-D or end of a piped script) */
            if (st->interactive) printf("\n");
            break;
        }
        n = cli_tokenize(line, &tok, &err);
        if (n < 0) {
            fprintf(stderr, "error: %s\n", err);
            errors++;
        } else if (n > 0) {
            if (!st->interactive) {
                printf("> %s\n", line); /* echo script lines so the output reads like a session */
            }
            if (cli_run(st, n, tok) != 0) errors++;
            fflush(stdout);
        }
        cli_free_tokens(tok, n > 0 ? n : 0);
        free(line);
    }
    if (st->net && st->dirty && st->interactive) printf("(note: the network had unsaved changes)\n");
    return errors ? 1 : 0;
}

int main(int argc, char **argv) {
    cli_state st;
    const char *model = NULL;
    char **rest;
    int n_rest = 0, i, rc;
    memset(&st, 0, sizeof st);
    cli_console_init(&argc, &argv);
    rest = calloc((size_t)argc + 1, sizeof *rest);
    if (!rest) return 2;
    for (i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "-m") || !strcmp(a, "--model") || !strcmp(a, "-o") || !strcmp(a, "--output")) {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: %s needs a file name\n", a);
                free(rest);
                return 2;
            }
            if (a[1] == 'm' || a[2] == 'm') model = argv[++i];
            else st.output = argv[++i];
        } else if (!strcmp(a, "--ascii")) {
            print_set_ascii(1);
        } else if (n_rest == 0 && (!strcmp(a, "-h") || !strcmp(a, "--help"))) {
            cli_print_help(NULL);
            free(rest);
            return 0;
        } else if (n_rest == 0 && (!strcmp(a, "-v") || !strcmp(a, "--version"))) {
            printf("neuromz %s\n", NMZ_VERSION_STRING);
            free(rest);
            return 0;
        } else {
            rest[n_rest++] = argv[i];
        }
    }
    st.oneshot = n_rest > 0;
    if (model) {
        char *load_argv[2];
        load_argv[0] = "load";
        load_argv[1] = (char *)model;
        if (cli_run(&st, 2, load_argv) != 0) {
            free(rest);
            return 1;
        }
    }
    if (n_rest == 0) {
        if (st.output) fprintf(stderr, "note: -o is ignored in the interactive shell; use `save FILE`\n");
        st.output = NULL;
        st.interactive = cli_stdin_is_tty();
        rc = repl(&st);
    } else {
        rc = cli_run(&st, n_rest, rest);
        if (rc == 0 && st.output) {
            char *save_argv[2];
            save_argv[0] = "save";
            save_argv[1] = (char *)st.output;
            rc = cli_run(&st, 2, save_argv);
        }
    }
    cli_state_free(&st);
    free(rest);
    return rc;
}
