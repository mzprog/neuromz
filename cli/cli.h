/* cli.h — internals of the `neuromz` command-line program. */
#ifndef NMZ_CLI_H
#define NMZ_CLI_H

#include <stdint.h>
#include <stdio.h>

#include "neuromz.h"

typedef struct cli_state {
    nmz_net *net;
    char *path;      /* file the net was loaded from / saved to (or NULL) */
    int dirty;       /* changed since last save                          */
    int interactive; /* stdin is a terminal                              */
    int quit;
    int oneshot;        /* running a single command from the shell argv   */
    const char *output; /* -o FILE in one-shot mode (saved after success) */
} cli_state;

/* line.c — no fixed-size buffers anywhere */
char *cli_read_line(FILE *f);            /* malloc'd, no '\n'; NULL at EOF */
int cli_tokenize(const char *line, char ***out, const char **err); /* count or -1 */
void cli_free_tokens(char **tok, int n);

/* commands.c */
/* Run one command (argv[0] is the command name). Returns 0 on success. */
int cli_run(cli_state *st, int argc, char **argv);
void cli_state_free(cli_state *st);
void cli_print_help(const char *topic);

/* print.c */
typedef struct print_opts {
    int max_rows;  /* table rows before eliding (0 = unlimited) */
    int max_terms; /* Σ terms written out before eliding        */
} print_opts;

void print_set_ascii(int ascii);
int print_is_ascii(void);
/* Pick the UTF-8 or the ASCII spelling of a symbol. */
const char *sym(const char *utf8, const char *ascii);
/* Format a number for tables (fixed width ~10). */
const char *fmt_num(char *buf, size_t n, double v);
/* Print a recorded forward/backward(/update) pass with formulas.
 * `applied`: whether the update was really applied (step) or a preview. */
void print_trace_all(const nmz_trace_recorder *rec, const print_opts *o, int applied);
void print_matrix(const char *title, const nmz_tensor *t, int max_rows);
void print_rule(void);

#endif
