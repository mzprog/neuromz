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

/* util.c */
#define CLI_SET_LR 1u
#define CLI_SET_MOMENTUM 2u
#define CLI_SET_BATCH 4u
#define CLI_SET_EPOCHS 8u
#define CLI_SET_TARGET 16u
#define CLI_SET_SHUFFLE 32u
#define CLI_SET_OPTIMIZER 64u
#define CLI_SET_BETAS 128u
#define CLI_SET_SCHEDULE 256u
#define CLI_SET_VAL 512u
#define CLI_SET_CLIP 1024u
#define CLI_SET_CARRY 2048u
#define CLI_SET_PATIENCE 4096u
#define CLI_SET_SEQLEN 8192u
#define CLI_SET_DECAY 16384u  /* --weight-decay, --decay-all */
#define CLI_SET_WARMUP 32768u
int cli_fail(const char *fmt, const char *arg); /* prints "error: ...", returns 1 */
int cli_lib_fail(const char *what);               /* adds nmz_last_error()          */
int cli_need_net(cli_state *st);
int cli_parse_int(const char *s, long lo, long hi, long *out);
int cli_parse_u64(const char *s, uint64_t *out);
int cli_parse_double(const char *s, double *out);
int cli_is_option(const char *s);
const char *cli_opt_value(int argc, char **argv, int *i);
char *cli_strdup(const char *s);
void cli_set_path(cli_state *st, const char *path);
void cli_replace_net(cli_state *st, nmz_net *net, const char *path);
void cli_print_train_config(const nmz_train_opts *o);
int cli_train_option(int argc, char **argv, int *i, nmz_train_opts *o, unsigned *mask);
/* seq_len: for sequence nets, time steps per sample (0 = the model's setting) */
int cli_load_data(cli_state *st, const char *spec, int want_test, int limit, int header, int seq_len,
                  nmz_dataset **out);
/* print one code point readably ("a", "⏎" for newline, "␣" for space; ASCII
 * mode: \n, ' ', U+0645) into buf */
const char *cli_char_label(uint32_t cp, char *buf, size_t n);
const char *cli_act_reason(nmz_act a);
const char *cli_init_reason(int init);

/* main.c (platform bits) */
double cli_now(void);       /* seconds, monotonic where available */
int cli_stdout_is_tty(void);

/* cmd_new.c */
int cmd_new(cli_state *st, int argc, char **argv);

/* commands.c */
/* Run one command (argv[0] is the command name). Returns 0 on success. */
int cli_run(cli_state *st, int argc, char **argv);
void cli_state_free(cli_state *st);
void cli_print_help(const char *topic);

/* print.c */
typedef struct print_opts {
    int max_rows;  /* table rows before eliding (0 = unlimited) */
    int max_terms; /* Σ terms written out before eliding        */
    int full;      /* --full: no eliding at all                 */
    int at_set;    /* --at C,Y,X given (stored 0-based)         */
    int at_c, at_y, at_x;
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

/* print_conv.c */
char heat_char(double v, double lo, double hi);
/* h × w values (row stride stride_row) as an ASCII heatmap scaled to their own
 * min..max; each value printed `cell` characters wide. */
void print_heatmap(const float *v, int h, int w, int stride_row, const char *indent, int cell);
void fmt_index(char *buf, size_t n, const nmz_tensor *t, size_t e);
int is_map(const nmz_tensor *t);
void pr_focus_reset(void);
void pr_normalize(const nmz_trace_event *ev, const print_opts *o);
void pr_conv_forward(const nmz_trace_event *ev, const print_opts *o);
void pr_conv_backward(const nmz_trace_event *ev, const print_opts *o);
void pr_pool_forward(const nmz_trace_event *ev, const print_opts *o);
void pr_pool_backward(const nmz_trace_event *ev, const print_opts *o);
void pr_flatten(const nmz_trace_event *ev, int backward);
void pr_dropout(const nmz_trace_event *ev, int backward);
void pr_act_map_forward(const nmz_trace_event *ev, const print_opts *o);
void pr_act_map_backward(const nmz_trace_event *ev, const print_opts *o);
void pr_adam_header(const nmz_trace_event *ev);
void pr_adam_update(const nmz_trace_event *ev, const print_opts *o);
void print_sgd_update(const nmz_trace_event *ev, const print_opts *o);

/* print_seq.c: the unrolled view of a character model (rnn, lstm, gru). */
/* How much of the LAST step's loss gradient reaches every step t, for one
 * recurrent layer (from a backward pass with only the last step's loss). */
typedef struct seq_survival {
    int T, layer;    /* steps; net layer index of the recurrent layer   */
    char kind[8];    /* "rnn", "lstm", "gru"                            */
    double *h;       /* |dL/dh_t|                                        */
    double *c;       /* lstm: |dL/dc_t|, else NULL                       */
    double *gate_prod; /* lstm: mean over cells of Π_{s>t} f_s; gru: of z */
    double *x;       /* |dL/de_t| at the embedding output (the input)    */
    int *n_keep;     /* lstm/gru: units whose Π f (Π z) from t on is >= 0.5 */
    int units;       /* H */
} seq_survival;
int seq_is_recurrent_type(const char *type);
/* layer: net layer index, < 0 = the first recurrent layer. 0 = ok. */
int seq_survival_collect(const nmz_trace_recorder *rec, int layer, int T, seq_survival *sv);
void seq_survival_free(seq_survival *sv);
/* geometric-mean factor per step of v[T-1] -> v[0]; *left = v[0]/v[T-1] */
double seq_survival_rate(const double *v, int T, double *left);
/* unit < 0 = automatic; layer = net layer index (< 0: the first recurrent
 * layer); sv (may be NULL) = the vanishing-gradient view */
void print_seq_trace(const nmz_trace_recorder *rec, const nmz_vocab *v, const print_opts *o, int applied, int unit,
                     int layer, const seq_survival *sv);
void print_update_section(const nmz_trace_recorder *rec, const print_opts *o, int applied);
/* print_attn.c: the Transformer's attention, step by step */
typedef struct attn_pats attn_pats; /* per-head pattern statistics */
attn_pats *attn_pats_new(void);
void attn_pats_free(attn_pats *ap);
/* add one attention event computed over ids (positions from 0); 1 on OOM */
int attn_pats_add(attn_pats *ap, const nmz_trace_event *ev, const int *ids, const nmz_vocab *v);
void attn_pats_print(const attn_pats *ap, const nmz_net *net);
void attn_heatmap(const nmz_tensor *w, int b, int h, const int *ids, const nmz_vocab *v, int from, int to);
void attn_put_char(const nmz_vocab *v, int id); /* one display column, RTL-safe */
void attn_line_start(void);
/* pos, head: 0-based or -1 (last position / the most decisive head);
 * attn_n: 1-based attention layer; grad_e (may be NULL): |dL/dx_t| of a
 * last-position-only backward pass */
void print_attn_trace(const nmz_trace_recorder *rec, const nmz_vocab *v, const print_opts *o, int pos, int attn_n,
                      int head, const double *grad_e, const nmz_net *net);
/* cmd_seq.c: commands for sequence models */
int cmd_compare(cli_state *st, int argc, char **argv);
int show_gates(cli_state *st, int argc, char **argv);
int show_attention(cli_state *st, int argc, char **argv);
int show_params(cli_state *st);
int net_has_attention(const nmz_net *net);
/* net layer index of the n-th (1-based) layer with attention, or -1 */
int seq_attention_layer(const nmz_net *net, int n);
/* net layer index of the n-th (1-based) recurrent layer, or -1 */
int seq_recurrent_layer(const nmz_net *net, int n);
void print_rule(void);

#endif
