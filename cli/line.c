/* line.c — read arbitrarily long lines and split them into tokens.
 * (v1 used a fixed 32-byte token buffer and overflowed it.) */
#include "cli.h"

#include <stdlib.h>
#include <string.h>

char *cli_read_line(FILE *f) {
    size_t len = 0, cap = 128;
    char *buf = malloc(cap);
    int c;
    if (!buf) return NULL;
    while ((c = fgetc(f)) != EOF && c != '\n') {
        if (len + 1 >= cap) {
            char *p = realloc(buf, cap *= 2);
            if (!p) {
                free(buf);
                return NULL;
            }
            buf = p;
        }
        buf[len++] = (char)c;
    }
    if (c == EOF && len == 0) {
        free(buf);
        return NULL;
    }
    if (len > 0 && buf[len - 1] == '\r') len--;
    buf[len] = '\0';
    return buf;
}

void cli_free_tokens(char **tok, int n) {
    int i;
    if (!tok) return;
    for (i = 0; i < n; i++) free(tok[i]);
    free(tok);
}

/* Whitespace-separated tokens; "double" or 'single' quotes group words.
 * A '#' outside quotes starts a comment. */
int cli_tokenize(const char *line, char ***out, const char **err) {
    char **tok = NULL;
    int n = 0, cap = 0;
    const char *p = line;
    *out = NULL;
    *err = NULL;
    for (;;) {
        size_t len = 0, tcap = 16;
        char *t;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '\0' || *p == '#') break;
        t = malloc(tcap);
        if (!t) goto oom;
        while (*p && *p != ' ' && *p != '\t') {
            char c = *p++;
            if (c == '"' || c == '\'') {
                char q = c;
                while (*p && *p != q) {
                    if (len + 1 >= tcap) {
                        char *nt = realloc(t, tcap *= 2);
                        if (!nt) {
                            free(t);
                            goto oom;
                        }
                        t = nt;
                    }
                    t[len++] = *p++;
                }
                if (*p != q) {
                    free(t);
                    cli_free_tokens(tok, n);
                    *err = "unterminated quote";
                    return -1;
                }
                p++;
                continue;
            }
            if (len + 1 >= tcap) {
                char *nt = realloc(t, tcap *= 2);
                if (!nt) {
                    free(t);
                    goto oom;
                }
                t = nt;
            }
            t[len++] = c;
        }
        t[len] = '\0';
        if (n == cap) {
            char **nt = realloc(tok, (size_t)(cap = cap ? cap * 2 : 8) * sizeof *nt);
            if (!nt) {
                free(t);
                goto oom;
            }
            tok = nt;
        }
        tok[n++] = t;
    }
    *out = tok;
    return n;
oom:
    cli_free_tokens(tok, n);
    *err = "out of memory";
    return -1;
}
