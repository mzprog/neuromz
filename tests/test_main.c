/* test_main.c — runs every test table; exit status 1 on any failure.
 * Usage: test_neuromz [substring]   (run only tests whose name contains it) */
#include "test.h"

#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

int g_checks = 0, g_failures = 0;
static int g_note_open = 0;

const char *test_path(const char *name) {
    static char buf[4][512];
    static int slot = 0;
    const char *dir = getenv("NMZ_TEST_DIR");
    char *b = buf[slot++ % 4];
    snprintf(b, sizeof buf[0], "%s/%s", dir && *dir ? dir : "build", name);
    return b;
}

void test_note(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    if (!g_note_open) fputc('\n', stdout);
    g_note_open = 1;
    fputs("      ", stdout);
    vprintf(fmt, ap);
    fputc('\n', stdout);
    va_end(ap);
}

int main(int argc, char **argv) {
    const test_case *tables[] = {core_tests, text_tests, math_tests, layer_tests, seq_tests, gated_tests, optim_tests, train_tests, io_tests};
    const char *filter = argc > 1 ? argv[1] : NULL;
    int t, run = 0, failed_tests = 0;
    for (t = 0; t < (int)(sizeof tables / sizeof tables[0]); t++) {
        const test_case *c;
        for (c = tables[t]; c->name; c++) {
            int before = g_failures;
            if (filter && !strstr(c->name, filter)) continue;
            printf("  %-40s", c->name);
            fflush(stdout);
            g_note_open = 0;
            c->fn();
            run++;
            if (g_note_open) printf("  %-40s", "");
            if (g_failures != before) {
                failed_tests++;
                printf(" FAILED\n");
            } else {
                printf(" ok\n");
            }
            fflush(stdout);
        }
    }
    printf("\n%d tests, %d checks, %d failed check(s) in %d test(s)\n", run, g_checks, g_failures,
           failed_tests);
    return g_failures ? 1 : 0;
}
