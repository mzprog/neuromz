/* test_gpu_main.c — the GPU tests (ctest label "gpu"): test_gpu [--parity] [substring]
 * Exit 77 (skipped) when this machine has no usable CUDA device. */
#include "test.h"

#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

int g_checks = 0, g_failures = 0;
static int g_note_open = 0;

extern const test_case gpu_tests[];
extern const test_case gpu_parity_tests[];

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
    const test_case *table = gpu_tests, *c;
    const char *filter = NULL;
    nmz_device_info di;
    int i, run = 0, failed_tests = 0;
    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--parity"))
            table = gpu_parity_tests;
        else
            filter = argv[i];
    }
    if (nmz_backend_query("cuda", &di) != NMZ_OK) {
        printf("no usable CUDA device: %s (skipped)\n", di.note);
        return 77;
    }
    printf("%s, compute %d.%d, %.0f MiB\n", di.device, di.cc_major, di.cc_minor, di.mem_total / 1048576.0);
    for (c = table; c->name; c++) {
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
        nmz_backend_select("cpu");
    }
    printf("\n%d tests, %d checks, %d failed check(s) in %d test(s)\n", run, g_checks, g_failures, failed_tests);
    return g_failures ? 1 : 0;
}
