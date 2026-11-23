/* test.h — a deliberately tiny test harness (no dependencies). */
#ifndef NMZ_TEST_H
#define NMZ_TEST_H

#include <math.h>
#include <stdio.h>

#include "neuromz.h"

extern int g_checks, g_failures;

#define CHECK(cond)                                                                       \
    do {                                                                                  \
        g_checks++;                                                                       \
        if (!(cond)) {                                                                    \
            g_failures++;                                                                 \
            fprintf(stderr, "  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);             \
        }                                                                                 \
    } while (0)

#define CHECK_MSG(cond, ...)                                                              \
    do {                                                                                  \
        g_checks++;                                                                       \
        if (!(cond)) {                                                                    \
            g_failures++;                                                                 \
            fprintf(stderr, "  FAIL %s:%d: %s: ", __FILE__, __LINE__, #cond);             \
            fprintf(stderr, __VA_ARGS__);                                                 \
            fputc('\n', stderr);                                                          \
        }                                                                                 \
    } while (0)

#define CHECK_OK(expr)                                                                    \
    do {                                                                                  \
        nmz_status s_ = (expr);                                                           \
        g_checks++;                                                                       \
        if (s_ != NMZ_OK) {                                                               \
            g_failures++;                                                                 \
            fprintf(stderr, "  FAIL %s:%d: %s -> %s (%s)\n", __FILE__, __LINE__, #expr,   \
                    nmz_status_str(s_), nmz_last_error());                                \
        }                                                                                 \
    } while (0)

#define CHECK_STATUS(expr, want)                                                          \
    do {                                                                                  \
        nmz_status s_ = (expr);                                                           \
        g_checks++;                                                                       \
        if (s_ != (want)) {                                                               \
            g_failures++;                                                                 \
            fprintf(stderr, "  FAIL %s:%d: %s -> %s, expected %s\n", __FILE__, __LINE__,  \
                    #expr, nmz_status_str(s_), nmz_status_str(want));                     \
        }                                                                                 \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                             \
    CHECK_MSG(fabs((double)(a) - (double)(b)) <= (tol), "%.9g vs %.9g (tol %g)",          \
              (double)(a), (double)(b), (double)(tol))

typedef void (*test_fn)(void);

/* Each test file exposes a table of tests. */
typedef struct test_case {
    const char *name;
    test_fn fn;
} test_case;

extern const test_case core_tests[];
extern const test_case math_tests[];
extern const test_case train_tests[];
extern const test_case io_tests[];
extern const test_case layer_tests[];

/* Scratch file path inside the build directory. */
const char *test_path(const char *name);
/* Print an informational line (shown in the test log). */
void test_note(const char *fmt, ...)
#if defined(__GNUC__)
    __attribute__((format(printf, 1, 2)))
#endif
    ;

#endif
