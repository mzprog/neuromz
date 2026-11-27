/* test_threads_util.c — run a function in several threads (Win32 or
 * pthreads), for the concurrency tests. The only file of the tests that
 * talks to the operating system's thread API. */
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <pthread.h>
#endif

#include <stdlib.h>

#include "test.h"

typedef struct start {
    test_thread_fn fn;
    void *arg;
} start;

#if defined(_WIN32)
static DWORD WINAPI trampoline(LPVOID p) {
    start *s = (start *)p;
    s->fn(s->arg);
    return 0;
}
#else
static void *trampoline(void *p) {
    start *s = (start *)p;
    s->fn(s->arg);
    return NULL;
}
#endif

int test_run_threads(int n, test_thread_fn fn, void **args) {
    start *st = calloc((size_t)n, sizeof *st);
    int i, ok = 1;
#if defined(_WIN32)
    HANDLE *h = calloc((size_t)n, sizeof *h);
    if (!st || !h) {
        free(st);
        free(h);
        return 0;
    }
    for (i = 0; i < n; i++) {
        st[i].fn = fn;
        st[i].arg = args[i];
        h[i] = CreateThread(NULL, 0, trampoline, &st[i], 0, NULL);
        if (!h[i]) ok = 0;
    }
    for (i = 0; i < n; i++)
        if (h[i]) {
            WaitForSingleObject(h[i], INFINITE);
            CloseHandle(h[i]);
        }
    free(h);
#else
    pthread_t *t = calloc((size_t)n, sizeof *t);
    int *started = calloc((size_t)n, sizeof *started);
    if (!st || !t || !started) {
        free(st);
        free(t);
        free(started);
        return 0;
    }
    for (i = 0; i < n; i++) {
        st[i].fn = fn;
        st[i].arg = args[i];
        started[i] = pthread_create(&t[i], NULL, trampoline, &st[i]) == 0;
        if (!started[i]) ok = 0;
    }
    for (i = 0; i < n; i++)
        if (started[i]) pthread_join(t[i], NULL);
    free(t);
    free(started);
#endif
    free(st);
    return ok;
}
