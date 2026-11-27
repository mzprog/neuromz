/* platform.c — everything the `neuromz` program needs from the operating
 * system, in one file, so the rest of the CLI (and the whole library) is
 * plain C11 (DESIGN decision 113):
 *
 *   terminal detection   isatty (POSIX)           _isatty (Windows CRT)
 *   terminal width       ioctl(TIOCGWINSZ)        GetConsoleScreenBufferInfo
 *   UTF-8 console        (the terminal's own)     SetConsoleOutputCP(CP_UTF8)
 *   UTF-8 argv           (already UTF-8)          GetCommandLineW → UTF-8
 *   file names           fopen                    _wfopen on the UTF-16 name
 *   absolute paths       realpath                 _wfullpath
 *   monotonic time       clock_gettime            QueryPerformanceCounter
 *
 * Anything else is the C standard library. */
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <io.h>
#include <shellapi.h>
#include <sys/stat.h>
#elif defined(__unix__) || defined(__APPLE__)
#define _XOPEN_SOURCE 700 /* realpath */
#define _DARWIN_C_SOURCE /* TIOCGWINSZ on macOS */
#include <limits.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#define NMZ_POSIX 1
#endif

#include "cli.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
/* UTF-8 → UTF-16 (malloc'd), NULL on failure */
static wchar_t *widen(const char *s) {
    int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, NULL, 0);
    wchar_t *w;
    if (n <= 0 || !(w = malloc((size_t)n * sizeof *w))) return NULL;
    MultiByteToWideChar(CP_UTF8, 0, s, -1, w, n);
    return w;
}

static char *narrow(const wchar_t *w) {
    int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, NULL, 0, NULL, NULL);
    char *s;
    if (n <= 0 || !(s = malloc((size_t)n))) return NULL;
    WideCharToMultiByte(CP_UTF8, 0, w, -1, s, n, NULL, NULL);
    return s;
}
#endif

double cli_now(void) {
#if defined(_WIN32)
    LARGE_INTEGER f, c;
    if (QueryPerformanceFrequency(&f) && QueryPerformanceCounter(&c)) return (double)c.QuadPart / (double)f.QuadPart;
#elif defined(NMZ_POSIX)
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t) == 0) return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
#endif
    return (double)clock() / CLOCKS_PER_SEC;
}

int cli_stdout_is_tty(void) {
#if defined(_WIN32)
    return _isatty(_fileno(stdout));
#elif defined(NMZ_POSIX)
    return isatty(STDOUT_FILENO);
#else
    return 0;
#endif
}

int cli_stdin_is_tty(void) {
#if defined(_WIN32)
    return _isatty(_fileno(stdin));
#elif defined(NMZ_POSIX)
    return isatty(STDIN_FILENO);
#else
    return 0;
#endif
}

int cli_term_width(void) {
    int w = 0;
#if defined(_WIN32)
    CONSOLE_SCREEN_BUFFER_INFO info;
    if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &info))
        w = info.srWindow.Right - info.srWindow.Left + 1;
#elif defined(NMZ_POSIX) && defined(TIOCGWINSZ)
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0) w = ws.ws_col;
#endif
    if (w <= 0) {
        const char *c = getenv("COLUMNS");
        w = c ? atoi(c) : 0;
    }
    return w >= 20 && w <= 1000 ? w : 80;
}

void cli_console_init(int *argc, char ***argv) {
#if defined(_WIN32)
    /* The console shows UTF-8 bytes as characters (Arabic, Greek, box
     * drawing) and the program receives its arguments as UTF-8, like on
     * every other system. Font support for Arabic is up to the console. */
    int n = 0, i;
    LPWSTR *wargv;
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    wargv = CommandLineToArgvW(GetCommandLineW(), &n);
    if (wargv && n == *argc) {
        char **u = calloc((size_t)n + 1, sizeof *u); /* kept for the life of the program */
        for (i = 0; u && i < n; i++)
            if (!(u[i] = narrow(wargv[i]))) u[i] = (*argv)[i];
        if (u) *argv = u;
    }
    if (wargv) LocalFree(wargv);
#else
    (void)argc;
    (void)argv;
#endif
}

FILE *cli_fopen(const char *path, const char *mode) {
#if defined(_WIN32)
    wchar_t *wp = widen(path), *wm = widen(mode);
    FILE *f = wp && wm ? _wfopen(wp, wm) : NULL;
    free(wp);
    free(wm);
    return f;
#else
    return fopen(path, mode);
#endif
}

int cli_file_exists(const char *path) {
#if defined(_WIN32)
    struct _stat st;
    wchar_t *wp = widen(path);
    int ok = wp && _wstat(wp, &st) == 0 && (st.st_mode & _S_IFREG);
    free(wp);
    return ok;
#elif defined(NMZ_POSIX)
    struct stat st;
    return stat(path, &st) == 0 && S_ISREG(st.st_mode);
#else
    FILE *f = fopen(path, "rb");
    if (f) fclose(f);
    return f != NULL;
#endif
}

char *cli_abspath(const char *path) {
#if defined(_WIN32)
    wchar_t *wp = widen(path), *full = wp ? _wfullpath(NULL, wp, 0) : NULL;
    char *s = full ? narrow(full) : NULL;
    free(wp);
    free(full);
    return s ? s : cli_strdup(path);
#elif defined(NMZ_POSIX)
    char *r = realpath(path, NULL);
    return r ? r : cli_strdup(path);
#else
    return cli_strdup(path);
#endif
}
