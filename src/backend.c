/* backend.c — the backend registry, selection, and the device copies of
 * tensors (DESIGN §11).
 *
 * Selection is process-wide: nmz_backend_select picks the backend every
 * net computes on from then on (a net notices at its next computation).
 * The CPU is the default, so a program that never selects anything runs
 * exactly as with v2.0.1. Device backends initialise lazily, the first
 * time they are selected or queried; until then nothing of theirs runs,
 * which is how a CUDA build works on a machine without a GPU or driver.
 *
 * A tensor's `data` is host memory. When a device backend computes on a
 * tensor, the tensor gets a device copy (struct nmz_tensor_dev) and a note
 * of which copy is newer. Asking for a pointer in some memory moves data
 * only when that copy is out of date (nmz__acquire), and a write makes the
 * other copy stale. Host code that keeps calling the accessors never sees
 * stale data, whichever backend ran last. */
#include "internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------- registry */

#define MAX_BACKENDS 8

typedef struct slot {
    const nmz_backend *be;
    int state; /* 0 = not tried, 1 = usable, -1 = unavailable (why) */
    char why[256];
} slot;

static slot g_slots[MAX_BACKENDS];
static int g_n, g_builtins;
static const nmz_backend *g_current; /* NULL: the CPU */
static char g_note[256] = "default (the CPU unless another backend is selected)";

static void add_slot(const nmz_backend *be) {
    int i;
    for (i = 0; i < g_n; i++)
        if (g_slots[i].be == be || strcmp(g_slots[i].be->name, be->name) == 0) return;
    if (g_n < MAX_BACKENDS) {
        g_slots[g_n].be = be;
        g_slots[g_n].state = be->host_memory ? 1 : 0;
        g_slots[g_n].why[0] = '\0';
        g_n++;
    }
}

static void builtins(void) {
    if (g_builtins) return;
    g_builtins = 1;
    add_slot(&nmz__cpu_backend);
#ifdef NMZ_HAVE_CUDA
    add_slot(nmz__cuda_backend());
#endif
    add_slot(&nmz__cpu_generic_backend);
}

static slot *find(const char *name) {
    int i;
    builtins();
    for (i = 0; i < g_n; i++)
        if (strcmp(g_slots[i].be->name, name) == 0) return &g_slots[i];
    return NULL;
}

nmz_status nmz_backend_register(const nmz_backend *be) {
    builtins();
    if (!be || !be->name || be->abi != NMZ_BACKEND_ABI)
        return nmz__fail(NMZ_ERR_ARG, "backend table is NULL or built for another ABI (expected %d)", NMZ_BACKEND_ABI);
    if (find(be->name)) return nmz__fail(NMZ_ERR_ARG, "a backend named '%s' is already registered", be->name);
    if (g_n >= MAX_BACKENDS) return nmz__fail(NMZ_ERR_NOMEM, "too many backends");
    add_slot(be);
    return NMZ_OK;
}

int nmz_backend_count(void) {
    builtins();
    return g_n;
}

const char *nmz_backend_name(int i) {
    builtins();
    return i >= 0 && i < g_n ? g_slots[i].be->name : NULL;
}

/* Initialise once; remember the answer (a missing driver will not appear
 * later in the same process). */
static int usable(slot *sl) {
    if (sl->state == 0) {
        nmz_status s = sl->be->init ? sl->be->init(sl->be, sl->why, sizeof sl->why) : NMZ_OK;
        sl->state = s == NMZ_OK ? 1 : -1;
        if (s != NMZ_OK && !sl->why[0]) snprintf(sl->why, sizeof sl->why, "initialisation failed");
    }
    return sl->state == 1;
}

const nmz_backend *nmz_backend_current(void) { return g_current ? g_current : &nmz__cpu_backend; }

nmz_status nmz_backend_select(const char *name) {
    slot *sl;
    if (!name) return nmz__fail(NMZ_ERR_ARG, "backend name is NULL");
    builtins();
    if (strcmp(name, "auto") == 0) {
        sl = find("cuda");
        if (!sl) {
            g_current = NULL;
            snprintf(g_note, sizeof g_note, "auto: this library was built without CUDA, using the CPU");
        } else if (usable(sl)) {
            nmz_device_info di;
            memset(&di, 0, sizeof di);
            sl->be->info(sl->be, &di);
            g_current = sl->be;
            snprintf(g_note, sizeof g_note, "auto: CUDA device found (%s)", di.device);
        } else {
            g_current = NULL;
            snprintf(g_note, sizeof g_note, "auto: no usable CUDA device (%.200s), using the CPU", sl->why);
        }
        return NMZ_OK;
    }
    sl = find(name);
    if (!sl) {
        if (strcmp(name, "cuda") == 0)
            return nmz__fail(NMZ_ERR_UNSUPPORTED, "this library was built without CUDA support (configure it with "
                                                  "-DNMZ_CUDA=ON)");
        return nmz__fail(NMZ_ERR_ARG, "unknown backend '%s' (expected auto, cpu or cuda)", name);
    }
    if (!usable(sl)) return nmz__fail(NMZ_ERR_UNSUPPORTED, "the %s backend cannot be used: %s", name, sl->why);
    g_current = sl->be == &nmz__cpu_backend ? NULL : sl->be;
    snprintf(g_note, sizeof g_note, "requested");
    return NMZ_OK;
}

nmz_status nmz_backend_memory_limit(size_t bytes) {
    const nmz_backend *be = nmz_backend_current();
    if (be->host_memory || !be->memory_limit) return NMZ_OK;
    return be->memory_limit(be, bytes);
}

static void fill_info(slot *sl, nmz_device_info *out) {
    memset(out, 0, sizeof *out);
    snprintf(out->backend, sizeof out->backend, "%s", sl->be->name);
    if (sl->be->info) sl->be->info(sl->be, out);
}

nmz_status nmz_backend_info(nmz_device_info *out) {
    const nmz_backend *be = nmz_backend_current();
    slot *sl;
    if (!out) return nmz__fail(NMZ_ERR_ARG, "out is NULL");
    sl = find(be->name);
    fill_info(sl, out);
    snprintf(out->note, sizeof out->note, "%s", g_note);
    return NMZ_OK;
}

nmz_status nmz_backend_query(const char *name, nmz_device_info *out) {
    slot *sl;
    if (!name || !out) return nmz__fail(NMZ_ERR_ARG, "NULL argument");
    sl = find(name);
    if (!sl) {
        memset(out, 0, sizeof *out);
        snprintf(out->backend, sizeof out->backend, "%.15s", name);
        snprintf(out->note, sizeof out->note, "%s",
                 strcmp(name, "cuda") == 0 ? "not built in (configure with -DNMZ_CUDA=ON)" : "unknown backend");
        return nmz__fail(NMZ_ERR_UNSUPPORTED, "backend '%s': %s", name, out->note);
    }
    if (!usable(sl)) {
        memset(out, 0, sizeof *out);
        snprintf(out->backend, sizeof out->backend, "%s", sl->be->name);
        snprintf(out->note, sizeof out->note, "%s", sl->why);
        return nmz__fail(NMZ_ERR_UNSUPPORTED, "the %s backend cannot be used: %s", name, sl->why);
    }
    fill_info(sl, out);
    snprintf(out->note, sizeof out->note, "%s", nmz_backend_current() == sl->be ? g_note : "available");
    return NMZ_OK;
}

/* --------------------------------------------------------- device copies */

static void download_now(nmz_tensor *t) {
    struct nmz_tensor_dev *d = t->dev;
    if (d->where == NMZ__DEVICE && t->size && d->ptr) {
        if (d->be->download(d->be, t->data, d->ptr, t->size * sizeof(float)) != NMZ_OK) return; /* message set */
        d->where = NMZ__BOTH;
    }
}

float *nmz__acquire(const nmz_backend *be, nmz_tensor *t, unsigned mode) {
    struct nmz_tensor_dev *d;
    if (!t) return NULL;
    d = t->dev;
    if (be->host_memory) {
        if (!d) return t->data;
        if (mode != NMZ_WRITE) download_now(t);
        if (mode != NMZ_READ) d->where = NMZ__HOST;
        return t->data;
    }
    if (d && d->be != be) { /* computed on another device before: via the host */
        nmz__tensor_to_host(t);
        d = NULL;
    }
    if (!d) {
        d = calloc(1, sizeof *d);
        if (!d) {
            nmz__fail(NMZ_ERR_NOMEM, "out of memory");
            return NULL;
        }
        d->be = be;
        d->where = NMZ__HOST;
        t->dev = d;
    }
    if (!d->ptr || t->size > d->cap) {
        size_t n = t->size ? t->size : 1;
        float *p;
        if (d->ptr) {
            be->release(be, d->ptr);
            d->ptr = NULL;
            d->cap = 0;
        }
        p = be->alloc(be, n * sizeof(float));
        if (!p) return NULL; /* the backend said why */
        d->ptr = p;
        d->cap = n;
        if (d->where == NMZ__BOTH) d->where = NMZ__HOST; /* the new memory holds nothing yet */
    }
    if (mode != NMZ_WRITE && d->where == NMZ__HOST && t->size) {
        if (be->upload(be, d->ptr, t->data, t->size * sizeof(float)) != NMZ_OK) return NULL;
        d->where = NMZ__BOTH;
    }
    if (mode != NMZ_READ) d->where = NMZ__DEVICE;
    return d->ptr;
}

float *nmz_tensor_ptr(const nmz_backend *be, nmz_tensor *t, unsigned mode) {
    if (!be || !t) return NULL;
    if (mode == NMZ_READ) return (float *)nmz__r(be, t);
    return mode == NMZ_WRITE ? nmz__w(be, t) : nmz__u(be, t);
}

void nmz__tensor_to_host(nmz_tensor *t) { nmz__tensor_drop_device(t, 1); }

nmz_status nmz__tensor_set(nmz_tensor *t, size_t i, float v) {
    struct nmz_tensor_dev *d;
    if (!t || i >= t->size) return nmz__fail(NMZ_ERR_ARG, "tensor index out of range");
    d = t->dev;
    if (d && d->where == NMZ__BOTH && d->ptr) { /* both copies current: change one number in each */
        nmz_status s;
        t->data[i] = v;
        if ((s = d->be->upload(d->be, d->ptr + i, &t->data[i], sizeof(float))) != NMZ_OK) return s;
        return NMZ_OK;
    }
    nmz__hu(t)[i] = v;
    return NMZ_OK;
}

nmz_status nmz__zero_on(const nmz_backend *be, nmz_tensor *t) {
    float *p;
    if (!t) return NMZ_OK;
    if (be->host_memory && !t->dev) {
        nmz_tensor_zero(t);
        return NMZ_OK;
    }
    if (!t->size) return NMZ_OK;
    if (!(p = nmz__w(be, t))) return nmz__op_failed();
    return be->fill(be, t->size, 0.0f, p);
}

void nmz__tensor_drop_device(nmz_tensor *t, int keep) {
    struct nmz_tensor_dev *d;
    if (!t || !(d = t->dev)) return;
    if (keep) download_now(t);
    if (d->ptr) d->be->release(d->be, d->ptr);
    free(d);
    t->dev = NULL;
}
