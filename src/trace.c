/* trace.c — trace events: emission, lookup, JSON-lines export, recorder.
 *
 * Cost when tracing is off: layers test one pointer (NMZ_TRACING) and skip
 * everything else. No allocation happens unless a recorder is used. */
#include "internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

void nmz_trace_emit(const nmz_ctx *ctx, nmz_trace_phase phase, const char *layer_type,
                    const char *detail, int n_scalars, const nmz_trace_scalar *scalars,
                    int n_items, const nmz_trace_item *items) {
    nmz_trace_event ev;
    int i;
    if (!NMZ_TRACING(ctx)) return;
    memset(&ev, 0, sizeof ev);
    ev.phase = phase;
    ev.step = ctx->step;
    ev.layer = ctx->layer_index;
    ev.time_step = -1;
    ev.layer_type = layer_type ? layer_type : "";
    ev.detail = detail ? detail : "";
    if (n_scalars > NMZ_TRACE_MAX_SCALARS) n_scalars = NMZ_TRACE_MAX_SCALARS;
    if (n_items > NMZ_TRACE_MAX_ITEMS) n_items = NMZ_TRACE_MAX_ITEMS;
    ev.n_scalars = n_scalars;
    for (i = 0; i < n_scalars; i++) ev.scalars[i] = scalars[i];
    ev.n_items = n_items;
    for (i = 0; i < n_items; i++) ev.items[i] = items[i];
    ctx->trace(&ev, ctx->trace_user);
}

const char *nmz_trace_phase_name(nmz_trace_phase p) {
    switch (p) {
    case NMZ_TRACE_FORWARD: return "forward";
    case NMZ_TRACE_LOSS: return "loss";
    case NMZ_TRACE_BACKWARD: return "backward";
    case NMZ_TRACE_UPDATE: return "update";
    }
    return "?";
}

const nmz_tensor *nmz_trace_find(const nmz_trace_event *ev, const char *name) {
    int i;
    if (!ev || !name) return NULL;
    for (i = 0; i < ev->n_items; i++)
        if (ev->items[i].name && strcmp(ev->items[i].name, name) == 0) return ev->items[i].t;
    return NULL;
}

int nmz_trace_scalar_get(const nmz_trace_event *ev, const char *name, double *out) {
    int i;
    if (!ev || !name) return 0;
    for (i = 0; i < ev->n_scalars; i++)
        if (ev->scalars[i].name && strcmp(ev->scalars[i].name, name) == 0) {
            if (out) *out = ev->scalars[i].value;
            return 1;
        }
    return 0;
}

/* ------------------------------------------------------------------ JSON */

static void json_string(FILE *f, const char *s) {
    fputc('"', f);
    for (; s && *s; s++) {
        unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\')
            fprintf(f, "\\%c", c);
        else if (c < 0x20)
            fprintf(f, "\\u%04x", c);
        else
            fputc(c, f);
    }
    fputc('"', f);
}

static void json_number(FILE *f, double v) {
    if (isfinite(v))
        fprintf(f, "%.9g", v);
    else
        fputs("null", f); /* JSON has no NaN/Inf */
}

nmz_status nmz_trace_event_write_json(const nmz_trace_event *ev, FILE *f) {
    int i, d;
    size_t e;
    if (!ev || !f) return nmz__fail(NMZ_ERR_ARG, "NULL argument");
    fprintf(f, "{\"step\":%ld,\"phase\":", ev->step);
    json_string(f, nmz_trace_phase_name(ev->phase));
    fprintf(f, ",\"layer\":%d,\"time_step\":%d,\"type\":", ev->layer, ev->time_step);
    json_string(f, ev->layer_type);
    fputs(",\"detail\":", f);
    json_string(f, ev->detail);
    fputs(",\"scalars\":{", f);
    for (i = 0; i < ev->n_scalars; i++) {
        if (i) fputc(',', f);
        json_string(f, ev->scalars[i].name);
        fputc(':', f);
        json_number(f, ev->scalars[i].value);
    }
    fputs("},\"items\":{", f);
    for (i = 0; i < ev->n_items; i++) {
        const nmz_tensor *t = ev->items[i].t;
        if (i) fputc(',', f);
        json_string(f, ev->items[i].name);
        fputs(":{\"shape\":[", f);
        for (d = 0; t && d < t->rank; d++) fprintf(f, d ? ",%d" : "%d", t->shape[d]);
        fputs("],\"data\":[", f);
        for (e = 0; t && e < t->size; e++) {
            if (e) fputc(',', f);
            json_number(f, t->data[e]);
        }
        fputs("]}", f);
    }
    fputs("}}\n", f);
    return ferror(f) ? nmz__fail(NMZ_ERR_IO, "write error") : NMZ_OK;
}

/* -------------------------------------------------------------- recorder */

typedef struct rec_event {
    nmz_trace_event ev;
    nmz_tensor tensors[NMZ_TRACE_MAX_ITEMS];
    char *strings[2 + NMZ_TRACE_MAX_ITEMS + NMZ_TRACE_MAX_SCALARS];
} rec_event;

/* Events are individually allocated: items point into their own rec_event,
 * so the event must never move after it is filled. */
struct nmz_trace_recorder {
    rec_event **events;
    size_t count, cap;
    nmz_status status;
};

nmz_status nmz_trace_recorder_create(nmz_trace_recorder **out) {
    if (!out) return nmz__fail(NMZ_ERR_ARG, "out is NULL");
    *out = calloc(1, sizeof **out);
    return *out ? NMZ_OK : nmz__fail(NMZ_ERR_NOMEM, "out of memory");
}

static void rec_event_free(rec_event *r) {
    size_t i;
    for (i = 0; i < NMZ_TRACE_MAX_ITEMS; i++) nmz_tensor_free(&r->tensors[i]);
    for (i = 0; i < sizeof r->strings / sizeof r->strings[0]; i++) free(r->strings[i]);
}

void nmz_trace_recorder_clear(nmz_trace_recorder *rec) {
    size_t i;
    if (!rec) return;
    for (i = 0; i < rec->count; i++) {
        rec_event_free(rec->events[i]);
        free(rec->events[i]);
    }
    rec->count = 0;
    rec->status = NMZ_OK;
}

void nmz_trace_recorder_free(nmz_trace_recorder *rec) {
    if (!rec) return;
    nmz_trace_recorder_clear(rec);
    free(rec->events);
    free(rec);
}

static int copy_str(rec_event *r, int slot, const char *s, const char **dst) {
    r->strings[slot] = nmz__strdup(s ? s : "");
    *dst = r->strings[slot];
    return r->strings[slot] != NULL;
}

void nmz_trace_recorder_fn(const nmz_trace_event *ev, void *user) {
    nmz_trace_recorder *rec = user;
    rec_event *r;
    int i, ok = 1;
    if (!rec || !ev || rec->status != NMZ_OK) return;
    if (rec->count == rec->cap) {
        size_t cap = rec->cap ? rec->cap * 2 : 64;
        rec_event **p = realloc(rec->events, cap * sizeof *p);
        if (!p) {
            rec->status = NMZ_ERR_NOMEM;
            return;
        }
        rec->events = p;
        rec->cap = cap;
    }
    r = calloc(1, sizeof *r);
    if (!r) {
        rec->status = NMZ_ERR_NOMEM;
        return;
    }
    r->ev = *ev;
    ok &= copy_str(r, 0, ev->layer_type, &r->ev.layer_type);
    ok &= copy_str(r, 1, ev->detail, &r->ev.detail);
    for (i = 0; i < ev->n_items && ok; i++) {
        ok &= copy_str(r, 2 + i, ev->items[i].name, &r->ev.items[i].name);
        if (ev->items[i].t) {
            ok &= nmz_tensor_copy(&r->tensors[i], ev->items[i].t) == NMZ_OK;
            r->ev.items[i].t = &r->tensors[i];
        }
    }
    for (i = 0; i < ev->n_scalars && ok; i++)
        ok &= copy_str(r, 2 + NMZ_TRACE_MAX_ITEMS + i, ev->scalars[i].name, &r->ev.scalars[i].name);
    if (!ok) {
        rec_event_free(r);
        free(r);
        rec->status = NMZ_ERR_NOMEM;
        return;
    }
    rec->events[rec->count++] = r;
}

size_t nmz_trace_recorder_count(const nmz_trace_recorder *rec) { return rec ? rec->count : 0; }

const nmz_trace_event *nmz_trace_recorder_get(const nmz_trace_recorder *rec, size_t i) {
    if (!rec || i >= rec->count) return NULL;
    return &rec->events[i]->ev;
}

nmz_status nmz_trace_recorder_status(const nmz_trace_recorder *rec) {
    return rec ? rec->status : NMZ_ERR_ARG;
}
