/* io.c — portable .nmz model files (docs/DESIGN.md §7).
 *
 * Every number is written byte by byte in little-endian order, so files are
 * identical on every platform; nothing is written with fwrite(&struct). */
#include "internal.h"

#include <errno.h>
#include <float.h>
#include <stdlib.h>
#include <string.h>

/* The format assumes IEEE-754 binary32/binary64. */
typedef char nmz_static_assert_float[(sizeof(float) == 4 && FLT_MANT_DIG == 24) ? 1 : -1];
typedef char nmz_static_assert_double[(sizeof(double) == 8 && DBL_MANT_DIG == 53) ? 1 : -1];

#define NMZ_FORMAT_VERSION 3u  /* written; 2 and 3 are read */
#define NMZ_MAX_SECTIONS 64u
#define NMZ_MAX_STRING 4096u
#define NMZ_MAX_LAYERS 100000u
#define NMZ_MAX_CLASSES 100000u

static const unsigned char k_magic[8] = {0x89, 'N', 'M', 'Z', '\r', '\n', 0x1a, '\n'};

/* ------------------------------------------------------------------ CRC-32 */

uint32_t nmz__crc32(uint32_t crc, const void *data, size_t n) {
    static uint32_t table[256];
    static int ready = 0; /* benign race: every thread computes the same table */
    const unsigned char *p = data;
    size_t i;
    if (!ready) {
        uint32_t k, c;
        int b;
        for (k = 0; k < 256; k++) {
            c = k;
            for (b = 0; b < 8; b++) c = (c & 1u) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            table[k] = c;
        }
        ready = 1;
    }
    crc = ~crc;
    for (i = 0; i < n; i++) crc = table[(crc ^ p[i]) & 0xffu] ^ (crc >> 8);
    return ~crc;
}

/* ------------------------------------------------------------ byte writer */

/* A writer with f == NULL only counts bytes (used to size layer configs). */
typedef struct writer_ext {
    struct nmz_writer w;
    size_t count;
} writer_ext;

static nmz_status put_bytes(nmz_writer *w, const void *p, size_t n) {
    writer_ext *x = (writer_ext *)w;
    if (w->err != NMZ_OK) return w->err;
    x->count += n;
    if (!w->f) return NMZ_OK;
    if (fwrite(p, 1, n, w->f) != n) {
        w->err = nmz__fail(NMZ_ERR_IO, "write failed: %s", strerror(errno));
        return w->err;
    }
    w->crc = nmz__crc32(w->crc, p, n);
    return NMZ_OK;
}

nmz_status nmz_write_u32(nmz_writer *w, uint32_t v) {
    unsigned char b[4];
    b[0] = (unsigned char)v;
    b[1] = (unsigned char)(v >> 8);
    b[2] = (unsigned char)(v >> 16);
    b[3] = (unsigned char)(v >> 24);
    return put_bytes(w, b, 4);
}

static nmz_status write_u64(nmz_writer *w, uint64_t v) {
    nmz_write_u32(w, (uint32_t)v);
    return nmz_write_u32(w, (uint32_t)(v >> 32));
}

nmz_status nmz_write_f32(nmz_writer *w, float v) {
    uint32_t u;
    memcpy(&u, &v, 4);
    return nmz_write_u32(w, u);
}

nmz_status nmz_write_f64(nmz_writer *w, double v) {
    uint64_t u;
    memcpy(&u, &v, 8);
    return write_u64(w, u);
}

static nmz_status write_str(nmz_writer *w, const char *s) {
    size_t n = strlen(s);
    if (n > NMZ_MAX_STRING) return w->err = nmz__fail(NMZ_ERR_ARG, "string too long to save");
    nmz_write_u32(w, (uint32_t)n);
    return put_bytes(w, s, n);
}

/* ------------------------------------------------------------ byte reader */

typedef struct reader_ext {
    struct nmz_reader r;
    size_t count;
} reader_ext;

static nmz_status get_bytes(nmz_reader *r, void *p, size_t n) {
    reader_ext *x = (reader_ext *)r;
    if (r->err != NMZ_OK) return r->err;
    if (fread(p, 1, n, r->f) != n) {
        r->err = nmz__fail(NMZ_ERR_FORMAT, "unexpected end of file (truncated model?)");
        return r->err;
    }
    r->crc = nmz__crc32(r->crc, p, n);
    x->count += n;
    return NMZ_OK;
}

nmz_status nmz_read_u32(nmz_reader *r, uint32_t *v) {
    unsigned char b[4];
    if (get_bytes(r, b, 4) != NMZ_OK) return r->err;
    *v = (uint32_t)b[0] | (uint32_t)b[1] << 8 | (uint32_t)b[2] << 16 | (uint32_t)b[3] << 24;
    return NMZ_OK;
}

static nmz_status read_u64(nmz_reader *r, uint64_t *v) {
    uint32_t lo = 0, hi = 0;
    nmz_read_u32(r, &lo);
    if (nmz_read_u32(r, &hi) != NMZ_OK) return r->err;
    *v = (uint64_t)hi << 32 | lo;
    return NMZ_OK;
}

nmz_status nmz_read_f32(nmz_reader *r, float *v) {
    uint32_t u = 0;
    if (nmz_read_u32(r, &u) != NMZ_OK) return r->err;
    memcpy(v, &u, 4);
    return NMZ_OK;
}

nmz_status nmz_read_f64(nmz_reader *r, double *v) {
    uint64_t u = 0;
    if (read_u64(r, &u) != NMZ_OK) return r->err;
    memcpy(v, &u, 8);
    return NMZ_OK;
}

static nmz_status read_str(nmz_reader *r, char **out) {
    uint32_t n = 0;
    char *s;
    *out = NULL;
    if (nmz_read_u32(r, &n) != NMZ_OK) return r->err;
    if (n > NMZ_MAX_STRING) return r->err = nmz__fail(NMZ_ERR_FORMAT, "corrupt string length %u", n);
    s = malloc((size_t)n + 1);
    if (!s) return r->err = nmz__fail(NMZ_ERR_NOMEM, "out of memory");
    if (get_bytes(r, s, n) != NMZ_OK) {
        free(s);
        return r->err;
    }
    s[n] = '\0';
    *out = s;
    return NMZ_OK;
}

/* --------------------------------------------------------------- registry */

static const nmz_layer_ops *const k_registry[] = {&nmz__dense_ops,     &nmz__activation_ops, &nmz__conv2d_ops,
                                                  &nmz__maxpool2d_ops, &nmz__flatten_ops,    &nmz__dropout_ops};

const nmz_layer_ops *nmz_layer_ops_find(const char *type) {
    size_t i;
    if (!type) return NULL;
    for (i = 0; i < sizeof k_registry / sizeof k_registry[0]; i++)
        if (strcmp(k_registry[i]->type, type) == 0) return k_registry[i];
    return NULL;
}

/* ------------------------------------------------------------------- save */

/* ------------------------------------------------- optional sections (format 3)
 * u32 count, then per section: string tag, u32 byte length, bytes.
 * Readers skip tags they do not know, so new sections never break old files. */

static void sec_normalize(nmz_net *net, nmz_writer *w) {
    int g;
    nmz_write_u32(w, (uint32_t)net->norm);
    nmz_write_u32(w, (uint32_t)net->norm_fitted);
    nmz_write_u32(w, (uint32_t)(net->norm_fitted ? net->norm_groups : 0));
    for (g = 0; net->norm_fitted && g < net->norm_groups; g++) nmz_write_f32(w, net->norm_offset[g]);
    for (g = 0; net->norm_fitted && g < net->norm_groups; g++) nmz_write_f32(w, net->norm_scale[g]);
}

static void sec_optimizer(nmz_net *net, nmz_writer *w) {
    int k;
    size_t e;
    nmz_write_u32(w, (uint32_t)net->opt_kind);
    write_u64(w, (uint64_t)net->opt_t);
    nmz_write_u32(w, (uint32_t)net->n_params);
    for (k = 0; k < net->n_params; k++) {
        nmz_write_u32(w, (uint32_t)net->opt_m[k].size);
        for (e = 0; e < net->opt_m[k].size; e++) nmz_write_f32(w, net->opt_m[k].data[e]);
        if (net->opt_kind == NMZ_OPT_ADAM)
            for (e = 0; e < net->opt_v[k].size; e++) nmz_write_f32(w, net->opt_v[k].data[e]);
    }
}

static void write_section(nmz_net *net, nmz_writer *w, const char *tag, void (*fn)(nmz_net *, nmz_writer *)) {
    writer_ext counter;
    memset(&counter, 0, sizeof counter);
    fn(net, &counter.w); /* measure */
    write_str(w, tag);
    nmz_write_u32(w, (uint32_t)counter.count);
    fn(net, w);
}

static nmz_status save_body(nmz_net *net, nmz_writer *w, unsigned flags) {
    int i, j, d;
    size_t e;
    const nmz_train_opts *o = &net->train;
    put_bytes(w, k_magic, sizeof k_magic);
    nmz_write_u32(w, NMZ_FORMAT_VERSION);
    nmz_write_u32(w, 0); /* flags */
    nmz_write_u32(w, (uint32_t)net->in_rank);
    for (i = 0; i < net->in_rank; i++) nmz_write_u32(w, (uint32_t)net->in_shape[i]);
    nmz_write_u32(w, (uint32_t)net->loss);
    write_u64(w, net->rng.state);
    write_u64(w, net->rng.inc);
    write_u64(w, (uint64_t)net->step);
    nmz_write_u32(w, (uint32_t)o->max_epochs);
    nmz_write_u32(w, (uint32_t)o->batch_size);
    nmz_write_f32(w, o->learning_rate);
    nmz_write_f32(w, o->momentum);
    nmz_write_u32(w, (uint32_t)(o->shuffle != 0));
    nmz_write_f64(w, o->target_loss);
    /* format 3: optimizer and schedule */
    nmz_write_u32(w, (uint32_t)o->optimizer);
    nmz_write_f32(w, o->beta1);
    nmz_write_f32(w, o->beta2);
    nmz_write_f32(w, o->epsilon);
    nmz_write_u32(w, (uint32_t)o->schedule);
    nmz_write_u32(w, (uint32_t)o->step_size);
    nmz_write_f32(w, o->gamma);
    nmz_write_f32(w, o->min_lr);
    nmz_write_u32(w, (uint32_t)net->n_classes);
    for (i = 0; i < net->n_classes; i++) write_str(w, net->class_names[i]);
    nmz_write_u32(w, (uint32_t)net->n_layers);
    for (i = 0; i < net->n_layers; i++) {
        nmz_layer *l = net->layers[i];
        writer_ext counter;
        int np = nmz_layer_param_count(l);
        memset(&counter, 0, sizeof counter);
        l->ops->save_config(l, &counter.w); /* first pass: measure */
        if (counter.w.err != NMZ_OK) return w->err = counter.w.err;
        write_str(w, l->ops->type);
        nmz_write_u32(w, (uint32_t)counter.count);
        l->ops->save_config(l, w);
        nmz_write_u32(w, (uint32_t)np);
        for (j = 0; j < np; j++) {
            const nmz_tensor *v = nmz_layer_param_value(l, j);
            write_str(w, nmz_layer_param_name(l, j));
            nmz_write_u32(w, (uint32_t)v->rank);
            for (d = 0; d < v->rank; d++) nmz_write_u32(w, (uint32_t)v->shape[d]);
            for (e = 0; e < v->size; e++) nmz_write_f32(w, v->data[e]);
        }
    }
    {
        int with_norm = net->norm != NMZ_NORM_NONE;
        int with_opt = net->opt_kind >= 0 && !(flags & NMZ_SAVE_NO_OPTIMIZER);
        nmz_write_u32(w, (uint32_t)(with_norm + with_opt));
        if (with_norm) write_section(net, w, "normalize", sec_normalize);
        if (with_opt) write_section(net, w, "optimizer", sec_optimizer);
    }
    return nmz_write_u32(w, w->crc); /* footer: CRC of everything above */
}

nmz_status nmz_net_save(nmz_net *net, const char *path) { return nmz_net_save_ex(net, path, 0); }

nmz_status nmz_net_save_ex(nmz_net *net, const char *path, unsigned flags) {
    writer_ext x;
    char *tmp;
    size_t n;
    nmz_status s;
    if (!net || !path || !*path) return nmz__fail(NMZ_ERR_ARG, "net or path is NULL/empty");
    if ((s = nmz_net_build(net)) != NMZ_OK) return s;
    n = strlen(path);
    tmp = malloc(n + 5);
    if (!tmp) return nmz__fail(NMZ_ERR_NOMEM, "out of memory");
    memcpy(tmp, path, n);
    memcpy(tmp + n, ".tmp", 5);
    memset(&x, 0, sizeof x);
    x.w.f = fopen(tmp, "wb");
    if (!x.w.f) {
        s = nmz__fail(NMZ_ERR_IO, "cannot write '%s': %s", tmp, strerror(errno));
        free(tmp);
        return s;
    }
    s = save_body(net, &x.w, flags);
    if (fclose(x.w.f) != 0 && s == NMZ_OK) s = nmz__fail(NMZ_ERR_IO, "write failed: %s", strerror(errno));
    /* write to a temp file, then rename: a failed save never destroys a good model */
    if (s == NMZ_OK && rename(tmp, path) != 0) {
        remove(path); /* rename() does not overwrite on some platforms */
        if (rename(tmp, path) != 0) s = nmz__fail(NMZ_ERR_IO, "cannot replace '%s': %s", path, strerror(errno));
    }
    if (s != NMZ_OK) remove(tmp);
    free(tmp);
    return s;
}

/* ------------------------------------------------------------------- load */

static nmz_status read_normalize(nmz_reader *r, nmz_net *net) {
    uint32_t kind = 0, fitted = 0, groups = 0, g;
    nmz_read_u32(r, &kind);
    nmz_read_u32(r, &fitted);
    if (nmz_read_u32(r, &groups) != NMZ_OK) return r->err;
    if (kind >= NMZ_NORM_COUNT || fitted > 1 || (fitted && (int)groups != nmz__norm_group_count(net)) ||
        (!fitted && groups != 0))
        return r->err = nmz__fail(NMZ_ERR_FORMAT, "corrupt normalize section");
    net->norm = (nmz_norm)kind;
    if (fitted) {
        if ((r->err = nmz__norm_alloc(net, (int)groups)) != NMZ_OK) return r->err;
        for (g = 0; g < groups; g++) nmz_read_f32(r, &net->norm_offset[g]);
        for (g = 0; g < groups; g++) nmz_read_f32(r, &net->norm_scale[g]);
        net->norm_fitted = 1;
    }
    return r->err;
}

static nmz_status read_optimizer(nmz_reader *r, nmz_net *net) {
    uint32_t kind = 0, n = 0, count = 0;
    uint64_t t = 0;
    int k;
    size_t e;
    nmz_read_u32(r, &kind);
    read_u64(r, &t);
    if (nmz_read_u32(r, &n) != NMZ_OK) return r->err;
    if (kind >= NMZ_OPT_COUNT || (int)n != net->n_params || t > 0x7fffffffu)
        return r->err = nmz__fail(NMZ_ERR_FORMAT, "corrupt optimizer section");
    for (k = 0; k < net->n_params; k++) {
        if (nmz_read_u32(r, &count) != NMZ_OK) return r->err;
        if (count != net->opt_m[k].size) return r->err = nmz__fail(NMZ_ERR_FORMAT, "optimizer state does not match the layers");
        for (e = 0; e < count; e++) nmz_read_f32(r, &net->opt_m[k].data[e]);
        if (kind == NMZ_OPT_ADAM)
            for (e = 0; e < count; e++) nmz_read_f32(r, &net->opt_v[k].data[e]);
    }
    if (r->err == NMZ_OK) {
        net->opt_kind = (int)kind;
        net->opt_t = (long)t;
    }
    return r->err;
}

static nmz_status read_sections(nmz_reader *r, nmz_net *net) {
    uint32_t n = 0, i, len = 0;
    if (nmz_read_u32(r, &n) != NMZ_OK) return r->err;
    if (n > NMZ_MAX_SECTIONS) return r->err = nmz__fail(NMZ_ERR_FORMAT, "corrupt section count %u", n);
    for (i = 0; i < n; i++) {
        char *tag = NULL;
        size_t before;
        if (read_str(r, &tag) != NMZ_OK) return r->err;
        if (nmz_read_u32(r, &len) != NMZ_OK) {
            free(tag);
            return r->err;
        }
        before = ((reader_ext *)r)->count;
        if (!strcmp(tag, "normalize"))
            read_normalize(r, net);
        else if (!strcmp(tag, "optimizer"))
            read_optimizer(r, net);
        else { /* unknown: skip (written by a newer NeuroMZ) */
            unsigned char buf[256];
            uint32_t left = len;
            while (left > 0 && r->err == NMZ_OK) {
                uint32_t chunk = left < sizeof buf ? left : (uint32_t)sizeof buf;
                get_bytes(r, buf, chunk);
                left -= chunk;
            }
        }
        if (r->err == NMZ_OK && ((reader_ext *)r)->count - before != len)
            r->err = nmz__fail(NMZ_ERR_FORMAT, "section '%s' has the wrong length", tag);
        free(tag);
        if (r->err != NMZ_OK) return r->err;
    }
    return NMZ_OK;
}

static nmz_status load_body(nmz_reader *r, nmz_net **out) {
    unsigned char magic[8] = {0};
    uint32_t version = 0, flags = 0, in_rank = 0, loss = 0, n_classes = 0, n_layers = 0, u = 0;
    uint64_t st = 0, inc = 0, step = 0;
    int in_shape[NMZ_MAX_RANK];
    nmz_train_opts o;
    nmz_net *net = NULL;
    nmz_status s;
    uint32_t i, j, d;
    size_t e;

    if (get_bytes(r, magic, sizeof magic) != NMZ_OK || memcmp(magic, k_magic, sizeof magic) != 0) {
        if (memcmp(magic, "1.", 2) == 0)
            return nmz__fail(NMZ_ERR_FORMAT, "this looks like a NeuroMZ v1 file; v1 files cannot be loaded by v2");
        return nmz__fail(NMZ_ERR_FORMAT, "not a NeuroMZ model file (bad magic number)");
    }
    nmz_read_u32(r, &version);
    nmz_read_u32(r, &flags);
    if (r->err) return r->err;
    if (version < 2 || version > NMZ_FORMAT_VERSION)
        return nmz__fail(NMZ_ERR_FORMAT, "unsupported format version %u (this build reads versions 2 to %u)",
                         version, NMZ_FORMAT_VERSION);
    if (flags != 0) return nmz__fail(NMZ_ERR_FORMAT, "unsupported format flags 0x%x", flags);
    if (nmz_read_u32(r, &in_rank) != NMZ_OK) return r->err;
    if (in_rank < 1 || in_rank > NMZ_MAX_RANK - 1) return nmz__fail(NMZ_ERR_FORMAT, "corrupt input rank %u", in_rank);
    for (i = 0; i < in_rank; i++) {
        if (nmz_read_u32(r, &u) != NMZ_OK) return r->err;
        if (u == 0 || u > NMZ_MAX_DIM) return nmz__fail(NMZ_ERR_FORMAT, "corrupt input shape");
        in_shape[i] = (int)u;
    }
    nmz_read_u32(r, &loss);
    read_u64(r, &st);
    read_u64(r, &inc);
    read_u64(r, &step);
    o = nmz_train_defaults(); /* format 2 has no optimizer/schedule fields */
    nmz_read_u32(r, &u);
    o.max_epochs = (int)u;
    nmz_read_u32(r, &u);
    o.batch_size = (int)u;
    nmz_read_f32(r, &o.learning_rate);
    nmz_read_f32(r, &o.momentum);
    nmz_read_u32(r, &u);
    o.shuffle = u != 0;
    nmz_read_f64(r, &o.target_loss);
    if (version >= 3) {
        uint32_t opt = 0, sched = 0, step_sz = 0;
        nmz_read_u32(r, &opt);
        nmz_read_f32(r, &o.beta1);
        nmz_read_f32(r, &o.beta2);
        nmz_read_f32(r, &o.epsilon);
        nmz_read_u32(r, &sched);
        nmz_read_u32(r, &step_sz);
        nmz_read_f32(r, &o.gamma);
        nmz_read_f32(r, &o.min_lr);
        if (opt >= NMZ_OPT_COUNT || sched >= NMZ_SCHED_COUNT || step_sz > 0x7fffffffu)
            return nmz__fail(NMZ_ERR_FORMAT, "corrupt training configuration");
        o.optimizer = (nmz_optimizer)opt;
        o.schedule = (nmz_schedule)sched;
        o.step_size = (int)step_sz;
    }
    if (r->err) return r->err;
    if (loss >= NMZ_LOSS_COUNT) return nmz__fail(NMZ_ERR_FORMAT, "corrupt loss id %u", loss);
    if ((inc & 1u) == 0 || u > 1 || o.max_epochs < 0 || o.batch_size < 0 || step > (uint64_t)0x7fffffff)
        return nmz__fail(NMZ_ERR_FORMAT, "corrupt header");
    if (nmz__train_opts_validate(&o) != NMZ_OK) {
        char why[256];
        snprintf(why, sizeof why, "%s", nmz_last_error()); /* never format the buffer into itself */
        return nmz__fail(NMZ_ERR_FORMAT, "corrupt training configuration: %s", why);
    }

    if ((s = nmz_net_create(&net, (int)in_rank, in_shape, 0)) != NMZ_OK) return s;
    net->rng.state = st;
    net->rng.inc = inc;
    net->step = (long)step;
    net->train = o;
    net->loss = (nmz_loss)loss;

    if (nmz_read_u32(r, &n_classes) != NMZ_OK) goto fail;
    if (n_classes > NMZ_MAX_CLASSES) {
        nmz__fail(NMZ_ERR_FORMAT, "corrupt class count");
        goto fail;
    }
    if (n_classes > 0) {
        net->class_names = calloc(n_classes, sizeof *net->class_names);
        if (!net->class_names) {
            nmz__fail(NMZ_ERR_NOMEM, "out of memory");
            goto fail;
        }
        for (i = 0; i < n_classes; i++) {
            if (read_str(r, &net->class_names[i]) != NMZ_OK) goto fail;
            net->n_classes = (int)i + 1;
        }
    }

    if (nmz_read_u32(r, &n_layers) != NMZ_OK) goto fail;
    if (n_layers == 0 || n_layers > NMZ_MAX_LAYERS) {
        nmz__fail(NMZ_ERR_FORMAT, "corrupt layer count %u", n_layers);
        goto fail;
    }
    for (i = 0; i < n_layers; i++) {
        char *type = NULL;
        const nmz_layer_ops *ops;
        nmz_layer *layer = NULL;
        uint32_t cfg_len = 0, np = 0;
        size_t before;
        if (read_str(r, &type) != NMZ_OK) goto fail;
        ops = nmz_layer_ops_find(type);
        if (!ops) {
            nmz__fail(NMZ_ERR_FORMAT, "unknown layer type '%s' (written by a newer NeuroMZ?)", type);
            free(type);
            goto fail;
        }
        free(type);
        if (nmz_read_u32(r, &cfg_len) != NMZ_OK) goto fail;
        before = ((reader_ext *)r)->count;
        if ((s = ops->load_config(&layer, r)) != NMZ_OK) {
            if (r->err == NMZ_OK) r->err = s;
            goto fail;
        }
        if (((reader_ext *)r)->count - before != cfg_len) {
            nmz_layer_free(layer);
            nmz__fail(NMZ_ERR_FORMAT, "layer %u config has the wrong length", i);
            goto fail;
        }
        if ((s = nmz_net_add(net, layer)) != NMZ_OK) {
            r->err = s;
            goto fail;
        }
        layer = net->layers[net->n_layers - 1];
        if (nmz_read_u32(r, &np) != NMZ_OK) goto fail;
        if ((int)np != nmz_layer_param_count(layer)) {
            nmz__fail(NMZ_ERR_FORMAT, "layer %u: expected %d parameters, file has %u", i,
                      nmz_layer_param_count(layer), np);
            goto fail;
        }
        for (j = 0; j < np; j++) {
            nmz_tensor *v = nmz_layer_param_value(layer, (int)j);
            char *name = NULL;
            uint32_t rank = 0;
            int ok;
            if (read_str(r, &name) != NMZ_OK) goto fail;
            ok = strcmp(name, nmz_layer_param_name(layer, (int)j)) == 0;
            free(name);
            if (nmz_read_u32(r, &rank) != NMZ_OK) goto fail;
            ok = ok && (int)rank == v->rank;
            for (d = 0; ok && d < rank; d++) {
                if (nmz_read_u32(r, &u) != NMZ_OK) goto fail;
                ok = (int)u == v->shape[d];
            }
            if (!ok) {
                nmz__fail(NMZ_ERR_FORMAT, "layer %u parameter %u does not match the layer's shape", i, j);
                goto fail;
            }
            for (e = 0; e < v->size; e++)
                if (nmz_read_f32(r, &v->data[e]) != NMZ_OK) goto fail;
        }
    }
    if ((s = nmz__net_finish_build(net)) != NMZ_OK) {
        r->err = s;
        goto fail;
    }
    if (version >= 3 && read_sections(r, net) != NMZ_OK) goto fail;
    {
        uint32_t expect = r->crc, stored = 0;
        unsigned char b[4];
        if (fread(b, 1, 4, r->f) != 4) {
            nmz__fail(NMZ_ERR_FORMAT, "unexpected end of file (missing checksum)");
            goto fail;
        }
        stored = (uint32_t)b[0] | (uint32_t)b[1] << 8 | (uint32_t)b[2] << 16 | (uint32_t)b[3] << 24;
        if (stored != expect) {
            nmz__fail(NMZ_ERR_FORMAT, "checksum mismatch: the file is corrupted");
            goto fail;
        }
        if (fgetc(r->f) != EOF) {
            nmz__fail(NMZ_ERR_FORMAT, "unexpected data after the end of the model");
            goto fail;
        }
    }
    *out = net;
    return NMZ_OK;
fail:
    nmz_net_free(net);
    return r->err != NMZ_OK ? r->err : NMZ_ERR_FORMAT;
}

nmz_status nmz_net_load(nmz_net **out, const char *path) {
    reader_ext x;
    nmz_status s;
    if (!out || !path) return nmz__fail(NMZ_ERR_ARG, "NULL argument");
    *out = NULL;
    memset(&x, 0, sizeof x);
    x.r.f = fopen(path, "rb");
    if (!x.r.f) return nmz__fail(NMZ_ERR_IO, "cannot open '%s': %s", path, strerror(errno));
    s = load_body(&x.r, out);
    fclose(x.r.f);
    return s;
}
