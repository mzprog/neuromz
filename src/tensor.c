/* tensor.c — contiguous row-major float tensor.
 *
 * `data` is always allocated in host memory (capacity ≥ size). A tensor a
 * device backend has computed on also has a device copy (dev, backend.c);
 * these functions keep the two consistent: reshape drops a device buffer
 * that became too small, zero and copy work where the newest data is. A
 * host-only tensor (dev == NULL) takes exactly the v2.0.1 code paths. */
#include "internal.h"

#include <stdlib.h>
#include <string.h>

nmz_status nmz_tensor_reshape(nmz_tensor *t, int rank, const int *shape) {
    size_t n;
    nmz_status s;
    int i;
    if (!t) return nmz__fail(NMZ_ERR_ARG, "tensor is NULL");
    s = nmz__shape_size(rank, shape, &n);
    if (s != NMZ_OK) return s;
    if (n > t->capacity) {
        float *p;
        if (t->dev && t->dev->where == NMZ__DEVICE) t->dev->where = NMZ__HOST; /* growing: contents unspecified */
        p = realloc(t->data, n * sizeof *p);
        if (!p) return nmz__fail(NMZ_ERR_NOMEM, "out of memory allocating %zu floats", n);
        t->data = p;
        t->capacity = n;
    }
    if (t->dev && n > t->dev->cap && t->dev->ptr) { /* the device buffer is reallocated when next needed */
        if (t->dev->where == NMZ__DEVICE) t->dev->where = NMZ__HOST;
        t->dev->be->release(t->dev->be, t->dev->ptr);
        t->dev->ptr = NULL;
        t->dev->cap = 0;
    }
    t->rank = rank;
    for (i = 0; i < NMZ_MAX_RANK; i++) t->shape[i] = i < rank ? shape[i] : 0;
    t->size = n;
    return NMZ_OK;
}

nmz_status nmz_tensor_init(nmz_tensor *t, int rank, const int *shape) {
    nmz_status s;
    if (!t) return nmz__fail(NMZ_ERR_ARG, "tensor is NULL");
    memset(t, 0, sizeof *t);
    s = nmz_tensor_reshape(t, rank, shape);
    if (s == NMZ_OK) nmz_tensor_zero(t);
    return s;
}

nmz_status nmz_tensor_copy(nmz_tensor *dst, const nmz_tensor *src) {
    nmz_status s;
    if (!dst || !src) return nmz__fail(NMZ_ERR_ARG, "tensor is NULL");
    if (dst == src) return NMZ_OK;
    s = nmz_tensor_reshape(dst, src->rank, src->shape);
    if (s != NMZ_OK) return s;
    if (!src->size) return NMZ_OK;
    if (src->dev && dst->dev && src->dev->be == dst->dev->be && src->dev->where != NMZ__HOST) {
        /* both on the same device and the device copy of src is current */
        const nmz_backend *be = src->dev->be;
        const float *sp = nmz__r(be, src);
        float *dp = nmz__w(be, dst);
        if (!sp || !dp) return nmz__op_failed();
        return be->copy(be, 1, src->size, sp, src->size, dp, src->size);
    }
    memcpy(nmz__hw(dst), nmz__hr(src), src->size * sizeof(float));
    return NMZ_OK;
}

void nmz_tensor_zero(nmz_tensor *t) {
    if (!t || !t->size) return;
    if (t->dev && t->dev->ptr && t->dev->cap >= t->size && t->dev->where != NMZ__HOST) {
        const nmz_backend *be = t->dev->be;
        if (be->fill(be, t->size, 0.0f, t->dev->ptr) == NMZ_OK) {
            t->dev->where = NMZ__DEVICE;
            return;
        }
    }
    if (t->data) memset(nmz__hw(t), 0, t->size * sizeof(float));
}

void nmz_tensor_free(nmz_tensor *t) {
    if (!t) return;
    if (t->dev) {
        if (t->dev->ptr) t->dev->be->release(t->dev->be, t->dev->ptr);
        free(t->dev);
    }
    free(t->data);
    memset(t, 0, sizeof *t);
}

int nmz_tensor_cols(const nmz_tensor *t) {
    if (!t || t->rank == 0) return 1;
    return t->shape[t->rank - 1];
}

size_t nmz_tensor_rows(const nmz_tensor *t) {
    int c = nmz_tensor_cols(t);
    if (!t || c <= 0) return 0;
    return t->size / (size_t)c;
}
