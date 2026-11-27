/* tensor.c — contiguous row-major float tensor. */
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
        float *p = realloc(t->data, n * sizeof *p);
        if (!p) return nmz__fail(NMZ_ERR_NOMEM, "out of memory allocating %zu floats", n);
        t->data = p;
        t->capacity = n;
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
    if (src->size) memcpy(dst->data, src->data, src->size * sizeof(float));
    return NMZ_OK;
}

void nmz_tensor_zero(nmz_tensor *t) {
    if (t && t->data && t->size) memset(t->data, 0, t->size * sizeof(float));
}

void nmz_tensor_free(nmz_tensor *t) {
    if (!t) return;
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
