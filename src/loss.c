/* loss.c — loss functions. Tensors are treated as rows × classes: the last
 * dimension holds the outputs of one sample (or one time step), every leading
 * dimension is a row. All losses are averaged over rows.
 *
 *   MSE:   L = (1/R) Σ_r ½ Σ_k (y_rk − t_rk)²      ∂L/∂y_rk = (y_rk − t_rk)/R
 *   CE:    L = −(1/R) Σ_r Σ_k t_rk · log p_rk      with p = softmax(z)
 *          fused gradient  ∂L/∂z_rk = (p_rk · Σ_j t_rj − t_rk)/R
 *          (= (p − t)/R for one-hot targets)                                  */
#include "internal.h"

#include <math.h>
#include <string.h>

static const char *const k_loss_names[NMZ_LOSS_COUNT] = {"mse", "cross_entropy"};
static const char *const k_init_names[NMZ_INIT_COUNT] = {"auto", "xavier", "he", "orthogonal"};

const char *nmz_loss_name(nmz_loss l) {
    if ((int)l < 0 || l >= NMZ_LOSS_COUNT) return "?";
    return k_loss_names[l];
}

const char *nmz_init_name(nmz_init i) {
    if ((int)i < 0 || i >= NMZ_INIT_COUNT) return "?";
    return k_init_names[i];
}

nmz_status nmz_loss_parse(const char *s, nmz_loss *out) {
    if (!s || !out) return nmz__fail(NMZ_ERR_ARG, "NULL argument");
    if (!strcmp(s, "mse")) { *out = NMZ_LOSS_MSE; return NMZ_OK; }
    if (!strcmp(s, "cross_entropy") || !strcmp(s, "ce") || !strcmp(s, "crossentropy")) {
        *out = NMZ_LOSS_CROSS_ENTROPY;
        return NMZ_OK;
    }
    return nmz__fail(NMZ_ERR_ARG, "unknown loss '%s' (expected mse or cross_entropy)", s);
}

nmz_status nmz_init_parse(const char *s, nmz_init *out) {
    if (!s || !out) return nmz__fail(NMZ_ERR_ARG, "NULL argument");
    if (!strcmp(s, "auto")) { *out = NMZ_INIT_AUTO; return NMZ_OK; }
    if (!strcmp(s, "xavier") || !strcmp(s, "glorot")) { *out = NMZ_INIT_XAVIER; return NMZ_OK; }
    if (!strcmp(s, "he") || !strcmp(s, "kaiming")) { *out = NMZ_INIT_HE; return NMZ_OK; }
    if (!strcmp(s, "orthogonal") || !strcmp(s, "orth")) { *out = NMZ_INIT_ORTHOGONAL; return NMZ_OK; }
    return nmz__fail(NMZ_ERR_ARG, "unknown init '%s' (expected auto, xavier, he or orthogonal)", s);
}

static nmz_status check_pair(const nmz_tensor *y, const nmz_tensor *t) {
    if (!y || !t) return nmz__fail(NMZ_ERR_ARG, "NULL tensor");
    if (!nmz__shape_equal(y->rank, y->shape, t->rank, t->shape))
        return nmz__fail(NMZ_ERR_SHAPE, "target shape does not match the output shape");
    return NMZ_OK;
}

/* The CPU kernels: the reference loops on raw rows × cols arrays. */
nmz_status nmz__cpu_loss_value(const nmz_backend *be, nmz_loss kind, size_t rows, int cols, const float *y,
                               const float *t, const float *z, double *out) {
    size_t r, size = rows * (size_t)cols;
    int k;
    double total = 0.0;
    (void)be;
    if (kind == NMZ_LOSS_MSE) {
        for (r = 0; r < size; r++) {
            double e = (double)y[r] - t[r];
            total += 0.5 * e * e;
        }
    } else if (kind == NMZ_LOSS_CROSS_ENTROPY) {
        for (r = 0; r < rows; r++) {
            const float *tr = t + r * (size_t)cols;
            if (z) {
                /* log p_k = z_k − log Σ_j exp(z_j), computed stably */
                const float *zr = z + r * (size_t)cols;
                double mx = zr[0], sum = 0.0, lse;
                for (k = 1; k < cols; k++)
                    if (zr[k] > mx) mx = zr[k];
                for (k = 0; k < cols; k++) sum += exp((double)zr[k] - mx);
                lse = mx + log(sum);
                for (k = 0; k < cols; k++)
                    if (tr[k] != 0.0f) total -= tr[k] * ((double)zr[k] - lse);
            } else {
                const float *pr = y + r * (size_t)cols;
                for (k = 0; k < cols; k++)
                    if (tr[k] != 0.0f) total -= tr[k] * log(fmax((double)pr[k], 1e-300));
            }
        }
    } else {
        return nmz__fail(NMZ_ERR_ARG, "invalid loss kind %d", (int)kind);
    }
    *out = total / (double)rows;
    return NMZ_OK;
}

nmz_status nmz__cpu_loss_grad(const nmz_backend *be, nmz_loss kind, size_t rows, int cols, const float *y,
                              const float *t, float *g) {
    size_t r, size = rows * (size_t)cols;
    int k;
    double inv = 1.0 / (double)rows;
    (void)be;
    if (kind == NMZ_LOSS_MSE) {
        for (r = 0; r < size; r++) g[r] = (float)(((double)y[r] - t[r]) * inv);
    } else if (kind == NMZ_LOSS_CROSS_ENTROPY) {
        for (r = 0; r < rows; r++) {
            const float *pr = y + r * (size_t)cols, *tr = t + r * (size_t)cols;
            float *gr = g + r * (size_t)cols;
            double tsum = 0.0;
            for (k = 0; k < cols; k++) tsum += tr[k];
            for (k = 0; k < cols; k++) gr[k] = (float)(((double)pr[k] * tsum - tr[k]) * inv);
        }
    } else {
        return nmz__fail(NMZ_ERR_ARG, "invalid loss kind %d", (int)kind);
    }
    return NMZ_OK;
}

/* Tensor-level entry points: the value is read back (one sync), or, for
 * _queued, written to *out at the backend's next sync. */
nmz_status nmz__loss_value_queued(const nmz_backend *be, nmz_loss kind, const nmz_tensor *y, const nmz_tensor *t,
                                  const nmz_tensor *z, double *out) {
    const float *py, *pt, *pz = NULL;
    nmz_status s = check_pair(y, t);
    if (s != NMZ_OK) return s;
    if ((int)kind < 0 || kind >= NMZ_LOSS_COUNT) return nmz__fail(NMZ_ERR_ARG, "invalid loss kind %d", (int)kind);
    py = nmz__r(be, y);
    pt = nmz__r(be, t);
    if (z && !(pz = nmz__r(be, z))) return nmz__op_failed();
    if (!py || !pt) return nmz__op_failed();
    return be->loss_value(be, kind, nmz_tensor_rows(y), nmz_tensor_cols(y), py, pt, pz, out);
}

nmz_status nmz__loss_value(const nmz_backend *be, nmz_loss kind, const nmz_tensor *y, const nmz_tensor *t,
                           const nmz_tensor *z, double *out) {
    nmz_status s = nmz__loss_value_queued(be, kind, y, t, z, out);
    return s == NMZ_OK ? be->sync(be) : s;
}

nmz_status nmz__loss_grad(const nmz_backend *be, nmz_loss kind, const nmz_tensor *y, const nmz_tensor *t,
                          nmz_tensor *g) {
    const float *py, *pt;
    float *pg;
    nmz_status s = check_pair(y, t);
    if (s != NMZ_OK) return s;
    if ((int)kind < 0 || kind >= NMZ_LOSS_COUNT) return nmz__fail(NMZ_ERR_ARG, "invalid loss kind %d", (int)kind);
    if ((s = nmz_tensor_reshape(g, y->rank, y->shape)) != NMZ_OK) return s;
    py = nmz__r(be, y);
    pt = nmz__r(be, t);
    pg = nmz__w(be, g);
    if (!py || !pt || !pg) return nmz__op_failed();
    return be->loss_grad(be, kind, nmz_tensor_rows(y), nmz_tensor_cols(y), py, pt, pg);
}
