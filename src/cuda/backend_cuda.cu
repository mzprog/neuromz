/* backend_cuda.cu — the CUDA backend (DESIGN §11.7).
 *
 * The C library sees one function, nmz__cuda_backend(), returning the
 * nmz_backend table (neuromz_backend.h); everything else here is private.
 *
 *   - Nothing runs until the backend is selected or queried: init() asks
 *     the runtime for a device, checks its compute capability, launches a
 *     probe kernel (a GPU older than the compiled code fails here, not
 *     later), and loads cuBLAS with LoadLibrary / dlopen. A machine
 *     without a GPU, a driver or the cuBLAS library gets a reason string,
 *     and "auto" stays on the CPU. The CUDA runtime is linked statically
 *     (it loads the driver itself, and reports a missing one as an error).
 *   - Matrix products are cuBLAS SGEMM (FP32; Turing has no TF32), with the
 *     row-major operands passed as the transposed column-major problem.
 *     Every other op is a kernel below, in float with double sums where the
 *     CPU sums in double (reductions, statistics, Adam).
 *   - One stream. Ops are asynchronous; scalars an op returns (losses,
 *     norms, counts) are written by a kernel to device memory, copied to
 *     pinned host memory on the stream, and delivered at sync().
 *   - Reductions are deterministic: partial sums per block, then one pass
 *     in a fixed order (no atomics), so a run repeats bit for bit. The one
 *     exception is the embedding gradient of a large vocabulary
 *     (vocabulary × positions above 2^24), which adds with atomics.
 *   - Device memory is counted (now and at most) for nmz_backend_info, and
 *     an allocation that fails says what was asked, what is held and what
 *     is free, and to try a smaller batch. */
#include <cuda_runtime.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "neuromz_backend.h"

extern "C" nmz_status nmz__op_failed(void);

/* The few cuBLAS declarations used here (stable ABI values from
 * cublas_api.h), so building needs only nvcc and the CUDA runtime: the
 * library itself is loaded at run time (cu_init). Its major version is the
 * toolkit's (cublas64_13.dll / libcublas.so.13 for CUDA 13). */
typedef struct nmz_cublas_context *cublasHandle_t;
typedef int cublasStatus_t;
typedef int cublasOperation_t;
#define CUBLAS_STATUS_SUCCESS 0
#define CUBLAS_OP_N 0
#define CUBLAS_OP_T 1
#ifdef _WIN32
#define CUBLASWINAPI __stdcall
#else
#define CUBLASWINAPI
#endif
#define CUBLAS_VER_MAJOR (CUDART_VERSION / 1000)

#ifndef NMZ_CUDA_MIN_CC
#define NMZ_CUDA_MIN_CC 75 /* the lowest architecture compiled (CMake passes the real value) */
#endif

/* ------------------------------------------------------------------ state */

typedef cublasStatus_t(CUBLASWINAPI *pf_create)(cublasHandle_t *);
typedef cublasStatus_t(CUBLASWINAPI *pf_set_stream)(cublasHandle_t, cudaStream_t);
typedef cublasStatus_t(CUBLASWINAPI *pf_sgemm)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
                                                const float *, const float *, int, const float *, int, const float *,
                                                float *, int);
typedef cublasStatus_t(CUBLASWINAPI *pf_get_version)(cublasHandle_t, int *);
typedef cublasStatus_t(CUBLASWINAPI *pf_sgemm_sb)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
                                                   const float *, const float *, int, long long, const float *, int,
                                                   long long, const float *, float *, int, long long, int);

#define MAX_PENDING 64

typedef struct alloc_slot {
    void *p;
    size_t n;
} alloc_slot;

static struct {
    int ready;
    int dev;
    cudaDeviceProp prop;
    cudaStream_t stream;
    void *lib;
    cublasHandle_t blas;
    pf_sgemm sgemm;
    pf_sgemm_sb sgemm_sb; /* strided batched: attention, one GEMM per head */
    int blas_version;
    size_t used, peak;
    size_t limit, limit_default; /* what alloc may hand out (decision 142) */
    alloc_slot *slots; /* open addressing: device pointer → bytes */
    size_t n_slots, n_live;
    double *pinned;            /* MAX_PENDING scalars on their way to the host */
    double *dscal;             /* the same on the device                       */
    double *pend_dst[MAX_PENDING];
    int n_pending;
    void *scratch;             /* reductions' partial sums, offsets           */
    size_t scratch_cap;
    float *work[4];            /* the fused sequences' buffers (grow only)    */
    size_t work_cap[4];
    nmz_status sticky;         /* an asynchronous error seen at sync          */
} g;

static nmz_backend g_be;

static nmz_status cuda_fail(cudaError_t e, const char *what) {
    (void)cudaGetLastError(); /* clear it: the next call should not see it again */
    return nmz_backend_error(e == cudaErrorMemoryAllocation ? NMZ_ERR_NOMEM : NMZ_ERR_STATE, "CUDA: %s: %s", what,
                             cudaGetErrorString(e));
}

static nmz_status check(const char *what) {
    cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? NMZ_OK : cuda_fail(e, what);
}

/* ------------------------------------------------------- allocation table */

static size_t slot_of(void *p) {
    uintptr_t h = (uintptr_t)p;
    h ^= h >> 17;
    h *= (uintptr_t)0x9E3779B97F4A7C15ull;
    return (size_t)(h ^ (h >> 31)) & (g.n_slots - 1);
}

static int table_put(void *p, size_t n) {
    size_t i;
    if ((g.n_live + 1) * 2 > g.n_slots) { /* grow: rehash */
        size_t old = g.n_slots, k;
        alloc_slot *os = g.slots;
        alloc_slot *ns = (alloc_slot *)calloc(old ? old * 2 : 256, sizeof *ns);
        if (!ns) return 0;
        g.slots = ns;
        g.n_slots = old ? old * 2 : 256;
        for (k = 0; k < old; k++)
            if (os[k].p) {
                i = slot_of(os[k].p);
                while (g.slots[i].p) i = (i + 1) & (g.n_slots - 1);
                g.slots[i] = os[k];
            }
        free(os);
    }
    i = slot_of(p);
    while (g.slots[i].p) i = (i + 1) & (g.n_slots - 1);
    g.slots[i].p = p;
    g.slots[i].n = n;
    g.n_live++;
    return 1;
}

static size_t table_take(void *p) {
    size_t i, n, j;
    if (!g.n_slots) return 0;
    i = slot_of(p);
    while (g.slots[i].p && g.slots[i].p != p) i = (i + 1) & (g.n_slots - 1);
    if (!g.slots[i].p) return 0;
    n = g.slots[i].n;
    g.slots[i].p = NULL;
    g.n_live--;
    /* re-insert the rest of the cluster (linear probing deletion) */
    for (j = (i + 1) & (g.n_slots - 1); g.slots[j].p; j = (j + 1) & (g.n_slots - 1)) {
        alloc_slot s = g.slots[j];
        size_t k = slot_of(s.p);
        g.slots[j].p = NULL;
        while (g.slots[k].p) k = (k + 1) & (g.n_slots - 1);
        g.slots[k] = s;
    }
    return n;
}

static void *cu_alloc(const nmz_backend *be, size_t bytes) {
    void *p = NULL;
    cudaError_t e;
    (void)be;
    if (!bytes) bytes = 4;
    if (g.limit && g.used + bytes > g.limit) { /* beyond the cap the driver would page memory out (WDDM) */
        nmz_backend_error(NMZ_ERR_NOMEM,
                          "out of GPU memory: %.1f MiB more were needed and the library already holds %.1f MiB of the "
                          "%.0f MiB it may use (%s). Try a smaller batch size (or a smaller model)",
                          bytes / 1048576.0, g.used / 1048576.0, g.limit / 1048576.0,
                          g.limit == g.limit_default ? "what was free on the device when it started; NMZ_CUDA_MEMORY "
                                                       "or nmz_backend_memory_limit change it"
                                                     : "the limit set with NMZ_CUDA_MEMORY or nmz_backend_memory_limit");
        return NULL;
    }
    e = cudaMalloc(&p, bytes);
    if (e != cudaSuccess) {
        size_t fr = 0, tot = 0;
        (void)cudaGetLastError();
        cudaMemGetInfo(&fr, &tot);
        nmz_backend_error(NMZ_ERR_NOMEM,
                          "out of GPU memory: %.1f MiB more were needed; the library holds %.1f MiB, %.1f MiB of %.0f "
                          "MiB are free. Try a smaller batch size (or a smaller model)",
                          bytes / 1048576.0, g.used / 1048576.0, fr / 1048576.0, tot / 1048576.0);
        return NULL;
    }
    if (!table_put(p, bytes)) {
        cudaFree(p);
        nmz_backend_error(NMZ_ERR_NOMEM, "out of host memory");
        return NULL;
    }
    g.used += bytes;
    if (g.used > g.peak) g.peak = g.used;
    return p;
}

static void cu_release(const nmz_backend *be, void *p) {
    (void)be;
    if (!p) return;
    g.used -= table_take(p);
    cudaFree(p);
}

/* grow-only work space for reductions */
static void *scratch(size_t bytes) {
    if (bytes > g.scratch_cap) {
        if (g.scratch) cu_release(&g_be, g.scratch);
        g.scratch_cap = 0;
        g.scratch = cu_alloc(&g_be, bytes < 65536 ? 65536 : bytes);
        if (!g.scratch) return NULL;
        g.scratch_cap = bytes < 65536 ? 65536 : bytes;
    }
    return g.scratch;
}

/* grow-only buffers of the fused sequence kernels (one op at a time on one
 * stream, so they are never in use by two ops) */
static float *work(int i, size_t floats) {
    if (floats > g.work_cap[i]) {
        if (g.work[i]) cu_release(&g_be, g.work[i]);
        g.work_cap[i] = 0;
        g.work[i] = (float *)cu_alloc(&g_be, floats * sizeof(float));
        if (!g.work[i]) return NULL;
        g.work_cap[i] = floats;
    }
    return g.work[i];
}

static nmz_status cu_upload(const nmz_backend *be, void *dev, const void *host, size_t bytes) {
    (void)be;
    if (!bytes) return NMZ_OK;
    if (cudaMemcpyAsync(dev, host, bytes, cudaMemcpyHostToDevice, g.stream) != cudaSuccess)
        return cuda_fail(cudaGetLastError(), "copy to the device");
    return NMZ_OK;
}

static nmz_status cu_sync(const nmz_backend *be);

static nmz_status cu_download(const nmz_backend *be, void *host, const void *dev, size_t bytes) {
    (void)be;
    if (!bytes) return NMZ_OK;
    if (cudaMemcpyAsync(host, dev, bytes, cudaMemcpyDeviceToHost, g.stream) != cudaSuccess)
        return cuda_fail(cudaGetLastError(), "copy to the host");
    return cu_sync(&g_be);
}

static nmz_status cu_sync(const nmz_backend *be) {
    cudaError_t e;
    int i;
    (void)be;
    e = cudaStreamSynchronize(g.stream);
    if (e != cudaSuccess) return cuda_fail(e, "a kernel failed");
    for (i = 0; i < g.n_pending; i++) *g.pend_dst[i] = g.pinned[i];
    g.n_pending = 0;
    return NMZ_OK;
}

/* the device slot an op's scalar goes to; it reaches *dst at the next sync */
static double *scalar_slot(double *dst) {
    if (g.n_pending == MAX_PENDING && cu_sync(&g_be) != NMZ_OK) return NULL;
    g.pend_dst[g.n_pending] = dst;
    return g.dscal + g.n_pending;
}

static nmz_status scalar_send(void) { /* after the kernel that wrote the slot */
    int i = g.n_pending++;
    if (cudaMemcpyAsync(&g.pinned[i], g.dscal + i, sizeof(double), cudaMemcpyDeviceToHost, g.stream) != cudaSuccess)
        return cuda_fail(cudaGetLastError(), "copy a result to the host");
    return check("a reduction");
}

/* ------------------------------------------------------------- launching */

#define TPB 256
static unsigned blocks_for(size_t n) {
    size_t b = (n + TPB - 1) / TPB;
    if (b < 1) b = 1;
    return (unsigned)(b > 8192 ? 8192 : b);
}
static unsigned warps_blocks(size_t warps) { /* 8 warps per block */
    size_t b = (warps + 7) / 8;
    return (unsigned)(b < 1 ? 1 : b > 65535 ? 65535 : b);
}
#define GRID_STRIDE(i, n) for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += (size_t)gridDim.x * blockDim.x)
#define WARP_ID() (((size_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5)
#define WARP_STRIDE() (((size_t)gridDim.x * blockDim.x) >> 5)
#define NEED(p) \
    if (!(p)) return nmz__op_failed()

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return __shfl_sync(0xffffffffu, v, 0);
}
__device__ __forceinline__ double warp_sumd(double v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return __shfl_sync(0xffffffffu, v, 0);
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, o));
    return __shfl_sync(0xffffffffu, v, 0);
}
/* the first index of the largest value of the warp's (value, index) pairs */
__device__ __forceinline__ void warp_argmax(float &v, int &i) {
    for (int o = 16; o > 0; o >>= 1) {
        float ov = __shfl_down_sync(0xffffffffu, v, o);
        int oi = __shfl_down_sync(0xffffffffu, i, o);
        if (ov > v || (ov == v && oi < i)) v = ov, i = oi;
    }
    v = __shfl_sync(0xffffffffu, v, 0);
    i = __shfl_sync(0xffffffffu, i, 0);
}

/* --------------------------------------------- reductions to one double */

/* partial[block] = Σ of its share; then one thread sums the partials in order */
__global__ void k_sum_partials(const double *part, int n, double scale, double *out, int add) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        double s = 0;
        for (int i = 0; i < n; i++) s += part[i];
        s *= scale;
        *out = add ? *out + s : s;
    }
}

__device__ __forceinline__ void block_reduce_store(double v, double *part) {
    __shared__ double sh[TPB / 32];
    v = warp_sumd(v);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x < 32) {
        double w = threadIdx.x < blockDim.x / 32 ? sh[threadIdx.x] : 0.0;
        w = warp_sumd(w);
        if (threadIdx.x == 0) part[blockIdx.x] = w;
    }
}

__global__ void k_sumsq(size_t n, const float *x, double *part) {
    double s = 0;
    GRID_STRIDE(i, n) s += (double)x[i] * x[i];
    block_reduce_store(s, part);
}

__global__ void k_sum_rows(size_t n, const double *v, double *part) {
    double s = 0;
    GRID_STRIDE(i, n) s += v[i];
    block_reduce_store(s, part);
}

__global__ void k_mse_sum(size_t n, const float *y, const float *t, double *part) {
    double s = 0;
    GRID_STRIDE(i, n) {
        double e = (double)y[i] - t[i];
        s += 0.5 * e * e;
    }
    block_reduce_store(s, part);
}

/* --------------------------------------------------------------- kernels */

__global__ void k_fill(size_t n, float v, float *x) {
    GRID_STRIDE(i, n) x[i] = v;
}
__global__ void k_add(size_t n, const float *a, const float *b, float *o) {
    GRID_STRIDE(i, n) o[i] = a[i] + b[i];
}
__global__ void k_mul(size_t n, const float *a, const float *b, float *o) {
    GRID_STRIDE(i, n) o[i] = a[i] * b[i];
}
__global__ void k_add_bias(size_t n, int cols, float *y, const float *b) {
    GRID_STRIDE(i, n) y[i] += b[i % (size_t)cols];
}
__global__ void k_add_bias3(size_t n, int B, int C, float *y, const float *bias) {
    GRID_STRIDE(i, n) y[i] += bias[(i / (size_t)C) % (size_t)B];
}
__global__ void k_transpose3(int A, int B, int C, const float *in, float *out) {
    size_t n = (size_t)A * B * C;
    GRID_STRIDE(i, n) { /* i = (b·A + a)·C + c in the output */
        size_t c = i % (size_t)C, ba = i / (size_t)C, a = ba % (size_t)A, b = ba / (size_t)A;
        out[i] = in[(a * B + b) * C + c];
    }
}

/* column sums of a (or a ⊙ b), block (32 columns × 8 lanes), partials [gy, cols] */
__global__ void k_colsum_part(size_t rows, int cols, const float *a, const float *b, double *part) {
    __shared__ double sh[8][33];
    int c = blockIdx.x * 32 + threadIdx.x;
    double s = 0;
    if (c < cols)
        for (size_t r = (size_t)blockIdx.y * 8 + threadIdx.y; r < rows; r += (size_t)gridDim.y * 8) {
            float v = a[r * cols + c];
            s += b ? (double)v * b[r * cols + c] : (double)v;
        }
    sh[threadIdx.y][threadIdx.x] = s;
    __syncthreads();
    if (threadIdx.y == 0 && c < cols) {
        for (int k = 1; k < 8; k++) s += sh[k][threadIdx.x];
        part[(size_t)blockIdx.y * cols + c] = s;
    }
}
__global__ void k_colsum_finish(int parts, int cols, const double *part, float *out) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < cols) {
        double s = 0;
        for (int p = 0; p < parts; p++) s += part[(size_t)p * cols + c];
        out[c] += (float)s;
    }
}

__global__ void k_rowsum(int rows, size_t cols, const float *a, float *out) {
    for (size_t r = WARP_ID(); r < (size_t)rows; r += WARP_STRIDE()) {
        int lane = threadIdx.x & 31;
        double s = 0;
        for (size_t c = lane; c < cols; c += 32) s += a[r * cols + c];
        s = warp_sumd(s);
        if (lane == 0) out[r] += (float)s;
    }
}

/* ----------------------------------------------------------- activations */

__device__ __forceinline__ float sigmoid_f(float z) {
    if (z >= 0.0f) return 1.0f / (1.0f + expf(-z));
    {
        float e = expf(z);
        return e / (1.0f + e);
    }
}
__device__ __forceinline__ float act_f(int a, float z, float alpha) {
    switch (a) {
    case NMZ_ACT_SIGMOID: return sigmoid_f(z);
    case NMZ_ACT_TANH: return tanhf(z);
    case NMZ_ACT_RELU: return z > 0.0f ? z : 0.0f;
    case NMZ_ACT_LEAKY_RELU: return z > 0.0f ? z : alpha * z;
    case NMZ_ACT_GELU: return z * 0.5f * (1.0f + erff(z * 0.70710678118654752440f));
    default: return z;
    }
}
__device__ __forceinline__ float act_d(int a, float z, float alpha) {
    switch (a) {
    case NMZ_ACT_LINEAR: return 1.0f;
    case NMZ_ACT_SIGMOID: {
        float s = sigmoid_f(z);
        return s * (1.0f - s);
    }
    case NMZ_ACT_TANH: {
        float t = tanhf(z);
        return 1.0f - t * t;
    }
    case NMZ_ACT_RELU: return z > 0.0f ? 1.0f : 0.0f;
    case NMZ_ACT_LEAKY_RELU: return z > 0.0f ? 1.0f : alpha;
    case NMZ_ACT_GELU:
        return 0.5f * (1.0f + erff(z * 0.70710678118654752440f)) + z * 0.39894228040143267794f * expf(-0.5f * z * z);
    default: return 1.0f;
    }
}
__global__ void k_act_fwd(int a, float alpha, size_t n, const float *z, float *y) {
    GRID_STRIDE(i, n) y[i] = act_f(a, z[i], alpha);
}
__global__ void k_act_bwd(int a, float alpha, size_t n, const float *z, const float *dy, float *dz, float *fp) {
    GRID_STRIDE(i, n) {
        float d = act_d(a, z[i], alpha);
        if (fp) fp[i] = d;
        dz[i] = dy[i] * d;
    }
}

/* one warp per row */
__global__ void k_softmax(size_t rows, int cols, const float *z, float *p) {
    for (size_t r = WARP_ID(); r < rows; r += WARP_STRIDE()) {
        const float *zr = z + r * cols;
        float *pr = p + r * cols;
        int lane = threadIdx.x & 31;
        float mx = -INFINITY;
        double sum = 0;
        for (int k = lane; k < cols; k += 32) mx = fmaxf(mx, zr[k]);
        mx = warp_max(mx);
        for (int k = lane; k < cols; k += 32) sum += expf(zr[k] - mx);
        sum = warp_sumd(sum);
        for (int k = lane; k < cols; k += 32) pr[k] = (float)(expf(zr[k] - mx) / sum);
    }
}
__global__ void k_softmax_bwd(size_t rows, int cols, const float *p, const float *dy, float *dz) {
    for (size_t r = WARP_ID(); r < rows; r += WARP_STRIDE()) {
        const float *pr = p + r * cols, *gr = dy + r * cols;
        float *dr = dz + r * cols;
        int lane = threadIdx.x & 31;
        double dot = 0;
        for (int k = lane; k < cols; k += 32) dot += (double)gr[k] * pr[k];
        dot = warp_sumd(dot);
        for (int k = lane; k < cols; k += 32) dr[k] = (float)(pr[k] * ((double)gr[k] - dot));
    }
}

/* per row: −Σ t·log p (from z by log-sum-exp, or from p) */
__global__ void k_ce_rows(size_t rows, int cols, const float *y, const float *t, const float *z, double *out) {
    for (size_t r = WARP_ID(); r < rows; r += WARP_STRIDE()) {
        const float *tr = t + r * cols;
        int lane = threadIdx.x & 31;
        double total = 0;
        if (z) {
            const float *zr = z + r * cols;
            float mx = -INFINITY;
            double sum = 0, lse;
            for (int k = lane; k < cols; k += 32) mx = fmaxf(mx, zr[k]);
            mx = warp_max(mx);
            for (int k = lane; k < cols; k += 32) sum += exp((double)zr[k] - mx);
            sum = warp_sumd(sum);
            lse = mx + log(sum);
            for (int k = lane; k < cols; k += 32)
                if (tr[k] != 0.0f) total -= tr[k] * ((double)zr[k] - lse);
        } else {
            const float *pr = y + r * cols;
            for (int k = lane; k < cols; k += 32)
                if (tr[k] != 0.0f) total -= tr[k] * log(fmax((double)pr[k], 1e-300));
        }
        total = warp_sumd(total);
        if (lane == 0) out[r] = total;
    }
}
__global__ void k_mse_grad(size_t n, double inv, const float *y, const float *t, float *g) {
    GRID_STRIDE(i, n) g[i] = (float)(((double)y[i] - t[i]) * inv);
}
__global__ void k_ce_grad(size_t rows, int cols, double inv, const float *p, const float *t, float *g) {
    for (size_t r = WARP_ID(); r < rows; r += WARP_STRIDE()) {
        const float *pr = p + r * cols, *tr = t + r * cols;
        float *gr = g + r * cols;
        int lane = threadIdx.x & 31;
        double tsum = 0;
        for (int k = lane; k < cols; k += 32) tsum += tr[k];
        tsum = warp_sumd(tsum);
        for (int k = lane; k < cols; k += 32) gr[k] = (float)(((double)pr[k] * tsum - tr[k]) * inv);
    }
}
/* mode 1: argmax(y) = argmax(t); 2: every output on the same side of 0.5 */
__global__ void k_correct(int mode, size_t rows, int cols, const float *y, const float *t, double *out) {
    for (size_t r = WARP_ID(); r < rows; r += WARP_STRIDE()) {
        const float *yr = y + r * cols, *tr = t + r * cols;
        int lane = threadIdx.x & 31, ok;
        if (mode == 1) {
            float vy = -INFINITY, vt = -INFINITY;
            int iy = cols, it = cols;
            for (int k = lane; k < cols; k += 32) {
                if (yr[k] > vy) vy = yr[k], iy = k;
                if (tr[k] > vt) vt = tr[k], it = k;
            }
            warp_argmax(vy, iy);
            warp_argmax(vt, it);
            ok = iy == it;
        } else {
            int bad = 0;
            for (int k = lane; k < cols; k += 32) bad |= (yr[k] >= 0.5f) != (tr[k] >= 0.5f);
            ok = !__any_sync(0xffffffffu, bad);
        }
        if (lane == 0) out[r] = ok;
    }
}

/* ---------------------------------------------------------- convolution */

__global__ void k_im2col(nmz_conv_shape s, const float *x, float *col) {
    const size_t P = (size_t)s.OH * s.OW, N = (size_t)s.B * P, K = (size_t)s.C * s.kh * s.kw;
    GRID_STRIDE(i, K * N) {
        size_t k = i / N, n = i % N, b = n / P, p = n % P;
        int kx = (int)(k % s.kw), ky = (int)((k / s.kw) % s.kh), ch = (int)(k / ((size_t)s.kw * s.kh));
        int oy = (int)(p / s.OW), ox = (int)(p % s.OW);
        int iy = oy * s.stride + ky - s.pad, ix = ox * s.stride + kx - s.pad;
        col[i] = (iy < 0 || iy >= s.H || ix < 0 || ix >= s.W) ? 0.0f : x[((b * s.C + ch) * s.H + iy) * s.W + ix];
    }
}
/* gather: each input pixel sums the column entries that copied it */
__global__ void k_col2im(nmz_conv_shape s, const float *col, float *dx) {
    const size_t P = (size_t)s.OH * s.OW, N = (size_t)s.B * P, n = (size_t)s.B * s.C * s.H * s.W;
    GRID_STRIDE(i, n) {
        int ix = (int)(i % s.W), iy = (int)((i / s.W) % s.H), ch = (int)((i / ((size_t)s.W * s.H)) % s.C);
        size_t b = i / ((size_t)s.W * s.H * s.C);
        float sum = 0.0f;
        for (int ky = 0; ky < s.kh; ky++) {
            int ty = iy + s.pad - ky, oy;
            if (ty < 0 || ty % s.stride) continue;
            oy = ty / s.stride;
            if (oy >= s.OH) continue;
            for (int kx = 0; kx < s.kw; kx++) {
                int tx = ix + s.pad - kx, ox;
                if (tx < 0 || tx % s.stride) continue;
                ox = tx / s.stride;
                if (ox >= s.OW) continue;
                sum += col[((size_t)(ch * s.kh + ky) * s.kw + kx) * N + b * P + (size_t)oy * s.OW + ox];
            }
        }
        dx[i] = sum;
    }
}
__global__ void k_pool_fwd(int planes, int H, int W, int size, int stride, int OH, int OW, const float *x, float *y,
                           float *am) {
    const size_t n = (size_t)planes * OH * OW;
    GRID_STRIDE(i, n) {
        int ox = (int)(i % OW), oy = (int)((i / OW) % OH);
        const float *xp = x + (i / ((size_t)OW * OH)) * (size_t)H * W;
        int best = (oy * stride) * W + ox * stride;
        for (int ky = 0; ky < size; ky++)
            for (int kx = 0; kx < size; kx++) {
                int idx = (oy * stride + ky) * W + ox * stride + kx;
                if (xp[idx] > xp[best]) best = idx;
            }
        y[i] = xp[best];
        am[i] = (float)best;
    }
}
/* gather: an input pixel adds the gradient of every window it won */
__global__ void k_pool_bwd(int planes, int H, int W, int size, int stride, int OH, int OW, const float *dy,
                           const float *am, float *dx) {
    const size_t n = (size_t)planes * H * W;
    GRID_STRIDE(i, n) {
        int ix = (int)(i % W), iy = (int)((i / W) % H), idx = iy * W + ix;
        size_t pl = i / ((size_t)W * H);
        const float *gp = dy + pl * (size_t)OH * OW, *ap = am + pl * (size_t)OH * OW;
        int oy0 = iy - size + 1 > 0 ? (iy - size + stride) / stride : 0, oy1 = iy / stride;
        int ox0 = ix - size + 1 > 0 ? (ix - size + stride) / stride : 0, ox1 = ix / stride;
        float sum = 0.0f;
        if (oy1 >= OH) oy1 = OH - 1;
        if (ox1 >= OW) ox1 = OW - 1;
        for (int oy = oy0; oy <= oy1; oy++)
            for (int ox = ox0; ox <= ox1; ox++)
                if ((int)ap[oy * OW + ox] == idx) sum += gp[oy * OW + ox];
        dx[i] = sum;
    }
}

/* ----------------------------------------------------- dropout, lookups */

__device__ __forceinline__ uint64_t mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}
/* counter-based: element i of a mask drawn with key */
__global__ void k_dropout(size_t n, float rate, float keep, uint64_t key, float *mask) {
    GRID_STRIDE(i, n) {
        float u = (float)(mix64(key + (i + 1) * 0x9E3779B97F4A7C15ull) >> 40) * (1.0f / 16777216.0f);
        mask[i] = u < rate ? 0.0f : keep;
    }
}
__global__ void k_embed_fwd(size_t n, int V, int D, const float *ids, const float *W, float *y) {
    GRID_STRIDE(i, n * (size_t)D) {
        size_t r = i / D;
        int id = (int)ids[r];
        id = id < 0 ? 0 : id >= V ? V - 1 : id; /* checked on the host; never read out of bounds */
        y[i] = W[(size_t)id * D + i % D];
    }
}
/* deterministic: one thread per (token, column) adds its occurrences in order */
__global__ void k_embed_bwd(size_t n, int V, int D, const float *ids, const float *dy, float *dW) {
    GRID_STRIDE(i, (size_t)V * D) {
        int v = (int)(i / D), k = (int)(i % D), hit = 0;
        float s = 0.0f;
        for (size_t r = 0; r < n; r++)
            if ((int)ids[r] == v) s += dy[r * D + k], hit = 1;
        if (hit) dW[i] += s;
    }
}
__global__ void k_embed_bwd_atomic(size_t n, int V, int D, const float *ids, const float *dy, float *dW) {
    GRID_STRIDE(i, n * (size_t)D) {
        int id = (int)ids[i / D];
        if (id >= 0 && id < V) atomicAdd(&dW[(size_t)id * D + i % D], dy[i]);
    }
}
__global__ void k_posenc_fwd(int B, int T, int D, int off0, const int *off, const float *x, const float *P, float *y) {
    GRID_STRIDE(i, (size_t)B * T * D) {
        size_t k = i % D, bt = i / D, b = bt / T, t = bt % T;
        int o = off ? off[b] : off0;
        y[i] = x[i] + P[((size_t)o + t) * D + k];
    }
}
__global__ void k_posenc_bwd_same(int B, int T, int D, int off0, const float *dy, float *dP) {
    GRID_STRIDE(i, (size_t)T * D) { /* every row at the same offset: sum over rows in order */
        size_t k = i % D, t = i / D;
        float s = 0.0f;
        for (int b = 0; b < B; b++) s += dy[((size_t)b * T + t) * D + k];
        dP[((size_t)off0 + t) * D + k] += s;
    }
}
__global__ void k_posenc_bwd_atomic(int B, int T, int D, const int *off, const float *dy, float *dP) {
    GRID_STRIDE(i, (size_t)B * T * D) {
        size_t k = i % D, bt = i / D, b = bt / T, t = bt % T;
        atomicAdd(&dP[((size_t)off[b] + t) * D + k], dy[i]);
    }
}

/* ------------------------------------------------------------ LayerNorm */

__global__ void k_ln_fwd(size_t rows, int D, float eps, const float *x, const float *g, const float *b, float *y,
                         float *xh, float *mean, float *rstd) {
    for (size_t r = WARP_ID(); r < rows; r += WARP_STRIDE()) {
        const float *xr = x + r * D;
        int lane = threadIdx.x & 31;
        double mu = 0, var = 0, rs;
        for (int k = lane; k < D; k += 32) mu += xr[k];
        mu = warp_sumd(mu) / D;
        for (int k = lane; k < D; k += 32) var += ((double)xr[k] - mu) * ((double)xr[k] - mu);
        var = warp_sumd(var) / D;
        rs = 1.0 / sqrt(var + eps);
        if (lane == 0) mean[r] = (float)mu, rstd[r] = (float)rs;
        for (int k = lane; k < D; k += 32) {
            float h = (float)(((double)xr[k] - mu) * rs);
            xh[r * D + k] = h;
            y[r * D + k] = g[k] * h + b[k];
        }
    }
}
__global__ void k_ln_bwd(size_t rows, int D, const float *dy, const float *xh, const float *rstd, const float *gain,
                         float *dx, float *dxhat) {
    for (size_t r = WARP_ID(); r < rows; r += WARP_STRIDE()) {
        const float *g = dy + r * D, *h = xh + r * D;
        int lane = threadIdx.x & 31;
        double m1 = 0, m2 = 0, rs = rstd[r];
        for (int k = lane; k < D; k += 32) {
            double d = (double)g[k] * gain[k];
            m1 += d;
            m2 += d * h[k];
            if (dxhat) dxhat[r * D + k] = (float)d;
        }
        m1 = warp_sumd(m1) / D;
        m2 = warp_sumd(m2) / D;
        if (dx)
            for (int k = lane; k < D; k += 32) dx[r * D + k] = (float)(rs * ((double)g[k] * gain[k] - m1 - h[k] * m2));
    }
}

/* ------------------------------------------------------------ attention */

/* one warp per query row (b, h, t): scores, softmax over keys 0 … Tp+t, context */
__global__ void k_attn_fwd(int B, int H, int T, int Tk, int Tp, int d, int E, float scale, const float *qkv,
                           const float *kv, float *P, float *S, float *ctx) {
    for (size_t w = WARP_ID(); w < (size_t)B * H * T; w += WARP_STRIDE()) {
        int t = (int)(w % T), h = (int)((w / T) % H), b = (int)(w / ((size_t)T * H)), last = Tp + t;
        int lane = threadIdx.x & 31;
        const float *q = qkv + ((size_t)b * T + t) * 3 * E + (size_t)h * d;
        float *p = P + (((size_t)b * H + h) * T + t) * Tk;
        float mx = -INFINITY, sum = 0.0f;
        for (int j = lane; j <= last; j += 32) {
            const float *k = kv + ((size_t)b * Tk + j) * 2 * E + (size_t)h * d;
            float s = 0.0f;
            for (int c = 0; c < d; c++) s += q[c] * k[c];
            s *= scale;
            p[j] = s;
            if (S) S[(((size_t)b * H + h) * T + t) * Tk + j] = s;
            mx = fmaxf(mx, s);
        }
        mx = warp_max(mx);
        for (int j = lane; j <= last; j += 32) {
            float e = expf(p[j] - mx);
            p[j] = e;
            sum += e;
        }
        sum = warp_sum(sum);
        for (int j = lane; j <= last; j += 32) p[j] = p[j] / sum;
        for (int j = last + 1 + lane; j < Tk; j += 32) p[j] = 0.0f; /* masked: the future */
        __syncwarp();
        for (int c = lane; c < d; c += 32) {
            const float *v = kv + (size_t)b * Tk * 2 * E + E + (size_t)h * d + c;
            float acc = 0.0f;
            for (int j = 0; j <= last; j++) acc += p[j] * v[(size_t)j * 2 * E];
            ctx[((size_t)b * T + t) * E + (size_t)h * d + c] = acc;
        }
    }
}
/* per query row: ∂L/∂scores (into dP) and ∂L/∂q (added into the q columns) */
__global__ void k_attn_bwd_q(int B, int H, int T, int Tk, int Tp, int d, int E, float scale, const float *qkv,
                             const float *kv, const float *P, const float *dctx, float *dP, float *dqkv) {
    for (size_t w = WARP_ID(); w < (size_t)B * H * T; w += WARP_STRIDE()) {
        int t = (int)(w % T), h = (int)((w / T) % H), b = (int)(w / ((size_t)T * H)), last = Tp + t;
        int lane = threadIdx.x & 31;
        const float *dc = dctx + ((size_t)b * T + t) * E + (size_t)h * d;
        const float *p = P + (((size_t)b * H + h) * T + t) * Tk;
        float *dp = dP + (((size_t)b * H + h) * T + t) * Tk;
        float dot = 0.0f;
        for (int j = lane; j <= last; j += 32) {
            const float *v = kv + ((size_t)b * Tk + j) * 2 * E + E + (size_t)h * d;
            float g = 0.0f;
            for (int c = 0; c < d; c++) g += dc[c] * v[c];
            dp[j] = g;
            dot += p[j] * g;
        }
        dot = warp_sum(dot);
        for (int j = lane; j <= last; j += 32) dp[j] = p[j] * (dp[j] - dot);
        for (int j = last + 1 + lane; j < Tk; j += 32) dp[j] = 0.0f;
        __syncwarp();
        for (int c = lane; c < d; c += 32) {
            const float *k = kv + (size_t)b * Tk * 2 * E + (size_t)h * d + c;
            float acc = 0.0f;
            for (int j = 0; j <= last; j++) acc += dp[j] * k[(size_t)j * 2 * E];
            dqkv[((size_t)b * T + t) * 3 * E + (size_t)h * d + c] += acc * scale;
        }
    }
}
/* per key row (b, h, j): ∂L/∂k_j and ∂L/∂v_j from every query that saw it */
__global__ void k_attn_bwd_kv(int B, int H, int T, int Tk, int Tp, int d, int E, float scale, const float *qkv,
                              const float *P, const float *dctx, const float *dP, float *dkv) {
    for (size_t w = WARP_ID(); w < (size_t)B * H * Tk; w += WARP_STRIDE()) {
        int j = (int)(w % Tk), h = (int)((w / Tk) % H), b = (int)(w / ((size_t)Tk * H));
        int lane = threadIdx.x & 31, t0 = j - Tp > 0 ? j - Tp : 0;
        for (int c = lane; c < d; c += 32) {
            float dk = 0.0f, dv = 0.0f;
            for (int t = t0; t < T; t++) {
                size_t pi = (((size_t)b * H + h) * T + t) * Tk + j;
                dv += P[pi] * dctx[((size_t)b * T + t) * E + (size_t)h * d + c];
                dk += dP[pi] * qkv[((size_t)b * T + t) * 3 * E + (size_t)h * d + c];
            }
            dkv[((size_t)b * Tk + j) * 2 * E + (size_t)h * d + c] += dk * scale;
            dkv[((size_t)b * Tk + j) * 2 * E + E + (size_t)h * d + c] += dv;
        }
    }
}


/* the attention softmax over the visible keys of each score row (b, h, t):
 * from S (raw scores, masked entries set to 0 for the trace) into P */
__global__ void k_attn_softmax(size_t rows, int T, int Tk, int Tp, float *S, float *P) {
    for (size_t r = WARP_ID(); r < rows; r += WARP_STRIDE()) {
        int lane = threadIdx.x & 31, last = Tp + (int)(r % T);
        float *sr = S + r * Tk, *pr = P + r * Tk, mx = -INFINITY;
        double sum = 0;
        for (int j = lane; j <= last; j += 32) mx = fmaxf(mx, sr[j]);
        mx = warp_max(mx);
        for (int j = lane; j <= last; j += 32) sum += expf(sr[j] - mx);
        sum = warp_sumd(sum);
        for (int j = lane; j < Tk; j += 32) {
            if (j <= last) {
                pr[j] = (float)(expf(sr[j] - mx) / sum);
            } else {
                pr[j] = 0.0f; /* masked: the future */
                if (S != P) sr[j] = 0.0f;
            }
        }
    }
}
/* dP (∂L/∂weights in) → ∂L/∂scores (out): p ⊙ (dp − Σ p·dp), 0 where masked */
__global__ void k_attn_softmax_bwd(size_t rows, int T, int Tk, int Tp, const float *P, float *dP) {
    for (size_t r = WARP_ID(); r < rows; r += WARP_STRIDE()) {
        int lane = threadIdx.x & 31, last = Tp + (int)(r % T);
        const float *pr = P + r * Tk;
        float *dr = dP + r * Tk;
        double dot = 0;
        for (int j = lane; j <= last; j += 32) dot += (double)pr[j] * dr[j];
        dot = warp_sumd(dot);
        for (int j = lane; j < Tk; j += 32) dr[j] = j <= last ? (float)(pr[j] * ((double)dr[j] - dot)) : 0.0f;
    }
}

/* ------------------------------------------------------ recurrent cells */

#define AT(r, b, j) ((r).p[(size_t)(b) * (r).ld + (j)])
#define REC(field, b, j) \
    if (rec.field) rec.field[((size_t)(b) * rec.T + t) * H + (j)]

__global__ void k_rnn_fwd(int B, int H, nmz_rows z, nmz_rows h) {
    GRID_STRIDE(i, (size_t)B * H) {
        int b = (int)(i / H), j = (int)(i % H);
        AT(h, b, j) = tanhf(AT(z, b, j));
    }
}
__global__ void k_rnn_bwd(int B, int H, nmz_rows h, nmz_rows dy, nmz_rows dn, nmz_rows dz, nmz_seq_record rec, int t) {
    GRID_STRIDE(i, (size_t)B * H) {
        int b = (int)(i / H), j = (int)(i % H);
        float hv = AT(h, b, j), n = AT(dn, b, j), d = AT(dy, b, j) + n;
        AT(dz, b, j) = d * (1.0f - hv * hv);
        REC(dh_next, b, j) = n;
        REC(dh, b, j) = d;
    }
}
__device__ __forceinline__ float sig(float a) { return 1.0f / (1.0f + expf(-a)); }
__global__ void k_lstm_fwd(int B, int H, nmz_rows a, nmz_rows cp, nmz_rows c, nmz_rows h) {
    GRID_STRIDE(i, (size_t)B * H) {
        int b = (int)(i / H), j = (int)(i % H);
        float ig = sig(AT(a, b, j)), fg = sig(AT(a, b, H + j)), gg = tanhf(AT(a, b, 2 * H + j)),
              og = sig(AT(a, b, 3 * H + j));
        float cv = fg * AT(cp, b, j) + ig * gg;
        AT(a, b, j) = ig;
        AT(a, b, H + j) = fg;
        AT(a, b, 2 * H + j) = gg;
        AT(a, b, 3 * H + j) = og;
        AT(c, b, j) = cv;
        AT(h, b, j) = og * tanhf(cv);
    }
}
__global__ void k_lstm_bwd(int B, int H, nmz_rows gt, nmz_rows c, nmz_rows cp, nmz_rows dy, nmz_rows dhn, nmz_rows dcn,
                           nmz_rows da, nmz_seq_record rec, int t) {
    GRID_STRIDE(i, (size_t)B * H) {
        int b = (int)(i / H), j = (int)(i % H);
        float ig = AT(gt, b, j), fg = AT(gt, b, H + j), gg = AT(gt, b, 2 * H + j), og = AT(gt, b, 3 * H + j);
        float tc = tanhf(AT(c, b, j)), hn = AT(dhn, b, j), cn = AT(dcn, b, j);
        float dh = AT(dy, b, j) + hn, dc = cn + dh * og * (1.0f - tc * tc);
        AT(da, b, j) = dc * gg * ig * (1.0f - ig);
        AT(da, b, H + j) = dc * AT(cp, b, j) * fg * (1.0f - fg);
        AT(da, b, 2 * H + j) = dc * ig * (1.0f - gg * gg);
        AT(da, b, 3 * H + j) = dh * tc * og * (1.0f - og);
        REC(dh_next, b, j) = hn;
        REC(dh, b, j) = dh;
        REC(dc_next, b, j) = cn;
        REC(dc, b, j) = dc;
        REC(dc_prev, b, j) = dc * fg;
        AT(dcn, b, j) = dc * fg;
    }
}
__global__ void k_gru_fwd1(int B, int H, nmz_rows a, nmz_rows hp, nmz_rows rh) {
    GRID_STRIDE(i, (size_t)B * H) {
        int b = (int)(i / H), j = (int)(i % H);
        float z = sig(AT(a, b, j)), r = sig(AT(a, b, H + j));
        AT(a, b, j) = z;
        AT(a, b, H + j) = r;
        AT(rh, b, j) = r * AT(hp, b, j);
    }
}
__global__ void k_gru_fwd2(int B, int H, nmz_rows a, nmz_rows hp, nmz_rows h) {
    GRID_STRIDE(i, (size_t)B * H) {
        int b = (int)(i / H), j = (int)(i % H);
        float z = AT(a, b, j), n = tanhf(AT(a, b, 2 * H + j));
        AT(a, b, 2 * H + j) = n;
        AT(h, b, j) = z * AT(hp, b, j) + (1.0f - z) * n;
    }
}
__global__ void k_gru_bwd1(int B, int H, nmz_rows gt, nmz_rows hp, nmz_rows dy, nmz_rows dhn, nmz_rows da,
                           nmz_rows direct, nmz_rows rh, nmz_seq_record rec, int t) {
    GRID_STRIDE(i, (size_t)B * H) {
        int b = (int)(i / H), j = (int)(i % H);
        float z = AT(gt, b, j), r = AT(gt, b, H + j), n = AT(gt, b, 2 * H + j), h = AT(hp, b, j), hn = AT(dhn, b, j);
        float dh = AT(dy, b, j) + hn;
        AT(da, b, j) = dh * (h - n) * z * (1.0f - z);
        AT(da, b, 2 * H + j) = dh * (1.0f - z) * (1.0f - n * n);
        AT(direct, b, j) = dh * z;
        AT(rh, b, j) = r * h;
        REC(dh_next, b, j) = hn;
        REC(dh, b, j) = dh;
        REC(direct, b, j) = dh * z;
    }
}
__global__ void k_gru_bwd2(int B, int H, nmz_rows gt, nmz_rows hp, nmz_rows drh, nmz_rows direct, nmz_rows da,
                           nmz_rows dhn) {
    GRID_STRIDE(i, (size_t)B * H) {
        int b = (int)(i / H), j = (int)(i % H);
        float r = AT(gt, b, H + j), d = AT(drh, b, j);
        AT(da, b, H + j) = d * AT(hp, b, j) * r * (1.0f - r);
        AT(dhn, b, j) = AT(direct, b, j) + d * r;
    }
}


/* ------------------------------------------- fused recurrent sequences
 * A recurrence is sequential in t but independent across rows: one thread
 * block per row runs every step of it, keeping h (and c) in shared memory
 * and reading W_h from the cache, so a sequence costs one launch instead
 * of two per step. The input product of all steps and the weight
 * gradients of all steps stay single cuBLAS GEMMs around these kernels. */

#define SEQ_REC(field, j) \
    if (rec.field) rec.field[row * H + (j)]

__global__ void k_rnn_seq_fwd(int T, int H, const float *h0, const float *Whh, float *z, float *y) {
    extern __shared__ float sh[]; /* h [H] */
    const int b = blockIdx.x, j = threadIdx.x;
    if (j < H) sh[j] = h0[(size_t)b * H + j];
    __syncthreads();
    for (int t = 0; t < T; t++) {
        size_t row = (size_t)b * T + t;
        float acc = 0.0f;
        if (j < H) {
            acc = z[row * H + j];
            for (int i = 0; i < H; i++) acc += sh[i] * Whh[(size_t)i * H + j];
        }
        __syncthreads();
        if (j < H) {
            float hv = tanhf(acc);
            z[row * H + j] = acc;
            y[row * H + j] = hv;
            sh[j] = hv;
        }
        __syncthreads();
    }
}

/* WhT = W_hᵀ ([G, H]): ∂L/∂h_{t−1} reads it by rows, coalesced */
__global__ void k_rnn_seq_bwd(int T, int H, const float *y, const float *dy, const float *WhT, float *dz,
                              float *dh_next, nmz_seq_record rec) {
    extern __shared__ float sh[]; /* dz_t [H] */
    const int b = blockIdx.x, j = threadIdx.x;
    float dn = 0.0f;
    for (int t = T - 1; t >= 0; t--) {
        size_t row = (size_t)b * T + t;
        if (j < H) {
            float hv = y[row * H + j], dh = dy[row * H + j] + dn, d = dh * (1.0f - hv * hv);
            dz[row * H + j] = d;
            sh[j] = d;
            SEQ_REC(dh_next, j) = dn;
            SEQ_REC(dh, j) = dh;
            SEQ_REC(da, j) = d;
        }
        __syncthreads();
        if (j < H) {
            float acc = 0.0f;
            for (int k = 0; k < H; k++) acc += sh[k] * WhT[(size_t)k * H + j];
            dn = acc;
            SEQ_REC(dh_prev, j) = acc;
        }
        __syncthreads();
    }
    if (j < H) dh_next[(size_t)b * H + j] = dn;
}

__global__ void k_lstm_seq_fwd(int T, int H, const float *h0, const float *c0, const float *Wh, float *gates, float *c,
                               float *y) {
    extern __shared__ float sh[]; /* h [H], a [4H] */
    const int b = blockIdx.x, j = threadIdx.x, G = 4 * H;
    float *h = sh, *a = sh + H, cprev = 0.0f;
    if (j < H) h[j] = h0[(size_t)b * H + j], cprev = c0[(size_t)b * H + j];
    __syncthreads();
    for (int t = 0; t < T; t++) {
        size_t row = (size_t)b * T + t;
        float *gt = gates + row * G;
        if (j < G) {
            float acc = gt[j];
            for (int i = 0; i < H; i++) acc += h[i] * Wh[(size_t)i * G + j];
            a[j] = acc;
        }
        __syncthreads();
        if (j < H) {
            float ig = sig(a[j]), fg = sig(a[H + j]), gg = tanhf(a[2 * H + j]), og = sig(a[3 * H + j]);
            float cv = fg * cprev + ig * gg, hv = og * tanhf(cv);
            cprev = cv;
            gt[j] = ig, gt[H + j] = fg, gt[2 * H + j] = gg, gt[3 * H + j] = og;
            c[row * H + j] = cv;
            y[row * H + j] = hv;
            h[j] = hv;
        }
        __syncthreads();
    }
}

__global__ void k_lstm_seq_bwd(int T, int H, const float *gates, const float *c, const float *c0, const float *dy,
                               const float *WhT, float *dA, float *dh_next, float *dc_next, nmz_seq_record rec) {
    extern __shared__ float sh[]; /* da_t [4H] */
    const int b = blockIdx.x, j = threadIdx.x, G = 4 * H;
    float dhn = 0.0f, dcn = 0.0f;
    for (int t = T - 1; t >= 0; t--) {
        size_t row = (size_t)b * T + t;
        if (j < H) {
            const float *gt = gates + row * G;
            float ig = gt[j], fg = gt[H + j], gg = gt[2 * H + j], og = gt[3 * H + j], tc = tanhf(c[row * H + j]);
            float cp = t ? c[(row - 1) * H + j] : c0[(size_t)b * H + j];
            float dh = dy[row * H + j] + dhn, dc = dcn + dh * og * (1.0f - tc * tc);
            sh[j] = dc * gg * ig * (1.0f - ig);
            sh[H + j] = dc * cp * fg * (1.0f - fg);
            sh[2 * H + j] = dc * ig * (1.0f - gg * gg);
            sh[3 * H + j] = dh * tc * og * (1.0f - og);
            SEQ_REC(dh_next, j) = dhn;
            SEQ_REC(dh, j) = dh;
            SEQ_REC(dc_next, j) = dcn;
            SEQ_REC(dc, j) = dc;
            SEQ_REC(dc_prev, j) = dc * fg;
            dcn = dc * fg;
        }
        __syncthreads();
        if (j < G) {
            dA[row * G + j] = sh[j];
            if (rec.da) rec.da[row * G + j] = sh[j];
        }
        if (j < H) {
            float acc = 0.0f;
            for (int k = 0; k < G; k++) acc += sh[k] * WhT[(size_t)k * H + j];
            dhn = acc;
            SEQ_REC(dh_prev, j) = acc;
        }
        __syncthreads();
    }
    if (j < H) dh_next[(size_t)b * H + j] = dhn, dc_next[(size_t)b * H + j] = dcn;
}

__global__ void k_gru_seq_fwd(int T, int H, const float *h0, const float *Wh, float *gates, float *y) {
    extern __shared__ float sh[]; /* h [H], a [2H], rh [H] */
    const int b = blockIdx.x, j = threadIdx.x, G = 3 * H;
    float *h = sh, *a = sh + H, *rh = sh + 3 * H;
    if (j < H) h[j] = h0[(size_t)b * H + j];
    __syncthreads();
    for (int t = 0; t < T; t++) {
        size_t row = (size_t)b * T + t;
        float *gt = gates + row * G;
        if (j < 2 * H) { /* z and r see h_{t−1} */
            float acc = gt[j];
            for (int i = 0; i < H; i++) acc += h[i] * Wh[(size_t)i * G + j];
            a[j] = acc;
        }
        __syncthreads();
        if (j < H) {
            float z = sig(a[j]), r = sig(a[H + j]);
            gt[j] = z, gt[H + j] = r;
            rh[j] = r * h[j];
        }
        __syncthreads();
        if (j < H) { /* the candidate sees r ⊙ h_{t−1} */
            float acc = gt[2 * H + j], n, z = gt[j];
            for (int i = 0; i < H; i++) acc += rh[i] * Wh[(size_t)i * G + 2 * H + j];
            n = tanhf(acc);
            gt[2 * H + j] = n;
            y[row * H + j] = z * h[j] + (1.0f - z) * n;
            h[j] = y[row * H + j]; /* only this thread reads h[j] in this phase */
        }
        __syncthreads();
    }
}

__global__ void k_gru_seq_bwd(int T, int H, const float *y, const float *h0, const float *gates, const float *dy,
                              const float *WhT, float *dA, float *RH, float *dh_next, nmz_seq_record rec) {
    extern __shared__ float sh[]; /* da_t [3H] */
    const int b = blockIdx.x, j = threadIdx.x, G = 3 * H;
    float dhn = 0.0f, direct = 0.0f, part = 0.0f, hp = 0.0f, r = 0.0f;
    for (int t = T - 1; t >= 0; t--) {
        size_t row = (size_t)b * T + t;
        if (j < H) {
            const float *gt = gates + row * G;
            float z = gt[j], n = gt[2 * H + j], dh;
            r = gt[H + j];
            hp = t ? y[(row - 1) * H + j] : h0[(size_t)b * H + j];
            dh = dy[row * H + j] + dhn;
            sh[j] = dh * (hp - n) * z * (1.0f - z);
            sh[2 * H + j] = dh * (1.0f - z) * (1.0f - n * n);
            direct = dh * z;
            RH[row * H + j] = r * hp;
            SEQ_REC(dh_next, j) = dhn;
            SEQ_REC(dh, j) = dh;
            SEQ_REC(direct, j) = direct;
        }
        __syncthreads();
        if (j < H) { /* ∂L/∂(r⊙h_{t−1}) = ∂L/∂a_ĥ·W_hĥᵀ, then ∂L/∂a_r */
            float drh = 0.0f;
            for (int k = 0; k < H; k++) drh += sh[2 * H + k] * WhT[(size_t)(2 * H + k) * H + j];
            sh[H + j] = drh * hp * r * (1.0f - r);
            part = direct + drh * r;
        }
        __syncthreads();
        if (j < G) {
            dA[row * G + j] = sh[j];
            if (rec.da) rec.da[row * G + j] = sh[j];
        }
        if (j < H) { /* + [∂L/∂a_z ∂L/∂a_r]·[W_hz W_hr]ᵀ */
            float acc = part;
            for (int k = 0; k < 2 * H; k++) acc += sh[k] * WhT[(size_t)k * H + j];
            dhn = acc;
            SEQ_REC(dh_prev, j) = acc;
        }
        __syncthreads();
    }
    if (j < H) dh_next[(size_t)b * H + j] = dhn;
}

/* ------------------------------------------------------------ optimizers */

__global__ void k_sgd(size_t n, float lr, float mu, int fresh, const float *g, const float *vp, float *vo) {
    GRID_STRIDE(i, n) vo[i] = (fresh ? 0.0f : mu * vp[i]) - lr * g[i];
}
__global__ void k_adam(size_t n, double lr, double b1, double b2, double eps, double bc1, double bc2, double decay,
                       float *w, const float *g, float *m, float *v) {
    GRID_STRIDE(i, n) {
        double gi = g[i];
        double mi = b1 * m[i] + (1.0 - b1) * gi, vi = b2 * v[i] + (1.0 - b2) * gi * gi;
        m[i] = (float)mi;
        v[i] = (float)vi;
        w[i] += (float)(-lr * ((mi / bc1) / (sqrt(vi / bc2) + eps) + decay * w[i]));
    }
}
__global__ void k_adam_preview(size_t n, double lr, double b1, double b2, double eps, double bc1, double bc2,
                               double decay, int fresh, const float *w, const float *g, const float *m, const float *v,
                               float *mo, float *vo, float *mh, float *vh, float *dl) {
    GRID_STRIDE(i, n) {
        double gi = g[i], m0 = fresh ? 0.0 : m[i], v0 = fresh ? 0.0 : v[i];
        double mi = b1 * m0 + (1.0 - b1) * gi, vi = b2 * v0 + (1.0 - b2) * gi * gi, a = mi / bc1, c = vi / bc2;
        mo[i] = (float)mi;
        vo[i] = (float)vi;
        mh[i] = (float)a;
        vh[i] = (float)c;
        dl[i] = (float)(-lr * (a / (sqrt(c) + eps) + decay * w[i]));
    }
}
__global__ void k_scale(size_t n, double f, float *x) {
    GRID_STRIDE(i, n) x[i] = (float)(x[i] * f);
}

/* ------------------------------------------------------- training loop */

__global__ void k_norm(size_t n, size_t per, size_t div, int groups, const float *off, const float *sc, float *x,
                       int grad) {
    GRID_STRIDE(i, n) {
        int gi = (int)(((i % per) / div) % (size_t)groups);
        x[i] = grad ? x[i] * sc[gi] : (x[i] - off[gi]) * sc[gi];
    }
}
__global__ void k_gather(int rows, size_t width, const float *idx, const float *src, float *dst) {
    GRID_STRIDE(i, (size_t)rows * width) {
        size_t r = i / width;
        dst[i] = src[(size_t)idx[r] * width + i % width];
    }
}
__global__ void k_onehot(size_t n, int V, const float *ids, float *out) {
    GRID_STRIDE(i, n) {
        int id = (int)ids[i];
        if (id >= 0 && id < V) out[i * V + id] = 1.0f;
    }
}

/* ======================================================= the ops (host) */

static nmz_status cu_fill(const nmz_backend *be, size_t n, float v, float *x) {
    (void)be;
    if (!n) return NMZ_OK;
    NEED(x);
    if (v == 0.0f && !signbit(v)) {
        if (cudaMemsetAsync(x, 0, n * sizeof(float), g.stream) != cudaSuccess) return check("fill");
        return NMZ_OK;
    }
    k_fill<<<blocks_for(n), TPB, 0, g.stream>>>(n, v, x);
    return check("fill");
}

static nmz_status cu_copy(const nmz_backend *be, size_t rows, size_t cols, const float *src, size_t lds, float *dst,
                          size_t ldd) {
    cudaError_t e;
    (void)be;
    if (!rows || !cols) return NMZ_OK;
    NEED(src && dst);
    if (rows == 1 || (lds == cols && ldd == cols))
        e = cudaMemcpyAsync(dst, src, rows * cols * sizeof(float), cudaMemcpyDeviceToDevice, g.stream);
    else
        e = cudaMemcpy2DAsync(dst, ldd * sizeof(float), src, lds * sizeof(float), cols * sizeof(float), rows,
                              cudaMemcpyDeviceToDevice, g.stream);
    return e == cudaSuccess ? NMZ_OK : cuda_fail(e, "copy");
}

static nmz_status cu_add(const nmz_backend *be, size_t n, const float *a, const float *b, float *o) {
    (void)be;
    if (!n) return NMZ_OK;
    NEED(a && b && o);
    k_add<<<blocks_for(n), TPB, 0, g.stream>>>(n, a, b, o);
    return check("add");
}

static nmz_status cu_mul(const nmz_backend *be, size_t n, const float *a, const float *b, float *o) {
    (void)be;
    if (!n) return NMZ_OK;
    NEED(a && b && o);
    k_mul<<<blocks_for(n), TPB, 0, g.stream>>>(n, a, b, o);
    return check("mul");
}

/* row-major C = op(A)·op(B) is column-major Cᵀ = op(B)ᵀ·op(A)ᵀ: swap the operands */
static nmz_status cu_gemm(const nmz_backend *be, unsigned flags, int M, int N, int K, const float *A, int lda,
                          const float *B, int ldb, float *C, int ldc) {
    const float one = 1.0f, beta = (flags & NMZ_MM_ACCUMULATE) ? 1.0f : 0.0f;
    int ta = (flags & NMZ_MM_TRANS_A) != 0, tb = (flags & NMZ_MM_TRANS_B) != 0;
    cublasStatus_t st;
    (void)be;
    if (M <= 0 || N <= 0) return NMZ_OK;
    if (ta && tb) return nmz_backend_error(NMZ_ERR_UNSUPPORTED, "gemm: A and B cannot both be transposed");
    NEED(C);
    if (K <= 0) {
        if (flags & NMZ_MM_ACCUMULATE) return NMZ_OK;
        if (cudaMemset2DAsync(C, (size_t)ldc * sizeof(float), 0, (size_t)N * sizeof(float), (size_t)M, g.stream) !=
            cudaSuccess)
            return check("gemm (K = 0)");
        return NMZ_OK;
    }
    NEED(A && B);
    st = g.sgemm(g.blas, tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N, N, M, K, &one, B, ldb, A, lda,
                 &beta, C, ldc);
    if (st != CUBLAS_STATUS_SUCCESS)
        return nmz_backend_error(NMZ_ERR_STATE, "cuBLAS SGEMM failed (status %d) for %dx%dx%d", (int)st, M, N, K);
    return check("gemm");
}

static nmz_status cu_add_bias(const nmz_backend *be, size_t rows, int cols, float *y, const float *b) {
    (void)be;
    if (!rows || cols <= 0) return NMZ_OK;
    NEED(y && b);
    k_add_bias<<<blocks_for(rows * cols), TPB, 0, g.stream>>>(rows * cols, cols, y, b);
    return check("add_bias");
}

static nmz_status cu_add_bias3(const nmz_backend *be, int A, int B, int C, float *y, const float *bias) {
    size_t n = (size_t)A * B * C;
    (void)be;
    if (!n) return NMZ_OK;
    NEED(y && bias);
    k_add_bias3<<<blocks_for(n), TPB, 0, g.stream>>>(n, B, C, y, bias);
    return check("add_bias3");
}

/* out[j] += Σ_r a[r][j] (·b[r][j]), two deterministic passes */
static nmz_status colsum(size_t rows, int cols, const float *a, const float *b, float *out) {
    int gy = (int)((rows + 255) / 256);
    double *part;
    if (!rows || cols <= 0) return NMZ_OK;
    if (gy > 256) gy = 256;
    if (!(part = (double *)scratch((size_t)gy * cols * sizeof(double)))) return nmz__op_failed();
    k_colsum_part<<<dim3((unsigned)((cols + 31) / 32), (unsigned)gy), dim3(32, 8), 0, g.stream>>>(rows, cols, a, b, part);
    k_colsum_finish<<<(unsigned)((cols + 127) / 128), 128, 0, g.stream>>>(gy, cols, part, out);
    return check("column sums");
}

static nmz_status cu_colsum_add(const nmz_backend *be, size_t rows, int cols, const float *a, float *out) {
    (void)be;
    if (!rows || cols <= 0) return NMZ_OK;
    NEED(a && out);
    return colsum(rows, cols, a, NULL, out);
}

static nmz_status cu_rowsum_add(const nmz_backend *be, int rows, size_t cols, const float *a, float *out) {
    (void)be;
    if (rows <= 0) return NMZ_OK;
    NEED(a && out);
    k_rowsum<<<warps_blocks((size_t)rows), TPB, 0, g.stream>>>(rows, cols, a, out);
    return check("row sums");
}

static nmz_status cu_transpose3(const nmz_backend *be, int A, int B, int C, const float *in, float *out) {
    size_t n = (size_t)A * B * C;
    (void)be;
    if (!n) return NMZ_OK;
    NEED(in && out);
    k_transpose3<<<blocks_for(n), TPB, 0, g.stream>>>(A, B, C, in, out);
    return check("transpose");
}

static nmz_status cu_act_forward(const nmz_backend *be, nmz_act act, float alpha, size_t n, const float *z, float *a) {
    (void)be;
    if (!n) return NMZ_OK;
    NEED(z && a);
    k_act_fwd<<<blocks_for(n), TPB, 0, g.stream>>>((int)act, alpha, n, z, a);
    return check("activation");
}

static nmz_status cu_act_backward(const nmz_backend *be, nmz_act act, float alpha, size_t n, const float *z,
                                  const float *dy, float *dz, float *fprime) {
    (void)be;
    if (!n) return NMZ_OK;
    NEED(z && dy && dz);
    k_act_bwd<<<blocks_for(n), TPB, 0, g.stream>>>((int)act, alpha, n, z, dy, dz, fprime);
    return check("activation backward");
}

static nmz_status cu_softmax_forward(const nmz_backend *be, size_t rows, int cols, const float *z, float *p) {
    (void)be;
    if (!rows) return NMZ_OK;
    NEED(z && p);
    k_softmax<<<warps_blocks(rows), TPB, 0, g.stream>>>(rows, cols, z, p);
    return check("softmax");
}

static nmz_status cu_softmax_backward(const nmz_backend *be, size_t rows, int cols, const float *p, const float *dy,
                                      float *dz) {
    (void)be;
    if (!rows) return NMZ_OK;
    NEED(p && dy && dz);
    k_softmax_bwd<<<warps_blocks(rows), TPB, 0, g.stream>>>(rows, cols, p, dy, dz);
    return check("softmax backward");
}

/* Σ over a per-block kernel's partials into a scalar slot (· scale) */
#define REDUCE_INTO(dst, scale, n, launch)                                                                           \
    do {                                                                                                             \
        unsigned nb_ = blocks_for(n);                                                                                \
        double *part_ = (double *)scratch((size_t)nb_ * sizeof(double)), *slot_;                                     \
        if (!part_) return nmz__op_failed();                                                                         \
        launch;                                                                                                      \
        if (!(slot_ = scalar_slot(dst))) return nmz__op_failed();                                                    \
        k_sum_partials<<<1, 1, 0, g.stream>>>(part_, (int)nb_, (scale), slot_, 0);                                   \
        return scalar_send();                                                                                        \
    } while (0)

static nmz_status cu_loss_value(const nmz_backend *be, nmz_loss kind, size_t rows, int cols, const float *y,
                                const float *t, const float *z, double *out) {
    size_t n = rows * (size_t)cols;
    (void)be;
    NEED(y && t && out);
    if (!rows) return nmz_backend_error(NMZ_ERR_ARG, "loss of an empty batch");
    if (kind == NMZ_LOSS_MSE) {
        REDUCE_INTO(out, 1.0 / (double)rows, n, (k_mse_sum<<<nb_, TPB, 0, g.stream>>>(n, y, t, part_)));
    } else {
        /* per-row losses after the block partials in the scratch buffer */
        unsigned nb = blocks_for(rows);
        double *part = (double *)scratch((size_t)nb * sizeof(double) + rows * sizeof(double)), *row_loss, *slot;
        if (!part) return nmz__op_failed();
        row_loss = part + nb;
        k_ce_rows<<<warps_blocks(rows), TPB, 0, g.stream>>>(rows, cols, y, t, z, row_loss);
        k_sum_rows<<<nb, TPB, 0, g.stream>>>(rows, row_loss, part);
        if (!(slot = scalar_slot(out))) return nmz__op_failed();
        k_sum_partials<<<1, 1, 0, g.stream>>>(part, (int)nb, 1.0 / (double)rows, slot, 0);
        return scalar_send();
    }
}

static nmz_status cu_loss_grad(const nmz_backend *be, nmz_loss kind, size_t rows, int cols, const float *y,
                               const float *t, float *gr) {
    (void)be;
    if (!rows) return NMZ_OK;
    NEED(y && t && gr);
    if (kind == NMZ_LOSS_MSE)
        k_mse_grad<<<blocks_for(rows * cols), TPB, 0, g.stream>>>(rows * cols, 1.0 / (double)rows, y, t, gr);
    else
        k_ce_grad<<<warps_blocks(rows), TPB, 0, g.stream>>>(rows, cols, 1.0 / (double)rows, y, t, gr);
    return check("loss gradient");
}

static nmz_status cu_im2col(const nmz_backend *be, const nmz_conv_shape *s, const float *x, float *col) {
    size_t n = (size_t)s->C * s->kh * s->kw * s->B * s->OH * s->OW;
    (void)be;
    if (!n) return NMZ_OK;
    NEED(x && col);
    k_im2col<<<blocks_for(n), TPB, 0, g.stream>>>(*s, x, col);
    return check("im2col");
}

static nmz_status cu_col2im(const nmz_backend *be, const nmz_conv_shape *s, const float *col, float *dx) {
    size_t n = (size_t)s->B * s->C * s->H * s->W;
    (void)be;
    if (!n) return NMZ_OK;
    NEED(col && dx);
    k_col2im<<<blocks_for(n), TPB, 0, g.stream>>>(*s, col, dx);
    return check("col2im");
}

static nmz_status cu_pool_fwd(const nmz_backend *be, int planes, int H, int W, int size, int stride, int OH, int OW,
                              const float *x, float *y, float *am) {
    size_t n = (size_t)planes * OH * OW;
    (void)be;
    if (!n) return NMZ_OK;
    NEED(x && y && am);
    k_pool_fwd<<<blocks_for(n), TPB, 0, g.stream>>>(planes, H, W, size, stride, OH, OW, x, y, am);
    return check("max-pool");
}

static nmz_status cu_pool_bwd(const nmz_backend *be, int planes, int H, int W, int size, int stride, int OH, int OW,
                              const float *dy, const float *am, float *dx) {
    size_t n = (size_t)planes * H * W;
    (void)be;
    if (!n) return NMZ_OK;
    NEED(dy && am && dx);
    k_pool_bwd<<<blocks_for(n), TPB, 0, g.stream>>>(planes, H, W, size, stride, OH, OW, dy, am, dx);
    return check("max-pool backward");
}

static nmz_status cu_dropout_mask(const nmz_backend *be, size_t n, float rate, nmz_rng *rng, float *mask) {
    uint64_t key;
    (void)be;
    if (!n) return NMZ_OK;
    NEED(mask);
    key = ((uint64_t)nmz_rng_u32(rng) << 32) | nmz_rng_u32(rng); /* the net's PRNG decides the mask */
    k_dropout<<<blocks_for(n), TPB, 0, g.stream>>>(n, rate, 1.0f / (1.0f - rate), key, mask);
    return check("dropout");
}

static nmz_status cu_embed_forward(const nmz_backend *be, size_t n, int V, int D, const float *ids, const float *W,
                                   float *y) {
    (void)be;
    if (!n) return NMZ_OK;
    NEED(ids && W && y);
    k_embed_fwd<<<blocks_for(n * D), TPB, 0, g.stream>>>(n, V, D, ids, W, y);
    return check("embedding");
}

static nmz_status cu_embed_backward(const nmz_backend *be, size_t n, int V, int D, const float *ids, const float *dy,
                                    float *dW) {
    (void)be;
    if (!n) return NMZ_OK;
    NEED(ids && dy && dW);
    if ((double)V * (double)n <= 16777216.0)
        k_embed_bwd<<<blocks_for((size_t)V * D), TPB, 0, g.stream>>>(n, V, D, ids, dy, dW);
    else
        k_embed_bwd_atomic<<<blocks_for(n * D), TPB, 0, g.stream>>>(n, V, D, ids, dy, dW);
    return check("embedding backward");
}

/* NULL when every row has the same offset (then *off0 is it); else a device copy */
static int *offsets(int B, const int *off, int *off0, nmz_status *s) {
    int b, *d;
    *s = NMZ_OK;
    *off0 = off[0];
    for (b = 1; b < B; b++)
        if (off[b] != off[0]) break;
    if (b == B) return NULL;
    d = (int *)scratch((size_t)B * sizeof(int));
    if (!d) {
        *s = nmz__op_failed();
        return NULL;
    }
    if (cudaMemcpyAsync(d, off, (size_t)B * sizeof(int), cudaMemcpyHostToDevice, g.stream) != cudaSuccess)
        *s = check("offsets");
    return d;
}

static nmz_status cu_posenc_forward(const nmz_backend *be, int B, int T, int D, const int *off, const float *x,
                                    const float *P, float *y) {
    nmz_status s;
    int o0, *d;
    (void)be;
    if (!B || !T) return NMZ_OK;
    NEED(x && P && y);
    d = offsets(B, off, &o0, &s);
    if (s != NMZ_OK) return s;
    k_posenc_fwd<<<blocks_for((size_t)B * T * D), TPB, 0, g.stream>>>(B, T, D, o0, d, x, P, y);
    return check("position encoding");
}

static nmz_status cu_posenc_backward(const nmz_backend *be, int B, int T, int D, const int *off, const float *dy,
                                     float *dP) {
    nmz_status s;
    int o0, *d;
    (void)be;
    if (!B || !T) return NMZ_OK;
    NEED(dy && dP);
    d = offsets(B, off, &o0, &s);
    if (s != NMZ_OK) return s;
    if (!d)
        k_posenc_bwd_same<<<blocks_for((size_t)T * D), TPB, 0, g.stream>>>(B, T, D, o0, dy, dP);
    else
        k_posenc_bwd_atomic<<<blocks_for((size_t)B * T * D), TPB, 0, g.stream>>>(B, T, D, d, dy, dP);
    return check("position encoding backward");
}

static nmz_status cu_ln_forward(const nmz_backend *be, size_t rows, int D, float eps, const float *x, const float *gn,
                                const float *b, float *y, float *xh, float *mean, float *rstd) {
    (void)be;
    if (!rows) return NMZ_OK;
    NEED(x && gn && b && y && xh && mean && rstd);
    k_ln_fwd<<<warps_blocks(rows), TPB, 0, g.stream>>>(rows, D, eps, x, gn, b, y, xh, mean, rstd);
    return check("layernorm");
}

static nmz_status cu_ln_backward(const nmz_backend *be, size_t rows, int D, const float *dy, const float *xh,
                                 const float *rstd, const float *gn, float *dx, float *dg, float *db, float *dxhat) {
    nmz_status s;
    (void)be;
    if (!rows) return NMZ_OK;
    NEED(dy && xh && rstd && gn && dg && db);
    k_ln_bwd<<<warps_blocks(rows), TPB, 0, g.stream>>>(rows, D, dy, xh, rstd, gn, dx, dxhat);
    if ((s = check("layernorm backward")) != NMZ_OK) return s;
    if ((s = colsum(rows, D, dy, xh, dg)) != NMZ_OK) return s; /* dg += Σ dy ⊙ x̂ */
    return colsum(rows, D, dy, NULL, db);                       /* db += Σ dy     */
}

/* row-major, strided batched: C_i = α·op(A_i)·op(B_i) + β·C_i, i = 0 … batch−1 */
static nmz_status gemm_sb(int ta, int tb, int M, int N, int K, float alpha, const float *A, int lda, long long sa,
                          const float *B, int ldb, long long sb, float beta, float *C, int ldc, long long sc, int batch) {
    cublasStatus_t st = g.sgemm_sb(g.blas, tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N, N, M, K, &alpha,
                                   B, ldb, sb, A, lda, sa, &beta, C, ldc, sc, batch);
    if (st != CUBLAS_STATUS_SUCCESS)
        return nmz_backend_error(NMZ_ERR_STATE, "cuBLAS strided batched SGEMM failed (status %d)", (int)st);
    return NMZ_OK;
}

/* Attention on the GPU. Long query blocks (training, a prompt): the
 * products of all (row, head) pairs as strided batched cuBLAS GEMMs (S =
 * Q·Kᵀ/√d for all keys, the causal softmax zeroes the future, C = P·V),
 * one call per head or per row (gemm_bh); the masked half of
 * S is computed and thrown away, which cuBLAS does faster than a masked
 * kernel. Short blocks (generation: one new position against the KV cache):
 * one warp per query row, one launch. */
#define ATTN_GEMM_MIN_T 16

/* where (row b, head h) of an attention tensor starts: X + b·rows + h·heads */
typedef struct bh_stride {
    size_t rows, heads;
} bh_stride;

/* the product for every (row, head) pair: one strided batched call per head
 * (batched over the rows) or per row (batched over the heads), whichever
 * makes fewer calls — each call costs a launch, 10–20 µs under WDDM */
static nmz_status gemm_bh(int B, int H, int ta, int tb, int M, int N, int K, float alpha, const float *A, int lda,
                          bh_stride sa, const float *Bm, int ldb, bh_stride sb, float beta, float *C, int ldc,
                          bh_stride sc) {
    const int by_head = B >= H, outer = by_head ? H : B, count = by_head ? B : H;
    int o;
    nmz_status s;
    for (o = 0; o < outer; o++) {
        size_t oa = (size_t)o * (by_head ? sa.heads : sa.rows), ob = (size_t)o * (by_head ? sb.heads : sb.rows),
               oc = (size_t)o * (by_head ? sc.heads : sc.rows);
        if ((s = gemm_sb(ta, tb, M, N, K, alpha, A + oa, lda, (long long)(by_head ? sa.rows : sa.heads), Bm + ob, ldb,
                         (long long)(by_head ? sb.rows : sb.heads), beta, C + oc, ldc,
                         (long long)(by_head ? sc.rows : sc.heads), count)) != NMZ_OK)
            return s;
    }
    return NMZ_OK;
}

static nmz_status cu_attn_forward(const nmz_backend *be, int B, int H, int T, int Tk, int Tp, int d, int E,
                                  const float *qkv, const float *kv, float *P, float *S, float *ctx) {
    const float scale = (float)(1.0 / sqrt((double)d));
    const bh_stride sq = {(size_t)T * 3 * E, (size_t)d}, skv = {(size_t)Tk * 2 * E, (size_t)d},
                    sp = {(size_t)H * T * Tk, (size_t)T * Tk}, sc = {(size_t)T * E, (size_t)d};
    nmz_status s;
    (void)be;
    if (!B || !T) return NMZ_OK;
    NEED(qkv && kv && P && ctx);
    if (T < ATTN_GEMM_MIN_T) {
        k_attn_fwd<<<warps_blocks((size_t)B * H * T), TPB, 0, g.stream>>>(B, H, T, Tk, Tp, d, E, scale, qkv, kv, P, S,
                                                                           ctx);
        return check("attention");
    }
    /* scores S = Q·Kᵀ/√d, [T, Tk] per (row, head); then the causal softmax */
    if ((s = gemm_bh(B, H, 0, 1, T, Tk, d, scale, qkv, 3 * E, sq, kv, 2 * E, skv, 0.0f, S ? S : P, Tk, sp)) != NMZ_OK)
        return s;
    k_attn_softmax<<<warps_blocks((size_t)B * H * T), TPB, 0, g.stream>>>((size_t)B * H * T, T, Tk, Tp, S ? S : P, P);
    if ((s = check("attention softmax")) != NMZ_OK) return s;
    /* context = weights · values, the heads side by side */
    return gemm_bh(B, H, 0, 0, T, d, Tk, 1.0f, P, Tk, sp, kv + E, 2 * E, skv, 0.0f, ctx, E, sc);
}

static nmz_status cu_attn_backward(const nmz_backend *be, int B, int H, int T, int Tk, int Tp, int d, int E,
                                   const float *qkv, const float *kv, const float *P, const float *dctx, float *dP,
                                   float *dqkv, float *dkv) {
    const float scale = (float)(1.0 / sqrt((double)d));
    const bh_stride sq = {(size_t)T * 3 * E, (size_t)d}, skv = {(size_t)Tk * 2 * E, (size_t)d},
                    sp = {(size_t)H * T * Tk, (size_t)T * Tk}, sc = {(size_t)T * E, (size_t)d};
    nmz_status s;
    (void)be;
    if (!B || !T) return NMZ_OK;
    NEED(qkv && kv && P && dctx && dP && dqkv && dkv);
    if (T < ATTN_GEMM_MIN_T) {
        k_attn_bwd_q<<<warps_blocks((size_t)B * H * T), TPB, 0, g.stream>>>(B, H, T, Tk, Tp, d, E, scale, qkv, kv, P,
                                                                             dctx, dP, dqkv);
        k_attn_bwd_kv<<<warps_blocks((size_t)B * H * Tk), TPB, 0, g.stream>>>(B, H, T, Tk, Tp, d, E, scale, qkv, P,
                                                                               dctx, dP, dkv);
        return check("attention backward");
    }
    /* ∂L/∂weights = ∂L/∂context · Vᵀ, then through the softmax: ∂L/∂S */
    if ((s = gemm_bh(B, H, 0, 1, T, Tk, d, 1.0f, dctx, E, sc, kv + E, 2 * E, skv, 0.0f, dP, Tk, sp)) != NMZ_OK) return s;
    k_attn_softmax_bwd<<<warps_blocks((size_t)B * H * T), TPB, 0, g.stream>>>((size_t)B * H * T, T, Tk, Tp, P, dP);
    if ((s = check("attention softmax backward")) != NMZ_OK) return s;
    /* ∂L/∂q += ∂L/∂S · K/√d   ∂L/∂k += ∂L/∂Sᵀ · Q/√d   ∂L/∂v += Pᵀ · ∂L/∂context */
    if ((s = gemm_bh(B, H, 0, 0, T, d, Tk, scale, dP, Tk, sp, kv, 2 * E, skv, 1.0f, dqkv, 3 * E, sq)) != NMZ_OK ||
        (s = gemm_bh(B, H, 1, 0, Tk, d, T, scale, dP, Tk, sp, qkv, 3 * E, sq, 1.0f, dkv, 2 * E, skv)) != NMZ_OK)
        return s;
    return gemm_bh(B, H, 1, 0, Tk, d, T, 1.0f, P, Tk, sp, dctx, E, sc, 1.0f, dkv + E, 2 * E, skv);
}

static nmz_seq_record rec_of(const nmz_seq_record *r) {
    nmz_seq_record z;
    if (r) return *r;
    memset(&z, 0, sizeof z);
    return z;
}

#define CELL(n) blocks_for((size_t)(n)), TPB, 0, g.stream
static nmz_status cu_rnn_cell_forward(const nmz_backend *be, int B, int H, nmz_rows z, nmz_rows h) {
    (void)be;
    NEED(z.p && h.p);
    k_rnn_fwd<<<CELL(B * H)>>>(B, H, z, h);
    return check("rnn cell");
}
static nmz_status cu_rnn_cell_backward(const nmz_backend *be, int B, int H, nmz_rows h, nmz_rows dy, nmz_rows dh_next,
                                       nmz_rows dz, const nmz_seq_record *rec, int t) {
    (void)be;
    NEED(h.p && dy.p && dh_next.p && dz.p);
    k_rnn_bwd<<<CELL(B * H)>>>(B, H, h, dy, dh_next, dz, rec_of(rec), t);
    return check("rnn cell backward");
}
static nmz_status cu_lstm_cell_forward(const nmz_backend *be, int B, int H, nmz_rows a, nmz_rows c_prev, nmz_rows c,
                                       nmz_rows h) {
    (void)be;
    NEED(a.p && c_prev.p && c.p && h.p);
    k_lstm_fwd<<<CELL(B * H)>>>(B, H, a, c_prev, c, h);
    return check("lstm cell");
}
static nmz_status cu_lstm_cell_backward(const nmz_backend *be, int B, int H, nmz_rows gates, nmz_rows c,
                                        nmz_rows c_prev, nmz_rows dy, nmz_rows dh_next, nmz_rows dc_next, nmz_rows da,
                                        const nmz_seq_record *rec, int t) {
    (void)be;
    NEED(gates.p && c.p && c_prev.p && dy.p && dh_next.p && dc_next.p && da.p);
    k_lstm_bwd<<<CELL(B * H)>>>(B, H, gates, c, c_prev, dy, dh_next, dc_next, da, rec_of(rec), t);
    return check("lstm cell backward");
}
static nmz_status cu_gru_cell_forward1(const nmz_backend *be, int B, int H, nmz_rows a, nmz_rows h_prev, nmz_rows rh) {
    (void)be;
    NEED(a.p && h_prev.p && rh.p);
    k_gru_fwd1<<<CELL(B * H)>>>(B, H, a, h_prev, rh);
    return check("gru cell");
}
static nmz_status cu_gru_cell_forward2(const nmz_backend *be, int B, int H, nmz_rows a, nmz_rows h_prev, nmz_rows h) {
    (void)be;
    NEED(a.p && h_prev.p && h.p);
    k_gru_fwd2<<<CELL(B * H)>>>(B, H, a, h_prev, h);
    return check("gru cell");
}
static nmz_status cu_gru_cell_backward1(const nmz_backend *be, int B, int H, nmz_rows gates, nmz_rows h_prev,
                                        nmz_rows dy, nmz_rows dh_next, nmz_rows da, nmz_rows direct, nmz_rows rh,
                                        const nmz_seq_record *rec, int t) {
    (void)be;
    NEED(gates.p && h_prev.p && dy.p && dh_next.p && da.p && direct.p && rh.p);
    k_gru_bwd1<<<CELL(B * H)>>>(B, H, gates, h_prev, dy, dh_next, da, direct, rh, rec_of(rec), t);
    return check("gru cell backward");
}
static nmz_status cu_gru_cell_backward2(const nmz_backend *be, int B, int H, nmz_rows gates, nmz_rows h_prev,
                                        nmz_rows drh, nmz_rows direct, nmz_rows da, nmz_rows dh_next) {
    (void)be;
    NEED(gates.p && h_prev.p && drh.p && direct.p && da.p && dh_next.p);
    k_gru_bwd2<<<CELL(B * H)>>>(B, H, gates, h_prev, drh, direct, da, dh_next);
    return check("gru cell backward");
}


/* ------------------------------------------------- fused sequences (host) */

static unsigned seq_threads(int n) { return (unsigned)((n + 31) / 32 * 32); }

/* W [H, G] → WhT [G, H] (work buffer 0) */
static float *transposed(const float *W, int H, int G) {
    float *t = work(0, (size_t)G * H);
    if (t) k_transpose3<<<blocks_for((size_t)G * H), TPB, 0, g.stream>>>(H, G, 1, W, t);
    return t;
}

/* the weight gradients of all steps at once, as in the layers' composition:
 * dWx += xᵀ·dA, dWh (columns c0 … c0+n) += prevᵀ·dA, db += Σ dA, dx = dA·Wxᵀ */
static nmz_status seq_grads(int B, int T, int D, int H, int G, const float *x, const float *y, const float *h0,
                            const float *Wx, const float *dA, float *dWx, float *dWh, float *db, float *dx, int cols) {
    float *prev = work(2, (size_t)B * T * H);
    nmz_status s;
    if (!prev) return nmz__op_failed();
    if ((s = cu_gemm(&g_be, NMZ_MM_TRANS_A | NMZ_MM_ACCUMULATE, D, G, B * T, x, D, dA, G, dWx, G)) != NMZ_OK) return s;
    if (T > 1 && (s = cu_copy(&g_be, (size_t)B, (size_t)(T - 1) * H, y, (size_t)T * H, prev + H, (size_t)T * H)) != NMZ_OK)
        return s;
    if ((s = cu_copy(&g_be, (size_t)B, (size_t)H, h0, (size_t)H, prev, (size_t)T * H)) != NMZ_OK) return s;
    if ((s = cu_gemm(&g_be, NMZ_MM_TRANS_A | NMZ_MM_ACCUMULATE, H, cols, B * T, prev, H, dA, G, dWh, G)) != NMZ_OK)
        return s;
    if ((s = colsum((size_t)B * T, G, dA, NULL, db)) != NMZ_OK) return s;
    return dx ? cu_gemm(&g_be, NMZ_MM_TRANS_B, B * T, D, G, dA, G, Wx, G, dx, D) : NMZ_OK;
}

static nmz_status cu_rnn_forward(const nmz_backend *be, int B, int T, int D, int H, const float *x, const float *h0,
                                 const float *Wxh, const float *Whh, const float *b, float *y, float *z) {
    nmz_status s;
    (void)be;
    if (H > 1024) return NMZ_ERR_UNSUPPORTED; /* one block per row: the layer composes it instead */
    if (!B || !T) return NMZ_OK;
    NEED(x && h0 && Wxh && Whh && b && y);
    if (!z && !(z = work(1, (size_t)B * T * H))) return nmz__op_failed();
    if ((s = cu_gemm(&g_be, 0, B * T, H, D, x, D, Wxh, H, z, H)) != NMZ_OK ||
        (s = cu_add_bias(&g_be, (size_t)B * T, H, z, b)) != NMZ_OK)
        return s;
    k_rnn_seq_fwd<<<(unsigned)B, seq_threads(H), (size_t)H * sizeof(float), g.stream>>>(T, H, h0, Whh, z, y);
    return check("rnn sequence");
}

static nmz_status cu_rnn_backward(const nmz_backend *be, int B, int T, int D, int H, const float *x, const float *y,
                                  const float *h0, const float *Wxh, const float *Whh, const float *dy, float *dWxh,
                                  float *dWhh, float *db, float *dx, float *dh_next, const nmz_seq_record *rec) {
    float *WhT, *dz;
    nmz_status s;
    (void)be;
    if (H > 1024) return NMZ_ERR_UNSUPPORTED;
    if (!B || !T) return NMZ_OK;
    NEED(x && y && h0 && Wxh && Whh && dy && dWxh && dWhh && db && dh_next);
    if (!(WhT = transposed(Whh, H, H)) || !(dz = work(1, (size_t)B * T * H))) return nmz__op_failed();
    k_rnn_seq_bwd<<<(unsigned)B, seq_threads(H), (size_t)H * sizeof(float), g.stream>>>(T, H, y, dy, WhT, dz, dh_next,
                                                                                         rec_of(rec));
    if ((s = check("rnn sequence backward")) != NMZ_OK) return s;
    return seq_grads(B, T, D, H, H, x, y, h0, Wxh, dz, dWxh, dWhh, db, dx, H);
}

static nmz_status cu_lstm_forward(const nmz_backend *be, int B, int T, int D, int H, const float *x, const float *h0,
                                  const float *c0, const float *Wx, const float *Wh, const float *b, float *y,
                                  float *gates, float *c) {
    const int G = 4 * H;
    nmz_status s;
    (void)be;
    if (G > 1024) return NMZ_ERR_UNSUPPORTED;
    if (!B || !T) return NMZ_OK;
    NEED(x && h0 && c0 && Wx && Wh && b && y && gates && c);
    if ((s = cu_gemm(&g_be, 0, B * T, G, D, x, D, Wx, G, gates, G)) != NMZ_OK ||
        (s = cu_add_bias(&g_be, (size_t)B * T, G, gates, b)) != NMZ_OK)
        return s;
    k_lstm_seq_fwd<<<(unsigned)B, seq_threads(G), (size_t)5 * H * sizeof(float), g.stream>>>(T, H, h0, c0, Wh, gates, c,
                                                                                             y);
    return check("lstm sequence");
}

static nmz_status cu_lstm_backward(const nmz_backend *be, int B, int T, int D, int H, const float *x, const float *y,
                                   const float *h0, const float *c0, const float *gates, const float *c,
                                   const float *Wx, const float *Wh, const float *dy, float *dWx, float *dWh,
                                   float *db, float *dx, float *dh_next, float *dc_next, const nmz_seq_record *rec) {
    const int G = 4 * H;
    float *WhT, *dA;
    nmz_status s;
    (void)be;
    if (G > 1024) return NMZ_ERR_UNSUPPORTED;
    if (!B || !T) return NMZ_OK;
    NEED(x && y && h0 && c0 && gates && c && Wx && Wh && dy && dWx && dWh && db && dh_next && dc_next);
    if (!(WhT = transposed(Wh, H, G)) || !(dA = work(1, (size_t)B * T * G))) return nmz__op_failed();
    k_lstm_seq_bwd<<<(unsigned)B, seq_threads(G), (size_t)G * sizeof(float), g.stream>>>(
        T, H, gates, c, c0, dy, WhT, dA, dh_next, dc_next, rec_of(rec));
    if ((s = check("lstm sequence backward")) != NMZ_OK) return s;
    return seq_grads(B, T, D, H, G, x, y, h0, Wx, dA, dWx, dWh, db, dx, G);
}

static nmz_status cu_gru_forward(const nmz_backend *be, int B, int T, int D, int H, const float *x, const float *h0,
                                 const float *Wx, const float *Wh, const float *b, float *y, float *gates) {
    const int G = 3 * H;
    nmz_status s;
    (void)be;
    if (2 * H > 1024) return NMZ_ERR_UNSUPPORTED;
    if (!B || !T) return NMZ_OK;
    NEED(x && h0 && Wx && Wh && b && y && gates);
    if ((s = cu_gemm(&g_be, 0, B * T, G, D, x, D, Wx, G, gates, G)) != NMZ_OK ||
        (s = cu_add_bias(&g_be, (size_t)B * T, G, gates, b)) != NMZ_OK)
        return s;
    k_gru_seq_fwd<<<(unsigned)B, seq_threads(2 * H), (size_t)4 * H * sizeof(float), g.stream>>>(T, H, h0, Wh, gates, y);
    return check("gru sequence");
}

static nmz_status cu_gru_backward(const nmz_backend *be, int B, int T, int D, int H, const float *x, const float *y,
                                  const float *h0, const float *gates, const float *Wx, const float *Wh,
                                  const float *dy, float *dWx, float *dWh, float *db, float *dx, float *dh_next,
                                  const nmz_seq_record *rec) {
    const int G = 3 * H;
    float *WhT, *dA, *RH;
    nmz_status s;
    (void)be;
    if (G > 1024) return NMZ_ERR_UNSUPPORTED;
    if (!B || !T) return NMZ_OK;
    NEED(x && y && h0 && gates && Wx && Wh && dy && dWx && dWh && db && dh_next);
    if (!(WhT = transposed(Wh, H, G)) || !(dA = work(1, (size_t)B * T * G)) || !(RH = work(3, (size_t)B * T * H)))
        return nmz__op_failed();
    k_gru_seq_bwd<<<(unsigned)B, seq_threads(G), (size_t)G * sizeof(float), g.stream>>>(T, H, y, h0, gates, dy, WhT, dA,
                                                                                         RH, dh_next, rec_of(rec));
    if ((s = check("gru sequence backward")) != NMZ_OK) return s;
    /* z and r saw h_{t−1}, the candidate r ⊙ h_{t−1}: two products for W_h */
    if ((s = seq_grads(B, T, D, H, G, x, y, h0, Wx, dA, dWx, dWh, db, dx, 2 * H)) != NMZ_OK) return s;
    return cu_gemm(&g_be, NMZ_MM_TRANS_A | NMZ_MM_ACCUMULATE, H, H, B * T, RH, H, dA + 2 * H, G, dWh + 2 * H, G);
}

static nmz_status cu_sgd_velocity(const nmz_backend *be, size_t n, float lr, float mu, int fresh, const float *gr,
                                  const float *v_prev, float *v_out) {
    (void)be;
    if (!n) return NMZ_OK;
    NEED(gr && v_prev && v_out);
    k_sgd<<<blocks_for(n), TPB, 0, g.stream>>>(n, lr, mu, fresh, gr, v_prev, v_out);
    return check("sgd");
}

static nmz_status cu_adam_update(const nmz_backend *be, size_t n, double lr, double b1, double b2, double eps,
                                 double bc1, double bc2, double decay, float *w, const float *gr, float *m, float *v) {
    (void)be;
    if (!n) return NMZ_OK;
    NEED(w && gr && m && v);
    k_adam<<<blocks_for(n), TPB, 0, g.stream>>>(n, lr, b1, b2, eps, bc1, bc2, decay, w, gr, m, v);
    return check("adam");
}

static nmz_status cu_adam_preview(const nmz_backend *be, size_t n, double lr, double b1, double b2, double eps,
                                  double bc1, double bc2, double decay, int fresh, const float *w, const float *gr,
                                  const float *m, const float *v, float *m_out, float *v_out, float *mhat, float *vhat,
                                  float *delta) {
    (void)be;
    if (!n) return NMZ_OK;
    NEED(w && gr && m && v && m_out && v_out && mhat && vhat && delta);
    k_adam_preview<<<blocks_for(n), TPB, 0, g.stream>>>(n, lr, b1, b2, eps, bc1, bc2, decay, fresh, w, gr, m, v, m_out,
                                                        v_out, mhat, vhat, delta);
    return check("adam preview");
}

static nmz_status cu_sumsq(const nmz_backend *be, int k, float *const *x, const size_t *n, double *out) {
    double *slot;
    int i;
    (void)be;
    NEED(out);
    if (!(slot = scalar_slot(out))) return nmz__op_failed();
    if (cudaMemsetAsync(slot, 0, sizeof(double), g.stream) != cudaSuccess) return check("clip");
    for (i = 0; i < k; i++) { /* one tensor after the other, in order */
        unsigned nb = blocks_for(n[i]);
        double *part;
        if (!n[i]) continue;
        NEED(x[i]);
        if (!(part = (double *)scratch((size_t)nb * sizeof(double)))) return nmz__op_failed();
        k_sumsq<<<nb, TPB, 0, g.stream>>>(n[i], x[i], part);
        k_sum_partials<<<1, 1, 0, g.stream>>>(part, (int)nb, 1.0, slot, 1);
    }
    return scalar_send();
}

static nmz_status cu_scale(const nmz_backend *be, int k, float *const *x, const size_t *n, double f) {
    int i;
    (void)be;
    for (i = 0; i < k; i++) {
        if (!n[i]) continue;
        NEED(x[i]);
        k_scale<<<blocks_for(n[i]), TPB, 0, g.stream>>>(n[i], f, x[i]);
    }
    return check("scale");
}

static nmz_status cu_norm_apply(const nmz_backend *be, size_t n, size_t per, size_t div, int groups,
                                const float *offset, const float *scale, float *x, int grad) {
    (void)be;
    if (!n || !per) return NMZ_OK;
    NEED(offset && scale && x);
    k_norm<<<blocks_for(n), TPB, 0, g.stream>>>(n, per, div, groups, offset, scale, x, grad);
    return check("normalization");
}

static nmz_status cu_gather_rows(const nmz_backend *be, int rows, size_t width, const float *idx, const float *src,
                                 float *dst) {
    (void)be;
    if (rows <= 0 || !width) return NMZ_OK;
    NEED(idx && src && dst);
    k_gather<<<blocks_for((size_t)rows * width), TPB, 0, g.stream>>>(rows, width, idx, src, dst);
    return check("gather");
}

static nmz_status cu_onehot(const nmz_backend *be, size_t n, int V, const float *ids, float *out) {
    (void)be;
    if (!n) return NMZ_OK;
    NEED(ids && out);
    if (cudaMemsetAsync(out, 0, n * (size_t)V * sizeof(float), g.stream) != cudaSuccess) return check("one-hot");
    k_onehot<<<blocks_for(n), TPB, 0, g.stream>>>(n, V, ids, out);
    return check("one-hot");
}

static nmz_status cu_count_correct(const nmz_backend *be, int mode, size_t rows, int cols, const float *y,
                                   const float *t, double *out) {
    unsigned nb = blocks_for(rows);
    double *part, *ok, *slot;
    (void)be;
    NEED(y && t && out);
    if (!rows) {
        *out = 0;
        return NMZ_OK;
    }
    if (!(part = (double *)scratch((size_t)nb * sizeof(double) + rows * sizeof(double)))) return nmz__op_failed();
    ok = part + nb;
    k_correct<<<warps_blocks(rows), TPB, 0, g.stream>>>(mode, rows, cols, y, t, ok);
    k_sum_rows<<<nb, TPB, 0, g.stream>>>(rows, ok, part);
    if (!(slot = scalar_slot(out))) return nmz__op_failed();
    k_sum_partials<<<1, 1, 0, g.stream>>>(part, (int)nb, 1.0, slot, 0);
    return scalar_send();
}

/* ============================================================ lifecycle */

__global__ void k_probe(int *p) { *p = 42; }

static void *load_lib(const char *name) {
#ifdef _WIN32
    HMODULE h = LoadLibraryA(name);
    if (!h) { /* next to the toolkit's DLLs, with their dependencies (cublasLt) */
        char path[1024];
        const char *cp = getenv("CUDA_PATH");
        const char *subs[2] = {"\\bin\\x64\\", "\\bin\\"};
        int i;
        for (i = 0; !h && cp && i < 2; i++) {
            snprintf(path, sizeof path, "%s%s%s", cp, subs[i], name);
            h = LoadLibraryExA(path, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
        }
    }
    return (void *)h;
#else
    void *h = dlopen(name, RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        char path[512];
        snprintf(path, sizeof path, "/usr/local/cuda/lib64/%s", name);
        h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    }
    return h;
#endif
}

static void *symbol(void *lib, const char *name) {
#ifdef _WIN32
    return (void *)GetProcAddress((HMODULE)lib, name);
#else
    return dlsym(lib, name);
#endif
}

static const char *cublas_lib(void) {
    static char name[32];
#ifdef _WIN32
    snprintf(name, sizeof name, "cublas64_%d.dll", CUBLAS_VER_MAJOR);
#else
    snprintf(name, sizeof name, "libcublas.so.%d", CUBLAS_VER_MAJOR);
#endif
    return name;
}
#define CUBLAS_LIB cublas_lib()

static nmz_status cu_init(const nmz_backend *be, char *why, size_t n) {
    int count = 0, *probe = NULL, got = 0, runtime = 0, driver = 0;
    cudaError_t e;
    pf_create create;
    pf_set_stream set_stream;
    pf_get_version get_version;
    (void)be;
    if (g.ready) return NMZ_OK;
    cudaRuntimeGetVersion(&runtime);
    cudaDriverGetVersion(&driver);
    e = cudaGetDeviceCount(&count);
    if (e != cudaSuccess || count == 0) {
        (void)cudaGetLastError();
        if (e == cudaErrorInsufficientDriver && driver == 0)
            snprintf(why, n, "no NVIDIA driver is installed");
        else if (e == cudaErrorInsufficientDriver)
            snprintf(why, n, "the NVIDIA driver supports CUDA %d.%d, this library needs %d.%d: update the driver",
                     driver / 1000, driver % 1000 / 10, runtime / 1000, runtime % 1000 / 10);
        else if (e == cudaErrorNoDevice || count == 0)
            snprintf(why, n, "no CUDA device found");
        else
            snprintf(why, n, "%s", cudaGetErrorString(e));
        return NMZ_ERR_UNSUPPORTED;
    }
    g.dev = 0;
    if ((e = cudaSetDevice(g.dev)) != cudaSuccess || (e = cudaGetDeviceProperties(&g.prop, g.dev)) != cudaSuccess) {
        (void)cudaGetLastError();
        snprintf(why, n, "%s", cudaGetErrorString(e));
        return NMZ_ERR_UNSUPPORTED;
    }
    if (g.prop.major * 10 + g.prop.minor < NMZ_CUDA_MIN_CC) {
        snprintf(why, n, "%s has compute capability %d.%d; this build needs %d.%d or newer (NMZ_CUDA_ARCHITECTURES)",
                 g.prop.name, g.prop.major, g.prop.minor, NMZ_CUDA_MIN_CC / 10, NMZ_CUDA_MIN_CC % 10);
        return NMZ_ERR_UNSUPPORTED;
    }
    /* can this device run the code in this library? */
    if (cudaMalloc((void **)&probe, sizeof(int)) == cudaSuccess) {
        k_probe<<<1, 1>>>(probe);
        e = cudaGetLastError();
        if (e == cudaSuccess) e = cudaMemcpy(&got, probe, sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(probe);
    } else {
        e = cudaGetLastError();
    }
    if (e != cudaSuccess || got != 42) {
        (void)cudaGetLastError();
        snprintf(why, n, "%s cannot run this library's kernels (%s): build with NMZ_CUDA_ARCHITECTURES including %d%d",
                 g.prop.name, e != cudaSuccess ? cudaGetErrorString(e) : "wrong result", g.prop.major, g.prop.minor);
        return NMZ_ERR_UNSUPPORTED;
    }
    g.lib = load_lib(CUBLAS_LIB);
    if (!g.lib) {
        snprintf(why, n, "the cuBLAS library (%s) was not found: install the CUDA runtime or put it next to the program",
                 CUBLAS_LIB);
        return NMZ_ERR_UNSUPPORTED;
    }
    create = (pf_create)symbol(g.lib, "cublasCreate_v2");
    set_stream = (pf_set_stream)symbol(g.lib, "cublasSetStream_v2");
    g.sgemm = (pf_sgemm)symbol(g.lib, "cublasSgemm_v2");
    g.sgemm_sb = (pf_sgemm_sb)symbol(g.lib, "cublasSgemmStridedBatched");
    get_version = (pf_get_version)symbol(g.lib, "cublasGetVersion_v2");
    if (!create || !set_stream || !g.sgemm || !g.sgemm_sb) {
        snprintf(why, n, "%s lacks cublasCreate_v2/cublasSgemm_v2/cublasSgemmStridedBatched", CUBLAS_LIB);
        return NMZ_ERR_UNSUPPORTED;
    }
    if (cudaStreamCreateWithFlags(&g.stream, cudaStreamNonBlocking) != cudaSuccess || create(&g.blas) != CUBLAS_STATUS_SUCCESS ||
        set_stream(g.blas, g.stream) != CUBLAS_STATUS_SUCCESS) {
        (void)cudaGetLastError();
        snprintf(why, n, "could not start cuBLAS on %s", g.prop.name);
        return NMZ_ERR_UNSUPPORTED;
    }
    if (get_version) get_version(g.blas, &g.blas_version);
    if (cudaMallocHost((void **)&g.pinned, MAX_PENDING * sizeof(double)) != cudaSuccess ||
        !(g.dscal = (double *)cu_alloc(&g_be, MAX_PENDING * sizeof(double)))) {
        (void)cudaGetLastError();
        snprintf(why, n, "out of memory starting CUDA");
        return NMZ_ERR_UNSUPPORTED;
    }
    { /* the memory cap: what is free now (WDDM would otherwise page to the host), or NMZ_CUDA_MEMORY MiB */
        size_t fr = 0, tot = 0;
        const char *env = getenv("NMZ_CUDA_MEMORY");
        cudaMemGetInfo(&fr, &tot);
        g.limit_default = fr + g.used;
        g.limit = env && atof(env) > 0 ? (size_t)(atof(env) * 1048576.0) : g.limit_default;
    }
    g.ready = 1;
    return NMZ_OK;
}

static nmz_status cu_memory_limit(const nmz_backend *be, size_t bytes) {
    (void)be;
    g.limit = bytes ? bytes : g.limit_default;
    return NMZ_OK;
}

static void cu_info(const nmz_backend *be, nmz_device_info *out) {
    size_t fr = 0, tot = 0;
    (void)be;
    if (!g.ready) return;
    snprintf(out->device, sizeof out->device, "%.127s", g.prop.name); /* prop.name has 256 */
    out->cc_major = g.prop.major;
    out->cc_minor = g.prop.minor;
    cudaMemGetInfo(&fr, &tot);
    out->mem_total = tot ? tot : g.prop.totalGlobalMem;
    out->mem_free = fr;
    out->mem_used = g.used;
    out->mem_peak = g.peak;
    out->mem_limit = g.limit;
}

extern "C" const nmz_backend *nmz__cuda_backend(void) {
    nmz_backend *b = &g_be;
    if (b->abi) return b;
    b->abi = NMZ_BACKEND_ABI;
    b->name = "cuda";
    b->summary = "an NVIDIA GPU (cuBLAS + kernels)";
    b->host_memory = 0;
    b->init = cu_init;
    b->info = cu_info;
    b->memory_limit = cu_memory_limit;
    b->alloc = cu_alloc;
    b->release = cu_release;
    b->upload = cu_upload;
    b->download = cu_download;
    b->sync = cu_sync;
    b->fill = cu_fill;
    b->copy = cu_copy;
    b->add = cu_add;
    b->mul = cu_mul;
    b->gemm = cu_gemm;
    b->add_bias = cu_add_bias;
    b->add_bias3 = cu_add_bias3;
    b->colsum_add = cu_colsum_add;
    b->rowsum_add = cu_rowsum_add;
    b->transpose3 = cu_transpose3;
    b->act_forward = cu_act_forward;
    b->act_backward = cu_act_backward;
    b->softmax_forward = cu_softmax_forward;
    b->softmax_backward = cu_softmax_backward;
    b->loss_value = cu_loss_value;
    b->loss_grad = cu_loss_grad;
    b->im2col = cu_im2col;
    b->col2im = cu_col2im;
    b->maxpool_forward = cu_pool_fwd;
    b->maxpool_backward = cu_pool_bwd;
    b->dropout_mask = cu_dropout_mask;
    b->embed_forward = cu_embed_forward;
    b->embed_backward = cu_embed_backward;
    b->posenc_forward = cu_posenc_forward;
    b->posenc_backward = cu_posenc_backward;
    b->layernorm_forward = cu_ln_forward;
    b->layernorm_backward = cu_ln_backward;
    b->attention_forward = cu_attn_forward;
    b->attention_backward = cu_attn_backward;
    b->rnn_cell_forward = cu_rnn_cell_forward;
    b->rnn_cell_backward = cu_rnn_cell_backward;
    b->lstm_cell_forward = cu_lstm_cell_forward;
    b->lstm_cell_backward = cu_lstm_cell_backward;
    b->gru_cell_forward1 = cu_gru_cell_forward1;
    b->gru_cell_forward2 = cu_gru_cell_forward2;
    b->gru_cell_backward1 = cu_gru_cell_backward1;
    b->gru_cell_backward2 = cu_gru_cell_backward2;
    b->sgd_velocity = cu_sgd_velocity;
    b->adam_update = cu_adam_update;
    b->adam_preview = cu_adam_preview;
    b->sumsq = cu_sumsq;
    b->scale = cu_scale;
    b->norm_apply = cu_norm_apply;
    b->gather_rows = cu_gather_rows;
    b->onehot = cu_onehot;
    b->count_correct = cu_count_correct;
    b->rnn_forward = cu_rnn_forward; /* the recurrences: one launch per sequence */
    b->rnn_backward = cu_rnn_backward;
    b->lstm_forward = cu_lstm_forward;
    b->lstm_backward = cu_lstm_backward;
    b->gru_forward = cu_gru_forward;
    b->gru_backward = cu_gru_backward;
    return b;
}
