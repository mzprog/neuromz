/* test_transformer.c — phase 5: the decoder-only Transformer. The blocked
 * matrix product against the naive one, LayerNorm / positions / attention
 * by hand, numerical gradient checks of every building block and of a whole
 * multi-block model, the causal mask, weight tying, the KV cache against
 * full recomputation, AdamW and warm-up against hand-computed steps,
 * save/load, the phase-4 fixture. */
#include "test.h"
#include "neuromz_layer.h"

#include <stdlib.h>
#include <string.h>

static void fill(nmz_tensor *t, nmz_rng *r, double lo, double hi) {
    size_t i;
    for (i = 0; i < t->size; i++) t->data[i] = (float)(lo + (hi - lo) * nmz_rng_uniform(r));
}

static void fill_ids(nmz_tensor *t, nmz_rng *r, int V) {
    size_t i;
    for (i = 0; i < t->size; i++) t->data[i] = (float)nmz_rng_below(r, (uint32_t)V);
}

static void onehot_targets(nmz_tensor *t, nmz_rng *r, int from) {
    int B = t->shape[0], T = t->shape[1], V = t->shape[2], b, s;
    nmz_tensor_zero(t);
    for (b = 0; b < B; b++)
        for (s = from; s < T; s++) t->data[((size_t)b * T + s) * V + nmz_rng_below(r, (uint32_t)V)] = 1.0f;
}

static void report(const char *label, const nmz_gradcheck_result *res) {
    CHECK_MSG(res->failures == 0 && res->checked > 0,
              "%s: %zu/%zu mismatches; worst layer %d %s[%zu]: analytic %.6g numeric %.6g", label, res->failures,
              res->checked, res->worst_layer, res->worst_param, res->worst_index, res->worst_analytic,
              res->worst_numeric);
    test_note("%-44s %5zu grads, max |err| %.2e%s", label, res->checked, res->max_abs_err,
              res->kinks ? " (relu kinks skipped)" : "");
}

/* every parameter of every layer (LayerNorm gains, biases, …) to random
 * values, so no gradient is trivially 0 or 1 */
static void randomize_params(nmz_net *net, nmz_rng *r, double scale) {
    int i, j;
    for (i = 0; i < nmz_net_layer_count(net); i++) {
        nmz_layer *l = nmz_net_layer(net, i);
        for (j = 0; j < nmz_layer_param_count(l); j++) {
            nmz_tensor *v = nmz_layer_param_value(l, j);
            size_t e;
            for (e = 0; e < v->size; e++) v->data[e] += (float)(scale * (2.0 * nmz_rng_uniform(r) - 1.0));
        }
    }
}

/* ------------------------------------------------------------- matmul */

static void test_matmul(void) {
    static const int sizes[][3] = {{1, 1, 1}, {3, 5, 7}, {33, 300, 129}, {70, 257, 300}, {128, 64, 513}, {2, 600, 3}};
    static const unsigned flags[] = {0, NMZ_MM_TRANS_A, NMZ_MM_TRANS_B, NMZ_MM_ACCUMULATE,
                                     NMZ_MM_TRANS_A | NMZ_MM_ACCUMULATE, NMZ_MM_TRANS_B | NMZ_MM_ACCUMULATE};
    size_t si, fi;
    nmz_rng r;
    int identical = 0, cases = 0;
    double worst = 0;
    nmz_rng_seed(&r, 5, 1);
    for (si = 0; si < sizeof sizes / sizeof sizes[0]; si++)
        for (fi = 0; fi < sizeof flags / sizeof flags[0]; fi++) {
            const int M = sizes[si][0], N = sizes[si][1], K = sizes[si][2];
            float *A = malloc((size_t)M * K * sizeof *A), *B = malloc((size_t)K * N * sizeof *B);
            float *C1 = malloc((size_t)M * N * sizeof *C1), *C2 = malloc((size_t)M * N * sizeof *C2);
            size_t e;
            int i, j, k;
            CHECK(A && B && C1 && C2);
            if (!A || !B || !C1 || !C2) {
                free(A), free(B), free(C1), free(C2);
                return;
            }
            for (e = 0; e < (size_t)M * K; e++) A[e] = (float)(2 * nmz_rng_uniform(&r) - 1);
            for (e = 0; e < (size_t)K * N; e++) B[e] = (float)(2 * nmz_rng_uniform(&r) - 1);
            for (e = 0; e < (size_t)M * N; e++) C1[e] = C2[e] = (float)(2 * nmz_rng_uniform(&r) - 1);
            CHECK_OK(nmz_matmul(M, N, K, A, B, C1, flags[fi]));
            CHECK_OK(nmz_matmul_naive(M, N, K, A, B, C2, flags[fi]));
            cases++;
            identical += memcmp(C1, C2, (size_t)M * N * sizeof *C1) == 0;
            CHECK_MSG(memcmp(C1, C2, (size_t)M * N * sizeof *C1) == 0, "%dx%dx%d flags %u: blocked != naive", M, N, K,
                      flags[fi]);
            /* and both are the product (double reference, on a few entries) */
            for (i = 0; i < M; i += M > 8 ? M / 7 : 1)
                for (j = 0; j < N; j += N > 8 ? N / 7 : 1) {
                    double ref = 0;
                    if (flags[fi] & NMZ_MM_ACCUMULATE) continue;
                    for (k = 0; k < K; k++) {
                        double a = (flags[fi] & NMZ_MM_TRANS_A) ? A[(size_t)k * M + i] : A[(size_t)i * K + k];
                        double b = (flags[fi] & NMZ_MM_TRANS_B) ? B[(size_t)j * K + k] : B[(size_t)k * N + j];
                        ref += a * b;
                    }
                    if (fabs(ref - C1[(size_t)i * N + j]) > worst) worst = fabs(ref - C1[(size_t)i * N + j]);
                }
            free(A), free(B), free(C1), free(C2);
        }
    CHECK(worst < 1e-4);
    test_note("%d/%d shapes x flags bit-identical to the naive loops; max |float - double| %.1e; %d thread(s)",
              identical, cases, worst, nmz_threads());
    {
        float a = 1, b = 1, c = 0;
        CHECK_STATUS(nmz_matmul(1, 1, 1, &a, &b, &c, NMZ_MM_TRANS_A | NMZ_MM_TRANS_B), NMZ_ERR_UNSUPPORTED);
        CHECK_STATUS(nmz_matmul(-1, 1, 1, &a, &b, &c, 0), NMZ_ERR_ARG);
    }
}

/* ------------------------------------------------------------ by hand */

static nmz_net *fixed_net(int T, int D, uint64_t seed) {
    nmz_net *net = NULL;
    int in[2];
    in[0] = T;
    in[1] = D;
    return nmz_net_create(&net, 2, in, seed) == NMZ_OK ? net : NULL;
}

static void test_layernorm_by_hand(void) {
    nmz_net *net = fixed_net(1, 4, 1);
    nmz_tensor x = NMZ_TENSOR_EMPTY;
    const nmz_tensor *y;
    int xs[3] = {1, 1, 4}, k;
    double rs = 1.0 / sqrt(1.25 + 1e-5);
    CHECK(net != NULL);
    if (!net) return;
    CHECK_OK(nmz_net_add(net, nmz_layer_layernorm()));
    CHECK_OK(nmz_net_build(net));
    CHECK_OK(nmz_tensor_init(&x, 3, xs));
    for (k = 0; k < 4; k++) x.data[k] = (float)(k + 1); /* mean 2.5, variance 1.25 */
    CHECK_OK(nmz_net_forward(net, &x, &y));
    for (k = 0; k < 4; k++) CHECK_NEAR(y->data[k], (k + 1 - 2.5) * rs, 1e-6);
    /* gain and bias */
    nmz_layer_param_value(nmz_net_layer(net, 0), 0)->data[1] = 2.0f;
    nmz_layer_param_value(nmz_net_layer(net, 0), 1)->data[1] = 0.5f;
    CHECK_OK(nmz_net_forward(net, &x, &y));
    CHECK_NEAR(y->data[1], 2.0 * (2 - 2.5) * rs + 0.5, 1e-6);
    nmz_tensor_free(&x);
    nmz_net_free(net);
}

static void test_posenc_by_hand(void) {
    nmz_net *net = fixed_net(5, 6, 1);
    nmz_tensor x = NMZ_TENSOR_EMPTY;
    const nmz_tensor *y;
    int xs[3] = {1, 5, 6}, t;
    CHECK(net != NULL);
    if (!net) return;
    CHECK_OK(nmz_net_add(net, nmz_layer_posenc(NMZ_POS_SINUSOIDAL, 8)));
    CHECK(nmz_net_param_count(net) == 0);
    CHECK(nmz_net_context_length(net) == 8);
    CHECK_OK(nmz_tensor_init(&x, 3, xs)); /* zeros: y = P */
    CHECK_OK(nmz_net_forward(net, &x, &y));
    for (t = 0; t < 5; t++) {
        CHECK_NEAR(y->data[t * 6 + 0], sin(t), 1e-6);
        CHECK_NEAR(y->data[t * 6 + 1], cos(t), 1e-6);
        CHECK_NEAR(y->data[t * 6 + 2], sin(t / pow(10000.0, 2.0 / 6)), 1e-6);
        CHECK_NEAR(y->data[t * 6 + 5], cos(t / pow(10000.0, 4.0 / 6)), 1e-6);
    }
    nmz_tensor_free(&x);
    nmz_net_free(net);
    /* too many positions for the window */
    net = fixed_net(5, 6, 1);
    if (!net) return;
    CHECK_STATUS(nmz_net_add(net, nmz_layer_posenc(NMZ_POS_LEARNED, 4)), NMZ_OK);
    CHECK_OK(nmz_tensor_init(&x, 3, xs));
    CHECK_STATUS(nmz_net_forward(net, &x, &y), NMZ_ERR_SHAPE);
    nmz_tensor_free(&x);
    nmz_net_free(net);
}

/* one head, q = k = v = x (identity projections): the weights are a softmax
 * of x_t·x_j/√2 over j ≤ t, and y_t = Σ_j a_tj x_j */
static void test_attention_by_hand(void) {
    static const double xs3[3][2] = {{1, 0}, {0, 1}, {1, 1}};
    nmz_net *net = fixed_net(3, 2, 1);
    nmz_tensor x = NMZ_TENSOR_EMPTY;
    const nmz_tensor *y;
    nmz_layer *l;
    int shape[3] = {1, 3, 2}, t, j, k;
    CHECK(net != NULL);
    if (!net) return;
    CHECK_OK(nmz_net_add(net, nmz_layer_attention(1, 0)));
    CHECK_OK(nmz_net_build(net));
    l = nmz_net_layer(net, 0);
    {
        nmz_tensor *Wqkv = nmz_layer_param_value(l, 0), *Wo = nmz_layer_param_value(l, 2);
        CHECK(Wqkv->shape[0] == 2 && Wqkv->shape[1] == 6 && Wo->shape[0] == 2 && Wo->shape[1] == 2);
        nmz_tensor_zero(Wqkv);
        nmz_tensor_zero(Wo);
        for (k = 0; k < 2; k++) {
            Wqkv->data[k * 6 + k] = Wqkv->data[k * 6 + 2 + k] = Wqkv->data[k * 6 + 4 + k] = 1.0f;
            Wo->data[k * 2 + k] = 1.0f;
        }
    }
    CHECK_OK(nmz_tensor_init(&x, 3, shape));
    for (t = 0; t < 3; t++)
        for (k = 0; k < 2; k++) x.data[t * 2 + k] = (float)xs3[t][k];
    CHECK_OK(nmz_net_forward(net, &x, &y));
    for (t = 0; t < 3; t++) {
        double s[3], mx = -1e9, sum = 0, out[2] = {0, 0};
        for (j = 0; j <= t; j++) {
            s[j] = (xs3[t][0] * xs3[j][0] + xs3[t][1] * xs3[j][1]) / sqrt(2.0);
            if (s[j] > mx) mx = s[j];
        }
        for (j = 0; j <= t; j++) sum += exp(s[j] - mx);
        for (j = 0; j <= t; j++)
            for (k = 0; k < 2; k++) out[k] += exp(s[j] - mx) / sum * xs3[j][k];
        for (k = 0; k < 2; k++) CHECK_NEAR(y->data[t * 2 + k], out[k], 1e-6);
    }
    nmz_tensor_free(&x);
    nmz_net_free(net);
}

/* ---------------------------------------------------------- gradchecks */

typedef enum { L_LN, L_POS_LEARNED, L_POS_SIN, L_ATTN, L_ATTN_HD, L_FFN_GELU, L_FFN_RELU, L_BLOCK, L_BLOCK_DROP } lkind;

static nmz_layer *make_layer(lkind k) {
    switch (k) {
    case L_LN: return nmz_layer_layernorm();
    case L_POS_LEARNED: return nmz_layer_posenc(NMZ_POS_LEARNED, 8);
    case L_POS_SIN: return nmz_layer_posenc(NMZ_POS_SINUSOIDAL, 8);
    case L_ATTN: return nmz_layer_attention(2, 0);
    case L_ATTN_HD: return nmz_layer_attention(3, 3); /* E = 9 ≠ D = 4 */
    case L_FFN_GELU: return nmz_layer_ffn(10, NMZ_ACT_GELU);
    case L_FFN_RELU: return nmz_layer_ffn(10, NMZ_ACT_RELU);
    case L_BLOCK: return nmz_layer_block(2, 0, 12, NMZ_ACT_GELU, 0.0f);
    case L_BLOCK_DROP: return nmz_layer_block(2, 3, 12, NMZ_ACT_GELU, 0.25f);
    }
    return NULL;
}

static void block_case(const char *label, lkind k, int training, uint64_t seed) {
    nmz_net *net = fixed_net(5, 4, seed);
    nmz_tensor x = NMZ_TENSOR_EMPTY, t = NMZ_TENSOR_EMPTY;
    nmz_gradcheck_result res;
    int xs[3] = {2, 5, 4}, ts[3] = {2, 5, 3};
    nmz_rng r;
    CHECK(net != NULL);
    if (!net) return;
    nmz_rng_seed(&r, seed, 3);
    CHECK_OK(nmz_net_add(net, make_layer(k)));
    CHECK_OK(nmz_net_add(net, nmz_layer_dense(3, NMZ_INIT_AUTO))); /* a fixed readout so every output matters */
    CHECK_OK(nmz_net_build(net));
    randomize_params(net, &r, 0.3);
    CHECK_OK(nmz_tensor_init(&x, 3, xs));
    CHECK_OK(nmz_tensor_init(&t, 3, ts));
    fill(&x, &r, -1, 1);
    fill(&t, &r, -1, 1);
    /* atol 3e-4: the forward sums in float, so L carries noise of ~1e-7 and
     * the difference quotient ~1e-4 — visible on gradients that are exactly
     * 0, such as the key bias (adding a constant to every score of a row
     * does not change its softmax) */
    CHECK_OK(nmz_net_gradcheck_mode(net, &x, &t, 1e-3, 3e-4, 0, training, &res));
    report(label, &res);
    nmz_tensor_free(&x);
    nmz_tensor_free(&t);
    nmz_net_free(net);
}

static void test_gradcheck_blocks(void) {
    block_case("layernorm (gain, bias, input)", L_LN, 0, 1);
    block_case("posenc learned", L_POS_LEARNED, 0, 2);
    block_case("posenc sinusoidal (input only)", L_POS_SIN, 0, 3);
    block_case("attention, 2 heads of 2", L_ATTN, 0, 4);
    block_case("attention, 3 heads of 3 (E != D)", L_ATTN_HD, 0, 5);
    block_case("ffn gelu", L_FFN_GELU, 0, 6);
    block_case("ffn relu", L_FFN_RELU, 0, 7);
    block_case("block (pre-LN, residuals)", L_BLOCK, 0, 8);
    block_case("block, dropout 0.25, evaluation mode", L_BLOCK_DROP, 0, 9);
    block_case("block, dropout 0.25, training mode", L_BLOCK_DROP, 1, 10);
}

/* embed → posenc → block → block → layernorm → output (tied or dense) → softmax */
static nmz_net *gpt_net(int V, int D, int ctx, int tied, nmz_posenc pk, int heads, uint64_t seed) {
    nmz_net *net = NULL;
    int in = NMZ_ANY_LENGTH;
    if (nmz_net_create(&net, 1, &in, seed) != NMZ_OK) return NULL;
    nmz_net_add(net, nmz_layer_embedding(V, D, NMZ_INIT_AUTO));
    nmz_net_add(net, nmz_layer_posenc(pk, ctx));
    nmz_net_add(net, nmz_layer_block(heads, 0, 2 * D, NMZ_ACT_GELU, 0.0f));
    nmz_net_add(net, nmz_layer_block(heads, 0, 2 * D, NMZ_ACT_RELU, 0.0f));
    nmz_net_add(net, nmz_layer_layernorm());
    if (tied)
        nmz_net_add(net, nmz_layer_tied_output(V));
    else
        nmz_net_add(net, nmz_layer_dense(V, NMZ_INIT_AUTO));
    nmz_net_add(net, nmz_layer_activation(NMZ_ACT_SOFTMAX, 0));
    nmz_net_set_loss(net, NMZ_LOSS_CROSS_ENTROPY);
    if (nmz_net_build(net) != NMZ_OK) {
        nmz_net_free(net);
        return NULL;
    }
    return net;
}

static void gpt_case(const char *label, int tied, nmz_posenc pk, int last_only) {
    const int V = 7, B = 2, T = 6;
    nmz_net *net = gpt_net(V, 8, 16, tied, pk, 2, 31);
    nmz_tensor x = NMZ_TENSOR_EMPTY, t = NMZ_TENSOR_EMPTY;
    nmz_gradcheck_result res;
    int xs[2] = {B, T}, ts[3] = {B, T, V};
    nmz_rng r;
    CHECK(net != NULL);
    if (!net) return;
    nmz_rng_seed(&r, 32, (uint64_t)tied);
    randomize_params(net, &r, 0.2);
    CHECK_OK(nmz_tensor_init(&x, 2, xs));
    CHECK_OK(nmz_tensor_init(&t, 3, ts));
    fill_ids(&x, &r, V);
    onehot_targets(&t, &r, last_only ? T - 1 : 0);
    CHECK_OK(nmz_net_gradcheck_mode(net, &x, &t, 1e-3, 0, 0, 0, &res));
    report(label, &res);
    nmz_tensor_free(&x);
    nmz_tensor_free(&t);
    nmz_net_free(net);
}

static void test_gradcheck_gpt(void) {
    gpt_case("2 blocks, learned pos, tied output, CE", 1, NMZ_POS_LEARNED, 0);
    gpt_case("2 blocks, sinusoidal, dense output, CE", 0, NMZ_POS_SINUSOIDAL, 0);
    gpt_case("2 blocks, tied, loss at the last step only", 1, NMZ_POS_LEARNED, 1);
}

/* the output at t (and so a loss there) never depends on x_j, j > t */
static void test_causal_mask(void) {
    int variant;
    for (variant = 0; variant < 2; variant++) {
        const int T = 6, D = 4, tq = 2;
        nmz_net *net = fixed_net(T, D, 41);
        nmz_tensor x = NMZ_TENSOR_EMPTY, y0 = NMZ_TENSOR_EMPTY, tg = NMZ_TENSOR_EMPTY;
        const nmz_tensor *y, *dx;
        int xs[3] = {1, 6, 4}, t, k, same = 1, zero = 1, nonzero = 0;
        nmz_rng r;
        double L;
        CHECK(net != NULL);
        if (!net) return;
        nmz_rng_seed(&r, 42, (uint64_t)variant);
        CHECK_OK(nmz_net_add(net, variant ? nmz_layer_block(2, 0, 8, NMZ_ACT_GELU, 0) : nmz_layer_attention(2, 0)));
        CHECK_OK(nmz_net_build(net));
        randomize_params(net, &r, 0.3);
        CHECK_OK(nmz_tensor_init(&x, 3, xs));
        fill(&x, &r, -1, 1);
        CHECK_OK(nmz_net_forward(net, &x, &y));
        CHECK_OK(nmz_tensor_copy(&y0, y));
        for (t = tq + 1; t < T; t++) /* change the future */
            for (k = 0; k < D; k++) x.data[t * D + k] += 5.0f;
        CHECK_OK(nmz_net_forward(net, &x, &y));
        for (t = 0; t <= tq; t++)
            for (k = 0; k < D; k++) same &= y->data[t * D + k] == y0.data[t * D + k];
        CHECK_MSG(same, "outputs up to position %d changed when later inputs changed", tq);
        /* a loss on position tq only: target = output everywhere else */
        CHECK_OK(nmz_tensor_copy(&tg, y));
        for (k = 0; k < D; k++) tg.data[tq * D + k] += 1.0f;
        CHECK_OK(nmz_net_backward(net, &tg, &L));
        dx = nmz_net_input_grad(net);
        CHECK(dx != NULL);
        if (dx) {
            for (t = 0; t < T; t++)
                for (k = 0; k < D; k++) {
                    if (t > tq) zero &= dx->data[t * D + k] == 0.0f;
                    else nonzero += dx->data[t * D + k] != 0.0f;
                }
        }
        CHECK_MSG(zero, "gradient reached a future position");
        CHECK(nonzero == (tq + 1) * D);
        test_note("%s: output at t=%d unchanged by x_3..x_5, dL/dx_j = 0 exactly for j > %d", variant ? "block" : "attention",
                  tq, tq);
        nmz_tensor_free(&x);
        nmz_tensor_free(&y0);
        nmz_tensor_free(&tg);
        nmz_net_free(net);
    }
}

static void test_weight_tying(void) {
    const int V = 5, D = 4;
    nmz_net *tied = gpt_net(V, D, 8, 1, NMZ_POS_LEARNED, 2, 51), *untied = gpt_net(V, D, 8, 0, NMZ_POS_LEARNED, 2, 51);
    nmz_tensor x = NMZ_TENSOR_EMPTY, t = NMZ_TENSOR_EMPTY;
    int xs[2] = {1, 4}, ts[3] = {1, 4, 5}, n;
    nmz_rng r;
    CHECK(tied && untied);
    if (!tied || !untied) {
        nmz_net_free(tied);
        nmz_net_free(untied);
        return;
    }
    n = nmz_net_layer_count(tied);
    /* one matrix fewer: the output uses the embedding's E */
    CHECK(nmz_net_param_count(untied) - nmz_net_param_count(tied) == (size_t)V * D);
    CHECK(!strcmp(nmz_layer_type(nmz_net_layer(tied, n - 2)), "tied_output"));
    CHECK(nmz_layer_param_count(nmz_net_layer(tied, n - 2)) == 1);
    /* the shared gradient = the input use + the output use: zero the input
     * use by feeding a token that is not in the batch, its row of dE still
     * gets the output-side gradient */
    nmz_rng_seed(&r, 52, 0);
    CHECK_OK(nmz_tensor_init(&x, 2, xs));
    CHECK_OK(nmz_tensor_init(&t, 3, ts));
    x.data[0] = 0, x.data[1] = 1, x.data[2] = 2, x.data[3] = 1; /* tokens 3 and 4 never appear */
    onehot_targets(&t, &r, 0);
    CHECK_OK(nmz_net_forward(tied, &x, NULL));
    CHECK_OK(nmz_net_backward(tied, &t, NULL));
    {
        const nmz_tensor *dE = nmz_layer_param_grad(nmz_net_layer(tied, 0), 0);
        double row4 = 0;
        int k;
        for (k = 0; k < D; k++) row4 += fabs(dE->data[4 * D + k]);
        CHECK_MSG(row4 > 0, "the output side did not add into the embedding gradient");
    }
    /* save/load keeps the tie */
    CHECK_OK(nmz_net_save(tied, test_path("tied.nmz")));
    {
        nmz_net *back = NULL;
        const nmz_tensor *y1, *y2;
        nmz_tensor keep = NMZ_TENSOR_EMPTY;
        CHECK_OK(nmz_net_load(&back, test_path("tied.nmz")));
        if (back) {
            CHECK_OK(nmz_net_forward(tied, &x, &y1));
            CHECK_OK(nmz_tensor_copy(&keep, y1));
            CHECK_OK(nmz_net_forward(back, &x, &y2));
            CHECK(memcmp(keep.data, y2->data, keep.size * sizeof(float)) == 0);
            /* changing E after loading changes the output: it is really shared */
            nmz_layer_param_value(nmz_net_layer(back, 0), 0)->data[0] += 1.0f;
            CHECK_OK(nmz_net_forward(back, &x, &y2));
            CHECK(memcmp(keep.data, y2->data, keep.size * sizeof(float)) != 0);
            nmz_net_free(back);
        }
        nmz_tensor_free(&keep);
    }
    {   /* no embedding before it: a clear error at build */
        nmz_net *bad = fixed_net(3, 4, 1);
        if (bad) {
            CHECK_OK(nmz_net_add(bad, nmz_layer_tied_output(5)));
            CHECK_STATUS(nmz_net_build(bad), NMZ_ERR_STATE);
            nmz_net_free(bad);
        }
    }
    nmz_tensor_free(&x);
    nmz_tensor_free(&t);
    nmz_net_free(tied);
    nmz_net_free(untied);
}

/* -------------------------------------------------------- generation */

static nmz_net *gpt_with_vocab(const char *text, int items, int ctx, nmz_vocab **vout) {
    uint32_t *cp = NULL;
    size_t n = 0, bad = 0;
    nmz_vocab *v = NULL;
    nmz_net *net;
    nmz_rng r;
    if (nmz_utf8_decode(text, strlen(text), &cp, &n, &bad) != NMZ_OK) return NULL;
    nmz_vocab_build(&v, cp, n);
    free(cp);
    if (!v) return NULL;
    v->items = items;
    net = gpt_net(v->size, 8, ctx, 1, NMZ_POS_LEARNED, 2, 61);
    if (net) {
        nmz_rng_seed(&r, 62, 0);
        randomize_params(net, &r, 0.5); /* sharper, more varied predictions than a fresh init */
        nmz_net_set_vocab(net, v);
    }
    *vout = v;
    return net;
}

static void test_kv_cache(void) {
    nmz_vocab *v = NULL;
    nmz_net *net = gpt_with_vocab("the cat sat on the mat. a bat ate a rat.\n", 0, 16, &v);
    int cfg, r, slides = 0;
    CHECK(net != NULL);
    if (!net) {
        nmz_vocab_free(v);
        return;
    }
    for (cfg = 0; cfg < 4; cfg++) {
        nmz_gen_opts g = nmz_gen_defaults();
        char *c[3] = {NULL}, *f[3] = {NULL};
        g.seed = 70 + (uint64_t)cfg;
        g.temperature = 0.9f;
        g.length = cfg == 0 ? 10 : 45;                      /* 0: stays inside the window */
        g.context = cfg == 2 ? 8 : 0;                        /* 2: a smaller window than the model's 16 */
        g.prime = cfg == 3 ? "the cat sat on the mat. a b" : "th"; /* 3: a prime longer than the window */
        g.top_k = cfg == 1 ? 4 : 0;
        CHECK_OK(nmz_net_generate_batch(net, &g, 3, c, NULL));
        g.no_cache = 1;
        CHECK_OK(nmz_net_generate_batch(net, &g, 3, f, NULL));
        for (r = 0; r < 3; r++) {
            CHECK_MSG(c[r] && f[r] && !strcmp(c[r], f[r]), "config %d row %d: cache '%s' vs recompute '%s'", cfg, r,
                      c[r] ? c[r] : "", f[r] ? f[r] : "");
            free(c[r]);
            free(f[r]);
        }
        if (cfg) slides++;
    }
    {   /* items: rows finish at different steps and leave the batch (their cache rows too) */
        nmz_gen_opts g = nmz_gen_defaults();
        char *c[4] = {NULL}, *f[4] = {NULL};
        v->items = 1;
        nmz_net_set_vocab(net, v);
        g.stop_at_end = 1;
        g.length = 30;
        g.seed = 80;
        CHECK_OK(nmz_net_generate_batch(net, &g, 4, c, NULL));
        g.no_cache = 1;
        CHECK_OK(nmz_net_generate_batch(net, &g, 4, f, NULL));
        for (r = 0; r < 4; r++) {
            CHECK_MSG(c[r] && f[r] && !strcmp(c[r], f[r]), "items row %d: '%s' vs '%s'", r, c[r] ? c[r] : "",
                      f[r] ? f[r] : "");
            free(c[r]);
            free(f[r]);
        }
    }
    {   /* a batch row equals the single run with its seed */
        nmz_gen_opts g = nmz_gen_defaults();
        char *b[3] = {NULL}, *one = NULL;
        v->items = 0;
        nmz_net_set_vocab(net, v);
        g.length = 40;
        g.seed = 90;
        CHECK_OK(nmz_net_generate_batch(net, &g, 3, b, NULL));
        g.seed = 92;
        CHECK_OK(nmz_net_generate(net, &g, &one, NULL));
        CHECK(b[2] && one && !strcmp(b[2], one));
        for (r = 0; r < 3; r++) free(b[r]);
        free(one);
        g.context = 17;
        CHECK_STATUS(nmz_net_generate(net, &g, &one, NULL), NMZ_ERR_ARG); /* longer than the model's window */
    }
    test_note("KV cache == recompute: 4 configs x 3 rows (window 16 and 8, slides, long prime), items, batch rows");
    (void)slides;
    nmz_vocab_free(v);
    nmz_net_free(net);
}

/* ------------------------------------------------------------- AdamW */

static void test_adamw_by_hand(void) {
    int mode;
    for (mode = 0; mode < 2; mode++) { /* 0: matrices only (default), 1: decay_all */
        nmz_net *net = NULL;
        int in = 3, i;
        nmz_tensor x = NMZ_TENSOR_EMPTY, t = NMZ_TENSOR_EMPTY, w0 = NMZ_TENSOR_EMPTY, b0 = NMZ_TENSOR_EMPTY;
        int xs[2] = {2, 3}, ts[2] = {2, 2};
        nmz_train_opts o = nmz_train_defaults();
        nmz_layer *l;
        nmz_rng r;
        const double lr = 0.1, wd = 0.5, eps = 1e-8;
        CHECK_OK(nmz_net_create(&net, 1, &in, 3));
        CHECK_OK(nmz_net_add(net, nmz_layer_dense(2, NMZ_INIT_AUTO)));
        CHECK_OK(nmz_net_build(net));
        l = nmz_net_layer(net, 0);
        nmz_rng_seed(&r, 7, 0);
        CHECK_OK(nmz_tensor_init(&x, 2, xs));
        CHECK_OK(nmz_tensor_init(&t, 2, ts));
        fill(&x, &r, -1, 1);
        fill(&t, &r, -1, 1);
        fill(nmz_layer_param_value(l, 1), &r, -1, 1);
        CHECK_OK(nmz_tensor_copy(&w0, nmz_layer_param_value(l, 0)));
        CHECK_OK(nmz_tensor_copy(&b0, nmz_layer_param_value(l, 1)));
        o.optimizer = NMZ_OPT_ADAMW;
        o.weight_decay = (float)wd;
        o.decay_all = mode;
        CHECK(nmz_layer_param_decays(l, 0, 0) == 1 && nmz_layer_param_decays(l, 1, 0) == 0);
        CHECK_OK(nmz_net_forward(net, &x, NULL));
        CHECK_OK(nmz_net_backward(net, &t, NULL));
        CHECK_OK(nmz_net_optimizer_step(net, &o, (float)lr));
        for (i = 0; i < 2; i++) {
            const nmz_tensor *v0 = i ? &b0 : &w0, *v1 = nmz_layer_param_value(l, i), *g = nmz_layer_param_grad(l, i);
            size_t e;
            int decays = i == 0 || mode;
            for (e = 0; e < v0->size; e++) {
                /* step 1: m̂ = g, v̂ = g² (bias correction undoes the 1−β) */
                double gi = g->data[e], m = (1 - 0.9) * gi, vv = (1 - 0.999) * gi * gi;
                double mhat = m / (1 - 0.9), vhat = vv / (1 - 0.999);
                double want = v0->data[e] - lr * (mhat / (sqrt(vhat) + eps) + (decays ? wd * v0->data[e] : 0.0));
                CHECK_NEAR(v1->data[e], want, 2e-6);
            }
        }
        nmz_tensor_free(&x);
        nmz_tensor_free(&t);
        nmz_tensor_free(&w0);
        nmz_tensor_free(&b0);
        nmz_net_free(net);
    }
    {   /* embeddings and position tables are not decayed by default; an unused
         * embedding row (zero gradient) stays put, with decay_all it shrinks */
        int all;
        for (all = 0; all < 2; all++) {
            nmz_net *net = gpt_net(5, 4, 8, 1, NMZ_POS_LEARNED, 1, 71);
            nmz_tensor x = NMZ_TENSOR_EMPTY, t = NMZ_TENSOR_EMPTY;
            int xs[2] = {1, 3}, ts[3] = {1, 3, 5};
            nmz_train_opts o = nmz_train_defaults();
            float before;
            nmz_rng r;
            CHECK(net != NULL);
            if (!net) return;
            CHECK(nmz_layer_param_decays(nmz_net_layer(net, 0), 0, 0) == 0); /* embedding */
            CHECK(nmz_layer_param_decays(nmz_net_layer(net, 1), 0, 0) == 0); /* posenc P   */
            CHECK(nmz_layer_param_decays(nmz_net_layer(net, 2), 0, 0) == 0); /* ln1.g      */
            CHECK(nmz_layer_param_decays(nmz_net_layer(net, 2), 2, 0) == 1); /* attn.W_qkv */
            CHECK(nmz_layer_param_decays(nmz_net_layer(net, 2), 3, 0) == 0); /* attn.b_qkv */
            nmz_rng_seed(&r, 72, 0);
            CHECK_OK(nmz_tensor_init(&x, 2, xs));
            CHECK_OK(nmz_tensor_init(&t, 3, ts));
            x.data[0] = 0, x.data[1] = 1, x.data[2] = 2;
            onehot_targets(&t, &r, 0);
            o.optimizer = NMZ_OPT_ADAMW;
            o.weight_decay = 0.5f;
            o.decay_all = all;
            /* untie: the tied output would give row 4 a gradient; look at the posenc row 7 instead */
            before = nmz_layer_param_value(nmz_net_layer(net, 1), 0)->data[7 * 4];
            CHECK_OK(nmz_net_forward(net, &x, NULL));
            CHECK_OK(nmz_net_backward(net, &t, NULL));
            CHECK_OK(nmz_net_optimizer_step(net, &o, 0.1f));
            {
                float after = nmz_layer_param_value(nmz_net_layer(net, 1), 0)->data[7 * 4];
                if (all)
                    CHECK_NEAR(after, before * (1 - 0.1 * 0.5), 1e-7);
                else
                    CHECK(after == before);
            }
            nmz_tensor_free(&x);
            nmz_tensor_free(&t);
            nmz_net_free(net);
        }
    }
}

static int collect_lr(const nmz_batch_info *bi, void *user) {
    float *lrs = user;
    lrs[(bi->epoch - 1) * bi->n_batches + bi->batch - 1] = bi->learning_rate;
    return 0;
}

static void test_warmup_cosine(void) {
    nmz_train_opts o = nmz_train_defaults();
    const double pi = 3.14159265358979323846;
    o.schedule = NMZ_SCHED_WARMUP_COSINE;
    o.learning_rate = 1.0f;
    o.min_lr = 0.1f;
    o.warmup_steps = 4;
    CHECK_NEAR(nmz_schedule_lr_step(&o, 1, 12, 3), 0.25, 1e-7);
    CHECK_NEAR(nmz_schedule_lr_step(&o, 3, 12, 3), 0.75, 1e-7);
    CHECK_NEAR(nmz_schedule_lr_step(&o, 4, 12, 3), 1.0, 1e-7);
    CHECK_NEAR(nmz_schedule_lr_step(&o, 5, 12, 3), 1.0, 1e-7); /* the cosine starts at η */
    CHECK_NEAR(nmz_schedule_lr_step(&o, 9, 12, 3), 0.1 + 0.45 * (1 + cos(pi / 2)), 1e-6);
    CHECK_NEAR(nmz_schedule_lr_step(&o, 12, 12, 3), 0.1 + 0.45 * (1 + cos(7 * pi / 8)), 1e-6); /* > η_min: no step is wasted */
    o.schedule = NMZ_SCHED_STEP; /* other schedules: per epoch, the epoch of the batch */
    o.step_size = 1;
    o.gamma = 0.5f;
    CHECK_NEAR(nmz_schedule_lr_step(&o, 4, 12, 3), 0.5, 1e-7);
    {   /* in training: one rate per batch, reported to on_batch */
        nmz_net *net = NULL;
        nmz_dataset *ds = NULL;
        int in = 2, b;
        float x[16], y[8], lrs[8];
        nmz_train_result res;
        for (b = 0; b < 8; b++) x[2 * b] = (float)b, x[2 * b + 1] = 1.0f, y[b] = 0.1f * b;
        CHECK_OK(nmz_net_create(&net, 1, &in, 5));
        CHECK_OK(nmz_net_add(net, nmz_layer_dense(1, NMZ_INIT_AUTO)));
        CHECK_OK(nmz_dataset_create(&ds, 8, 2, 1, x, y));
        o = nmz_train_defaults();
        o.optimizer = NMZ_OPT_ADAMW;
        o.weight_decay = 0.01f;
        o.schedule = NMZ_SCHED_WARMUP_COSINE;
        o.warmup_steps = 3;
        o.learning_rate = 0.01f;
        o.batch_size = 2; /* 4 batches per epoch, 2 epochs = 8 steps */
        o.max_epochs = 2;
        o.target_loss = 0;
        o.on_batch = collect_lr;
        o.user = lrs;
        CHECK_OK(nmz_net_train(net, ds, &o, &res));
        for (b = 0; b < 8; b++) CHECK_NEAR(lrs[b], nmz_schedule_lr_step(&o, b + 1, 8, 4), 1e-9);
        CHECK_NEAR(lrs[0], 0.01 / 3, 1e-9);
        CHECK_NEAR(lrs[2], 0.01, 1e-9);
        CHECK(lrs[7] > 0 && lrs[7] < 0.001);
        test_note("warm-up 3 + cosine, 8 batches: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f", lrs[0], lrs[1], lrs[2],
                  lrs[3], lrs[4], lrs[5], lrs[6], lrs[7]);
        /* the settings are saved with the model */
        *nmz_net_train_config(net) = o;
        CHECK_OK(nmz_net_save(net, test_path("adamw.nmz")));
        {
            nmz_net *back = NULL;
            CHECK_OK(nmz_net_load(&back, test_path("adamw.nmz")));
            if (back) {
                const nmz_train_opts *bo = nmz_net_train_config(back);
                long t1 = 0, t2 = 0;
                CHECK(bo->optimizer == NMZ_OPT_ADAMW && bo->schedule == NMZ_SCHED_WARMUP_COSINE);
                CHECK(bo->warmup_steps == 3 && bo->weight_decay == 0.01f && bo->decay_all == 0);
                CHECK(nmz_net_optimizer_state(back, &t1) == NMZ_OPT_ADAMW);
                nmz_net_optimizer_state(net, &t2);
                CHECK(t1 == 8 && t2 == 8);
                nmz_net_free(back);
            }
        }
        nmz_dataset_free(ds);
        nmz_net_free(net);
    }
}

/* --------------------------------------------------------- save/load */

static void test_gpt_save_load(void) {
    nmz_vocab *v = NULL;
    nmz_net *net = gpt_with_vocab("abcabcabd\n", 0, 12, &v), *back = NULL;
    nmz_tensor x = NMZ_TENSOR_EMPTY, keep = NMZ_TENSOR_EMPTY;
    const nmz_tensor *y;
    int xs[2] = {2, 5}, i;
    nmz_rng r;
    CHECK(net != NULL);
    if (!net) {
        nmz_vocab_free(v);
        return;
    }
    nmz_rng_seed(&r, 3, 3);
    CHECK_OK(nmz_tensor_init(&x, 2, xs));
    fill_ids(&x, &r, v->size);
    CHECK_OK(nmz_net_forward(net, &x, &y));
    CHECK_OK(nmz_tensor_copy(&keep, y));
    CHECK_OK(nmz_net_save(net, test_path("gpt.nmz")));
    CHECK_OK(nmz_net_load(&back, test_path("gpt.nmz")));
    if (back) {
        CHECK(nmz_net_layer_count(back) == nmz_net_layer_count(net));
        CHECK(nmz_net_param_count(back) == nmz_net_param_count(net));
        CHECK(nmz_net_context_length(back) == 12);
        for (i = 0; i < nmz_net_layer_count(net); i++) {
            char d1[128], d2[128];
            nmz_layer_describe(nmz_net_layer(net, i), d1, sizeof d1);
            nmz_layer_describe(nmz_net_layer(back, i), d2, sizeof d2);
            CHECK_MSG(!strcmp(d1, d2), "layer %d: '%s' vs '%s'", i, d1, d2);
        }
        CHECK_OK(nmz_net_forward(back, &x, &y));
        CHECK(memcmp(keep.data, y->data, keep.size * sizeof(float)) == 0);
        nmz_net_free(back);
    }
    {
        char d[128];
        nmz_layer_describe(nmz_net_layer(net, 2), d, sizeof d);
        test_note("%s", d);
    }
    nmz_tensor_free(&x);
    nmz_tensor_free(&keep);
    nmz_vocab_free(v);
    nmz_net_free(net);
}

/* a model written by the last phase-4 commit (348bcaf): embed, lstm,
 * dropout, gru, dense — Adam state included */
static void test_phase4_file_loads(void) {
    static const char *const want[4] = {"tgLalret\n", "rCcMlaaashdlJade\n", "WLasrSmd\n", "eFkiHbni\n"};
    nmz_net *net = NULL;
    nmz_gen_opts g = nmz_gen_defaults();
    int s;
    CHECK_OK(nmz_net_load(&net, "tests/data/v3_phase4_names_en.nmz"));
    if (!net) return;
    CHECK(nmz_net_layer_count(net) == 6);
    CHECK(!strcmp(nmz_layer_type(nmz_net_layer(net, 1)), "lstm") && !strcmp(nmz_layer_type(nmz_net_layer(net, 3)), "gru"));
    CHECK(nmz_net_optimizer_state(net, NULL) == NMZ_OPT_ADAM);
    CHECK(nmz_net_train_config(net)->weight_decay == 0.0f && nmz_net_train_config(net)->warmup_steps == 0);
    CHECK(nmz_net_context_length(net) == 0);
    g.stop_at_end = 1;
    g.length = 64;
    for (s = 0; s < 4; s++) {
        char *out = NULL;
        g.seed = (uint64_t)(3 + s);
        CHECK_OK(nmz_net_generate(net, &g, &out, NULL));
        CHECK_MSG(out && !strcmp(out, want[s]), "seed %d: '%s'", 3 + s, out ? out : "");
        free(out);
    }
    {
        char *out = NULL;
        g = nmz_gen_defaults();
        g.seed = 9;
        g.length = 30;
        g.prime = "Ma";
        g.temperature = 0.8f;
        g.top_k = 5;
        CHECK_OK(nmz_net_generate(net, &g, &out, NULL));
        CHECK(out && !strcmp(out, "n\n\neaa\neni\n\naaa\naaa\neae\nin\naei"));
        free(out);
    }
    nmz_net_free(net);
}

/* the attention trace event: weights sum to 1 over visible keys, 0 beyond */
static void test_attention_trace(void) {
    nmz_vocab *v = NULL;
    nmz_net *net = gpt_with_vocab("abcd\n", 0, 8, &v);
    nmz_trace_recorder *rec = NULL;
    nmz_tensor x = NMZ_TENSOR_EMPTY;
    int xs[2] = {1, 5}, found = 0, n_ln = 0, n_block = 0;
    size_t i;
    CHECK(net != NULL);
    if (!net) {
        nmz_vocab_free(v);
        return;
    }
    CHECK_OK(nmz_trace_recorder_create(&rec));
    CHECK_OK(nmz_tensor_init(&x, 2, xs));
    for (i = 0; i < 5; i++) x.data[i] = (float)(i % 4);
    nmz_net_set_trace(net, nmz_trace_recorder_fn, rec);
    CHECK_OK(nmz_net_forward(net, &x, NULL));
    nmz_net_set_trace(net, NULL, NULL);
    for (i = 0; i < nmz_trace_recorder_count(rec); i++) {
        const nmz_trace_event *ev = nmz_trace_recorder_get(rec, i);
        if (!strcmp(ev->layer_type, "layernorm") && ev->detail && !strncmp(ev->detail, "ln", 2)) n_ln++;
        if (!strcmp(ev->layer_type, "block")) n_block++;
        if (!strcmp(ev->layer_type, "attention")) {
            const nmz_tensor *w = nmz_trace_find(ev, "weights"), *sc = nmz_trace_find(ev, "scores");
            int t, j, ok = 1;
            CHECK(ev->detail && !strcmp(ev->detail, "attn"));
            CHECK(w && sc && w->rank == 4 && w->shape[1] == 2 && w->shape[2] == 5 && w->shape[3] == 5);
            if (!w || w->rank != 4) continue;
            for (t = 0; t < 5; t++) {
                double sum = 0;
                for (j = 0; j < 5; j++) {
                    float a = w->data[t * 5 + j];
                    if (j > t) ok &= a == 0.0f;
                    sum += a;
                }
                ok &= fabs(sum - 1.0) < 1e-6;
            }
            CHECK(ok);
            found++;
        }
    }
    CHECK(found == 2 && n_block == 2 && n_ln == 4);
    nmz_tensor_free(&x);
    nmz_trace_recorder_free(rec);
    nmz_vocab_free(v);
    nmz_net_free(net);
}

const test_case transformer_tests[] = {{"tf/matmul_blocked_equals_naive", test_matmul},
                                       {"tf/layernorm_by_hand", test_layernorm_by_hand},
                                       {"tf/posenc_by_hand", test_posenc_by_hand},
                                       {"tf/attention_by_hand", test_attention_by_hand},
                                       {"tf/gradcheck_building_blocks", test_gradcheck_blocks},
                                       {"tf/gradcheck_whole_model", test_gradcheck_gpt},
                                       {"tf/causal_mask", test_causal_mask},
                                       {"tf/weight_tying", test_weight_tying},
                                       {"tf/kv_cache_equals_recompute", test_kv_cache},
                                       {"tf/adamw_by_hand", test_adamw_by_hand},
                                       {"tf/warmup_cosine", test_warmup_cosine},
                                       {"tf/save_load", test_gpt_save_load},
                                       {"tf/phase4_file_loads", test_phase4_file_loads},
                                       {"tf/attention_trace", test_attention_trace},
                                       {NULL, NULL}};
