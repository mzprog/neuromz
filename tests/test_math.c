/* test_math.c — the math must be right: numerical gradient checks for every
 * layer, activation and loss, plus small hand-computed examples. */
#include "test.h"
#include "neuromz_layer.h"

#include <string.h>

static void fill(nmz_tensor *t, nmz_rng *r, double lo, double hi) {
    size_t i;
    for (i = 0; i < t->size; i++) t->data[i] = (float)(lo + (hi - lo) * nmz_rng_uniform(r));
}

/* one-hot rows (or soft targets if soft != 0) */
static void fill_targets(nmz_tensor *t, nmz_rng *r, int onehot, int soft) {
    size_t rows = nmz_tensor_rows(t), i;
    int k, cols = nmz_tensor_cols(t);
    if (!onehot && !soft) {
        fill(t, r, 0.0, 1.0);
        return;
    }
    for (i = 0; i < rows; i++) {
        float *tr = t->data + i * (size_t)cols;
        for (k = 0; k < cols; k++) tr[k] = soft ? (float)(0.1 + 0.4 * nmz_rng_uniform(r)) : 0.0f;
        if (onehot) tr[nmz_rng_below(r, (uint32_t)cols)] = 1.0f;
    }
}

typedef struct spec {
    int units;
    nmz_act act;
    float alpha;
} spec;

/* Build net, random batch, run gradcheck; returns failures. */
static size_t run_gradcheck(const char *label, int in_rank, const int *in_shape, const spec *layers,
                            int n_layers, nmz_loss loss, int batch, int onehot, int soft, uint64_t seed) {
    nmz_net *net = NULL;
    nmz_tensor x = NMZ_TENSOR_EMPTY, t = NMZ_TENSOR_EMPTY;
    nmz_gradcheck_result res;
    nmz_rng r;
    int i, shape[NMZ_MAX_RANK];
    const nmz_tensor *y;
    memset(&res, 0, sizeof res);
    nmz_rng_seed(&r, seed, 3);
    CHECK_OK(nmz_net_create(&net, in_rank, in_shape, seed));
    for (i = 0; i < n_layers; i++) {
        CHECK_OK(nmz_net_add(net, nmz_layer_dense(layers[i].units, NMZ_INIT_AUTO)));
        if (layers[i].act != NMZ_ACT_LINEAR)
            CHECK_OK(nmz_net_add(net, nmz_layer_activation(layers[i].act, layers[i].alpha)));
    }
    CHECK_OK(nmz_net_set_loss(net, loss));
    CHECK_OK(nmz_net_build(net));
    /* random biases too, so they are not all zero during the check */
    for (i = 0; i < nmz_net_layer_count(net); i++) {
        nmz_layer *l = nmz_net_layer(net, i);
        if (nmz_layer_param_count(l) == 2) fill(nmz_layer_param_value(l, 1), &r, -0.5, 0.5);
    }
    shape[0] = batch;
    for (i = 0; i < in_rank; i++) shape[i + 1] = in_shape[i];
    CHECK_OK(nmz_tensor_init(&x, in_rank + 1, shape));
    fill(&x, &r, -1.5, 1.5);
    CHECK_OK(nmz_net_forward(net, &x, &y));
    CHECK_OK(nmz_tensor_init(&t, y->rank, y->shape));
    fill_targets(&t, &r, onehot, soft);
    CHECK_OK(nmz_net_gradcheck(net, &x, &t, 1e-3, 0, 0, &res));
    CHECK_MSG(res.failures == 0 && res.checked > 0,
              "%s: %zu/%zu mismatches; worst layer %d %s[%zu]: analytic %.6g numeric %.6g", label,
              res.failures, res.checked, res.worst_layer, res.worst_param, res.worst_index,
              res.worst_analytic, res.worst_numeric);
    test_note("%-28s %4zu grads, max |err| %.2e", label, res.checked, res.max_abs_err);
    nmz_tensor_free(&x);
    nmz_tensor_free(&t);
    nmz_net_free(net);
    return res.failures;
}

static void test_gradcheck_activations(void) {
    int in = 3;
    spec s[2];
    int a;
    for (a = 0; a < NMZ_ACT_COUNT; a++) {
        char label[64];
        if (a == NMZ_ACT_SOFTMAX) continue;
        s[0].units = 4;
        s[0].act = (nmz_act)a;
        s[0].alpha = a == NMZ_ACT_LEAKY_RELU ? 0.1f : 0.0f;
        s[1].units = 2;
        s[1].act = NMZ_ACT_LINEAR;
        s[1].alpha = 0;
        snprintf(label, sizeof label, "dense+%s / mse", nmz_act_name((nmz_act)a));
        run_gradcheck(label, 1, &in, s, 2, NMZ_LOSS_MSE, 5, 0, 0, 11 + (uint64_t)a);
    }
}

static void test_gradcheck_outputs(void) {
    int in = 3;
    spec sig[2] = {{4, NMZ_ACT_TANH, 0}, {3, NMZ_ACT_SIGMOID, 0}};
    spec smx[2] = {{4, NMZ_ACT_TANH, 0}, {3, NMZ_ACT_SOFTMAX, 0}};
    run_gradcheck("sigmoid output / mse", 1, &in, sig, 2, NMZ_LOSS_MSE, 4, 0, 0, 21);
    run_gradcheck("softmax / mse (Jacobian)", 1, &in, smx, 2, NMZ_LOSS_MSE, 4, 0, 0, 22);
    run_gradcheck("softmax+CE one-hot (fused)", 1, &in, smx, 2, NMZ_LOSS_CROSS_ENTROPY, 4, 1, 0, 23);
    run_gradcheck("softmax+CE soft targets", 1, &in, smx, 2, NMZ_LOSS_CROSS_ENTROPY, 4, 0, 1, 24);
}

static void test_gradcheck_deep(void) {
    int in = 3, in2[2] = {3, 4};
    spec deep[4] = {{5, NMZ_ACT_TANH, 0}, {4, NMZ_ACT_RELU, 0}, {4, NMZ_ACT_SIGMOID, 0}, {2, NMZ_ACT_LINEAR, 0}};
    spec seq[2] = {{5, NMZ_ACT_TANH, 0}, {2, NMZ_ACT_LINEAR, 0}};
    spec one[1] = {{3, NMZ_ACT_LINEAR, 0}};
    run_gradcheck("deep tanh/relu/sigmoid/lin", 1, &in, deep, 4, NMZ_LOSS_MSE, 3, 0, 0, 31);
    run_gradcheck("single dense, batch 1", 1, &in, one, 1, NMZ_LOSS_MSE, 1, 0, 0, 32);
    /* per-sample shape [3,4]: Dense works on the last dim (rows = B*3) */
    run_gradcheck("rank-2 samples (rows)", 2, in2, seq, 2, NMZ_LOSS_MSE, 2, 0, 0, 33);
}

static void test_activation_derivatives(void) {
    /* scalar f'(z) against central differences in double precision */
    const double h = 1e-3;
    int a;
    double z;
    for (a = 0; a < NMZ_ACT_COUNT; a++) {
        if (a == NMZ_ACT_SOFTMAX) continue;
        for (z = -4.05; z <= 4.0; z += 0.5) { /* avoids the kink at exactly 0 */
            double num = ((double)nmz_act_apply((nmz_act)a, (float)(z + h), 0.1f) -
                          (double)nmz_act_apply((nmz_act)a, (float)(z - h), 0.1f)) / (2 * h);
            double ana = nmz_act_derivative((nmz_act)a, (float)z, 0.1f);
            CHECK_MSG(fabs(num - ana) < 2e-3, "%s'(%g): analytic %g numeric %g", nmz_act_name((nmz_act)a), z,
                      ana, num);
        }
    }
    /* known values */
    CHECK_NEAR(nmz_act_apply(NMZ_ACT_SIGMOID, 0.0f, 0), 0.5, 1e-7);
    CHECK_NEAR(nmz_act_derivative(NMZ_ACT_SIGMOID, 0.0f, 0), 0.25, 1e-7);
    CHECK_NEAR(nmz_act_derivative(NMZ_ACT_TANH, 0.0f, 0), 1.0, 1e-7);
    CHECK_NEAR(nmz_act_apply(NMZ_ACT_LEAKY_RELU, -2.0f, 0.01f), -0.02, 1e-7);
    CHECK(nmz_act_apply(NMZ_ACT_SIGMOID, -1000.0f, 0) == 0.0f); /* stable, no NaN */
    CHECK(nmz_act_apply(NMZ_ACT_SIGMOID, 1000.0f, 0) == 1.0f);
}

/* A 2→2 linear layer with hand-set weights: checks forward, MSE, and that
 * EVERY column of dW gets its own gradient (the v1 bug updated column 0 only). */
static void test_hand_computed(void) {
    nmz_net *net = NULL;
    nmz_tensor x = NMZ_TENSOR_EMPTY, t = NMZ_TENSOR_EMPTY;
    const nmz_tensor *y;
    nmz_layer *d;
    float *W, *b, *dW, *db;
    int in = 2, xs[2] = {1, 2};
    double L;
    CHECK_OK(nmz_net_create(&net, 1, &in, 1));
    CHECK_OK(nmz_net_add(net, nmz_layer_dense(2, NMZ_INIT_XAVIER)));
    CHECK_OK(nmz_net_build(net));
    d = nmz_net_layer(net, 0);
    W = nmz_layer_param_value(d, 0)->data;
    b = nmz_layer_param_value(d, 1)->data;
    /* W = [[1, 2], [3, 4]] (row i = input i), b = [0.5, -1] */
    W[0] = 1; W[1] = 2; W[2] = 3; W[3] = 4;
    b[0] = 0.5f; b[1] = -1;
    CHECK_OK(nmz_tensor_init(&x, 2, xs));
    CHECK_OK(nmz_tensor_init(&t, 2, xs));
    x.data[0] = 1; x.data[1] = 2;   /* z0 = 1*1 + 2*3 + 0.5 = 7.5 ; z1 = 1*2 + 2*4 - 1 = 9 */
    t.data[0] = 7; t.data[1] = 10;  /* errors: +0.5, -1 */
    CHECK_OK(nmz_net_forward(net, &x, &y));
    CHECK_NEAR(y->data[0], 7.5, 1e-6);
    CHECK_NEAR(y->data[1], 9.0, 1e-6);
    CHECK_OK(nmz_net_backward(net, &t, &L));
    CHECK_NEAR(L, 0.5 * (0.25 + 1.0), 1e-9); /* ½Σ(y−t)² */
    dW = nmz_layer_param_grad(d, 0)->data;
    db = nmz_layer_param_grad(d, 1)->data;
    /* dW_ij = x_i * (y_j − t_j) */
    CHECK_NEAR(dW[0], 1 * 0.5, 1e-6);
    CHECK_NEAR(dW[1], 1 * -1.0, 1e-6);
    CHECK_NEAR(dW[2], 2 * 0.5, 1e-6);
    CHECK_NEAR(dW[3], 2 * -1.0, 1e-6);
    CHECK_NEAR(db[0], 0.5, 1e-6);
    CHECK_NEAR(db[1], -1.0, 1e-6);
    /* dL/dx_i = Σ_j W_ij δ_j : x0: 1*0.5 + 2*-1 = -1.5 ; x1: 3*0.5 + 4*-1 = -2.5 */
    CHECK_NEAR(nmz_net_input_grad(net)->data[0], -1.5, 1e-6);
    CHECK_NEAR(nmz_net_input_grad(net)->data[1], -2.5, 1e-6);
    /* plain SGD step, lr 0.1: W ← W − 0.1·dW */
    CHECK_OK(nmz_net_update(net, 0.1f, 0.0f));
    CHECK_NEAR(W[1], 2 + 0.1, 1e-6);
    CHECK_NEAR(W[3], 4 + 0.2, 1e-6);
    /* momentum: second identical step: v = 0.9·v − η·g */
    CHECK_OK(nmz_net_forward(net, &x, &y));
    CHECK_OK(nmz_net_backward(net, &t, NULL));
    {
        float g = dW[0], w_before = W[0];
        CHECK_OK(nmz_net_update(net, 0.1f, 0.9f));
        CHECK_NEAR(W[0], w_before + (0.9 * (-0.1 * 0.5) - 0.1 * g), 1e-6);
    }
    nmz_tensor_free(&x);
    nmz_tensor_free(&t);
    nmz_net_free(net);
}

static void test_softmax_ce_values(void) {
    nmz_net *net = NULL;
    nmz_tensor x = NMZ_TENSOR_EMPTY, t = NMZ_TENSOR_EMPTY;
    const nmz_tensor *y;
    nmz_layer *d;
    int in = 2, xs[2] = {2, 2}, ts[2] = {2, 3};
    double L;
    size_t i;
    CHECK_OK(nmz_net_create(&net, 1, &in, 1));
    CHECK_OK(nmz_net_add_dense(net, 3, NMZ_ACT_SOFTMAX, NMZ_INIT_XAVIER));
    CHECK_OK(nmz_net_set_loss(net, NMZ_LOSS_CROSS_ENTROPY));
    CHECK_OK(nmz_net_build(net));
    d = nmz_net_layer(net, 0);
    nmz_tensor_zero(nmz_layer_param_value(d, 0)); /* z = 0 → p = 1/3 each */
    CHECK_OK(nmz_tensor_init(&x, 2, xs));
    CHECK_OK(nmz_tensor_init(&t, 2, ts));
    t.data[0] = 1; t.data[5] = 1;
    CHECK_OK(nmz_net_forward(net, &x, &y));
    for (i = 0; i < y->size; i++) CHECK_NEAR(y->data[i], 1.0 / 3, 1e-6);
    CHECK_OK(nmz_net_backward(net, &t, &L));
    CHECK_NEAR(L, log(3.0), 1e-6);
    /* fused δ = (p − t)/B  → dL/db = Σ_rows δ = [(1/3−1)/2 + (1/3)/2, ...] */
    CHECK_NEAR(nmz_layer_param_grad(d, 1)->data[0], (1.0 / 3 - 1) / 2 + (1.0 / 3) / 2, 1e-6);
    CHECK_NEAR(nmz_layer_param_grad(d, 1)->data[1], (1.0 / 3) / 2 + (1.0 / 3) / 2, 1e-6);
    /* huge logits stay finite */
    nmz_layer_param_value(d, 1)->data[0] = 1000.0f;
    CHECK_OK(nmz_net_forward(net, &x, &y));
    CHECK(y->data[0] == 1.0f && y->data[1] == 0.0f);
    CHECK_OK(nmz_net_backward(net, &t, &L));
    /* row 1 is confidently right (≈0), row 2 confidently wrong (≈1000): mean ≈ 500 */
    CHECK_NEAR(L, 500.0, 1.0);
    nmz_tensor_free(&x);
    nmz_tensor_free(&t);
    nmz_net_free(net);
}

static void test_init_distribution(void) {
    nmz_net *net = NULL;
    int in = 200;
    nmz_tensor *W;
    double sum = 0, sq = 0, lim_x = sqrt(6.0 / (200 + 100)), lim_h = sqrt(6.0 / 200), mx = 0;
    size_t i;
    CHECK_OK(nmz_net_create(&net, 1, &in, 5));
    CHECK_OK(nmz_net_add_dense(net, 100, NMZ_ACT_TANH, NMZ_INIT_AUTO)); /* → xavier */
    CHECK_OK(nmz_net_add_dense(net, 50, NMZ_ACT_RELU, NMZ_INIT_AUTO));  /* → he */
    CHECK_OK(nmz_net_build(net));
    W = nmz_layer_param_value(nmz_net_layer(net, 0), 0);
    for (i = 0; i < W->size; i++) {
        sum += W->data[i];
        sq += (double)W->data[i] * W->data[i];
        if (fabs(W->data[i]) > mx) mx = fabs(W->data[i]);
    }
    CHECK_NEAR(sum / W->size, 0.0, 0.01);
    CHECK_NEAR(sq / W->size, 2.0 / 300, 0.0005); /* Glorot variance */
    CHECK(mx <= lim_x);
    W = nmz_layer_param_value(nmz_net_layer(net, 2), 0);
    for (i = 0, sq = 0, mx = 0; i < W->size; i++) {
        sq += (double)W->data[i] * W->data[i];
        if (fabs(W->data[i]) > mx) mx = fabs(W->data[i]);
    }
    CHECK_NEAR(sq / W->size, 2.0 / 100, 0.002); /* He variance, fan_in = 100 */
    CHECK(mx <= sqrt(6.0 / 100) && lim_h > 0);
    nmz_net_free(net);
}

const test_case math_tests[] = {
    {"math/activation_derivatives", test_activation_derivatives},
    {"math/gradcheck_each_activation", test_gradcheck_activations},
    {"math/gradcheck_output_losses", test_gradcheck_outputs},
    {"math/gradcheck_deep_and_shapes", test_gradcheck_deep},
    {"math/hand_computed_dense_mse_sgd", test_hand_computed},
    {"math/softmax_ce_values", test_softmax_ce_values},
    {"math/xavier_he_init", test_init_distribution},
    {NULL, NULL}};
