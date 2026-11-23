/* test_layers.c — Conv2D, MaxPool2D, Flatten, Dropout: gradient checks over
 * stride/padding/kernel combinations, hand-computed values, modes, errors. */
#include "test.h"
#include "neuromz_layer.h"

#include <stdlib.h>
#include <string.h>

static void fill(nmz_tensor *t, nmz_rng *r, double lo, double hi) {
    size_t i;
    for (i = 0; i < t->size; i++) t->data[i] = (float)(lo + (hi - lo) * nmz_rng_uniform(r));
}

/* distinct values (spacing 0.05, shuffled): max-pool has no near-ties, so a
 * ±eps perturbation never changes which input wins */
static void fill_distinct(nmz_tensor *t, nmz_rng *r) {
    size_t i;
    for (i = 0; i < t->size; i++) t->data[i] = (float)(0.05 * (double)i - 0.025 * (double)t->size);
    for (i = t->size - 1; i > 0; i--) {
        size_t j = nmz_rng_below(r, (uint32_t)i + 1);
        float tmp = t->data[i];
        t->data[i] = t->data[j];
        t->data[j] = tmp;
    }
}

/* Gradcheck an already-built net on a random batch. */
static void check(const char *label, nmz_net *net, int batch, int distinct_input, int onehot, int training,
                  uint64_t seed) {
    nmz_tensor x = NMZ_TENSOR_EMPTY, t = NMZ_TENSOR_EMPTY;
    nmz_gradcheck_result res;
    const nmz_tensor *y;
    nmz_rng r;
    int i, shape[NMZ_MAX_RANK];
    memset(&res, 0, sizeof res);
    nmz_rng_seed(&r, seed, 9);
    CHECK_OK(nmz_net_build(net));
    for (i = 0; i < nmz_net_layer_count(net); i++) { /* non-zero biases */
        nmz_layer *l = nmz_net_layer(net, i);
        if (nmz_layer_param_count(l) == 2) fill(nmz_layer_param_value(l, 1), &r, -0.3, 0.3);
    }
    {
        nmz_layer *l0 = nmz_net_layer(net, 0);
        shape[0] = batch;
        for (i = 0; i < l0->in_rank; i++) shape[i + 1] = l0->in_shape[i];
        CHECK_OK(nmz_tensor_init(&x, l0->in_rank + 1, shape));
    }
    if (distinct_input)
        fill_distinct(&x, &r);
    else
        fill(&x, &r, -1.0, 1.0);
    CHECK_OK(nmz_net_forward(net, &x, &y));
    CHECK_OK(nmz_tensor_init(&t, y->rank, y->shape));
    if (onehot) {
        size_t rr, rows = nmz_tensor_rows(&t);
        for (rr = 0; rr < rows; rr++) t.data[rr * nmz_tensor_cols(&t) + nmz_rng_below(&r, (uint32_t)nmz_tensor_cols(&t))] = 1;
    } else {
        fill(&t, &r, 0.0, 1.0);
    }
    CHECK_OK(nmz_net_gradcheck_mode(net, &x, &t, 1e-3, 0, 0, training, &res));
    CHECK_MSG(res.failures == 0 && res.checked > 0,
              "%s: %zu/%zu mismatches; worst layer %d %s[%zu]: analytic %.6g numeric %.6g", label, res.failures,
              res.checked, res.worst_layer, res.worst_param, res.worst_index, res.worst_analytic, res.worst_numeric);
    test_note("%-40s %5zu grads, max |err| %.2e", label, res.checked, res.max_abs_err);
    nmz_tensor_free(&x);
    nmz_tensor_free(&t);
}

static nmz_net *net3(int c, int h, int w, uint64_t seed) {
    nmz_net *net = NULL;
    int shape[3];
    shape[0] = c;
    shape[1] = h;
    shape[2] = w;
    CHECK_OK(nmz_net_create(&net, 3, shape, seed));
    return net;
}

static void head(nmz_net *net, int outputs) { /* flatten → dense (linear) */
    CHECK_OK(nmz_net_add(net, nmz_layer_flatten()));
    CHECK_OK(nmz_net_add(net, nmz_layer_dense(outputs, NMZ_INIT_XAVIER)));
}

static void test_gradcheck_conv(void) {
    struct {
        const char *label;
        int c, h, w, filters, kh, kw, stride, pad;
    } cases[] = {
        {"conv 1->2, 3x3, s1 p0, 5x5", 1, 5, 5, 2, 3, 3, 1, 0},
        {"conv 2->3, 3x3, s2 p1, 6x7", 2, 6, 7, 3, 3, 3, 2, 1},
        {"conv 2->2, 3x3, s1 pad=same, 5x5", 2, 5, 5, 2, 3, 3, 1, NMZ_PAD_SAME},
        {"conv 1->2, 2x3 kernel, s1 p0, 4x6", 1, 4, 6, 2, 2, 3, 1, 0},
        {"conv 1->2, 3x3, s2 p0, 6x6 (edge cut)", 1, 6, 6, 2, 3, 3, 2, 0},
        {"conv 3->2, 1x1, s1 p0, 3x3", 3, 3, 3, 2, 1, 1, 1, 0},
        {"conv 1->1, 5x5, s3 p2, 7x7", 1, 7, 7, 1, 5, 5, 3, 2},
    };
    size_t i;
    for (i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        nmz_net *net = net3(cases[i].c, cases[i].h, cases[i].w, 100 + i);
        CHECK_OK(nmz_net_add(net, nmz_layer_conv2d(cases[i].filters, cases[i].kh, cases[i].kw, cases[i].stride,
                                                   cases[i].pad, NMZ_INIT_AUTO)));
        head(net, 3);
        check(cases[i].label, net, 2, 0, 0, 1, 200 + i);
        nmz_net_free(net);
    }
    {
        /* activation between conv and head, softmax + CE output */
        nmz_net *net = net3(2, 6, 6, 7);
        CHECK_OK(nmz_net_add(net, nmz_layer_conv2d(3, 3, 3, 1, 1, NMZ_INIT_AUTO)));
        CHECK_OK(nmz_net_add(net, nmz_layer_activation(NMZ_ACT_TANH, 0)));
        CHECK_OK(nmz_net_add(net, nmz_layer_flatten()));
        CHECK_OK(nmz_net_add_dense(net, 4, NMZ_ACT_SOFTMAX, NMZ_INIT_AUTO));
        CHECK_OK(nmz_net_set_loss(net, NMZ_LOSS_CROSS_ENTROPY));
        check("conv+tanh -> softmax/CE", net, 3, 0, 1, 1, 77);
        nmz_net_free(net);
    }
}

static void test_gradcheck_pool_flatten(void) {
    nmz_net *net = net3(2, 4, 4, 1);
    CHECK_OK(nmz_net_add(net, nmz_layer_maxpool2d(2, 0)));
    head(net, 3);
    check("maxpool 2x2 s2 on 2x4x4", net, 2, 1, 0, 1, 11);
    nmz_net_free(net);

    net = net3(1, 5, 5, 2); /* overlapping windows: gradients add */
    CHECK_OK(nmz_net_add(net, nmz_layer_maxpool2d(3, 2)));
    head(net, 2);
    check("maxpool 3x3 s2 (overlapping)", net, 2, 1, 0, 1, 12);
    nmz_net_free(net);

    {
        nmz_net *f = NULL;
        int in[2] = {3, 4};
        CHECK_OK(nmz_net_create(&f, 2, in, 3));
        CHECK_OK(nmz_net_add(f, nmz_layer_flatten()));
        CHECK_OK(nmz_net_add_dense(f, 3, NMZ_ACT_SIGMOID, NMZ_INIT_AUTO));
        check("flatten [3,4] -> dense", f, 2, 0, 0, 1, 13);
        nmz_net_free(f);
    }
    /* a whole small CNN */
    net = net3(1, 8, 8, 4);
    CHECK_OK(nmz_net_add(net, nmz_layer_conv2d(3, 3, 3, 1, 0, NMZ_INIT_AUTO)));
    CHECK_OK(nmz_net_add(net, nmz_layer_activation(NMZ_ACT_RELU, 0)));
    CHECK_OK(nmz_net_add(net, nmz_layer_maxpool2d(2, 0)));
    CHECK_OK(nmz_net_add(net, nmz_layer_flatten()));
    CHECK_OK(nmz_net_add_dense(net, 5, NMZ_ACT_RELU, NMZ_INIT_AUTO));
    CHECK_OK(nmz_net_add_dense(net, 3, NMZ_ACT_SOFTMAX, NMZ_INIT_AUTO));
    CHECK_OK(nmz_net_set_loss(net, NMZ_LOSS_CROSS_ENTROPY));
    check("conv-relu-pool-flatten-dense-softmax", net, 2, 0, 1, 1, 14);
    nmz_net_free(net);
}

static void test_gradcheck_dropout(void) {
    int in = 6;
    nmz_net *net = NULL;
    CHECK_OK(nmz_net_create(&net, 1, &in, 5));
    CHECK_OK(nmz_net_add_dense(net, 8, NMZ_ACT_TANH, NMZ_INIT_AUTO));
    CHECK_OK(nmz_net_add(net, nmz_layer_dropout(0.5f)));
    CHECK_OK(nmz_net_add_dense(net, 2, NMZ_ACT_LINEAR, NMZ_INIT_AUTO));
    check("dropout 0.5, eval mode (identity)", net, 3, 0, 0, 0, 21);
    check("dropout 0.5, training (mask replayed)", net, 3, 0, 0, 1, 22);
    nmz_net_free(net);
}

/* hand-computed 1×3×3 conv with a 2×2 kernel */
static void test_conv_by_hand(void) {
    nmz_net *net = net3(1, 3, 3, 1);
    nmz_tensor x = NMZ_TENSOR_EMPTY, t = NMZ_TENSOR_EMPTY;
    const nmz_tensor *y;
    nmz_layer *c;
    int xs[4] = {1, 1, 3, 3}, ts[4] = {1, 1, 2, 2}, i;
    float *W, *dW, *dx;
    CHECK_OK(nmz_net_add(net, nmz_layer_conv2d(1, 2, 2, 1, 0, NMZ_INIT_HE)));
    CHECK_OK(nmz_net_build(net));
    c = nmz_net_layer(net, 0);
    W = nmz_layer_param_value(c, 0)->data;
    W[0] = 1; W[1] = 0; W[2] = 0; W[3] = -1; /* [[1,0],[0,-1]]: top-left minus bottom-right */
    nmz_layer_param_value(c, 1)->data[0] = 0.5f;
    CHECK_OK(nmz_tensor_init(&x, 4, xs));
    for (i = 0; i < 9; i++) x.data[i] = (float)(i + 1); /* [[1,2,3],[4,5,6],[7,8,9]] */
    CHECK_OK(nmz_net_forward(net, &x, &y));
    CHECK(y->rank == 4 && y->shape[1] == 1 && y->shape[2] == 2 && y->shape[3] == 2);
    for (i = 0; i < 4; i++) CHECK_NEAR(y->data[i], 1 - 5 + 0.5, 1e-6); /* every patch: a − (a+4) + 0.5 */
    CHECK_OK(nmz_tensor_init(&t, 4, ts)); /* t = 0 */
    CHECK_OK(nmz_net_backward(net, &t, NULL));
    /* MSE averages over rows = B·F·OH = 2, so δ = (z − t)/2 */
    dW = nmz_layer_param_grad(c, 0)->data;
    /* dW[ky,kx] = Σ_pos δ·x[pos+(ky,kx)], δ = −3.5/2 = −1.75 everywhere */
    CHECK_NEAR(dW[0], -1.75 * (1 + 2 + 4 + 5), 1e-5);
    CHECK_NEAR(dW[3], -1.75 * (5 + 6 + 8 + 9), 1e-5);
    CHECK_NEAR(nmz_layer_param_grad(c, 1)->data[0], -1.75 * 4, 1e-5);
    dx = (float *)nmz_net_input_grad(net)->data;
    CHECK_NEAR(dx[0], -1.75 * 1, 1e-6);       /* x[0,0] only meets W[0,0] */
    CHECK_NEAR(dx[4], -1.75 * (1 - 1), 1e-6); /* centre meets every weight: 1+0+0−1 */
    CHECK_NEAR(dx[8], -1.75 * -1, 1e-6);      /* x[2,2] only meets W[1,1] */
    nmz_tensor_free(&x);
    nmz_tensor_free(&t);
    nmz_net_free(net);
}

static void test_pool_by_hand(void) {
    nmz_net *net = net3(1, 4, 4, 1);
    nmz_tensor x = NMZ_TENSOR_EMPTY, t = NMZ_TENSOR_EMPTY;
    const nmz_tensor *y;
    static const float v[16] = {1, 9, 2, 3, 4, 0, 8, 7, 5, 5, 6, 1, 2, 3, 0, 6};
    int xs[4] = {1, 1, 4, 4}, ts[4] = {1, 1, 2, 2};
    const float *dx;
    CHECK_OK(nmz_net_add(net, nmz_layer_maxpool2d(2, 2)));
    CHECK_OK(nmz_tensor_init(&x, 4, xs));
    memcpy(x.data, v, sizeof v);
    CHECK_OK(nmz_net_forward(net, &x, &y));
    CHECK(y->data[0] == 9 && y->data[1] == 8 && y->data[2] == 5 && y->data[3] == 6);
    CHECK_OK(nmz_tensor_init(&t, 4, ts));
    CHECK_OK(nmz_net_backward(net, &t, NULL));
    dx = nmz_net_input_grad(net)->data;
    /* rows = 2: δ = y/2; winners at indices 1, 6, 8 and 10 (ties → first in row-major order) */
    CHECK(dx[1] == 4.5f && dx[6] == 4.0f && dx[8] == 2.5f && dx[10] == 3.0f);
    CHECK(dx[9] == 0.0f && dx[15] == 0.0f && dx[0] == 0.0f && dx[7] == 0.0f);
    nmz_tensor_free(&x);
    nmz_tensor_free(&t);
    nmz_net_free(net);
}

static void test_dropout_modes(void) {
    int in = 1000, i, kept = 0, B = 1;
    nmz_net *a = NULL, *b = NULL;
    nmz_tensor x = NMZ_TENSOR_EMPTY, t = NMZ_TENSOR_EMPTY;
    const nmz_tensor *y;
    int xs[2];
    float y1[1000];
    xs[0] = B;
    xs[1] = in;
    CHECK_OK(nmz_net_create(&a, 1, &in, 3));
    CHECK_OK(nmz_net_add(a, nmz_layer_dropout(0.25f)));
    CHECK_OK(nmz_tensor_init(&x, 2, xs));
    for (i = 0; i < in; i++) x.data[i] = 1.0f;
    /* eval: identity */
    CHECK_OK(nmz_net_forward(a, &x, &y));
    for (i = 0; i < in; i++) CHECK(y->data[i] == 1.0f);
    /* training mode: run one training epoch and read the mask from the trace */
    nmz_net_free(a);
    a = NULL;
    CHECK_OK(nmz_net_create(&a, 1, &in, 3));
    CHECK_OK(nmz_net_add(a, nmz_layer_dropout(0.25f)));
    CHECK_OK(nmz_net_create(&b, 1, &in, 3));
    CHECK_OK(nmz_net_add(b, nmz_layer_dropout(0.25f)));
    {
        nmz_trace_recorder *rec = NULL;
        nmz_dataset *ds = NULL;
        nmz_train_opts o = nmz_train_defaults();
        nmz_train_result r;
        const nmz_tensor *mask = NULL;
        size_t k;
        float *ones = calloc((size_t)in, sizeof(float));
        for (i = 0; ones && i < in; i++) ones[i] = 1.0f;
        CHECK_OK(nmz_dataset_create(&ds, 1, in, in, ones, ones));
        o.max_epochs = 1;
        o.shuffle = 0;
        CHECK_OK(nmz_trace_recorder_create(&rec));
        nmz_net_set_trace(a, nmz_trace_recorder_fn, rec);
        CHECK_OK(nmz_net_train(a, ds, &o, &r));
        nmz_net_set_trace(a, NULL, NULL);
        for (k = 0; k < nmz_trace_recorder_count(rec); k++) {
            const nmz_trace_event *ev = nmz_trace_recorder_get(rec, k);
            double tr = 0;
            if (ev->phase == NMZ_TRACE_FORWARD && !strcmp(ev->layer_type, "dropout") &&
                nmz_trace_scalar_get(ev, "training", &tr) && tr == 1.0)
                mask = nmz_trace_find(ev, "mask");
        }
        CHECK(mask != NULL);
        if (mask) {
            for (i = 0; i < in; i++) {
                CHECK(mask->data[i] == 0.0f || fabsf(mask->data[i] - 1.0f / 0.75f) < 1e-6f);
                kept += mask->data[i] != 0.0f;
                y1[i] = mask->data[i];
            }
            CHECK_MSG(kept > 700 && kept < 800, "kept %d of 1000 at rate 0.25", kept);
        }
        /* same seed → same mask */
        nmz_trace_recorder_clear(rec);
        nmz_net_set_trace(b, nmz_trace_recorder_fn, rec);
        CHECK_OK(nmz_net_train(b, ds, &o, &r));
        nmz_net_set_trace(b, NULL, NULL);
        mask = NULL;
        for (k = 0; k < nmz_trace_recorder_count(rec); k++) {
            const nmz_trace_event *ev = nmz_trace_recorder_get(rec, k);
            if (ev->phase == NMZ_TRACE_FORWARD && !strcmp(ev->layer_type, "dropout") && !mask) mask = nmz_trace_find(ev, "mask");
        }
        CHECK(mask != NULL && memcmp(mask->data, y1, sizeof y1) == 0);
        nmz_trace_recorder_free(rec);
        nmz_dataset_free(ds);
        free(ones);
    }
    nmz_tensor_free(&x);
    nmz_tensor_free(&t);
    nmz_net_free(a);
    nmz_net_free(b);
}

static void test_shape_errors(void) {
    nmz_net *net = NULL;
    int in = 10;
    CHECK_OK(nmz_net_create(&net, 1, &in, 1));
    CHECK_STATUS(nmz_net_add(net, nmz_layer_conv2d(2, 3, 3, 1, 0, NMZ_INIT_AUTO)), NMZ_ERR_SHAPE);
    CHECK_STATUS(nmz_net_add(net, nmz_layer_maxpool2d(2, 0)), NMZ_ERR_SHAPE);
    nmz_net_free(net);
    net = net3(1, 4, 4, 1);
    CHECK_STATUS(nmz_net_add(net, nmz_layer_conv2d(2, 5, 5, 1, 0, NMZ_INIT_AUTO)), NMZ_ERR_SHAPE);
    CHECK_STATUS(nmz_net_add(net, nmz_layer_conv2d(2, 2, 2, 1, NMZ_PAD_SAME, NMZ_INIT_AUTO)), NMZ_ERR_ARG);
    CHECK(nmz_layer_conv2d(0, 3, 3, 1, 0, NMZ_INIT_AUTO) == NULL);
    CHECK(nmz_layer_conv2d(2, 3, 3, 0, 0, NMZ_INIT_AUTO) == NULL);
    CHECK(nmz_layer_dropout(1.0f) == NULL && nmz_layer_dropout(-0.1f) == NULL);
    CHECK(nmz_layer_maxpool2d(0, 0) == NULL);
    CHECK_OK(nmz_net_add(net, nmz_layer_conv2d(2, 3, 3, 1, NMZ_PAD_SAME, NMZ_INIT_AUTO)));
    CHECK(nmz_layer_output_size(nmz_net_layer(net, 0)) == 2 * 4 * 4);
    CHECK(nmz_layer_init_used(nmz_net_layer(net, 0)) == NMZ_INIT_AUTO); /* resolved at build */
    CHECK_OK(nmz_net_add(net, nmz_layer_activation(NMZ_ACT_RELU, 0)));
    CHECK_OK(nmz_net_build(net));
    CHECK(nmz_layer_init_used(nmz_net_layer(net, 0)) == NMZ_INIT_HE);
    nmz_net_free(net);
}

/* a relu input placed right at the kink: the check must skip, not fail */
static void test_gradcheck_kink(void) {
    nmz_net *net = NULL;
    nmz_tensor x = NMZ_TENSOR_EMPTY, t = NMZ_TENSOR_EMPTY;
    nmz_gradcheck_result r;
    int in = 2, xs[2] = {1, 2}, ts[2] = {1, 1};
    nmz_layer *d;
    CHECK_OK(nmz_net_create(&net, 1, &in, 1));
    CHECK_OK(nmz_net_add_dense(net, 2, NMZ_ACT_RELU, NMZ_INIT_HE));
    CHECK_OK(nmz_net_add_dense(net, 1, NMZ_ACT_LINEAR, NMZ_INIT_XAVIER));
    CHECK_OK(nmz_net_build(net));
    d = nmz_net_layer(net, 0);
    nmz_layer_param_value(d, 0)->data[0] = 1.0f; /* z_1 = 5·1 + 0·w + b */
    nmz_layer_param_value(d, 0)->data[2] = 0.3f;
    nmz_layer_param_value(d, 1)->data[0] = -4.9999f; /* → z_1 = 0.0001, inside ±eps·x */
    nmz_layer_param_value(nmz_net_layer(net, 2), 0)->data[0] = 1.0f;
    CHECK_OK(nmz_tensor_init(&x, 2, xs));
    CHECK_OK(nmz_tensor_init(&t, 2, ts));
    x.data[0] = 5.0f;
    x.data[1] = 0.0f;
    t.data[0] = 3.0f;
    CHECK_OK(nmz_net_gradcheck(net, &x, &t, 1e-3, 0, 0, &r));
    CHECK_MSG(r.kinks > 0 && r.failures == 0, "kinks %zu failures %zu", r.kinks, r.failures);
    nmz_tensor_free(&x);
    nmz_tensor_free(&t);
    nmz_net_free(net);
}

const test_case layer_tests[] = {
    {"layers/gradcheck_skips_kinks", test_gradcheck_kink},
    {"layers/gradcheck_conv2d", test_gradcheck_conv},
    {"layers/gradcheck_pool_flatten_cnn", test_gradcheck_pool_flatten},
    {"layers/gradcheck_dropout", test_gradcheck_dropout},
    {"layers/conv_by_hand", test_conv_by_hand},
    {"layers/maxpool_by_hand", test_pool_by_hand},
    {"layers/dropout_modes_seeded", test_dropout_modes},
    {"layers/shape_errors", test_shape_errors},
    {NULL, NULL}};
