/* test_core.c — tensors, PRNG, error handling, CSV loader. */
#include "test.h"

#include <stdlib.h>
#include <string.h>

static void write_file(const char *path, const char *text) {
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fputs(text, f);
    fclose(f);
}

static void test_tensor(void) {
    nmz_tensor t = NMZ_TENSOR_EMPTY, c = NMZ_TENSOR_EMPTY;
    int shape[2] = {3, 4}, big[2] = {5, 5}, bad[2] = {0, 3};
    size_t i;
    CHECK_OK(nmz_tensor_init(&t, 2, shape));
    CHECK(t.size == 12 && t.rank == 2 && t.shape[1] == 4);
    for (i = 0; i < t.size; i++) CHECK(t.data[i] == 0.0f);
    CHECK(nmz_tensor_rows(&t) == 3 && nmz_tensor_cols(&t) == 4);
    t.data[5] = 2.5f;
    CHECK_OK(nmz_tensor_copy(&c, &t));
    CHECK(c.size == 12 && c.data[5] == 2.5f);
    CHECK_OK(nmz_tensor_reshape(&t, 2, big));
    CHECK(t.size == 25 && t.capacity >= 25);
    CHECK_STATUS(nmz_tensor_reshape(&t, 2, bad), NMZ_ERR_SHAPE);
    CHECK_STATUS(nmz_tensor_reshape(&t, 7, big), NMZ_ERR_SHAPE);
    nmz_tensor_free(&t);
    nmz_tensor_free(&c);
    CHECK(t.data == NULL && t.size == 0);
    nmz_tensor_free(&t); /* double free of an empty tensor is harmless */
}

static void test_rng(void) {
    nmz_rng a, b;
    int i, counts[10] = {0};
    double sum = 0, sq = 0;
    nmz_rng_seed(&a, 42, 1);
    nmz_rng_seed(&b, 42, 1);
    for (i = 0; i < 1000; i++) CHECK(nmz_rng_u32(&a) == nmz_rng_u32(&b));
    /* Reference values of PCG32 (pcg32_srandom(42, 54)) from the PCG paper's demo */
    nmz_rng_seed(&a, 42u, 54u);
    CHECK(nmz_rng_u32(&a) == 0xa15c02b7u);
    CHECK(nmz_rng_u32(&a) == 0x7b47f409u);
    CHECK(nmz_rng_u32(&a) == 0xba1d3330u);
    for (i = 0; i < 100000; i++) {
        double u = nmz_rng_uniform(&a), n = nmz_rng_normal(&a);
        CHECK(u >= 0.0 && u < 1.0);
        counts[nmz_rng_below(&a, 10)]++;
        sum += n;
        sq += n * n;
    }
    for (i = 0; i < 10; i++) CHECK_MSG(counts[i] > 9500 && counts[i] < 10500, "bucket %d = %d", i, counts[i]);
    CHECK_NEAR(sum / 100000, 0.0, 0.02);
    CHECK_NEAR(sq / 100000, 1.0, 0.03);
    CHECK(nmz_rng_below(&a, 1) == 0 && nmz_rng_below(&a, 0) == 0);
}

static void test_names(void) {
    nmz_act a;
    nmz_init in;
    nmz_loss l;
    int i;
    for (i = 0; i < NMZ_ACT_COUNT; i++) {
        CHECK_OK(nmz_act_parse(nmz_act_name((nmz_act)i), &a));
        CHECK(a == (nmz_act)i);
    }
    CHECK_OK(nmz_act_parse("lrelu", &a));
    CHECK(a == NMZ_ACT_LEAKY_RELU);
    CHECK_STATUS(nmz_act_parse("swish", &a), NMZ_ERR_ARG);
    CHECK(strstr(nmz_last_error(), "swish") != NULL);
    CHECK_OK(nmz_init_parse("he", &in));
    CHECK(in == NMZ_INIT_HE);
    CHECK_OK(nmz_loss_parse("ce", &l));
    CHECK(l == NMZ_LOSS_CROSS_ENTROPY);
    CHECK(strcmp(nmz_status_str(NMZ_ERR_SHAPE), "NMZ_ERR_SHAPE") == 0);
}

static void test_errors(void) {
    nmz_net *net = NULL;
    int in = 2, bad = 0;
    nmz_tensor x = NMZ_TENSOR_EMPTY, t = NMZ_TENSOR_EMPTY;
    const nmz_tensor *y;
    int xs[2] = {1, 3}, ts[2] = {1, 2};
    float out[4];
    /* NULL handles never crash */
    CHECK_STATUS(nmz_net_add(NULL, nmz_layer_dense(3, NMZ_INIT_AUTO)), NMZ_ERR_ARG);
    CHECK_STATUS(nmz_net_add(NULL, NULL), NMZ_ERR_ARG);
    CHECK_STATUS(nmz_net_build(NULL), NMZ_ERR_ARG);
    CHECK_STATUS(nmz_net_forward(NULL, &x, &y), NMZ_ERR_ARG);
    CHECK_STATUS(nmz_net_backward(NULL, &t, NULL), NMZ_ERR_ARG);
    CHECK_STATUS(nmz_net_train(NULL, NULL, NULL, NULL), NMZ_ERR_ARG);
    CHECK_STATUS(nmz_net_save(NULL, "x"), NMZ_ERR_ARG);
    CHECK(nmz_net_layer_count(NULL) == 0 && nmz_net_output_size(NULL) == 0);
    nmz_net_free(NULL);
    CHECK(nmz_layer_dense(0, NMZ_INIT_AUTO) == NULL);
    CHECK(nmz_layer_activation((nmz_act)99, 0) == NULL);
    CHECK_STATUS(nmz_net_create(&net, 1, &bad, 1), NMZ_ERR_SHAPE);
    CHECK(net == NULL);

    CHECK_OK(nmz_net_create(&net, 1, &in, 1));
    CHECK_STATUS(nmz_net_build(net), NMZ_ERR_STATE); /* no layers */
    CHECK_OK(nmz_net_add_dense(net, 2, NMZ_ACT_SIGMOID, NMZ_INIT_AUTO));
    /* cross-entropy needs a softmax output */
    CHECK_OK(nmz_net_set_loss(net, NMZ_LOSS_CROSS_ENTROPY));
    CHECK_STATUS(nmz_net_build(net), NMZ_ERR_ARG);
    CHECK(strstr(nmz_last_error(), "softmax") != NULL);
    CHECK_OK(nmz_net_set_loss(net, NMZ_LOSS_MSE));
    CHECK_OK(nmz_net_build(net));
    CHECK_STATUS(nmz_net_add_dense(net, 2, NMZ_ACT_TANH, NMZ_INIT_AUTO), NMZ_ERR_STATE);
    /* backward before forward */
    CHECK_OK(nmz_tensor_init(&t, 2, ts));
    CHECK_STATUS(nmz_net_backward(net, &t, NULL), NMZ_ERR_STATE);
    /* wrong input width */
    CHECK_OK(nmz_tensor_init(&x, 2, xs));
    CHECK_STATUS(nmz_net_forward(net, &x, &y), NMZ_ERR_SHAPE);
    /* wrong target width */
    xs[1] = 2;
    CHECK_OK(nmz_tensor_reshape(&x, 2, xs));
    CHECK_OK(nmz_net_forward(net, &x, &y));
    ts[1] = 3;
    CHECK_OK(nmz_tensor_reshape(&t, 2, ts));
    CHECK_STATUS(nmz_net_backward(net, &t, NULL), NMZ_ERR_SHAPE);
    CHECK_STATUS(nmz_net_update(net, -1.0f, 0.0f), NMZ_ERR_ARG);
    CHECK_STATUS(nmz_net_update(net, 0.1f, 1.0f), NMZ_ERR_ARG);
    CHECK_OK(nmz_net_predict(net, x.data, out));
    nmz_net_free(net);
    CHECK_STATUS(nmz_net_load(&net, "/nonexistent/dir/model.nmz"), NMZ_ERR_IO);
    CHECK(net == NULL); /* load always resets *out */
    nmz_tensor_free(&x);
    nmz_tensor_free(&t);
}

static void test_errors_no_leak(void) {
    /* add() takes ownership even when it fails */
    nmz_net *net;
    int in = 3;
    CHECK_OK(nmz_net_create(&net, 1, &in, 7));
    CHECK_OK(nmz_net_add_dense(net, 4, NMZ_ACT_RELU, NMZ_INIT_AUTO));
    CHECK_OK(nmz_net_build(net));
    CHECK_STATUS(nmz_net_add(net, nmz_layer_dense(2, NMZ_INIT_HE)), NMZ_ERR_STATE);
    nmz_net_free(net);
}

static void test_csv(void) {
    nmz_dataset *ds = NULL;
    nmz_csv_opts o = nmz_csv_defaults();
    const char *p = test_path("t_iris_like.csv");
    write_file(p, "# comment line\n"
                  "a, b ,label\n"
                  "1.5,2,cat\r\n"
                  "\n"
                  "-3,4e1,\"dog\"\n"
                  "0,0,cat\n");
    CHECK_OK(nmz_dataset_load_csv(&ds, p, &o));
    if (ds) {
        CHECK(ds->n_samples == 3 && ds->n_inputs == 2 && ds->n_outputs == 2 && ds->n_classes == 2);
        CHECK(strcmp(ds->class_names[0], "cat") == 0 && strcmp(ds->class_names[1], "dog") == 0);
        CHECK(ds->x[0] == 1.5f && ds->x[3] == 40.0f && ds->x[2] == -3.0f);
        CHECK(ds->y[0] == 1.0f && ds->y[1] == 0.0f && ds->y[2] == 0.0f && ds->y[3] == 1.0f);
        nmz_dataset_free(ds);
        ds = NULL;
    }
    /* numeric, no header, 2 targets */
    p = test_path("t_num.csv");
    write_file(p, "0,0,0,1\n0,1,1,0\n1,0,1,0\n1,1,0,1\n");
    o.n_targets = 2;
    CHECK_OK(nmz_dataset_load_csv(&ds, p, &o));
    if (ds) {
        CHECK(ds->n_samples == 4 && ds->n_inputs == 2 && ds->n_outputs == 2 && ds->n_classes == 0);
        CHECK(ds->y[6] == 0.0f && ds->y[7] == 1.0f);
        nmz_dataset_free(ds);
        ds = NULL;
    }
    /* errors carry file:line */
    p = test_path("t_bad.csv");
    write_file(p, "1,2,3\n4,5\n");
    o = nmz_csv_defaults();
    CHECK_STATUS(nmz_dataset_load_csv(&ds, p, &o), NMZ_ERR_FORMAT);
    CHECK_MSG(strstr(nmz_last_error(), ":2:") != NULL, "%s", nmz_last_error());
    write_file(p, "1,x,3\n4,5,6\n");
    o.header = 0;
    CHECK_STATUS(nmz_dataset_load_csv(&ds, p, &o), NMZ_ERR_FORMAT);
    CHECK(strstr(nmz_last_error(), "not a number") != NULL);
    write_file(p, "");
    CHECK_STATUS(nmz_dataset_load_csv(&ds, p, &o), NMZ_ERR_FORMAT);
    CHECK_STATUS(nmz_dataset_load_csv(&ds, test_path("does_not_exist.csv"), &o), NMZ_ERR_IO);
    o.n_targets = 5;
    write_file(p, "1,2,3\n");
    CHECK_STATUS(nmz_dataset_load_csv(&ds, p, &o), NMZ_ERR_FORMAT);
    CHECK(ds == NULL);
}

static void put_be32(FILE *f, unsigned v) {
    fputc((int)(v >> 24) & 255, f);
    fputc((int)(v >> 16) & 255, f);
    fputc((int)(v >> 8) & 255, f);
    fputc((int)v & 255, f);
}

static void test_idx(void) {
    const char *pi = test_path("t-images-idx3-ubyte"), *pl = test_path("t-labels-idx1-ubyte");
    FILE *f = fopen(pi, "wb");
    nmz_dataset *ds = NULL;
    int i;
    if (!f) return;
    put_be32(f, 0x803);
    put_be32(f, 3); /* 3 images of 2x3 */
    put_be32(f, 2);
    put_be32(f, 3);
    for (i = 0; i < 18; i++) fputc(i * 10, f);
    fclose(f);
    f = fopen(pl, "wb");
    if (!f) return;
    put_be32(f, 0x801);
    put_be32(f, 3);
    fputc(2, f);
    fputc(0, f);
    fputc(4, f);
    fclose(f);
    CHECK_OK(nmz_dataset_load_idx(&ds, pi, pl, 0));
    if (ds) {
        CHECK(ds->n_samples == 3 && ds->n_inputs == 6 && ds->n_outputs == 5 && ds->n_classes == 5);
        CHECK(ds->sample_rank == 3 && ds->sample_shape[0] == 1 && ds->sample_shape[1] == 2 && ds->sample_shape[2] == 3);
        CHECK(ds->x[0] == 0.0f && ds->x[7] == 70.0f && ds->x[17] == 170.0f); /* raw 0..255 */
        CHECK(ds->y[2] == 1.0f && ds->y[5] == 1.0f && ds->y[14] == 1.0f && ds->y[0] == 0.0f);
        CHECK(!strcmp(ds->class_names[4], "4"));
        CHECK_OK(nmz_dataset_truncate(ds, 2));
        CHECK(ds->n_samples == 2);
        nmz_dataset_free(ds);
        ds = NULL;
    }
    CHECK_OK(nmz_dataset_load_idx(&ds, pi, pl, 2));
    CHECK(ds && ds->n_samples == 2);
    nmz_dataset_free(ds);
    ds = NULL;
    /* wrong kinds, truncation, count mismatch */
    CHECK_STATUS(nmz_dataset_load_idx(&ds, pl, pi, 0), NMZ_ERR_FORMAT);
    CHECK(strstr(nmz_last_error(), "magic") != NULL);
    f = fopen(pl, "wb");
    if (f) {
        put_be32(f, 0x801);
        put_be32(f, 4);
        fclose(f);
    }
    CHECK_STATUS(nmz_dataset_load_idx(&ds, pi, pl, 0), NMZ_ERR_FORMAT);
    f = fopen(pi, "wb");
    if (f) {
        put_be32(f, 0x803);
        put_be32(f, 4);
        put_be32(f, 2);
        put_be32(f, 2);
        fputc(1, f);
        fclose(f);
    }
    f = fopen(pl, "wb");
    if (f) {
        put_be32(f, 0x801);
        put_be32(f, 4);
        fputs("abcd", f);
        fclose(f);
    }
    CHECK_STATUS(nmz_dataset_load_idx(&ds, pi, pl, 0), NMZ_ERR_FORMAT);
    CHECK(strstr(nmz_last_error(), "truncated") != NULL);
    CHECK_STATUS(nmz_dataset_load_idx(&ds, test_path("nope"), pl, 0), NMZ_ERR_IO);
    CHECK(ds == NULL);
}

const test_case core_tests[] = {
    {"core/idx_loader", test_idx},
    {"core/tensor", test_tensor},
    {"core/rng_pcg32", test_rng},
    {"core/names", test_names},
    {"core/errors_no_crash", test_errors},
    {"core/add_takes_ownership", test_errors_no_leak},
    {"core/csv_loader", test_csv},
    {NULL, NULL}};
