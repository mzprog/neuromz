/* test_io.c — portable model files: round-trip, byte layout, corruption. */
#include "test.h"

#include <stdlib.h>
#include <string.h>

static long file_size(const char *p) {
    FILE *f = fopen(p, "rb");
    long n;
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    n = ftell(f);
    fclose(f);
    return n;
}

static unsigned char *slurp(const char *p, long *n) {
    FILE *f = fopen(p, "rb");
    unsigned char *b;
    *n = file_size(p);
    if (!f || *n <= 0) {
        if (f) fclose(f);
        return NULL;
    }
    b = malloc((size_t)*n);
    if (b && fread(b, 1, (size_t)*n, f) != (size_t)*n) {
        free(b);
        b = NULL;
    }
    fclose(f);
    return b;
}

static void spit(const char *p, const unsigned char *b, long n) {
    FILE *f = fopen(p, "wb");
    if (!f) return;
    fwrite(b, 1, (size_t)n, f);
    fclose(f);
}

static nmz_net *make_net(void) {
    nmz_net *net = NULL;
    int in = 4;
    const char *names[3] = {"setosa", "versicolor", "virginica"};
    nmz_train_opts *o;
    if (nmz_net_create(&net, 1, &in, 1234) != NMZ_OK) return NULL;
    nmz_net_add_dense(net, 6, NMZ_ACT_LEAKY_RELU, NMZ_INIT_AUTO);
    nmz_net_add_dense(net, 5, NMZ_ACT_TANH, NMZ_INIT_XAVIER);
    nmz_net_add_dense(net, 3, NMZ_ACT_SOFTMAX, NMZ_INIT_AUTO);
    nmz_net_set_loss(net, NMZ_LOSS_CROSS_ENTROPY);
    nmz_net_set_class_names(net, 3, names);
    o = nmz_net_train_config(net);
    o->learning_rate = 0.025f;
    o->batch_size = 16;
    o->target_loss = 0.07;
    if (nmz_net_build(net) != NMZ_OK) {
        nmz_net_free(net);
        return NULL;
    }
    return net;
}

static void test_roundtrip(void) {
    nmz_net *a = make_net(), *b = NULL;
    const char *p = test_path("t_model.nmz");
    float x[4] = {5.1f, 3.5f, 1.4f, 0.2f}, ya[3], yb[3];
    int i, j, k;
    char sa[1024], sb[1024];
    CHECK(a != NULL);
    if (!a) return;
    CHECK_OK(nmz_net_save(a, p));
    CHECK_OK(nmz_net_load(&b, p));
    CHECK(b != NULL);
    if (b) {
        CHECK(nmz_net_layer_count(a) == nmz_net_layer_count(b));
        for (i = 0; i < nmz_net_layer_count(a); i++)
            for (j = 0; j < nmz_layer_param_count(nmz_net_layer(a, i)); j++) {
                nmz_tensor *ta = nmz_layer_param_value(nmz_net_layer(a, i), j);
                nmz_tensor *tb = nmz_layer_param_value(nmz_net_layer(b, i), j);
                CHECK(ta->size == tb->size && memcmp(ta->data, tb->data, ta->size * sizeof(float)) == 0);
            }
        CHECK_OK(nmz_net_predict(a, x, ya));
        CHECK_OK(nmz_net_predict(b, x, yb));
        for (k = 0; k < 3; k++) CHECK(ya[k] == yb[k]); /* bit-identical */
        CHECK(nmz_net_loss_kind(b) == NMZ_LOSS_CROSS_ENTROPY);
        CHECK(nmz_net_class_count(b) == 3 && strcmp(nmz_net_class_name(b, 2), "virginica") == 0);
        CHECK(nmz_net_train_config(b)->learning_rate == 0.025f && nmz_net_train_config(b)->batch_size == 16);
        CHECK(nmz_net_train_config(b)->target_loss == 0.07);
        nmz_net_summary(a, sa, sizeof sa);
        nmz_net_summary(b, sb, sizeof sb);
        CHECK_MSG(strcmp(sa, sb) == 0, "\n%s\nvs\n%s", sa, sb);
        CHECK(strstr(sa, "leaky_relu") && strstr(sa, "he init"));
        /* saving the loaded copy gives the same bytes */
        CHECK_OK(nmz_net_save(b, test_path("t_model2.nmz")));
        {
            long n1, n2;
            unsigned char *b1 = slurp(p, &n1), *b2 = slurp(test_path("t_model2.nmz"), &n2);
            CHECK(b1 && b2 && n1 == n2 && memcmp(b1, b2, (size_t)n1) == 0);
            free(b1);
            free(b2);
        }
        nmz_net_free(b);
    }
    nmz_net_free(a);
}

static void test_byte_layout(void) {
    nmz_net *a = make_net();
    const char *p = test_path("t_layout.nmz");
    long n;
    unsigned char *b;
    static const unsigned char magic[8] = {0x89, 'N', 'M', 'Z', '\r', '\n', 0x1a, '\n'};
    if (!a) return;
    CHECK_OK(nmz_net_save(a, p));
    b = slurp(p, &n);
    CHECK(b != NULL && n > 40);
    if (b) {
        CHECK(memcmp(b, magic, 8) == 0);
        CHECK(b[8] == 3 && b[9] == 0 && b[10] == 0 && b[11] == 0); /* version 3, little-endian */
        CHECK(b[16] == 1 && b[20] == 4);                            /* input rank 1, size 4    */
        free(b);
    }
    nmz_net_free(a);
}

static void test_corruption(void) {
    nmz_net *a = make_net(), *b = NULL;
    const char *p = test_path("t_corrupt.nmz"), *q = test_path("t_corrupt_bad.nmz");
    long n, i;
    unsigned char *buf;
    if (!a) return;
    CHECK_OK(nmz_net_save(a, p));
    buf = slurp(p, &n);
    if (!buf) {
        nmz_net_free(a);
        return;
    }
    /* flip one bit somewhere in the weights → checksum mismatch */
    buf[n - 40] ^= 0x10;
    spit(q, buf, n);
    CHECK_STATUS(nmz_net_load(&b, q), NMZ_ERR_FORMAT);
    CHECK_MSG(strstr(nmz_last_error(), "checksum") != NULL, "%s", nmz_last_error());
    buf[n - 40] ^= 0x10;
    /* every truncation is rejected cleanly (no crash, no leak) */
    for (i = 0; i < n; i += (i < 64 ? 1 : 7)) {
        spit(q, buf, i);
        CHECK_STATUS(nmz_net_load(&b, q), NMZ_ERR_FORMAT);
        CHECK(b == NULL);
    }
    /* trailing garbage */
    {
        unsigned char *longer = malloc((size_t)n + 1);
        if (longer) {
            memcpy(longer, buf, (size_t)n);
            longer[n] = 0;
            spit(q, longer, n + 1);
            CHECK_STATUS(nmz_net_load(&b, q), NMZ_ERR_FORMAT);
            free(longer);
        }
    }
    /* wrong magic / v1 file */
    spit(q, (const unsigned char *)"1.1\0\0\0\0\0\0\0\0\0\0\0\0\0", 16);
    CHECK_STATUS(nmz_net_load(&b, q), NMZ_ERR_FORMAT);
    CHECK(strstr(nmz_last_error(), "v1") != NULL);
    spit(q, (const unsigned char *)"hello world, not a model", 24);
    CHECK_STATUS(nmz_net_load(&b, q), NMZ_ERR_FORMAT);
    /* future version */
    buf[8] = 99;
    spit(q, buf, n);
    CHECK_STATUS(nmz_net_load(&b, q), NMZ_ERR_FORMAT);
    CHECK(strstr(nmz_last_error(), "version") != NULL);
    free(buf);
    /* a failed save into a missing directory reports IO and leaves nothing behind */
    CHECK_STATUS(nmz_net_save(a, "/nonexistent-dir/x.nmz"), NMZ_ERR_IO);
    nmz_net_free(a);
}

static void test_train_after_load(void) {
    /* loading keeps the PRNG state: train(load(save(net))) == train(net) */
    static const float X[8] = {0, 0, 0, 1, 1, 0, 1, 1}, Y[4] = {0, 1, 1, 0};
    nmz_dataset *ds = NULL;
    nmz_net *a = NULL, *b = NULL;
    nmz_train_result ra, rb;
    int sizes[3] = {2, 3, 1};
    nmz_train_opts o;
    CHECK_OK(nmz_dataset_create(&ds, 4, 2, 1, X, Y));
    CHECK_OK(nmz_net_create_quick(&a, 3, sizes, 5, NULL));
    CHECK_OK(nmz_net_save(a, test_path("t_resume.nmz")));
    CHECK_OK(nmz_net_load(&b, test_path("t_resume.nmz")));
    o = *nmz_net_train_config(a);
    o.max_epochs = 30;
    o.batch_size = 1;
    CHECK_OK(nmz_net_train(a, ds, &o, &ra));
    CHECK_OK(nmz_net_train(b, ds, &o, &rb));
    CHECK(ra.loss == rb.loss);
    nmz_net_free(a);
    nmz_net_free(b);
    nmz_dataset_free(ds);
}

const test_case io_tests[] = {
    {"io/save_load_roundtrip", test_roundtrip},
    {"io/little_endian_layout", test_byte_layout},
    {"io/corrupt_truncated_foreign", test_corruption},
    {"io/resume_is_deterministic", test_train_after_load},
    {NULL, NULL}};
