/* api_mnist.c — the MNIST regression from C: a small CNN (the CLI's
 * mnist-cnn preset) trained for one epoch on the first 20 000 training
 * images, then tested on the 10 000 test images. Fails (exit 1) below 95%
 * test accuracy; exits 77 (CTest: skipped) when the data is missing.
 * Usage: api_mnist [DATA_DIR]   (default: data; get it with --target data) */
#include <stdio.h>
#include <string.h>

#include "neuromz.h"

#define TRY(expr)                                                                 \
    do {                                                                          \
        if ((expr) != NMZ_OK) {                                                   \
            fprintf(stderr, "%s failed: %s\n", #expr, nmz_last_error());          \
            return 1;                                                             \
        }                                                                         \
    } while (0)

static int exists(const char *p) {
    FILE *f = fopen(p, "rb");
    if (f) fclose(f);
    return f != NULL;
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "data";
    char ti[512], tl[512], vi[512], vl[512];
    nmz_dataset *train = NULL, *test = NULL;
    nmz_net *net = NULL;
    nmz_train_opts o;
    nmz_train_result res;
    double loss, acc;
    int shape[3] = {1, 28, 28};
    snprintf(ti, sizeof ti, "%s/train-images-idx3-ubyte", dir);
    snprintf(tl, sizeof tl, "%s/train-labels-idx1-ubyte", dir);
    snprintf(vi, sizeof vi, "%s/t10k-images-idx3-ubyte", dir);
    snprintf(vl, sizeof vl, "%s/t10k-labels-idx1-ubyte", dir);
    if (!exists(ti) || !exists(tl) || !exists(vi) || !exists(vl)) {
        printf("MNIST not found in %s/ (download it with: cmake --build build --target data): skipped\n", dir);
        return 77;
    }
    TRY(nmz_dataset_load_idx(&train, ti, tl, 20000));
    TRY(nmz_dataset_load_idx(&test, vi, vl, 0));
    TRY(nmz_net_create(&net, 3, shape, 1));
    TRY(nmz_net_add(net, nmz_layer_conv2d(8, 3, 3, 1, 0, NMZ_INIT_AUTO)));
    TRY(nmz_net_add(net, nmz_layer_activation(NMZ_ACT_RELU, 0)));
    TRY(nmz_net_add(net, nmz_layer_maxpool2d(2, 2)));
    TRY(nmz_net_add(net, nmz_layer_conv2d(16, 3, 3, 1, 0, NMZ_INIT_AUTO)));
    TRY(nmz_net_add(net, nmz_layer_activation(NMZ_ACT_RELU, 0)));
    TRY(nmz_net_add(net, nmz_layer_maxpool2d(2, 2)));
    TRY(nmz_net_add(net, nmz_layer_flatten()));
    TRY(nmz_net_add_dense(net, 64, NMZ_ACT_RELU, NMZ_INIT_AUTO));
    TRY(nmz_net_add_dense(net, 10, NMZ_ACT_SOFTMAX, NMZ_INIT_AUTO));
    TRY(nmz_net_set_loss(net, NMZ_LOSS_CROSS_ENTROPY));
    TRY(nmz_net_set_normalization(net, NMZ_NORM_ZSCORE));
    o = nmz_train_defaults();
    o.optimizer = NMZ_OPT_ADAM;
    o.learning_rate = 0.001f;
    o.batch_size = 32;
    o.max_epochs = 1;
    o.target_loss = 0;
    TRY(nmz_net_train(net, train, &o, &res));
    TRY(nmz_net_evaluate(net, test, &loss, &acc));
    printf("MNIST CNN, 1 epoch on 20000 images: test loss %.4f, test accuracy %.2f%%\n", loss, 100.0 * acc);
    nmz_net_free(net);
    nmz_dataset_free(train);
    nmz_dataset_free(test);
    return acc >= 0.95 ? 0 : 1;
}
