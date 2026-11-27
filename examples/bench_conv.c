/* bench_conv.c — naive loops vs im2col for the mnist-cnn architecture.
 * Build/run: cmake --build build --target bench   (needs `--target data`).
 * Prints seconds per epoch. Usage: bench_conv [N_SAMPLES] */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "neuromz.h"

static double now(void) {
    struct timespec t;
    timespec_get(&t, TIME_UTC); /* C11: every compiler, Windows included */
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static nmz_net *cnn(nmz_conv_algo algo) {
    nmz_net *net = NULL;
    int shape[3] = {1, 28, 28}, i;
    nmz_net_create(&net, 3, shape, 42);
    nmz_net_add(net, nmz_layer_conv2d(8, 3, 3, 1, 0, NMZ_INIT_AUTO));
    nmz_net_add(net, nmz_layer_activation(NMZ_ACT_RELU, 0));
    nmz_net_add(net, nmz_layer_maxpool2d(2, 0));
    nmz_net_add(net, nmz_layer_conv2d(16, 3, 3, 1, 0, NMZ_INIT_AUTO));
    nmz_net_add(net, nmz_layer_activation(NMZ_ACT_RELU, 0));
    nmz_net_add(net, nmz_layer_maxpool2d(2, 0));
    nmz_net_add(net, nmz_layer_flatten());
    nmz_net_add_dense(net, 64, NMZ_ACT_RELU, NMZ_INIT_AUTO);
    nmz_net_add_dense(net, 10, NMZ_ACT_SOFTMAX, NMZ_INIT_AUTO);
    nmz_net_set_loss(net, NMZ_LOSS_CROSS_ENTROPY);
    nmz_net_set_normalization(net, NMZ_NORM_ZSCORE);
    for (i = 0; i < nmz_net_layer_count(net); i++)
        if (nmz_layer_init_used(nmz_net_layer(net, i)) >= 0 && nmz_net_layer(net, i))
            nmz_layer_conv2d_set_algo(nmz_net_layer(net, i), algo); /* fails harmlessly on dense */
    return net;
}

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 10000, a;
    nmz_dataset *ds = NULL;
    double times[2], losses[2];
    if (nmz_dataset_load_idx(&ds, "data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte", n) != NMZ_OK) {
        fprintf(stderr, "%s (run make data)\n", nmz_last_error());
        return 1;
    }
    for (a = 0; a < 2; a++) {
        nmz_net *net = cnn((nmz_conv_algo)a);
        nmz_train_opts o = nmz_train_defaults();
        nmz_train_result r;
        double t0;
        o.optimizer = NMZ_OPT_ADAM;
        o.learning_rate = 0.001f;
        o.max_epochs = 1;
        o.target_loss = 0;
        t0 = now();
        if (nmz_net_train(net, ds, &o, &r) != NMZ_OK) {
            fprintf(stderr, "%s\n", nmz_last_error());
            return 1;
        }
        times[a] = now() - t0;
        losses[a] = r.loss;
        printf("%-7s %6.2f s for 1 epoch on %d images (incl. whole-set loss), loss %.6f\n", a ? "naive" : "im2col",
               times[a], ds->n_samples, r.loss);
        nmz_net_free(net);
    }
    printf("speedup im2col vs naive: %.2fx (same loss to %.1e)\n", times[1] / times[0], losses[0] - losses[1]);
    nmz_dataset_free(ds);
    return 0;
}
