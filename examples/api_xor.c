/* api_xor.c — using libneuromz from C: the beginner path, the advanced path,
 * and the trace callback. Build: cmake --build build (see README.md); run
 * from the repository root: build/api_xor. Exit status 1 if XOR is not
 * learned (CTest uses it as a regression test). */
#include <stdio.h>
#include <stdlib.h>

#include "neuromz.h"

#define TRY(expr)                                                                 \
    do {                                                                          \
        nmz_status s_ = (expr);                                                   \
        if (s_ != NMZ_OK) {                                                       \
            fprintf(stderr, "%s failed: %s\n", #expr, nmz_last_error());          \
            return 1;                                                             \
        }                                                                         \
    } while (0)

/* print only the loss and the output-layer delta of each traced pass */
static void on_trace(const nmz_trace_event *ev, void *user) {
    double L;
    (void)user;
    if (ev->phase == NMZ_TRACE_LOSS && nmz_trace_scalar_get(ev, "L", &L))
        printf("  pass %ld: loss %.5f\n", ev->step, L);
}

int main(void) {
    static const float X[8] = {0, 0, 0, 1, 1, 0, 1, 1};
    static const float Y[4] = {0, 1, 1, 0};
    nmz_dataset *ds = NULL;
    nmz_net *net = NULL;
    nmz_train_result res;
    nmz_train_opts opts;
    float out;
    int i, sizes[3] = {2, 4, 1}, n_in = 2;

    TRY(nmz_dataset_create(&ds, 4, 2, 1, X, Y));

    /* 1. Beginner: sizes in, sensible defaults chosen for you */
    TRY(nmz_net_create_quick(&net, 3, sizes, 42, NULL));
    TRY(nmz_net_train(net, ds, NULL, &res));
    printf("quick:    %s after %d epochs, loss %.6f\n", res.converged ? "converged" : "stopped", res.epochs, res.loss);
    if (!res.converged) return 1;
    for (i = 0; i < 4; i++) {
        TRY(nmz_net_predict(net, X + 2 * i, &out));
        printf("  %g xor %g -> %.3f\n", X[2 * i], X[2 * i + 1], out);
    }
    {
        char path[1024];
        const char *dir = getenv("NMZ_TEST_DIR");
        snprintf(path, sizeof path, "%s/api_xor.nmz", dir && *dir ? dir : ".");
        TRY(nmz_net_save(net, path));
    }
    nmz_net_free(net);

    /* 2. Advanced: every layer, init, loss and training knob explicit */
    TRY(nmz_net_create(&net, 1, &n_in, 7));
    TRY(nmz_net_add_dense(net, 8, NMZ_ACT_LEAKY_RELU, NMZ_INIT_HE));
    TRY(nmz_net_add(net, nmz_layer_dense(1, NMZ_INIT_XAVIER)));
    TRY(nmz_net_add(net, nmz_layer_activation(NMZ_ACT_SIGMOID, 0)));
    TRY(nmz_net_set_loss(net, NMZ_LOSS_MSE));
    opts = nmz_train_defaults();
    opts.max_epochs = 3000;
    opts.batch_size = 4;
    opts.learning_rate = 0.3f;
    opts.momentum = 0.8f;
    opts.target_loss = 5e-4;
    TRY(nmz_net_train(net, ds, &opts, &res));
    printf("advanced: %s after %d epochs, loss %.6f\n", res.converged ? "converged" : "stopped", res.epochs, res.loss);
    if (!res.converged) return 1;

    /* 3. Trace: watch three training steps on one sample */
    {
        nmz_tensor x = NMZ_TENSOR_EMPTY, t = NMZ_TENSOR_EMPTY;
        int xs[2] = {1, 2}, ts[2] = {1, 1};
        TRY(nmz_tensor_init(&x, 2, xs));
        TRY(nmz_tensor_init(&t, 2, ts));
        x.data[0] = 1;
        t.data[0] = 1;
        nmz_net_set_trace(net, on_trace, NULL);
        printf("trace (1 0 -> 1):\n");
        for (i = 0; i < 3; i++) {
            TRY(nmz_net_forward(net, &x, NULL));
            TRY(nmz_net_backward(net, &t, NULL));
            TRY(nmz_net_update(net, 0.3f, 0.0f));
        }
        nmz_net_set_trace(net, NULL, NULL);
        nmz_tensor_free(&x);
        nmz_tensor_free(&t);
    }
    nmz_net_free(net);
    nmz_dataset_free(ds);
    return 0;
}
