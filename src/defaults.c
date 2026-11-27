/* defaults.c — what "quick mode" chooses, and why. The CLI prints the
 * reasons; library users get the same choices from nmz_net_create_quick. */
#include "internal.h"

#include <string.h>

nmz_status nmz_quick_defaults(int n_sizes, const int *sizes, nmz_quick *q) {
    int i, n_out;
    if (!sizes || !q || n_sizes < 2)
        return nmz__fail(NMZ_ERR_ARG, "need at least an input size and an output size");
    for (i = 0; i < n_sizes; i++)
        if (sizes[i] <= 0 || sizes[i] > NMZ_MAX_DIM)
            return nmz__fail(NMZ_ERR_ARG, "layer sizes must be positive integers (got %d)", sizes[i]);
    memset(q, 0, sizeof *q);
    n_out = sizes[n_sizes - 1];

    q->hidden_act = NMZ_ACT_TANH;
    q->why_hidden_act = n_sizes > 2
        ? "zero-centred and smooth; in small nets it trains more reliably than relu, whose units can die"
        : "(no hidden layers)";
    q->init = NMZ_INIT_XAVIER;
    q->why_init = "Glorot/Xavier keeps the signal variance stable through tanh/sigmoid layers (He is for relu)";

    if (n_out == 1) {
        q->output_act = NMZ_ACT_SIGMOID;
        q->why_output_act = "1 output -> squashed into (0,1), a natural fit for 0/1 targets";
        q->loss = NMZ_LOSS_MSE;
        q->why_loss = "squared error is the textbook pairing for a sigmoid output: delta = (y-t)*f'(z)";
    } else {
        q->output_act = NMZ_ACT_SOFTMAX;
        q->why_output_act = "several outputs -> class probabilities that sum to 1 (assumes one class per "
                            "sample; for multi-label use N:sigmoid --loss mse)";
        q->loss = NMZ_LOSS_CROSS_ENTROPY;
        q->why_loss = "cross-entropy fused with softmax has the simple, stable gradient p - t";
    }

    return nmz_quick_train_defaults(q->loss, q);
}

nmz_status nmz_quick_train_defaults(nmz_loss loss, nmz_quick *q) {
    if (!q || (int)loss < 0 || loss >= NMZ_LOSS_COUNT) return nmz__fail(NMZ_ERR_ARG, "invalid loss");
    q->train = nmz_train_defaults();
    q->train.momentum = 0.9f;
    q->why_momentum = "0.9 averages recent gradients: smoother steps and faster progress across plateaus";
    q->train.batch_size = 32;
    q->why_batch = "mini-batches of 32 (full batch when the dataset is smaller): stable gradients, several updates per epoch";
    if (loss == NMZ_LOSS_MSE) {
        q->train.learning_rate = 0.5f;
        q->why_learning_rate = "0.5 suits squared error on sigmoid outputs, whose gradients are small (<= 0.25*error)";
        q->train.target_loss = 0.001;
        q->why_target_loss = "stop when the average 1/2*error^2 <= 0.001 (outputs within ~0.045 of their targets)";
    } else {
        q->train.learning_rate = 0.02f;
        q->why_learning_rate = "0.02 (x10 effective with momentum 0.9): stable for softmax+cross-entropy; 0.05 sometimes stalls";
        q->train.target_loss = 0.07;
        q->why_target_loss = "stop when the average cross-entropy <= 0.07 (~93% probability on the right class); lower mostly memorises the data";
    }
    q->train.max_epochs = 5000;
    q->why_epochs = "at most 5000 passes over the data, so training always ends";
    return NMZ_OK;
}

nmz_status nmz_net_create_quick(nmz_net **out, int n_sizes, const int *sizes, uint64_t seed,
                                nmz_quick *quick) {
    nmz_quick q;
    nmz_net *net = NULL;
    nmz_status s;
    int i;
    if (!out) return nmz__fail(NMZ_ERR_ARG, "out is NULL");
    *out = NULL;
    if ((s = nmz_quick_defaults(n_sizes, sizes, &q)) != NMZ_OK) return s;
    if ((s = nmz_net_create(&net, 1, &sizes[0], seed)) != NMZ_OK) return s;
    for (i = 1; i < n_sizes - 1 && s == NMZ_OK; i++) s = nmz_net_add_dense(net, sizes[i], q.hidden_act, q.init);
    if (s == NMZ_OK) s = nmz_net_add_dense(net, sizes[n_sizes - 1], q.output_act, NMZ_INIT_XAVIER);
    if (s == NMZ_OK) s = nmz_net_set_loss(net, q.loss);
    if (s == NMZ_OK) s = nmz_net_build(net);
    if (s != NMZ_OK) {
        nmz_net_free(net);
        return s;
    }
    *nmz_net_train_config(net) = q.train;
    if (quick) *quick = q;
    *out = net;
    return NMZ_OK;
}
