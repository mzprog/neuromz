/* net.c — the network: a sequential stack of layers plus the buffers that
 * connect them.
 *
 *   acts[0]   = input x                 dacts[0]   = ∂L/∂x
 *   acts[i+1] = layer_i(acts[i])        dacts[i+1] = ∂L/∂acts[i+1]
 *
 * backward() computes EVERY gradient first; update() changes weights in a
 * separate pass, so no gradient is ever computed from already-updated
 * weights (a v1 bug). */
#include "internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ----------------------------------------------------------- layer helpers */

nmz_layer *nmz_layer_alloc(const nmz_layer_ops *ops, void *self) {
    nmz_layer *l = calloc(1, sizeof *l);
    if (!l) {
        nmz__fail(NMZ_ERR_NOMEM, "out of memory");
        return NULL;
    }
    l->ops = ops;
    l->self = self;
    return l;
}

void nmz_layer_free(nmz_layer *l) {
    if (!l) return;
    if (l->ops && l->ops->destroy) l->ops->destroy(l);
    free(l);
}

const char *nmz_layer_type(const nmz_layer *l) { return l && l->ops ? l->ops->type : "?"; }

void nmz_layer_describe(const nmz_layer *l, char *buf, size_t n) {
    if (!buf || n == 0) return;
    if (!l || !l->ops || !l->ops->describe)
        snprintf(buf, n, "?");
    else
        l->ops->describe(l, buf, n);
}

int nmz_layer_param_count(const nmz_layer *l) {
    return l && l->ops ? l->ops->param_count(l) : 0;
}

static nmz_param layer_param(const nmz_layer *l, int i) {
    nmz_param none = {NULL, NULL, NULL};
    if (!l || i < 0 || i >= nmz_layer_param_count(l)) return none;
    return l->ops->param((nmz_layer *)l, i);
}

const char *nmz_layer_param_name(const nmz_layer *l, int i) { return layer_param(l, i).name; }
nmz_tensor *nmz_layer_param_value(nmz_layer *l, int i) { return layer_param(l, i).value; }
nmz_tensor *nmz_layer_param_grad(nmz_layer *l, int i) { return layer_param(l, i).grad; }

int nmz_layer_activation_kind(const nmz_layer *l) {
    if (!l || l->ops != &nmz__activation_ops) return -1;
    return (int)nmz__activation_layer_act(l);
}

/* Product of the dimensions; NMZ_ANY_LENGTH (0) dims count as 1, so for
 * sequence nets sizes are "per time step". */
static int shape_prod(int rank, const int *shape) {
    int i, p = 1;
    for (i = 0; i < rank; i++) p *= shape[i] ? shape[i] : 1;
    return p;
}

int nmz_layer_output_size(const nmz_layer *l) { return l ? shape_prod(l->out_rank, l->out_shape) : 0; }
int nmz_layer_input_size(const nmz_layer *l) { return l ? shape_prod(l->in_rank, l->in_shape) : 0; }

/* ------------------------------------------------------------- lifecycle */

nmz_train_opts nmz_train_defaults(void) {
    nmz_train_opts o;
    memset(&o, 0, sizeof o);
    o.max_epochs = 1000;
    o.batch_size = 32;
    o.learning_rate = 0.1f;
    o.momentum = 0.9f;
    o.shuffle = 1;
    o.target_loss = 1e-3;
    o.optimizer = NMZ_OPT_SGD;
    o.beta1 = 0.9f;
    o.beta2 = 0.999f;
    o.epsilon = 1e-8f;
    o.schedule = NMZ_SCHED_CONSTANT;
    o.step_size = 10;
    o.gamma = 0.5f;
    o.min_lr = 0.0f;
    o.val_fraction = 0.0f;
    o.clip_norm = 0.0f;
    o.carry_state = 0;
    o.seq_len = 0;
    o.patience = 0;
    return o;
}

nmz_status nmz_net_create(nmz_net **out, int input_rank, const int *input_shape, uint64_t seed) {
    nmz_net *net;
    size_t n;
    nmz_status s;
    int i;
    if (!out) return nmz__fail(NMZ_ERR_ARG, "out is NULL");
    *out = NULL;
    if (input_rank < 1 || input_rank > NMZ_MAX_RANK - 1)
        return nmz__fail(NMZ_ERR_ARG, "input rank must be 1..%d (batch dim is added)", NMZ_MAX_RANK - 1);
    if (!input_shape) return nmz__fail(NMZ_ERR_ARG, "input shape is NULL");
    if (input_shape[0] == NMZ_ANY_LENGTH) { /* sequence: any number of time steps */
        if (input_rank > 1 && (s = nmz__shape_size(input_rank - 1, input_shape + 1, &n)) != NMZ_OK) return s;
    } else if ((s = nmz__shape_size(input_rank, input_shape, &n)) != NMZ_OK) {
        return s;
    }
    net = calloc(1, sizeof *net);
    if (!net) return nmz__fail(NMZ_ERR_NOMEM, "out of memory");
    net->in_rank = input_rank;
    for (i = 0; i < input_rank; i++) net->in_shape[i] = input_shape[i];
    net->loss = NMZ_LOSS_MSE;
    net->train = nmz_train_defaults();
    net->opt_kind = -1;
    nmz_rng_seed(&net->rng, seed, 0x4e4d5aULL /* "NMZ" stream */);
    *out = net;
    return NMZ_OK;
}

static void free_buffers(nmz_net *net) {
    int i;
    if (net->acts)
        for (i = 0; i <= net->n_layers; i++) nmz_tensor_free(&net->acts[i]);
    if (net->dacts)
        for (i = 0; i <= net->n_layers; i++) nmz_tensor_free(&net->dacts[i]);
    for (i = 0; i < net->n_params; i++) {
        if (net->opt_m) nmz_tensor_free(&net->opt_m[i]);
        if (net->opt_v) nmz_tensor_free(&net->opt_v[i]);
    }
    free(net->acts);
    free(net->dacts);
    free(net->opt_m);
    free(net->opt_v);
    free(net->param_refs);
    net->acts = net->dacts = net->opt_m = net->opt_v = NULL;
    net->param_refs = NULL;
    net->n_params = 0;
}

void nmz_net_free(nmz_net *net) {
    int i;
    if (!net) return;
    free_buffers(net);
    for (i = 0; i < net->n_layers; i++) nmz_layer_free(net->layers[i]);
    free(net->layers);
    for (i = 0; i < net->n_classes; i++) free(net->class_names[i]);
    free(net->class_names);
    for (i = 0; i < 6; i++) nmz_tensor_free(&net->scratch[i]);
    nmz__norm_free(net);
    nmz_tensor_free(&net->in1);
    nmz_tensor_free(&net->out1);
    nmz_vocab_free(net->vocab);
    free(net->data_path);
    free(net->item_hash);
    free(net);
}

nmz_status nmz_net_add(nmz_net *net, nmz_layer *layer) {
    const int *in_shape;
    int in_rank;
    nmz_status s;
    if (!layer) return nmz__fail(NMZ_ERR_ARG, "layer is NULL (did its factory fail?)");
    if (!net) {
        nmz_layer_free(layer);
        return nmz__fail(NMZ_ERR_ARG, "net is NULL");
    }
    if (net->built) {
        nmz_layer_free(layer);
        return nmz__fail(NMZ_ERR_STATE, "cannot add layers after the network was built");
    }
    if (net->n_layers == net->cap_layers) {
        int cap = net->cap_layers ? net->cap_layers * 2 : 8;
        nmz_layer **p = realloc(net->layers, (size_t)cap * sizeof *p);
        if (!p) {
            nmz_layer_free(layer);
            return nmz__fail(NMZ_ERR_NOMEM, "out of memory");
        }
        net->layers = p;
        net->cap_layers = cap;
    }
    if (net->n_layers == 0) {
        in_rank = net->in_rank;
        in_shape = net->in_shape;
    } else {
        nmz_layer *prev = net->layers[net->n_layers - 1];
        in_rank = prev->out_rank;
        in_shape = prev->out_shape;
    }
    layer->in_rank = in_rank;
    memcpy(layer->in_shape, in_shape, sizeof(int) * (size_t)in_rank);
    s = layer->ops->setup(layer, in_rank, in_shape, &layer->out_rank, layer->out_shape);
    if (s != NMZ_OK) {
        nmz_layer_free(layer);
        return s;
    }
    net->layers[net->n_layers++] = layer;
    return NMZ_OK;
}

nmz_status nmz_net_add_dense(nmz_net *net, int units, nmz_act act, nmz_init init) {
    nmz_status s = nmz_net_add(net, nmz_layer_dense(units, init));
    if (s != NMZ_OK || act == NMZ_ACT_LINEAR) return s;
    return nmz_net_add(net, nmz_layer_activation(act, 0.0f));
}

nmz_status nmz_net_set_loss(nmz_net *net, nmz_loss loss) {
    if (!net) return nmz__fail(NMZ_ERR_ARG, "net is NULL");
    if ((int)loss < 0 || loss >= NMZ_LOSS_COUNT) return nmz__fail(NMZ_ERR_ARG, "invalid loss");
    if (loss == NMZ_LOSS_CROSS_ENTROPY && net->built && !nmz__net_fused_ce_possible(net))
        return nmz__fail(NMZ_ERR_ARG, "cross_entropy requires a softmax output layer");
    net->loss = loss;
    return NMZ_OK;
}

int nmz__net_fused_ce_possible(const nmz_net *net) {
    return net->n_layers >= 1 &&
           nmz_layer_activation_kind(net->layers[net->n_layers - 1]) == NMZ_ACT_SOFTMAX;
}

int nmz__net_fused_ce(const nmz_net *net) {
    return net->loss == NMZ_LOSS_CROSS_ENTROPY && nmz__net_fused_ce_possible(net);
}

nmz_status nmz__net_finish_build(nmz_net *net) {
    int i, j, k;
    nmz_status s;
    if (net->n_layers < 1) return nmz__fail(NMZ_ERR_STATE, "the network has no layers");
    if (net->loss == NMZ_LOSS_CROSS_ENTROPY && !nmz__net_fused_ce_possible(net))
        return nmz__fail(NMZ_ERR_ARG,
                         "cross_entropy requires a softmax output layer (it is computed fused "
                         "with softmax); use --loss mse or make the last activation softmax");
    for (i = 0; i < net->n_layers; i++) { /* layers that use another layer's parameter (tied weights) */
        nmz_layer *l = net->layers[i];
        if (l->ops->bind && (s = l->ops->bind(l, net->layers, net->n_layers, i)) != NMZ_OK) return s;
    }
    free_buffers(net);
    net->acts = calloc((size_t)net->n_layers + 1, sizeof *net->acts);
    net->dacts = calloc((size_t)net->n_layers + 1, sizeof *net->dacts);
    if (!net->acts || !net->dacts) return nmz__fail(NMZ_ERR_NOMEM, "out of memory");
    for (i = 0; i < net->n_layers; i++) net->n_params += nmz_layer_param_count(net->layers[i]);
    if (net->n_params > 0) {
        net->param_refs = calloc((size_t)net->n_params, sizeof *net->param_refs);
        net->opt_m = calloc((size_t)net->n_params, sizeof *net->opt_m);
        net->opt_v = calloc((size_t)net->n_params, sizeof *net->opt_v);
        if (!net->param_refs || !net->opt_m || !net->opt_v) return nmz__fail(NMZ_ERR_NOMEM, "out of memory");
    }
    for (i = 0, k = 0; i < net->n_layers; i++) {
        int np = nmz_layer_param_count(net->layers[i]);
        for (j = 0; j < np; j++, k++) {
            nmz_tensor *v = nmz_layer_param_value(net->layers[i], j);
            net->param_refs[k].layer = i;
            net->param_refs[k].index = j;
            if ((s = nmz_tensor_init(&net->opt_m[k], v->rank, v->shape)) != NMZ_OK) return s;
            if ((s = nmz_tensor_init(&net->opt_v[k], v->rank, v->shape)) != NMZ_OK) return s;
        }
    }
    net->built = 1;
    net->have_forward = net->have_grads = 0;
    return NMZ_OK;
}

static nmz_status init_all_params(nmz_net *net) {
    int i;
    for (i = 0; i < net->n_layers; i++) {
        nmz_layer *next = i + 1 < net->n_layers ? net->layers[i + 1] : NULL;
        nmz_status s = net->layers[i]->ops->init_params(net->layers[i], &net->rng, next);
        if (s != NMZ_OK) return s;
    }
    for (i = 0; i < net->n_params; i++) {
        nmz_tensor_zero(&net->opt_m[i]);
        nmz_tensor_zero(&net->opt_v[i]);
    }
    net->opt_kind = -1;
    net->opt_t = 0;
    return NMZ_OK;
}

nmz_status nmz_net_build(nmz_net *net) {
    nmz_status s;
    if (!net) return nmz__fail(NMZ_ERR_ARG, "net is NULL");
    if (net->built) return NMZ_OK;
    if ((s = nmz__net_finish_build(net)) != NMZ_OK) return s;
    if ((s = init_all_params(net)) != NMZ_OK) {
        net->built = 0;
        return s;
    }
    return NMZ_OK;
}

nmz_status nmz_net_reinit(nmz_net *net, uint64_t seed) {
    nmz_status s;
    if (!net) return nmz__fail(NMZ_ERR_ARG, "net is NULL");
    if ((s = nmz_net_build(net)) != NMZ_OK) return s;
    nmz_rng_seed(&net->rng, seed, 0x4e4d5aULL);
    net->step = 0;
    net->have_forward = net->have_grads = 0;
    return init_all_params(net);
}

/* ------------------------------------------------------------ accessors */

int nmz_net_layer_count(const nmz_net *net) { return net ? net->n_layers : 0; }

nmz_layer *nmz_net_layer(nmz_net *net, int i) {
    if (!net || i < 0 || i >= net->n_layers) return NULL;
    return net->layers[i];
}

int nmz_net_input_size(const nmz_net *net) { return net ? shape_prod(net->in_rank, net->in_shape) : 0; }

int nmz_net_output_size(const nmz_net *net) {
    if (!net) return 0;
    if (net->n_layers == 0) return nmz_net_input_size(net);
    return nmz_layer_output_size(net->layers[net->n_layers - 1]);
}

nmz_loss nmz_net_loss_kind(const nmz_net *net) { return net ? net->loss : NMZ_LOSS_MSE; }
int nmz_net_is_built(const nmz_net *net) { return net ? net->built : 0; }
long nmz_net_step(const nmz_net *net) { return net ? net->step : 0; }
nmz_train_opts *nmz_net_train_config(nmz_net *net) { return net ? &net->train : NULL; }

size_t nmz_net_param_count(const nmz_net *net) {
    size_t total = 0;
    int i, j;
    if (!net) return 0;
    for (i = 0; i < net->n_layers; i++)
        for (j = 0; j < nmz_layer_param_count(net->layers[i]); j++)
            total += nmz_layer_param_value(net->layers[i], j)->size;
    return total;
}

nmz_status nmz_net_set_class_names(nmz_net *net, int n, const char *const *names) {
    char **copy = NULL;
    int i;
    if (!net || n < 0 || (n > 0 && !names)) return nmz__fail(NMZ_ERR_ARG, "invalid class names");
    if (n > 0) {
        copy = calloc((size_t)n, sizeof *copy);
        if (!copy) return nmz__fail(NMZ_ERR_NOMEM, "out of memory");
        for (i = 0; i < n; i++) {
            copy[i] = nmz__strdup(names[i] ? names[i] : "");
            if (!copy[i]) {
                while (i--) free(copy[i]);
                free(copy);
                return nmz__fail(NMZ_ERR_NOMEM, "out of memory");
            }
        }
    }
    for (i = 0; i < net->n_classes; i++) free(net->class_names[i]);
    free(net->class_names);
    net->class_names = copy;
    net->n_classes = n;
    return NMZ_OK;
}

int nmz_net_class_count(const nmz_net *net) { return net ? net->n_classes : 0; }

const char *nmz_net_class_name(const nmz_net *net, int i) {
    if (!net || i < 0 || i >= net->n_classes) return NULL;
    return net->class_names[i];
}

void nmz_net_set_trace(nmz_net *net, nmz_trace_fn fn, void *user) {
    if (!net) return;
    net->trace = fn;
    net->trace_user = user;
}

const nmz_tensor *nmz_net_input_grad(const nmz_net *net) {
    if (!net || !net->have_grads) return NULL;
    return &net->dacts[0];
}

const nmz_tensor *nmz_net_activation(const nmz_net *net, int i) {
    if (!net || !net->have_forward || i < -1 || i >= net->n_layers) return NULL;
    return &net->acts[i + 1];
}

const nmz_tensor *nmz_net_activation_grad(const nmz_net *net, int i) {
    if (!net || !net->have_grads || i < -1 || i >= net->n_layers) return NULL;
    return &net->dacts[i + 1];
}

int nmz_layer_param_decays(const nmz_layer *l, int i, int decay_all) {
    const nmz_tensor *v;
    if (!l || i < 0 || i >= nmz_layer_param_count(l)) return 0;
    if (decay_all) return 1;
    /* embeddings and position tables are looked up, not multiplied: decaying
     * them pulls rarely seen characters/positions towards 0 (decision 94) */
    if (l->ops->flags & (NMZ_LAYER_DISCRETE_INPUT | NMZ_LAYER_NO_DECAY)) return 0;
    v = layer_param(l, i).value;
    return v && v->rank >= 2; /* weight matrices; biases and LayerNorm gains are vectors */
}

/* -------------------------------------------------------------- compute */

/* Trace events carry the 1-based number of the forward/backward pass they
 * belong to: a forward pass is numbered step+1 (the backward pass that
 * follows it will increment step to that number). */
static nmz_ctx make_ctx_step(nmz_net *net, int layer, int training, long step) {
    nmz_ctx c;
    c.rng = &net->rng;
    c.carry = net->carry;
    c.trace = net->trace;
    c.trace_user = net->trace_user;
    c.step = step;
    c.layer_index = layer;
    c.training = training;
    return c;
}

static nmz_ctx make_ctx(nmz_net *net, int layer, int training) {
    nmz_ctx c;
    c.rng = &net->rng;
    c.carry = net->carry;
    c.trace = net->trace;
    c.trace_user = net->trace_user;
    c.step = net->step;
    c.layer_index = layer;
    c.training = training;
    return c;
}

/* [batch, shape...]; NMZ_ANY_LENGTH dims become `steps` (the actual T). */
static nmz_status shape_with_batch_t(int batch, int steps, int rank, const int *shape, int *out) {
    int i;
    if (rank + 1 > NMZ_MAX_RANK) return nmz__fail(NMZ_ERR_SHAPE, "rank too large");
    out[0] = batch;
    for (i = 0; i < rank; i++) out[i + 1] = shape[i] == NMZ_ANY_LENGTH ? steps : shape[i];
    return NMZ_OK;
}

static nmz_status shape_with_batch(int batch, int rank, const int *shape, int *out) {
    return shape_with_batch_t(batch, 1, rank, shape, out);
}

/* does x fit the per-sample input shape (wildcard dims accept any size)? */
static int input_fits(const nmz_net *net, const nmz_tensor *x) {
    int d;
    if (x->rank != net->in_rank + 1) return 0;
    for (d = 0; d < net->in_rank; d++)
        if (net->in_shape[d] != NMZ_ANY_LENGTH && x->shape[d + 1] != net->in_shape[d]) return 0;
    return 1;
}

nmz_status nmz__net_forward(nmz_net *net, const nmz_tensor *x, int training,
                            const nmz_tensor **y) {
    int i, shape[NMZ_MAX_RANK];
    nmz_status s;
    if (!net || !x) return nmz__fail(NMZ_ERR_ARG, "NULL argument");
    if ((s = nmz_net_build(net)) != NMZ_OK) return s;
    if (!input_fits(net, x)) {
        if (nmz_net_is_sequence(net))
            return nmz__fail(NMZ_ERR_SHAPE, "a sequence net needs input of shape [batch, steps%s]",
                             net->in_rank > 1 ? ", ..." : "");
        return nmz__fail(NMZ_ERR_SHAPE, "input must have shape [batch, %d] (got %d values per sample)",
                         nmz_net_input_size(net), x->rank >= 1 ? (int)(x->size / (size_t)x->shape[0]) : 0);
    }
    net->have_forward = net->have_grads = 0;
    if ((s = nmz_tensor_copy(&net->acts[0], x)) != NMZ_OK) return s;
    if (net->norm != NMZ_NORM_NONE) {
        if (!net->norm_fitted)
            return nmz__fail(NMZ_ERR_STATE, "input normalization (%s) is not fitted yet: train the network first "
                                            "(or call nmz_net_fit_normalization)", nmz_norm_name(net->norm));
        nmz__norm_apply(net, &net->acts[0]);
        if (net->trace) {
            nmz_ctx ctx = make_ctx_step(net, -1, training, net->step + 1);
            nmz_tensor off, sc;
            memset(&off, 0, sizeof off);
            off.rank = 1;
            off.shape[0] = net->norm_groups;
            off.size = off.capacity = (size_t)net->norm_groups;
            sc = off;
            off.data = net->norm_offset; /* views, never freed */
            sc.data = net->norm_scale;
            {
                nmz_trace_item it[4] = {{"x", x}, {"offset", &off}, {"scale", &sc}, {"x_norm", &net->acts[0]}};
                nmz_trace_emit(&ctx, NMZ_TRACE_FORWARD, "normalize", nmz_norm_name(net->norm), 0, NULL, 4, it);
            }
        }
    }
    for (i = 0; i < net->n_layers; i++) {
        nmz_layer *l = net->layers[i];
        nmz_ctx ctx = make_ctx_step(net, i, training, net->step + 1);
        if ((s = shape_with_batch_t(x->shape[0], x->rank > 1 ? x->shape[1] : 1, l->out_rank, l->out_shape, shape)) !=
            NMZ_OK)
            return s;
        if ((s = nmz_tensor_reshape(&net->acts[i + 1], l->out_rank + 1, shape)) != NMZ_OK) return s;
        if ((s = l->ops->forward(l, &net->acts[i], &net->acts[i + 1], &ctx)) != NMZ_OK) return s;
    }
    net->have_forward = 1;
    if (y) *y = &net->acts[net->n_layers];
    return NMZ_OK;
}

nmz_status nmz_net_forward(nmz_net *net, const nmz_tensor *x, const nmz_tensor **y) {
    return nmz__net_forward(net, x, 0, y);
}

nmz_status nmz_net_forward_mode(nmz_net *net, const nmz_tensor *x, int training, const nmz_tensor **y) {
    nmz_status s;
    if (!net) return nmz__fail(NMZ_ERR_ARG, "net is NULL");
    s = nmz__net_forward(net, x, training != 0, y);
    net->training = training != 0; /* the backward pass that follows is traced with the same mode */
    return s;
}

nmz_status nmz_net_predict(nmz_net *net, const float *x, float *y) {
    const nmz_tensor *out;
    int shape[NMZ_MAX_RANK];
    nmz_status s;
    if (!net || !x || !y) return nmz__fail(NMZ_ERR_ARG, "NULL argument");
    if (nmz_net_is_sequence(net))
        return nmz__fail(NMZ_ERR_UNSUPPORTED, "predict takes one fixed-size sample; for a sequence net use "
                                              "nmz_net_forward with [1, steps] or nmz_net_generate");
    if ((s = shape_with_batch(1, net->in_rank, net->in_shape, shape)) != NMZ_OK) return s;
    if ((s = nmz_tensor_reshape(&net->in1, net->in_rank + 1, shape)) != NMZ_OK) return s;
    memcpy(net->in1.data, x, net->in1.size * sizeof(float));
    if ((s = nmz_net_forward(net, &net->in1, &out)) != NMZ_OK) return s;
    memcpy(y, out->data, out->size * sizeof(float));
    return NMZ_OK;
}

static nmz_status check_target(const nmz_net *net, const nmz_tensor *t) {
    const nmz_tensor *y = &net->acts[net->n_layers];
    if (!t) return nmz__fail(NMZ_ERR_ARG, "target is NULL");
    if (!nmz__shape_equal(y->rank, y->shape, t->rank, t->shape))
        return nmz__fail(NMZ_ERR_SHAPE, "target must have shape [batch, %d] like the output",
                         nmz_net_output_size(net));
    return NMZ_OK;
}

nmz_status nmz_net_compute_loss(nmz_net *net, const nmz_tensor *t, double *loss) {
    const nmz_tensor *z;
    nmz_status s;
    if (!net || !loss) return nmz__fail(NMZ_ERR_ARG, "NULL argument");
    if (!net->have_forward) return nmz__fail(NMZ_ERR_STATE, "call forward before computing the loss");
    if ((s = check_target(net, t)) != NMZ_OK) return s;
    z = nmz__net_fused_ce(net) ? &net->acts[net->n_layers - 1] : NULL;
    return nmz__loss_value(net->loss, &net->acts[net->n_layers], t, z, loss);
}

nmz_status nmz_net_backward(nmz_net *net, const nmz_tensor *t, double *loss) {
    int i, start, fused;
    double L = 0.0;
    nmz_status s;
    const nmz_tensor *y;
    if (!net) return nmz__fail(NMZ_ERR_ARG, "net is NULL");
    if (!net->have_forward) return nmz__fail(NMZ_ERR_STATE, "call forward before backward");
    if ((s = check_target(net, t)) != NMZ_OK) return s;
    y = &net->acts[net->n_layers];
    fused = nmz__net_fused_ce(net);
    net->step++;

    /* 1. zero every gradient: layers ADD into them */
    for (i = 0; i < net->n_params; i++)
        nmz_tensor_zero(nmz_layer_param_grad(net->layers[net->param_refs[i].layer],
                                             net->param_refs[i].index));

    /* 2. loss and its gradient */
    if ((s = nmz__loss_value(net->loss, y, t, fused ? &net->acts[net->n_layers - 1] : NULL, &L)) != NMZ_OK)
        return s;
    if (net->trace) {
        nmz_ctx ctx = make_ctx(net, -1, net->training);
        nmz_trace_scalar sc = {"L", L};
        nmz_trace_item it[2] = {{"y", y}, {"t", t}};
        nmz_trace_emit(&ctx, NMZ_TRACE_LOSS, "loss", nmz_loss_name(net->loss), 1, &sc, 2, it);
    }
    if (fused) {
        /* ∂L/∂z for the softmax input; the softmax layer itself is skipped */
        start = net->n_layers - 2;
        if ((s = nmz__loss_grad(net->loss, y, t, &net->dacts[net->n_layers - 1])) != NMZ_OK) return s;
        if (net->trace) {
            nmz_ctx ctx = make_ctx(net, -1, net->training);
            nmz_trace_item it[3] = {{"p", y}, {"t", t}, {"dL/dz", &net->dacts[net->n_layers - 1]}};
            nmz_trace_emit(&ctx, NMZ_TRACE_BACKWARD, "loss", "softmax+cross_entropy", 0, NULL, 3, it);
        }
    } else {
        start = net->n_layers - 1;
        if ((s = nmz__loss_grad(net->loss, y, t, &net->dacts[net->n_layers])) != NMZ_OK) return s;
        if (net->trace) {
            nmz_ctx ctx = make_ctx(net, -1, net->training);
            nmz_trace_item it[3] = {{"y", y}, {"t", t}, {"dL/dy", &net->dacts[net->n_layers]}};
            nmz_trace_emit(&ctx, NMZ_TRACE_BACKWARD, "loss", nmz_loss_name(net->loss), 0, NULL, 3, it);
        }
    }

    /* 3. walk the stack backwards; every layer gets ∂L/∂(its output) */
    for (i = start; i >= 0; i--) {
        nmz_layer *l = net->layers[i];
        nmz_ctx ctx = make_ctx(net, i, net->training);
        if ((s = nmz_tensor_reshape(&net->dacts[i], net->acts[i].rank, net->acts[i].shape)) != NMZ_OK)
            return s;
        s = l->ops->backward(l, &net->acts[i], &net->acts[i + 1], &net->dacts[i + 1], &net->dacts[i], &ctx);
        if (s != NMZ_OK) return s;
    }
    nmz__norm_scale_grad(net, &net->dacts[0]); /* ∂L/∂(raw input) */
    net->have_grads = 1;
    if (loss) *loss = L;
    if (!isfinite(L)) return nmz__fail(NMZ_ERR_NUMERIC, "loss became %g (try a smaller learning rate)", L);
    return NMZ_OK;
}

/* --------------------------------------------------------------- optimizers
 *
 * SGD + momentum:  v ← μ·v − η·g ;  w ← w + v
 * Adam:            t ← t + 1
 *                  m ← β1·m + (1−β1)·g        v ← β2·v + (1−β2)·g²
 *                  m̂ = m/(1−β1^t)             v̂ = v/(1−β2^t)      (bias correction)
 *                  w ← w − η·m̂/(√v̂ + ε)
 * With apply == 0 ("preview") the same numbers are computed into scratch
 * tensors and traced; weights and optimizer state stay untouched. */

/* fresh: treat the stored state as zero (preview after an optimizer switch) */
static nmz_status sgd_param(nmz_net *net, int k, float lr, float mu, int apply, int fresh) {
    nmz_layer *l = net->layers[net->param_refs[k].layer];
    int pi = net->param_refs[k].index;
    nmz_tensor *w = nmz_layer_param_value(l, pi), *g = nmz_layer_param_grad(l, pi);
    nmz_tensor *v = &net->opt_m[k], *v_new = v;
    const nmz_tensor *v_prev = v;
    nmz_status s;
    size_t e;
    if (!w || !g) return nmz__fail(NMZ_ERR_STATE, "layer %d returned no parameter %d", net->param_refs[k].layer, pi);
    if (net->trace) {
        /* keep a copy of the old velocity (apply) or compute the new one aside (preview) */
        if ((s = nmz_tensor_copy(&net->scratch[0], v)) != NMZ_OK) return s;
        if (fresh) nmz_tensor_zero(&net->scratch[0]);
        if (apply)
            v_prev = &net->scratch[0];
        else
            v_new = &net->scratch[0];
    }
    for (e = 0; e < w->size; e++) v_new->data[e] = (fresh ? 0.0f : mu * v_prev->data[e]) - lr * g->data[e];
    if (net->trace) {
        nmz_ctx ctx = make_ctx(net, net->param_refs[k].layer, 1);
        nmz_trace_scalar sc[2] = {{"lr", lr}, {"momentum", mu}};
        nmz_trace_item it[4] = {{"value", w}, {"grad", g}, {"velocity_prev", v_prev}, {"delta", v_new}};
        nmz_trace_emit(&ctx, NMZ_TRACE_UPDATE, nmz_layer_type(l), nmz_layer_param_name(l, pi), 2, sc, 4, it);
    }
    if (apply)
        for (e = 0; e < w->size; e++) w->data[e] += v->data[e];
    return NMZ_OK;
}

/* AdamW (decay > 0): the decoupled weight decay λ·w is added to the Adam
 * step, using the weight before the step: Δ = −η·(m̂/(√v̂ + ε) + λ·w). */
static nmz_status adam_param(nmz_net *net, int k, const nmz_train_opts *o, float lr, long t, int apply,
                             int fresh, double decay) {
    nmz_layer *l = net->layers[net->param_refs[k].layer];
    int pi = net->param_refs[k].index, j;
    nmz_tensor *w = nmz_layer_param_value(l, pi), *g = nmz_layer_param_grad(l, pi);
    nmz_tensor *m = &net->opt_m[k], *v = &net->opt_v[k];
    const double b1 = o->beta1, b2 = o->beta2, eps = o->epsilon;
    const double bc1 = 1.0 - pow(b1, (double)t), bc2 = 1.0 - pow(b2, (double)t);
    nmz_status s;
    size_t e;
    if (!w || !g) return nmz__fail(NMZ_ERR_STATE, "layer %d returned no parameter %d", net->param_refs[k].layer, pi);
    if (!net->trace) {
        if (!apply) return NMZ_OK;
        for (e = 0; e < w->size; e++) { /* fast path: in place */
            double gi = g->data[e];
            double mi = b1 * m->data[e] + (1.0 - b1) * gi, vi = b2 * v->data[e] + (1.0 - b2) * gi * gi;
            m->data[e] = (float)mi;
            v->data[e] = (float)vi;
            w->data[e] += (float)(-lr * ((mi / bc1) / (sqrt(vi / bc2) + eps) + decay * w->data[e]));
        }
        return NMZ_OK;
    }
    /* traced (or preview): compute m, v, m̂, v̂, Δ aside, emit, then apply */
    for (j = 0; j < 5; j++)
        if ((s = nmz_tensor_reshape(&net->scratch[j], w->rank, w->shape)) != NMZ_OK) return s;
    for (e = 0; e < w->size; e++) {
        double gi = g->data[e], m0 = fresh ? 0.0 : m->data[e], v0 = fresh ? 0.0 : v->data[e];
        double mi = b1 * m0 + (1.0 - b1) * gi, vi = b2 * v0 + (1.0 - b2) * gi * gi;
        double mhat = mi / bc1, vhat = vi / bc2;
        net->scratch[0].data[e] = (float)mi;
        net->scratch[1].data[e] = (float)vi;
        net->scratch[2].data[e] = (float)mhat;
        net->scratch[3].data[e] = (float)vhat;
        net->scratch[4].data[e] = (float)(-lr * (mhat / (sqrt(vhat) + eps) + decay * w->data[e]));
    }
    {
        nmz_ctx ctx = make_ctx(net, net->param_refs[k].layer, 1);
        nmz_trace_scalar sc[6] = {{"lr", lr}, {"beta1", b1}, {"beta2", b2}, {"epsilon", eps}, {"t", (double)t},
                                  {"weight_decay", decay}};
        nmz_trace_item it[7] = {{"value", w},         {"grad", g},           {"m", &net->scratch[0]},
                                {"v", &net->scratch[1]}, {"m_hat", &net->scratch[2]}, {"v_hat", &net->scratch[3]},
                                {"delta", &net->scratch[4]}};
        nmz_trace_emit(&ctx, NMZ_TRACE_UPDATE, nmz_layer_type(l), nmz_layer_param_name(l, pi),
                       o->optimizer == NMZ_OPT_ADAMW ? 6 : 5, sc, 7, it);
    }
    if (apply)
        for (e = 0; e < w->size; e++) {
            m->data[e] = net->scratch[0].data[e];
            v->data[e] = net->scratch[1].data[e];
            w->data[e] += net->scratch[4].data[e];
        }
    return NMZ_OK;
}

static nmz_status optimize(nmz_net *net, const nmz_train_opts *o, float lr, int apply) {
    int k, fresh;
    long t;
    nmz_status s;
    if (!net || !o) return nmz__fail(NMZ_ERR_ARG, "NULL argument");
    if (!net->built) return nmz__fail(NMZ_ERR_STATE, "network is not built");
    if (!(lr > 0.0f) || !isfinite(lr)) return nmz__fail(NMZ_ERR_ARG, "learning rate must be > 0");
    if ((int)o->optimizer < 0 || o->optimizer >= NMZ_OPT_COUNT) return nmz__fail(NMZ_ERR_ARG, "invalid optimizer");
    if (o->optimizer == NMZ_OPT_SGD && !(o->momentum >= 0.0f && o->momentum < 1.0f))
        return nmz__fail(NMZ_ERR_ARG, "momentum must be in [0, 1)");
    if (o->optimizer != NMZ_OPT_SGD && !(o->beta1 >= 0.0f && o->beta1 < 1.0f && o->beta2 >= 0.0f &&
                                         o->beta2 < 1.0f && o->epsilon > 0.0f))
        return nmz__fail(NMZ_ERR_ARG, "Adam needs 0 <= beta1, beta2 < 1 and epsilon > 0");
    if (o->optimizer == NMZ_OPT_ADAMW && !(o->weight_decay >= 0.0f && o->weight_decay < 10.0f))
        return nmz__fail(NMZ_ERR_ARG, "weight decay must be in [0, 10)");
    if (!apply && !net->trace) return NMZ_OK;
    fresh = net->opt_kind != (int)o->optimizer;
    if (fresh && apply) {
        /* the stored state belongs to another optimizer (or none): start over */
        for (k = 0; k < net->n_params; k++) {
            nmz_tensor_zero(&net->opt_m[k]);
            nmz_tensor_zero(&net->opt_v[k]);
        }
        net->opt_t = 0;
        net->opt_kind = (int)o->optimizer;
        fresh = 0; /* the state is now genuinely zero */
    }
    t = (fresh ? 0 : net->opt_t) + 1;
    for (k = 0; k < net->n_params; k++) {
        if (o->optimizer == NMZ_OPT_SGD) {
            s = sgd_param(net, k, lr, o->momentum, apply, fresh);
        } else {
            double decay = 0.0;
            if (o->optimizer == NMZ_OPT_ADAMW &&
                nmz_layer_param_decays(net->layers[net->param_refs[k].layer], net->param_refs[k].index, o->decay_all))
                decay = o->weight_decay;
            s = adam_param(net, k, o, lr, t, apply, fresh, decay);
        }
        if (s != NMZ_OK) return s;
    }
    if (apply) {
        if (o->optimizer != NMZ_OPT_SGD) net->opt_t = t;
        net->have_forward = 0; /* cached activations no longer match the weights */
    }
    return NMZ_OK;
}

nmz_status nmz_net_optimizer_step(nmz_net *net, const nmz_train_opts *o, float learning_rate) {
    return optimize(net, o, learning_rate, 1);
}

nmz_status nmz_net_optimizer_preview(nmz_net *net, const nmz_train_opts *o, float learning_rate) {
    return optimize(net, o, learning_rate, 0);
}

static nmz_train_opts sgd_opts(float momentum) {
    nmz_train_opts o = nmz_train_defaults();
    o.optimizer = NMZ_OPT_SGD;
    o.momentum = momentum;
    return o;
}

nmz_status nmz_net_update(nmz_net *net, float learning_rate, float momentum) {
    nmz_train_opts o = sgd_opts(momentum);
    return optimize(net, &o, learning_rate, 1);
}

nmz_status nmz_net_preview_update(nmz_net *net, float learning_rate, float momentum) {
    nmz_train_opts o = sgd_opts(momentum);
    return optimize(net, &o, learning_rate, 0);
}

int nmz_net_optimizer_state(const nmz_net *net, long *t) {
    if (t) *t = net ? net->opt_t : 0;
    return net ? net->opt_kind : -1;
}

nmz_status nmz_net_clip_gradients(nmz_net *net, double max_norm, double *norm, double *scale) {
    double sq = 0.0, n, f = 1.0;
    int k;
    size_t e;
    if (!net || !(max_norm > 0)) return nmz__fail(NMZ_ERR_ARG, "clip norm must be > 0");
    if (!net->built) return nmz__fail(NMZ_ERR_STATE, "network is not built");
    for (k = 0; k < net->n_params; k++) {
        const nmz_tensor *g = nmz_layer_param_grad(net->layers[net->param_refs[k].layer], net->param_refs[k].index);
        for (e = 0; e < g->size; e++) sq += (double)g->data[e] * g->data[e];
    }
    n = sqrt(sq);
    if (n > max_norm) {
        f = max_norm / n;
        for (k = 0; k < net->n_params; k++) {
            nmz_tensor *g = nmz_layer_param_grad(net->layers[net->param_refs[k].layer], net->param_refs[k].index);
            for (e = 0; e < g->size; e++) g->data[e] = (float)(g->data[e] * f);
        }
    }
    if (net->trace) {
        nmz_ctx ctx = make_ctx(net, -1, 1);
        nmz_trace_scalar sc[3] = {{"norm", n}, {"max_norm", max_norm}, {"scale", f}};
        nmz_trace_emit(&ctx, NMZ_TRACE_UPDATE, "clip", f < 1.0 ? "clipped" : "unchanged", 3, sc, 0, NULL);
    }
    if (norm) *norm = n;
    if (scale) *scale = f;
    return NMZ_OK;
}

void nmz_net_set_carry(nmz_net *net, int on) {
    if (net) net->carry = on != 0;
}

void nmz_net_reset_state(nmz_net *net) {
    int i;
    if (!net) return;
    for (i = 0; i < net->n_layers; i++) {
        nmz_layer *l = net->layers[i];
        if (l->ops->state) nmz_tensor_free(l->ops->state(l));
    }
}

nmz_status nmz_net_set_vocab(nmz_net *net, const nmz_vocab *v) {
    nmz_vocab *copy = NULL;
    nmz_status s;
    if (!net) return nmz__fail(NMZ_ERR_ARG, "net is NULL");
    if (v && (s = nmz_vocab_copy(&copy, v)) != NMZ_OK) return s;
    nmz_vocab_free(net->vocab);
    net->vocab = copy;
    return NMZ_OK;
}

const nmz_vocab *nmz_net_vocab(const nmz_net *net) { return net ? net->vocab : NULL; }

/* ------------------------------------------------- training data source
 * The CLI's `generate --new-only` needs to know which items the model saw.
 * Keeping 8 bytes per item (not the items) keeps model files small and
 * private enough to share; a false "known" needs a 64-bit collision
 * (≈ n²/2⁶⁵, about 10⁻¹¹ for 20 000 names). */
static uint64_t item_hash(const char *s) {
    uint64_t h = 14695981039346656037ull; /* FNV-1a 64 */
    for (; *s; s++) h = (h ^ (unsigned char)*s) * 1099511628211ull;
    return h;
}

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

void nmz__net_set_item_hashes(nmz_net *net, uint64_t *h, int n) {
    free(net->item_hash);
    net->item_hash = h;
    net->n_item_hash = h ? n : 0;
}

nmz_status nmz_net_set_data_source(nmz_net *net, const char *path, const char *const *items, int n_items) {
    char *p = NULL;
    uint64_t *h = NULL;
    int i, n = 0;
    if (!net || n_items < 0 || (n_items > 0 && !items)) return nmz__fail(NMZ_ERR_ARG, "bad data source arguments");
    if (path && strlen(path) > 4096) return nmz__fail(NMZ_ERR_ARG, "data path too long");
    if (n_items > 10000000) return nmz__fail(NMZ_ERR_ARG, "too many items (at most 10 000 000)");
    if (path && !(p = nmz__strdup(path))) return nmz__fail(NMZ_ERR_NOMEM, "out of memory");
    if (n_items > 0) {
        if (!(h = malloc((size_t)n_items * sizeof *h))) {
            free(p);
            return nmz__fail(NMZ_ERR_NOMEM, "out of memory");
        }
        for (i = 0; i < n_items; i++) {
            if (!items[i]) {
                free(p);
                free(h);
                return nmz__fail(NMZ_ERR_ARG, "item %d is NULL", i);
            }
            h[i] = item_hash(items[i]);
        }
        qsort(h, (size_t)n_items, sizeof *h, cmp_u64);
        for (i = 0; i < n_items; i++)
            if (n == 0 || h[i] != h[n - 1]) h[n++] = h[i];
    }
    free(net->data_path);
    net->data_path = p;
    nmz__net_set_item_hashes(net, h, n);
    return NMZ_OK;
}

const char *nmz_net_data_path(const nmz_net *net) { return net ? net->data_path : NULL; }

int nmz_net_item_count(const nmz_net *net) { return net ? net->n_item_hash : 0; }

int nmz_net_knows_item(const nmz_net *net, const char *item) {
    uint64_t h;
    if (!net || !net->item_hash || !item) return -1;
    h = item_hash(item);
    return bsearch(&h, net->item_hash, (size_t)net->n_item_hash, sizeof h, cmp_u64) != NULL;
}

int nmz_net_is_sequence(const nmz_net *net) { return net && net->in_shape[0] == NMZ_ANY_LENGTH; }

/* -------------------------------------------------------------- summary */

int nmz_net_summary(const nmz_net *net, char *buf, size_t n) {
    int i, len = 0, w;
    char desc[128];
#define APPEND(...)                                                                    \
    do {                                                                               \
        w = snprintf(buf ? buf + (len < (int)n ? len : (int)n) : NULL,                 \
                     buf && len < (int)n ? n - (size_t)len : 0, __VA_ARGS__);          \
        if (w > 0) len += w;                                                           \
    } while (0)
    if (!net) return 0;
    if (buf && n) buf[0] = '\0';
    APPEND("input: %d value(s) per sample\n", nmz_net_input_size(net));
    for (i = 0; i < net->n_layers; i++) {
        const nmz_layer *l = net->layers[i];
        int p, np = nmz_layer_param_count(l);
        size_t count = 0;
        for (p = 0; p < np; p++) count += nmz_layer_param_value((nmz_layer *)l, p)->size;
        nmz_layer_describe(l, desc, sizeof desc);
        APPEND("  [%d] %-40s out=%-5d params=%zu\n", i, desc, nmz_layer_output_size(l), count);
    }
    APPEND("loss: %s\n", nmz_loss_name(net->loss));
    APPEND("parameters: %zu\n", nmz_net_param_count(net));
#undef APPEND
    return len;
}
