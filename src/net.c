/* net.c — the network: a sequential stack of layers plus the buffers that
 * connect them.
 *
 *   acts[0]   = input x                 dacts[0]   = ∂L/∂x
 *   acts[i+1] = layer_i(acts[i])        dacts[i+1] = ∂L/∂acts[i+1]
 *
 * backward() computes EVERY gradient first; update() changes weights in a
 * separate pass, so no gradient is ever computed from already-updated
 * weights (a v1 bug).
 *
 * Every computation goes through the backend the net is on (net->be, which
 * follows nmz_backend_select): layers get it in their nmz_ctx, and the
 * optimizer and clipping call its ops. On the CPU these are the v2.0.1
 * loops; nothing else changes (DESIGN §11). */
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

nmz_param nmz__param(const nmz_layer *l, int i) { return layer_param(l, i); }

const char *nmz_layer_param_name(const nmz_layer *l, int i) { return layer_param(l, i).name; }
/* The public accessors hand out host data the caller may change: the host
 * copy becomes the newer one (a device copy is refreshed before its next
 * use). */
nmz_tensor *nmz_layer_param_value(nmz_layer *l, int i) {
    nmz_tensor *t = layer_param(l, i).value;
    nmz__hu(t);
    return t;
}
nmz_tensor *nmz_layer_param_grad(nmz_layer *l, int i) {
    nmz_tensor *t = layer_param(l, i).grad;
    nmz__hu(t);
    return t;
}

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
    net->be = &nmz__cpu_backend;
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
    nmz_tensor_free(&net->norm_t);
    nmz_tensor_free(&net->tgt);
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
            nmz_tensor *v = layer_param(net->layers[i], j).value;
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
            total += layer_param(net->layers[i], j).value->size;
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

/* host copies are brought up to date (the tensors are the net's own) */
const nmz_tensor *nmz_net_input_grad(const nmz_net *net) {
    if (!net || !net->have_grads) return NULL;
    nmz__hr(&net->dacts[0]);
    return &net->dacts[0];
}

const nmz_tensor *nmz_net_activation(const nmz_net *net, int i) {
    if (!net || !net->have_forward || i < -1 || i >= net->n_layers) return NULL;
    nmz__hr(&net->acts[i + 1]);
    return &net->acts[i + 1];
}

const nmz_tensor *nmz_net_activation_grad(const nmz_net *net, int i) {
    if (!net || !net->have_grads || i < -1 || i >= net->n_layers) return NULL;
    nmz__hr(&net->dacts[i + 1]);
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
    c.backend = net->be;
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
    c.backend = net->be;
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

/* ------------------------------------------------------------- backends */

/* Give device memory back: the activations and the layers' caches (keep:
 * their contents come to the host first) and, unless only_buffers, the
 * parameters, gradients and optimizer state (always kept). */
static void release_device(nmz_net *net, int only_buffers, int keep) {
    int i, j, n;
    for (i = 0; net->acts && i <= net->n_layers; i++) {
        nmz__tensor_drop_device(&net->acts[i], keep);
        nmz__tensor_drop_device(&net->dacts[i], keep);
    }
    for (i = 0; i < 6; i++) nmz__tensor_drop_device(&net->scratch[i], keep);
    nmz__tensor_drop_device(&net->tgt, keep);
    for (i = 0; i < net->n_layers; i++) {
        nmz_layer *l = net->layers[i];
        nmz_tensor *buf[64];
        if (!l->ops->buffers) continue;
        n = l->ops->buffers(l, buf, 64);
        for (j = 0; j < n && j < 64; j++) nmz__tensor_drop_device(buf[j], keep);
    }
    if (only_buffers) return;
    for (i = 0; i < net->n_params; i++) {
        nmz__tensor_to_host(&net->opt_m[i]);
        nmz__tensor_to_host(&net->opt_v[i]);
    }
    for (i = 0; i < net->n_layers; i++)
        for (j = 0; j < nmz_layer_param_count(net->layers[i]); j++) {
            nmz_param p = layer_param(net->layers[i], j);
            nmz__tensor_to_host(p.value);
            nmz__tensor_to_host(p.grad);
        }
    nmz__tensor_to_host(&net->norm_t);
}

/* Moving to a host backend: every tensor comes back to the host. */
static void to_host(nmz_net *net) { release_device(net, 0, 1); }

void nmz__net_drop_buffers(nmz_net *net) {
    if (net && !net->be->host_memory) release_device(net, 1, 0);
}

const nmz_backend *nmz__net_backend(nmz_net *net) {
    const nmz_backend *be = nmz_backend_current();
    if (net->be != be) {
        if (!net->be->host_memory && be->host_memory) to_host(net);
        net->be = be;
    }
    return be;
}

/* the normalization's offset and scale in be's memory */
static nmz_status norm_ptrs(nmz_net *net, const nmz_backend *be, const float **off, const float **sc) {
    if (be->host_memory) {
        *off = net->norm_offset;
        *sc = net->norm_scale;
        return NMZ_OK;
    }
    if (net->norm_t_ver != net->norm_ver || net->norm_t.size != (size_t)2 * net->norm_groups) {
        int n = 2 * net->norm_groups;
        nmz_status s;
        float *h;
        if ((s = nmz_tensor_reshape(&net->norm_t, 1, &n)) != NMZ_OK) return s;
        h = nmz__hw(&net->norm_t);
        memcpy(h, net->norm_offset, (size_t)net->norm_groups * sizeof(float));
        memcpy(h + net->norm_groups, net->norm_scale, (size_t)net->norm_groups * sizeof(float));
        net->norm_t_ver = net->norm_ver;
    }
    *off = nmz__r(be, &net->norm_t);
    *sc = *off ? *off + net->norm_groups : NULL;
    return *off ? NMZ_OK : nmz__op_failed();
}

static nmz_status norm_apply(nmz_net *net, const nmz_backend *be, nmz_tensor *t, int grad) {
    const float *off, *sc;
    size_t per = 1, div = 1;
    float *p;
    nmz_status s;
    int i;
    if (!net->norm_fitted || net->norm == NMZ_NORM_NONE) return NMZ_OK;
    for (i = 0; i < net->in_rank; i++) per *= (size_t)net->in_shape[i];
    if (net->in_rank == 3) div = (size_t)net->in_shape[1] * net->in_shape[2];
    if ((s = norm_ptrs(net, be, &off, &sc)) != NMZ_OK) return s;
    if (!(p = nmz__u(be, t))) return nmz__op_failed();
    return be->norm_apply(be, t->size, per, div, net->norm_groups, off, sc, p, grad);
}

/* A layer without NMZ_LAYER_BACKEND is host code: on a device, its input
 * (and gradient) and parameters come to the host first, and what it writes
 * is marked as changed there. */
static int host_layer(const nmz_net *net, const nmz_layer *l) {
    return !net->be->host_memory && !(l->ops->flags & NMZ_LAYER_BACKEND);
}

static void layer_params_to_host(nmz_layer *l, int grads) {
    int j;
    for (j = 0; j < nmz_layer_param_count(l); j++) {
        nmz_param p = layer_param(l, j);
        nmz__hu(p.value);
        if (grads) nmz__hu(p.grad);
    }
}

nmz_status nmz__net_forward(nmz_net *net, const nmz_tensor *x, int training,
                            const nmz_tensor **y) {
    int i, shape[NMZ_MAX_RANK];
    nmz_status s;
    if (!net || !x) return nmz__fail(NMZ_ERR_ARG, "NULL argument");
    if ((s = nmz_net_build(net)) != NMZ_OK) return s;
    nmz__net_backend(net);
    if (!input_fits(net, x)) {
        if (nmz_net_is_sequence(net))
            return nmz__fail(NMZ_ERR_SHAPE, "a sequence net needs input of shape [batch, steps%s]",
                             net->in_rank > 1 ? ", ..." : "");
        return nmz__fail(NMZ_ERR_SHAPE, "input must have shape [batch, %d] (got %d values per sample)",
                         nmz_net_input_size(net), x->rank >= 1 ? (int)(x->size / (size_t)x->shape[0]) : 0);
    }
    net->have_forward = net->have_grads = 0;
    if (!net->be->host_memory && !x->dev && (net->layers[0]->ops->flags & NMZ_LAYER_DISCRETE_INPUT) &&
        (s = nmz__check_ids(net->layers[0], x->data, x->size)) != NMZ_OK) /* a device kernel cannot report a bad id */
        return s;
    if ((s = nmz_tensor_copy(&net->acts[0], x)) != NMZ_OK) return s;
    if (net->norm != NMZ_NORM_NONE) {
        if (!net->norm_fitted)
            return nmz__fail(NMZ_ERR_STATE, "input normalization (%s) is not fitted yet: train the network first "
                                            "(or call nmz_net_fit_normalization)", nmz_norm_name(net->norm));
        if ((s = norm_apply(net, net->be, &net->acts[0], 0)) != NMZ_OK) return s;
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
        if (host_layer(net, l)) {
            nmz__hr(&net->acts[i]);
            layer_params_to_host(l, 0);
            nmz__hw(&net->acts[i + 1]);
            ctx.backend = &nmz__cpu_backend;
        }
        if ((s = l->ops->forward(l, &net->acts[i], &net->acts[i + 1], &ctx)) != NMZ_OK) return s;
    }
    net->have_forward = 1;
    if (y) *y = &net->acts[net->n_layers];
    return NMZ_OK;
}

nmz_status nmz_net_forward(nmz_net *net, const nmz_tensor *x, const nmz_tensor **y) {
    nmz_status s = nmz__net_forward(net, x, 0, y);
    if (s == NMZ_OK && y) nmz__hr(*y); /* the caller reads the output on the host */
    return s;
}

nmz_status nmz_net_forward_mode(nmz_net *net, const nmz_tensor *x, int training, const nmz_tensor **y) {
    nmz_status s;
    if (!net) return nmz__fail(NMZ_ERR_ARG, "net is NULL");
    s = nmz__net_forward(net, x, training != 0, y);
    net->training = training != 0; /* the backward pass that follows is traced with the same mode */
    if (s == NMZ_OK && y) nmz__hr(*y);
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
    memcpy(nmz__hw(&net->in1), x, net->in1.size * sizeof(float));
    if ((s = nmz_net_forward(net, &net->in1, &out)) != NMZ_OK) return s;
    memcpy(y, nmz__hr(out), out->size * sizeof(float));
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

/* The target in the net's memory: a caller's host tensor is copied into the
 * net (never given a device copy of its own: the caller owns it). */
static const nmz_tensor *target_on(nmz_net *net, const nmz_tensor *t, nmz_status *s) {
    *s = NMZ_OK;
    if (net->be->host_memory || t->dev) return t;
    *s = nmz_tensor_copy(&net->tgt, t);
    return &net->tgt;
}

nmz_status nmz_net_compute_loss(nmz_net *net, const nmz_tensor *t, double *loss) {
    const nmz_tensor *z;
    nmz_status s;
    if (!net || !loss) return nmz__fail(NMZ_ERR_ARG, "NULL argument");
    if (!net->have_forward) return nmz__fail(NMZ_ERR_STATE, "call forward before computing the loss");
    if ((s = check_target(net, t)) != NMZ_OK) return s;
    t = target_on(net, t, &s);
    if (s != NMZ_OK) return s;
    z = nmz__net_fused_ce(net) ? &net->acts[net->n_layers - 1] : NULL;
    return nmz__loss_value(net->be, net->loss, &net->acts[net->n_layers], t, z, loss);
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
    {
        const nmz_tensor *user_t = t;
        t = target_on(net, t, &s);
        if (s != NMZ_OK) return s;
        net->step++;
        /* 1. zero every gradient: layers ADD into them */
        for (i = 0; i < net->n_params; i++)
            if ((s = nmz__zero_on(net->be, layer_param(net->layers[net->param_refs[i].layer], net->param_refs[i].index)
                                          .grad)) != NMZ_OK)
                return s;
        /* 2. loss and its gradient */
        if ((s = nmz__loss_value(net->be, net->loss, y, t, fused ? &net->acts[net->n_layers - 1] : NULL, &L)) !=
            NMZ_OK)
            return s;
        t = net->trace ? user_t : t; /* trace items: the caller's own tensor */
    }
    if (net->trace) {
        nmz_ctx ctx = make_ctx(net, -1, net->training);
        nmz_trace_scalar sc = {"L", L};
        nmz_trace_item it[2] = {{"y", y}, {"t", t}};
        nmz_trace_emit(&ctx, NMZ_TRACE_LOSS, "loss", nmz_loss_name(net->loss), 1, &sc, 2, it);
    }
    if (fused) {
        /* ∂L/∂z for the softmax input; the softmax layer itself is skipped */
        start = net->n_layers - 2;
        if ((s = nmz__loss_grad(net->be, net->loss, y, target_on(net, t, &s), &net->dacts[net->n_layers - 1])) !=
            NMZ_OK)
            return s;
        if (net->trace) {
            nmz_ctx ctx = make_ctx(net, -1, net->training);
            nmz_trace_item it[3] = {{"p", y}, {"t", t}, {"dL/dz", &net->dacts[net->n_layers - 1]}};
            nmz_trace_emit(&ctx, NMZ_TRACE_BACKWARD, "loss", "softmax+cross_entropy", 0, NULL, 3, it);
        }
    } else {
        start = net->n_layers - 1;
        if ((s = nmz__loss_grad(net->be, net->loss, y, target_on(net, t, &s), &net->dacts[net->n_layers])) != NMZ_OK)
            return s;
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
        if (host_layer(net, l)) {
            nmz__hr(&net->acts[i]);
            nmz__hr(&net->acts[i + 1]);
            nmz__hr(&net->dacts[i + 1]);
            layer_params_to_host(l, 1);
            nmz__hw(&net->dacts[i]);
            ctx.backend = &nmz__cpu_backend;
        }
        s = l->ops->backward(l, &net->acts[i], &net->acts[i + 1], &net->dacts[i + 1], &net->dacts[i], &ctx);
        if (s != NMZ_OK) return s;
    }
    if ((s = norm_apply(net, net->be, &net->dacts[0], 1)) != NMZ_OK) return s; /* ∂L/∂(raw input) */
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
    const nmz_backend *be = net->be;
    nmz_layer *l = net->layers[net->param_refs[k].layer];
    int pi = net->param_refs[k].index;
    nmz_param pr = layer_param(l, pi);
    nmz_tensor *w = pr.value, *g = pr.grad;
    nmz_tensor *v = &net->opt_m[k], *v_new = v;
    const nmz_tensor *v_prev = v;
    const float *pg, *pv;
    float *pn;
    nmz_status s;
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
    pg = nmz__r(be, g);
    pv = nmz__r(be, v_prev);
    pn = v_new == v_prev ? nmz__u(be, v_new) : nmz__w(be, v_new);
    if (!pg || !pv || !pn) return nmz__op_failed();
    if ((s = be->sgd_velocity(be, w->size, lr, mu, fresh, pg, pv, pn)) != NMZ_OK) return s;
    if (net->trace) {
        nmz_ctx ctx = make_ctx(net, net->param_refs[k].layer, 1);
        nmz_trace_scalar sc[2] = {{"lr", lr}, {"momentum", mu}};
        nmz_trace_item it[4] = {{"value", w}, {"grad", g}, {"velocity_prev", v_prev}, {"delta", v_new}};
        nmz_trace_emit(&ctx, NMZ_TRACE_UPDATE, nmz_layer_type(l), nmz_layer_param_name(l, pi), 2, sc, 4, it);
    }
    if (apply) {
        float *pw = nmz__u(be, w);
        const float *pvel = nmz__r(be, v);
        if (!pw || !pvel) return nmz__op_failed();
        return be->add(be, w->size, pw, pvel, pw);
    }
    return NMZ_OK;
}

/* AdamW (decay > 0): the decoupled weight decay λ·w is added to the Adam
 * step, using the weight before the step: Δ = −η·(m̂/(√v̂ + ε) + λ·w). */
static nmz_status adam_param(nmz_net *net, int k, const nmz_train_opts *o, float lr, long t, int apply,
                             int fresh, double decay) {
    const nmz_backend *be = net->be;
    nmz_layer *l = net->layers[net->param_refs[k].layer];
    int pi = net->param_refs[k].index, j;
    nmz_param pr = layer_param(l, pi);
    nmz_tensor *w = pr.value, *g = pr.grad;
    nmz_tensor *m = &net->opt_m[k], *v = &net->opt_v[k];
    const double b1 = o->beta1, b2 = o->beta2, eps = o->epsilon;
    const double bc1 = 1.0 - pow(b1, (double)t), bc2 = 1.0 - pow(b2, (double)t);
    float *sp[5];
    nmz_status s;
    if (!w || !g) return nmz__fail(NMZ_ERR_STATE, "layer %d returned no parameter %d", net->param_refs[k].layer, pi);
    if (!net->trace) {
        float *pw, *pm, *pv;
        const float *pg;
        if (!apply) return NMZ_OK;
        pw = nmz__u(be, w); /* fast path: in place */
        pg = nmz__r(be, g);
        pm = nmz__u(be, m);
        pv = nmz__u(be, v);
        if (!pw || !pg || !pm || !pv) return nmz__op_failed();
        return be->adam_update(be, w->size, (double)lr, b1, b2, eps, bc1, bc2, decay, pw, pg, pm, pv);
    }
    /* traced (or preview): compute m, v, m̂, v̂, Δ aside, emit, then apply */
    for (j = 0; j < 5; j++) {
        if ((s = nmz_tensor_reshape(&net->scratch[j], w->rank, w->shape)) != NMZ_OK) return s;
        if (!(sp[j] = nmz__w(be, &net->scratch[j]))) return nmz__op_failed();
    }
    {
        const float *pw = nmz__r(be, w), *pg = nmz__r(be, g), *pm = nmz__r(be, m), *pv = nmz__r(be, v);
        if (!pw || !pg || !pm || !pv) return nmz__op_failed();
        if ((s = be->adam_preview(be, w->size, (double)lr, b1, b2, eps, bc1, bc2, decay, fresh, pw, pg, pm, pv, sp[0],
                                  sp[1], sp[2], sp[3], sp[4])) != NMZ_OK)
            return s;
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
    if (apply) {
        size_t n = w->size;
        float *pm = nmz__w(be, m), *pv = nmz__w(be, v), *pw = nmz__u(be, w);
        const float *s0 = nmz__r(be, &net->scratch[0]), *s1 = nmz__r(be, &net->scratch[1]),
                    *s4 = nmz__r(be, &net->scratch[4]);
        if (!pm || !pv || !pw || !s0 || !s1 || !s4) return nmz__op_failed();
        if ((s = be->copy(be, 1, n, s0, n, pm, n)) != NMZ_OK || (s = be->copy(be, 1, n, s1, n, pv, n)) != NMZ_OK)
            return s;
        return be->add(be, n, pw, s4, pw);
    }
    return NMZ_OK;
}

static nmz_status optimize(nmz_net *net, const nmz_train_opts *o, float lr, int apply) {
    int k, fresh;
    long t;
    nmz_status s;
    if (!net || !o) return nmz__fail(NMZ_ERR_ARG, "NULL argument");
    if (!net->built) return nmz__fail(NMZ_ERR_STATE, "network is not built");
    nmz__net_backend(net);
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
            if ((s = nmz__zero_on(net->be, &net->opt_m[k])) != NMZ_OK || (s = nmz__zero_on(net->be, &net->opt_v[k])) != NMZ_OK)
                return s;
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
    float *stack_p[64], **gp = stack_p;
    size_t stack_n[64], *gn = stack_n;
    const nmz_backend *be;
    nmz_status s = NMZ_OK;
    int k;
    if (!net || !(max_norm > 0)) return nmz__fail(NMZ_ERR_ARG, "clip norm must be > 0");
    if (!net->built) return nmz__fail(NMZ_ERR_STATE, "network is not built");
    be = nmz__net_backend(net);
    if (net->n_params > 64) {
        gp = malloc((size_t)net->n_params * sizeof *gp);
        gn = malloc((size_t)net->n_params * sizeof *gn);
        if (!gp || !gn) {
            free(gp == stack_p ? NULL : gp);
            free(gn == stack_n ? NULL : gn);
            return nmz__fail(NMZ_ERR_NOMEM, "out of memory");
        }
    }
    for (k = 0; k < net->n_params && s == NMZ_OK; k++) {
        nmz_tensor *g = layer_param(net->layers[net->param_refs[k].layer], net->param_refs[k].index).grad;
        gn[k] = g->size;
        if (!(gp[k] = nmz__u(be, g)) && g->size) s = nmz__op_failed();
    }
    if (s == NMZ_OK && (s = be->sumsq(be, net->n_params, gp, gn, &sq)) == NMZ_OK) s = be->sync(be);
    n = sqrt(sq);
    if (s == NMZ_OK && n > max_norm) {
        f = max_norm / n;
        s = be->scale(be, net->n_params, gp, gn, f);
    }
    if (gp != stack_p) free(gp);
    if (gn != stack_n) free(gn);
    if (s != NMZ_OK) return s;
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
        for (p = 0; p < np; p++) count += layer_param(l, p).value->size;
        nmz_layer_describe(l, desc, sizeof desc);
        APPEND("  [%d] %-40s out=%-5d params=%zu\n", i, desc, nmz_layer_output_size(l), count);
    }
    APPEND("loss: %s\n", nmz_loss_name(net->loss));
    APPEND("parameters: %zu\n", nmz_net_param_count(net));
#undef APPEND
    return len;
}
