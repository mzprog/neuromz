# NeuroMZ v2 — build with any C11 compiler; depends only on libc and libm.
#
#   make            library + CLI (same as `make lib cli`)
#   make lib        build/libneuromz.a
#   make cli        build/neuromz
#   make test       build and run the test suite
#   make sanitize   build everything with ASan + UBSan, run tests + CLI smoke test
#   make examples   train the bundled examples (XOR, XOR+AND, Iris)
#   make data       download MNIST into data/ and verify checksums
#   make data-text  download the texts (Alice; Kalila wa-Dimna in Arabic) into data/, checksum-verified
#   make bench      slow benchmarks (seeds sweep, MNIST, RNN vs LSTM vs GRU); not part of `make test`
#   make clean

CC      ?= cc
AR      ?= ar
# make OMP=1 ...: multi-threaded matrix products and attention heads (OpenMP);
# builds into build/omp so it never mixes with the default single-threaded objects
OMP     ?= 0
WARN    := -Wall -Wextra -Werror -pedantic -Wshadow -Wstrict-prototypes -Wmissing-prototypes
CFLAGS  ?= -O3 -g
ifeq ($(OMP),1)
OMPFLAGS := -fopenmp
BUILD   ?= build/omp
else
OMPFLAGS :=
BUILD   ?= build
endif
ALL_CFLAGS := -std=c11 $(WARN) $(CFLAGS) $(OMPFLAGS) -Iinclude
LDLIBS  := -lm

LIB_SRC := src/error.c src/tensor.c src/rng.c src/activation.c src/dense.c src/loss.c \
           src/net.c src/train.c src/trace.c src/gradcheck.c src/io.c src/dataset.c \
           src/defaults.c src/conv2d.c src/shape_layers.c src/normalize.c \
           src/text.c src/recurrent.c src/lstm.c src/gru.c src/generate.c \
           src/matmul.c src/attention.c src/transformer.c
CLI_SRC := cli/main.c cli/commands.c cli/cmd_new.c cli/util.c cli/print.c cli/print_conv.c cli/print_seq.c cli/print_attn.c cli/cmd_seq.c cli/line.c
TEST_SRC := tests/test_main.c tests/test_core.c tests/test_math.c tests/test_train.c \
            tests/test_io.c tests/test_layers.c tests/test_optim.c \
            tests/test_text.c tests/test_seq.c tests/test_gated.c tests/test_transformer.c

LIB_OBJ  := $(LIB_SRC:%.c=$(BUILD)/%.o)
CLI_OBJ  := $(CLI_SRC:%.c=$(BUILD)/%.o)
TEST_OBJ := $(TEST_SRC:%.c=$(BUILD)/%.o)
LIB      := $(BUILD)/libneuromz.a
CLI      := $(BUILD)/neuromz
TESTBIN  := $(BUILD)/test_neuromz

.PHONY: all lib cli test sanitize examples data data-text bench clean

all: lib cli

lib: $(LIB)
cli: $(CLI)

$(LIB): $(LIB_OBJ)
	$(AR) rcs $@ $^

$(CLI): $(CLI_OBJ) $(LIB)
	$(CC) $(ALL_CFLAGS) -o $@ $(CLI_OBJ) $(LIB) $(LDLIBS)

$(TESTBIN): $(TEST_OBJ) $(LIB)
	$(CC) $(ALL_CFLAGS) -o $@ $(TEST_OBJ) $(LIB) $(LDLIBS)

$(BUILD)/%.o: %.c include/neuromz.h include/neuromz_layer.h src/internal.h
	@mkdir -p $(dir $@)
	$(CC) $(ALL_CFLAGS) -MMD -MP -c $< -o $@

test: $(TESTBIN)
	./$(TESTBIN)

# Sanitizer build lives in its own directory so it never mixes with normal objects.
SAN_FLAGS := -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all
SAN_DIR := build/san$(if $(filter 1,$(OMP)),-omp,)
sanitize:
	$(MAKE) BUILD=$(SAN_DIR) CFLAGS="$(SAN_FLAGS)" $(SAN_DIR)/test_neuromz $(SAN_DIR)/neuromz
	ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=print_stacktrace=1 ./$(SAN_DIR)/test_neuromz
	ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=print_stacktrace=1 sh tests/cli_smoke.sh ./$(SAN_DIR)/neuromz

$(BUILD)/api_xor: examples/api_xor.c $(LIB)
	$(CC) $(ALL_CFLAGS) -o $@ $< $(LIB) $(LDLIBS)

examples: $(CLI) $(BUILD)/api_xor
	./$(BUILD)/api_xor
	sh examples/run_examples.sh ./$(CLI)

data:
	sh scripts/get_mnist.sh data

data-text:
	sh scripts/get_text.sh data

$(BUILD)/bench_conv: examples/bench_conv.c $(LIB)
	$(CC) $(ALL_CFLAGS) -o $@ $< $(LIB) $(LDLIBS)

$(BUILD)/bench_matmul: examples/bench_matmul.c $(LIB)
	$(CC) $(ALL_CFLAGS) -o $@ $< $(LIB) $(LDLIBS)

bench: $(CLI) $(BUILD)/bench_conv
	sh examples/benchmark.sh ./$(CLI)
	sh examples/benchmark_mnist.sh ./$(CLI)
	sh examples/benchmark_gated.sh ./$(CLI)

clean:
	rm -rf build

-include $(LIB_OBJ:.o=.d) $(CLI_OBJ:.o=.d) $(TEST_OBJ:.o=.d)
