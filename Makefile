# NeuroMZ v2 — build with any C11 compiler; depends only on libc and libm.
#
#   make            library + CLI (same as `make lib cli`)
#   make lib        build/libneuromz.a
#   make cli        build/neuromz
#   make test       build and run the test suite
#   make sanitize   build everything with ASan + UBSan, run tests + CLI smoke test
#   make examples   train the bundled examples (XOR, XOR+AND, Iris)
#   make clean

CC      ?= cc
AR      ?= ar
WARN    := -Wall -Wextra -Werror -pedantic -Wshadow -Wstrict-prototypes -Wmissing-prototypes
CFLAGS  ?= -O2 -g
ALL_CFLAGS := -std=c11 $(WARN) $(CFLAGS) -Iinclude
LDLIBS  := -lm

BUILD   ?= build

LIB_SRC := src/error.c src/tensor.c src/rng.c src/activation.c src/dense.c src/loss.c \
           src/net.c src/train.c src/trace.c src/gradcheck.c src/io.c src/dataset.c \
           src/defaults.c
CLI_SRC := cli/main.c cli/commands.c cli/print.c cli/line.c
TEST_SRC := tests/test_main.c tests/test_core.c tests/test_math.c tests/test_train.c \
            tests/test_io.c

LIB_OBJ  := $(LIB_SRC:%.c=$(BUILD)/%.o)
CLI_OBJ  := $(CLI_SRC:%.c=$(BUILD)/%.o)
TEST_OBJ := $(TEST_SRC:%.c=$(BUILD)/%.o)
LIB      := $(BUILD)/libneuromz.a
CLI      := $(BUILD)/neuromz
TESTBIN  := $(BUILD)/test_neuromz

.PHONY: all lib cli test sanitize examples clean

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
sanitize:
	$(MAKE) BUILD=build/san CFLAGS="$(SAN_FLAGS)" build/san/test_neuromz build/san/neuromz
	ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=print_stacktrace=1 ./build/san/test_neuromz
	ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=print_stacktrace=1 sh tests/cli_smoke.sh ./build/san/neuromz

$(BUILD)/api_xor: examples/api_xor.c $(LIB)
	$(CC) $(ALL_CFLAGS) -o $@ $< $(LIB) $(LDLIBS)

examples: $(CLI) $(BUILD)/api_xor
	./$(BUILD)/api_xor
	sh examples/run_examples.sh ./$(CLI)

clean:
	rm -rf build

-include $(LIB_OBJ:.o=.d) $(CLI_OBJ:.o=.d) $(TEST_OBJ:.o=.d)
