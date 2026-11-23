# NeuroMZ v2

**An educational neural-network engine in C.** Build a network in one
command, train it, and then *look inside*: NeuroMZ can print every number of a
forward and backward pass — `z`, `a`, the loss, every `δ`, every gradient and
every weight update — next to the formula that produced it. That includes
convolutions: the receptive-field patch, the kernel, the products and their
sum, and which max-pool input "won".

* `libneuromz`: a small, dependency-free C11 library (libc + libm only, no globals)
* `neuromz`: a command-line program with a beginner path and an advanced path
* Dense, Conv2D, MaxPool2D, Flatten, Dropout layers; SGD+momentum and Adam;
  learning-rate schedules; input normalization stored with the model
* Embedding and RNN layers (backpropagation through time inside the layer,
  truncated BPTT with a carried hidden state, gradient clipping)
* Trains a small CNN on MNIST to 98.8% test accuracy in about 2 minutes on a 2017
  laptop CPU (single thread), and character RNNs that invent English and
  Arabic names or write Alice-in-Wonderland-flavoured text
* Every derivative is verified against numerical gradients; everything runs clean
  under AddressSanitizer and UndefinedBehaviorSanitizer.

NeuroMZ started in 2017 as a beginner's project. That code is preserved in
[`legacy/`](legacy/) (git tag `v1-legacy`); v2 is a rewrite. The design is in
[`docs/DESIGN.md`](docs/DESIGN.md).

---

## Quick start

```bash
make            # builds build/libneuromz.a and build/neuromz
make test       # unit tests, gradient checks, convergence tests (~1 s)
make sanitize   # the same tests + a CLI smoke test under ASan + UBSan
make examples   # trains XOR, XOR+AND and Iris
make data       # downloads MNIST (~11 MB) into data/ and verifies checksums
make data-text  # downloads "Alice's Adventures in Wonderland" (151 KB) into data/
make bench      # slow: seed sweeps, full MNIST runs, conv benchmark, character RNNs
```

Requirements: a C11 compiler and `make` (plus `curl` or `wget` and `gunzip`
for `make data`). Nothing else.

---

## The beginner path

Give the layer sizes; NeuroMZ picks everything else **and tells you why**:

```
$ ./build/neuromz new 2 4 1 -o xor.nmz
Input 2   2 value(s) per sample
  dense 2->4               tanh            [default] zero-centred and smooth; in small nets it trains more reliably than relu, ...
  output 4->1              sigmoid         [default] 1 output -> squashed into (0,1), a natural fit for 0/1 targets
    init layer 0 (dense)   xavier          [auto]    Glorot/Xavier: Var(W) = 2/(fan_in+fan_out), keeps tanh/sigmoid layers in their useful range
    init layer 2 (dense)   xavier          [auto]    Glorot/Xavier: Var(W) = 2/(fan_in+fan_out), keeps tanh/sigmoid layers in their useful range
  loss                     mse             [default] squared error is the textbook pairing for a sigmoid output: delta = (y-t)*f'(z)
  normalize                none            [default] inputs are used as they are
  optimizer                sgd             [default] gradient descent with momentum: the textbook method
  learning rate            0.5             [default] 0.5 suits squared error on sigmoid outputs, whose gradients are small (<= 0.25*error)
  momentum                 0.9             [default] 0.9 averages recent gradients: smoother steps and faster progress across plateaus
  schedule                 constant        [default] the same learning rate in every epoch
  batch size               32              [default] mini-batches of 32 (full batch when the dataset is smaller): ...
  max epochs               5000            [default] at most 5000 passes over the data, so training always ends
  target loss              0.001           [default] stop when the average 1/2*error^2 <= 0.001 (outputs within ~0.045 of their targets)
  shuffle                  on              [default] visit samples in a new random order every epoch
  seed                     42              [default] same seed => same initial weights and same shuffling, every run
Created a network with 17 parameters.
Saved to xor.nmz

$ ./build/neuromz -m xor.nmz train examples/xor.csv -o xor.nmz
Training on examples/xor.csv: 4 samples, 2 input(s), 1 output(s)
  sgd lr 0.5, momentum 0.9, batch 32, max epochs 5000, target loss 0.001, shuffle on
  (batch 32 > 4 samples: using the full dataset as one batch)
  epoch      1   loss 0.119855     accuracy  50.0%   lr 0.5           0.0s
Converged after 172 epoch(s): loss 0.000993896 <= target 0.001, accuracy 100.0% (on the training data), 0.0s
Saved to xor.nmz

$ ./build/neuromz -m xor.nmz predict 1 0
output:  0.953803
```

With two or more outputs, quick mode uses softmax + cross-entropy (one class
per sample). A CSV whose last column is a text label is one-hot encoded
automatically, and the class names are stored with the model:

```
$ ./build/neuromz new 4 8 3 -o iris.nmz
$ ./build/neuromz -m iris.nmz train examples/iris_train.csv -o iris.nmz
$ ./build/neuromz -m iris.nmz test examples/iris_test.csv
Tested on examples/iris_test.csv: 30 samples (0.0s)
  loss 0.0482473   accuracy 100.0%
  confusion matrix (rows = true class, columns = predicted)
                     setosa versicolor  virginica
          setosa         10          0          0
      versicolor          0         10          0
       virginica          0          0         10
```

Run `neuromz` with no arguments for an interactive shell (same commands; type
`help`). Scripts can be piped into it: `./build/neuromz < lesson.txt`.

## MNIST: a CNN from the command line

```
$ make data
$ ./build/neuromz new --preset mnist-cnn -o cnn.nmz
Preset mnist-cnn: small convolutional network for MNIST digits
Input 1x28x28   MNIST digits: 1 channel (grey levels 0..255), 28x28 pixels
  conv 1x28x28->8x26x26    8x 3x3 relu     [preset]  8 filters of 3x3: small kernels find strokes and edges; relu + He init
  maxpool 8x26x26->8x13x13 2x2             [preset]  2x2 max-pool keeps the strongest response and halves the map (26 -> 13)
  conv 8x13x13->16x11x11   16x 3x3 relu    [preset]  16 filters combine edges into parts of digits (13 -> 11)
  maxpool 16x11x11->16x5x5 2x2             [preset]  halves again (11 -> 5): 16x5x5 = 400 features
  flatten 16x5x5->400                      [auto]    dense layers need a flat vector: the maps are laid out in one row
  dense 400->64            relu            [preset]  64 hidden units combine the 400 features (flatten is inserted automatically)
  output 64->10            softmax         [preset]  10 digits -> class probabilities
  ...
  normalize                zscore          [preset]  pixels 0..255 -> mean 0, std 1 (fitted on the training set, stored in the model)
  optimizer                adam            [preset]  Adam adapts the step per weight: conv and dense weights get very different gradients
  schedule                 cosine          [preset]  cosine: large steps first, a smooth decay to 0 by the last epoch for fine-tuning
  max epochs               3               [preset]  3 passes: about a minute each on a laptop CPU, enough for > 98%
  ...
Created a network with 27562 parameters.

$ ./build/neuromz -m cnn.nmz train mnist -o cnn.nmz
Training on mnist: 60000 samples, 784 input(s), 10 output(s)
  adam lr 0.001 (beta1 0.9, beta2 0.999, eps 1e-08), cosine schedule to 0, batch 32, max epochs 3, target loss 0, shuffle on
  epoch      1   loss 0.0618284    accuracy  98.1%   lr 0.001        33.0s
  epoch      2   loss 0.0518255    accuracy  98.4%   lr 0.00075      76.3s
  epoch      3   loss 0.0259508    accuracy  99.2%   lr 0.00025     114.0s
Finished 3 epoch(s): loss 0.0259508, accuracy  99.2% (on the training data), 114.0s
Saved to cnn.nmz

$ ./build/neuromz -m cnn.nmz test mnist
Tested on mnist: 10000 samples (1.2s)
  loss 0.0367732   accuracy  98.8%
  confusion matrix (rows = true class, columns = predicted)
                      0      1      2      3      4      5      6      7      8      9
               0    974      0      0      0      2      0      1      1      2      0
               1      0   1128      3      0      0      0      1      1      2      0
               2      4      0   1019      0      2      0      0      6      1      0
               3      1      0      1    995      1      6      0      2      4      0
               4      0      0      0      0    975      0      1      0      0      6
               5      2      0      1      2      0    881      1      0      5      0
               6      4      1      0      0      2      4    945      0      2      0
               7      1      3      5      1      1      0      0   1013      1      3
               8      2      0      1      1      1      1      0      3    960      5
               9      1      4      1      0      7      3      0      4      2    987
```

On a terminal, training shows a live line
(`epoch 2/3 [#########.....] 61%  batch loss 0.0412  97.3s`) that is left out
when the output is redirected. `--limit N` trains or tests on the first N
images for quick experiments: `train mnist --limit 2000 --epochs 1`.

The preset `mnist-mlp` (784 → 128 relu → 10 softmax) is the fully connected
baseline. Look at what the network learned and at single digits (the empty
border rows of the digit are trimmed here):

```
$ ./build/neuromz -m cnn.nmz show sample 1
Sample 1 of mnist:test, label 7
  +--------------------------------------------------------+
  |            --##**++::..                                |
  |            %%@@@@@@@@@@################**::            |
  |            ::==--==**%%@@%%@@@@@@@@%%@@@@++            |
  |                      ..::  ::::::::..%%@@==            |
  |                                    --@@##..            |
  |                                  ..%%@@--              |
  |                                  ++@@%%::              |
  |                                ::@@@@::                |
  |                                ++@@##                  |
  |                                ##@@::                  |
  |                              ==@@**                    |
  |                            --@@%%::                    |
  |                          ..%%@@**                      |
  |                          ##@@%%..                      |
  |                        ..@@@@--                        |
  |                      ..%%@@==                          |
  |                      ++@@@@::                          |
  |                    ::@@@@@@::                          |
  |                    ==@@@@%%..                          |
  |                    ==@@##..                            |
  +--------------------------------------------------------+
  prediction: 7 (100.0%)
       0   0.00% 
       1   0.00% 
       2   0.00% 
       3   0.00% 
       4   0.00% 
       5   0.00% 
       6   0.00% 
       7 100.00% ########################################  <
       8   0.00% 
       9   0.00%

$ ./build/neuromz -m cnn.nmz show filters
[layer 0] conv2d: 8 filters of 3x3 over 1 channel(s)   (each filter scaled to its own min..max: ' ' lowest, '@' highest)
   f1         f2         f3         f4         f5         f6         f7         f8        
  |@@@@++|  |::--@@|  |@@++..|  |......|  |@@**%%|  |##@@##|  |##****|  |%%  @@|  
  |--##++|  |..  **|  |--::..|  |--==::|  |::**  |  |..%%--|  |  @@##|  |..@@::|  
  |==  %%|  |++**..|  |--    |  |**@@  |  |::  ++|  |  %%##|  |==@@%%|  |::%%##|  
  filter 1, channel 1, row by row:  [ 0.809  0.734  0.212]  [-0.104  0.561  0.170]  [ 0.046 -0.613  0.643]
  ...
```

## Text: a character RNN

A character model reads text one Unicode character at a time and learns to
predict the next one. Names work best as one item per line: the newline is
the "end of name" token, so the model can be asked for whole new names.

```
$ ./build/neuromz new --preset names -o names.nmz          # vocabulary from examples/names_en.txt
Preset names: character RNN that invents names (one name per line)
  embed 53 chars->16                       [preset]  each of the V characters gets a learned 16-number vector ...
  rnn Tx16->Tx128          tanh            [preset]  128 hidden units carry what was written so far: h_t = tanh(x_t W_xh + h_{t-1} W_hh + b)
  output Tx128->Tx53       softmax         [preset]  at every step: probabilities for the next character (incl. end of name)
  ...
  vocabulary               53 chars        [data]    from examples/names_en.txt: 36123 characters; stored in the model, ...
  sequence length          16              [preset]  16 steps cover two or three names: enough context for one name, cheap to train
  validation               10%             [preset]  the last part of the text is held out: train and validation loss are reported every epoch
  clip norm                5               [preset]  if the gradient's global L2 norm exceeds this, scale it down (exploding gradients)
  patience                 3               [preset]  stop when the validation loss has not improved for this many epochs; keep the best

$ ./build/neuromz -m names.nmz train examples/names_en.txt -o names.nmz
  epoch      1   loss 2.4637     (ppl  11.75)  val 2.4816    (ppl  11.96) *  accuracy  26.0%   lr 0.003       3.4s
  ...
  epoch     12   loss 1.9542     (ppl   7.06)  val 2.1124    (ppl   8.27) *  accuracy  32.6%   lr 0.003      31.2s
  ...
Early stop after 15 epoch(s): kept the weights of epoch 12 (best validation loss), loss 1.95418, ...
  validation (last 10% of the data): loss 2.1124, perplexity 8.268, accuracy 32.6%

$ ./build/neuromz -m names.nmz generate --count 10 --data examples/names_en.txt
   1. Dauri
   2. Corlerte
   3. Shanniea
   ...
   9. Tamika   (in the training data)
  10. Danyly
```

`*` marks a new best validation loss; with `--patience` training stops when
it has not improved for that many epochs and keeps the best weights.
`generate` options: `--prime TEXT`, `--length N`, `--temperature X` (below 1
safer, above 1 wilder), `--top-k K`, `--seed N` (same seed, same text),
`--count N` (distinct items), `--data FILE --new-only` (skip names that are
already in the training data). `show vocab` lists the characters with their
counts. The perplexity e^loss reads as "as unsure as a uniform choice among
that many characters".

**Arabic** works the same way, because tokens are code points, not bytes
(each Arabic letter is 2 bytes of UTF-8; invalid UTF-8 is read as U+FFFD and
reported):

```
$ ./build/neuromz new --preset names --data examples/names_ar.txt -o ar.nmz
$ ./build/neuromz -m ar.nmz train examples/names_ar.txt -o ar.nmz
$ ./build/neuromz -m ar.nmz generate --count 10
```

> Terminals differ in how they display right-to-left text: some shape and
> order Arabic correctly, others show isolated letters or reverse the order.
> NeuroMZ prints the characters in logical order as UTF-8; where the trace
> prints one character per column you see the isolated letter forms the
> model actually works with. `--ascii` prints code points (`U+0645`) instead.

**Running text** uses the hidden state across batches (truncated
backpropagation through time):

```
$ make data-text
$ ./build/neuromz new --preset char-rnn -o alice.nmz         # embed:32 rnn:128 vocab, 64-step windows, state carried
$ ./build/neuromz -m alice.nmz train data/alice.txt -o alice.nmz
$ ./build/neuromz -m alice.nmz generate --prime "Alice " --length 300 --temperature 0.7
```

Advanced text models use the same spec style: `new text embed:32 rnn:256
vocab --data FILE --seq-len 100 --carry --clip 1 --patience 5 --val 0.05`.

### Trace a sequence: recurrence and BPTT, step by step

`trace --text TEXT` (or `--sample N --data FILE`) runs one sequence through
the net and prints it unrolled through time — for one hidden unit
(`--unit J`; by default the one that changes most) every term of
h_t = tanh(x_t·W_xh + h_{t−1}·W_hh + b), and the next-character
probabilities:

```
$ ./build/neuromz -m names.nmz trace --text "Christopher"
RNN UNROLLED THROUGH TIME   the same weights at every step; h carries the past forward
  h_t = tanh(x_t·W_xh + h_{t-1}·W_hh + b)      shown for hidden unit j = 72 of 128 (the unit that changes most here; ...)
  x_t = e_t (from the embedding), h_{-1} = 0

  t=0   input C    h_t[72] = tanh(1.3162 + 0.0000 + 0.1901) = tanh(1.5063) = 0.9063
         x_t·W_xh[:,j]   h_{t-1}·W_hh[:,j]   b[j]
        next: a     32.6% ██████████
              h     18.9% ██████   <- target
              l     17.1% █████
              o     13.5% ████
              e      4.4% █
        target h: p = 0.1892, loss -log p = 1.6649

  t=1   input h    h_t[72] = tanh(0.9507 + 0.4411 + 0.1901) = tanh(1.5819) = 0.9189
  ...
HIDDEN STATE OVER TIME   rows = hidden units, columns = time steps
  ' ' = -1 ... '+' = 0 ... '@' = +1   (tanh range)
         Christophe
      1 %@@@@@@@@@
      2 -@@@%%+:#*
      3 %@@@@@@%%@
  ...
```

The backward part shows backpropagation through time — how the gradient for
h_t is assembled from the output at t and from step t+1, shrunk by tanh′, and
passed back — and then the key picture: the gradient of the **last**
character's loss alone, followed back through time:

```
VANISHING / EXPLODING GRADIENTS   the loss of the LAST character only, followed back in time
  t=9    e         0.1557           ████████████████████████████████████
  t=8    h         0.1597   x1.026  ████████████████████████████████████
  t=7    p         0.1374   x0.860  ██████████████████████████████████
  t=6    o         0.1107   x0.806  ██████████████████████████████
  ...
  t=1    h         0.0368   x0.874  █████████████
  t=0    C         0.0264   x0.718  ████████

  bars: log scale over 1.0 decade(s). On average each step back multiplies the gradient by 0.821;
  after 9 steps only 0.17 of it is left: a VANISHING gradient. A vanilla RNN therefore learns
  short-range patterns easily and long-range ones hardly at all - the problem LSTMs fix
  with an additive cell state.
```

On a 64-character window of Alice the same view ends with "after 47 steps
only 0.00018 of it is left". The trace then shows the gradient clipping
decision (`global gradient norm 4.53 <= clip 5 -> unchanged`) and the Adam
update for every weight. `step` applies it.

## The advanced path

Every layer is one argument, every training knob is an option. What you set
is marked `[yours]`; the rest is still explained.

```bash
./build/neuromz new 4 16:relu:he 8:leaky_relu=0.05 3:softmax \
    --loss ce --lr 0.01 --momentum 0.9 --batch 16 --epochs 2000 --target-loss 0.05 --seed 7 -o iris.nmz
./build/neuromz new 1x28x28 conv:16:5:relu:pad=same pool:2 conv:32:3:relu:stride=2 dropout:0.25 128:relu 10:softmax \
    --optimizer adam --lr 0.002 --schedule step --step-size 2 --gamma 0.5 --normalize minmax --epochs 6 -o big.nmz
```

| | choices |
|---|---|
| input | a size (`784`) or an image shape (`1x28x28`, `28x28`) |
| dense | `N[:ACT][:INIT]` (or `dense:N:...`) |
| conv | `conv:FILTERS:K[:ACT][:INIT][:stride=S][:pad=P\|same]`, `K` = `3` or `3x5` |
| pooling, other | `pool:K[:stride=S]`, `flatten` (automatic before dense), `dropout:RATE` |
| activations | `linear`, `sigmoid`, `tanh`, `relu`, `leaky_relu[=ALPHA]`, `softmax` |
| init | `auto` (He before relu and for conv, Xavier otherwise), `xavier`, `he` |
| loss | `mse` (½·squared error), `cross_entropy`/`ce` (fused with a softmax output) |
| optimizer | `--optimizer sgd` (`--momentum`) or `adam` (`--beta1 --beta2 --eps`), `--lr` |
| schedule | `--schedule constant\|step\|cosine`, `--step-size N --gamma X`, `--min-lr X` |
| data | `--normalize none\|minmax\|zscore` (fitted at the first `train`, saved) |
| text | input `text` with `--data FILE` (vocabulary), `embed:D`, `rnn:H`, `vocab[:ACT]` (one output per character); `--items`/`--no-items` |
| training | `--batch`, `--epochs`, `--target-loss`, `--no-shuffle`, `--seed` |
| validation, stopping | `--val FRACTION` (the last part of the data is held out), `--patience N` (early stopping, keeps the best weights) |
| recurrent | `--seq-len N` (BPTT window), `--carry`/`--no-carry` (hidden state carried between batches), `--clip NORM` |

* `train DATA [options]` overrides the stored settings for one run;
  `set --optimizer adam --lr 0.001 ...` changes the settings saved with the model.
  `DATA` is a CSV file, `mnist` (training set for `train`, test set for
  `test`), `mnist:train`/`mnist:test`, or an IDX images file.
* `show` / `show weights` / `show filters` / `show sample N [--data SPEC]`,
  `test DATA`, `predict X... | --sample N`, `reinit [SEED]`,
  `save [FILE] [--no-optimizer]`, `load FILE`, `clear`, `help [COMMAND]`.
* One-shot use composes in shell scripts: `-m FILE` loads a model before the
  command, `-o FILE` saves it afterwards. `--ascii` replaces Greek letters.

Each epoch shuffles the data (seeded), runs every mini-batch — forward, then
**all** gradients, then the optimizer step — and finally measures the loss
over the whole training set; training stops when that loss reaches the target
or at the epoch limit. Model files keep the optimizer state (SGD velocity or
Adam's moments and step count), so `train` continues exactly where it stopped.

## The trace command: see the math

`trace` runs **one sample** forward and backward and prints every value with
its formula. Nothing changes; `step` does the same as a real training step
(dropout active) and applies the update.

```
$ ./build/neuromz new 2 2 1 -o tiny.nmz
$ ./build/neuromz -m tiny.nmz trace 1 0 --target 1
FORWARD PASS   input -> output
[layer 0] dense 2 → 2   (forward)
  z_j = b_j + Σ_i x_i·W_ij        x = this layer's input, W_ij = weight from input i to unit j
  z_1 = 0.0000 + 1.0000·1.1884 + 0.0000·0.1973 = 1.1884
  z_2 = 0.0000 + 1.0000·1.1158 + 0.0000·(-0.2618) = 1.1158
[layer 1] activation tanh   (forward)
  a_1 = tanh(1.1884) = 0.8301
  ...
LOSS  (mse)
  L = ½ Σ_k (y_k − t_k)²       y = network output, t = target
  (y_1 − t_1)² = (0.6656 − 1.0000)² = 0.1118
  L = ½ · 0.1118 = 0.0559
BACKWARD PASS   output -> input: every gradient is computed first
[layer 3] activation sigmoid   (backward)
  δ_j = ∂L/∂a_j · f'(z_j)        f'(z) = σ(z)·(1 − σ(z)) = a·(1 − a)
  f'(z_1) = 0.6656·(1 − 0.6656) = 0.2226
  δ_1 = -0.3344 · 0.2226 = -0.0744
[layer 2] dense 2 → 1   (backward)
  ∂L/∂W_ij = x_i · δ_j        e.g. ∂L/∂W_11 = 0.8301 · (-0.0744) = -0.0618
  ∂L/∂x_i = Σ_j W_ij · δ_j        (passed down: it is ∂L/∂a of the layer below)
  ...
WEIGHT UPDATE (preview only: not applied; use `step`)   η = 0.5, μ = 0.9
  v ← μ·v − η·∂L/∂w        w ← w + v
[layer 2] dense W  (2 values)
       i,j        w old        ∂L/∂w        v old    v new = Δ        w new
       1,1       0.8229      -0.0618       0.0000       0.0309       0.8538
```

For a CNN, `trace --sample N` takes a digit from the test set (its label is the
target). Each spatial layer is shown at one output position — by default where
the filter responds most, or `--at FILTER,ROW,COL` — and the backward pass
returns to the same position:

```
$ ./build/neuromz -m cnn.nmz trace --sample 1
[layer 0] conv2d 1x28x28 → 8x26x26   (forward)   8 filters 3x3, stride 1, pad 0
  z[f,y,x] = b[f] + Σ_c Σ_ky Σ_kx x[c, y·s+ky−p, x·s+kx−p] · W[f,c,ky,kx]
  in focus: filter 1, output position y=9, x=18, where this filter responds most  (choose another with --at F,Y,X)
  channel 1: rows 9..11, cols 18..20 of the input
    input patch x                 kernel W                      x · W (element-wise)
        2.096    2.096    2.096       0.809    0.734    0.212       1.696    1.538    0.445
        2.758    2.491    2.809      -0.104    0.561    0.170      -0.287    1.397    0.478
        0.327   -0.157    2.580       0.046   -0.613    0.643       0.015    0.096    1.660
    Σ of the products = 7.0384
  z = 7.0384 (all 1 channel) + b = 7.0384 + -0.0217 = 7.0168

[layer 1] activation relu   (forward)   applied to all 5408 values of the 8x26x26 maps
  in focus [c=1,y=9,x=18]: a = relu(7.0168) = 7.0168      (3110 of 5408 values are 0 after relu)

[layer 2] maxpool2d 8x26x26 → 8x13x13   (forward)   window 2x2, stride 2
  y = max of the window; the winner's position is remembered for the backward pass
  in focus: channel 1, output y=5, x=9  -> input rows 9..10, cols 17..18
         5.055  [   7.017]
         5.563      6.342 
  winner [ ] at row 9, col 18  ->  y = 7.0168

  ...
[layer 0] conv2d   (backward)   in focus: filter 1, y=9, x=18, δ = ∂L/∂z = 1.649e-06
  this position's share of ∂L/∂W[f] is δ · patch (channel 1 shown):
    input patch x                 δ · x                       
        2.096    2.096    2.096     3.5e-06  3.5e-06  3.5e-06
        2.758    2.491    2.809     4.5e-06  4.1e-06  4.6e-06
        0.327   -0.157    2.580     5.4e-07 -2.6e-07  4.3e-06
  and it sends δ · W[f] back to the same patch of the input (∂L/∂x, channel 1):
    kernel W                      δ · W                       
        0.809    0.734    0.212     1.3e-06  1.2e-06  3.5e-07
       -0.104    0.561    0.170    -1.7e-07  9.2e-07  2.8e-07
        0.046   -0.613    0.643     7.6e-08 -1.0e-06  1.1e-06
  summed over all 676 output positions:
    ∂L/∂b[1] = Σ δ = -7.911e-06
    ∂L/∂W[1] (channel 1):
     -5.9e-05 -4.3e-05 -6.3e-06
     -3.2e-05  4.9e-07  2.9e-05
      2.7e-05  6.7e-05  6.1e-05
    ∂L/∂x: 606 of 784 input values received a gradient (the input image: nothing to update there)
  ...
```

* `--target` can be a class name: `trace 5.1 3.5 1.4 0.2 --target setosa`.
* With softmax + cross-entropy the trace shows the combined gradient `δ = p − t`.
* With Adam, the update table shows `m`, `v`, `m̂`, `v̂` and `Δ` for every weight.
* `--full` prints every row and channel; `--json FILE` writes the raw trace
  events as JSON lines (one object per event, with every tensor), ready for a
  plotting script or a future GUI.
* `gradcheck` compares every backprop gradient with `(L(w+ε) − L(w−ε)) / 2ε` —
  a way to *verify* the math. Checks where a relu or a max-pool switches within
  ±ε are reported separately: the loss has a kink there, so the difference
  quotient is not a derivative.

## Using the library from C

```c
#include "neuromz.h"

nmz_net *net;
nmz_dataset *train;
nmz_train_result res;
nmz_train_opts opts = nmz_train_defaults();
int shape[3] = {1, 28, 28};

nmz_dataset_load_idx(&train, "data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte", 0);
nmz_net_create(&net, 3, shape, /*seed*/ 42);
nmz_net_add(net, nmz_layer_conv2d(8, 3, 3, /*stride*/ 1, /*pad*/ 0, NMZ_INIT_AUTO));
nmz_net_add(net, nmz_layer_activation(NMZ_ACT_RELU, 0));
nmz_net_add(net, nmz_layer_maxpool2d(2, 0));
nmz_net_add(net, nmz_layer_flatten());
nmz_net_add_dense(net, 10, NMZ_ACT_SOFTMAX, NMZ_INIT_AUTO);
nmz_net_set_loss(net, NMZ_LOSS_CROSS_ENTROPY);
nmz_net_set_normalization(net, NMZ_NORM_ZSCORE);
opts.optimizer = NMZ_OPT_ADAM;
opts.learning_rate = 0.001f;
opts.max_epochs = 2;
opts.target_loss = 0;
if (nmz_net_train(net, train, &opts, &res) != NMZ_OK)
    fprintf(stderr, "%s\n", nmz_last_error());
nmz_net_save(net, "small_cnn.nmz");
```

Build against `build/libneuromz.a` with `-Iinclude -lm`. The beginner API
(`nmz_net_create_quick`), forward/backward/update by hand, trace callbacks and
the trace recorder are shown in [`examples/api_xor.c`](examples/api_xor.c); all
functions are documented in [`include/neuromz.h`](include/neuromz.h). New layer
types implement the vtable in [`include/neuromz_layer.h`](include/neuromz_layer.h).

## Results

**MNIST** (`sh examples/benchmark_mnist.sh`): 60 000 training images, the
10 000 test images, seed 42, one thread on an Intel Core i7-7500U (2017
laptop, 2.7 GHz) built with `-O3`. Times include the whole-training-set loss
measured after every epoch.

| model | parameters | epochs | test accuracy | training time |
|---|---|---|---|---|
| `--preset mnist-cnn` (conv 8 → pool → conv 16 → pool → 64 → 10), Adam, cosine | 27 562 | 3 | **98.8%** | 114 s (38 s/epoch) |
| `--preset mnist-mlp` (784 → 128 → 10), Adam, cosine | 101 770 | 5 | **97.0%** | 100 s (20 s/epoch) |

Convolution is implemented twice: naive loops that mirror the formula, and
im2col (receptive fields copied into a matrix, then contiguous matrix
products; the default). Both pass the same gradient checks and agree to
1e-5. One epoch of the CNN on 10 000 images (`build/bench_conv`):

| conv implementation | 1 epoch, 10 000 images |
|---|---|
| naive loops | 10.3 s |
| im2col (default) | 5.0 s (2.1× faster; 2.1–2.4× over repeated runs) |

With `-O2` instead of `-O3` im2col was only ~1.1× faster: GCC 12's `-O2` does
not vectorise the inner loops, so `-O3` is now the default (DESIGN decision 46).

**Character RNNs** (`sh examples/benchmark_text.sh`): the presets, seed 42,
same machine; the validation split is the last 10 % of each text, early
stopping with patience 3.

| model | data | parameters | stopped / best epoch | validation loss (nats/char) | perplexity | next-char accuracy | training time |
|---|---|---|---|---|---|---|---|
| `--preset names` (embed 16, rnn 128) | 5163 English names, 36 k chars | 26 245 | 15 / 12 | 2.112 | 8.27 | 32.6% | 31 s |
| `--preset names --data examples/names_ar.txt` | 504 Arabic names, 2.7 k chars | 23 925 | 18 / 15 | 2.655 | 14.22 | 28.9% | 3 s |
| `--preset char-rnn` (embed 32, rnn 128, carried state) | Alice, 145 k chars | 32 683 | 16 / 13 | 1.634 | 5.12 | 53.9% | 155 s |

Generated with `--seed 1` (unedited):

* **English names**: Dauri, Corlerte, Shanniea, Rananda, Elexie, Kathalle,
  Sasely, Irbin, Tamika *(in the training data)*, Danyly.
* **Arabic names**: رامى، حهيد، نسدة، ماسة، سححير، طالح، ناوب، ششام، نؤة، راريءف
* **Alice** (`--prime "Alice " --length 300 --temperature 0.7`):

  > Alice said to Alice, as she had not quite all the ever, and said this was
  > the Queen’s be a throat for a serpeation and said, “Soupan. “Fithouted the
  > other she went on, “treat is again, so she had see in a trying and more
  > that she said would be in say?” the Hatter upoomester.

Honest assessment: the **English names** are pronounceable and mostly look
like plausible names (Shanniea, Rananda, Elexie, Kathalle); a few are odd
(Corlerte). The **Arabic names** are mixed: ماسة is a real given name and رامى
a common spelling variant of رامي; نسدة and ناوب follow Arabic letter
patterns but are not names; طالح is a real word but not a name (it is the opposite of the name
صالح, which the model has seen); حهيد, ششام, راريءف break Arabic spelling
habits. That is about what 504 training names allow — the model starts to
overfit after ~15 epochs. The **Alice** text has the book's
style — real words, quotes, punctuation, "said the Hatter", line breaks — but
no grammar or meaning beyond a few words: what a 128-unit vanilla RNN
trained for 2.5 minutes on 145 k characters can do. The vanishing gradient
shown above is the main limit; an LSTM is the next step.

**Small datasets** (`sh examples/benchmark.sh ./build/neuromz 20`, seeds 1–20,
defaults unless shown). Iris: 120 training rows, accuracy on the 30 held-out
rows of `examples/iris_test.csv`.

| example (seeds 1..20) | converged | median epochs | mean test accuracy |
|---|---|---|---|
| XOR, quick `new 2 4 1` | 20/20 | 176 | – |
| XOR, minimal `new 2 2 1` | 10/20 | 249 | – |
| XOR+AND, `new 2 4 2:sigmoid --loss mse` | 20/20 | 202 | – |
| Iris, quick `new 4 8 3` | 20/20 | 175 | 99.1% |
| Iris, `new 4 16:relu:he 8:leaky_relu=0.05 3:softmax`, lr 0.01, batch 16 | 20/20 | 105 | 98.8% |

The minimal 2-2-1 XOR network fails for half the seeds: with only two hidden
units the loss surface has well-known local minima. Two extra hidden units make
it reliable — a good thing to try with `trace` and `reinit`.

"Converged" means the whole-training-set loss reached the target (0.001 for
MSE, 0.07 for cross-entropy) before 5000 epochs.

## Tests

`make test` runs 57 tests (~105 000 checks) in a few seconds:

* **numerical gradient checks** for Dense, every activation (incl. the softmax
  Jacobian), MSE, fused softmax+cross-entropy, a deep mixed network, batch
  size 1, rank-2 samples; **Conv2D** over 7 stride/padding/kernel combinations
  (both implementations), **MaxPool2D** (incl. overlapping windows), Flatten,
  a whole CNN, **Dropout** in evaluation and training mode, and normalization
  (gradients w.r.t. the raw input);
* hand-computed values: dense/SGD/momentum, a 2×2 convolution, max-pool
  winners and gradient routing, **Adam** over three steps;
* **XOR over 20 seeds** (SGD and Adam with schedules), **XOR+AND** with two
  outputs (the case v1 got wrong), Iris-style CSV loading;
* determinism, resuming Adam after save/load (bit-identical), early stop, NaN
  detection, seeded dropout masks;
* model files: round trip, little-endian layout, checksum/truncation, **a real
  format-2 file from phase 1 still loads**, unknown sections are skipped;
* **sequences**: gradient checks for the RNN over 5 steps (incl. input
  gradients), embed → rnn → per-step dense → softmax/CE, a loss on the last
  step only (so earlier gradients can only arrive through h_{t−1}), and the
  RNN with a carried state h₀ ≠ 0; the RNN against hand-computed BPTT; the
  sparse embedding gradient with exact (dense) Adam; gradient clipping;
  learning a repeating text with and without carried state; validation
  splits for text and tabular data; early stopping restores the best epoch;
* **text**: UTF-8 with Arabic and 4-byte characters and every class of
  invalid sequence, vocabulary order/counts, line-per-item normalisation,
  the vocabulary round trip through save/load, deterministic generation
  (seed, greedy top-1, temperature), an Arabic model generating valid UTF-8,
  and **a format-3 file from phase 2 (without the new sections) still loads**;
* CSV and IDX parsing errors; every API entry point with `NULL`/bad input.

`make sanitize` runs all of that plus `tests/cli_smoke.sh` (130 CLI cases,
including image layers on a generated dataset, English and Arabic name
models, generation determinism, traces in UTF-8 and ASCII, invalid UTF-8,
no-network commands, malformed input, a 300 KB line and random binary input,
and MNIST and Alice when present) under
ASan + UBSan with leak detection. The slow benchmarks are in `make bench`.

## Layout

```
include/neuromz.h        public API          src/        libneuromz (portable C11)
include/neuromz_layer.h  layer-author API    cli/        the neuromz program
docs/DESIGN.md           design + decisions  tests/      test suite + CLI smoke test
examples/                data, C demo, benchmarks        scripts/    MNIST / text download
legacy/                  v1 (2017–2020)      data/       MNIST, Alice (downloaded, gitignored)
```

Dataset sources and licenses: [`examples/DATASETS.md`](examples/DATASETS.md).

Model files (`.nmz`) use a portable little-endian format with a magic number,
version, optional sections (normalization, optimizer state, vocabulary,
text-training settings) and CRC-32 (see DESIGN §7).

## Roadmap

Done: dense networks (phase 1); convolution, pooling, dropout, Adam,
schedules, normalization and MNIST (phase 2); text, embeddings, a vanilla RNN
with truncated BPTT, clipping, validation/early stopping and generation
(phase 3). Next: an LSTM layer — DESIGN §3.1 describes how it plugs in with no
engine change — then a tiny Transformer trained on text, and a GUI that
consumes the trace events.
