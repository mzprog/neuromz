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
* Embedding, RNN, **LSTM** and **GRU** layers (backpropagation through time
  inside the layer, truncated BPTT with a carried state, stacking, gradient
  clipping, orthogonal initialisation)
* `trace` shows every gate of an LSTM/GRU at every time step with real
  numbers, and `compare` puts side by side how much of a gradient survives
  N steps back in an RNN, an LSTM (along h and along the cell state) and a
  GRU — *why gates fix vanishing gradients*, measured
* A decoder-only **Transformer** (GPT-style: position encodings, LayerNorm,
  causal multi-head attention, GELU feed-forward, pre-LN residual blocks,
  weight tying) with AdamW, warm-up + cosine, and a KV cache for generation;
  `trace` shows attention at one position with real numbers — the query,
  every key, the scores, the mask, the softmax weights as bars labelled
  with the characters — plus heatmaps, what each head attends to (measured)
  and the residual stream
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
make data-text  # downloads "Alice's Adventures in Wonderland" and Kalila wa-Dimna into data/
make bench      # slow: seed sweeps, MNIST, conv and matmul benchmarks, RNN/LSTM/GRU/Transformer
make OMP=1      # optional: the same, multi-threaded with OpenMP (into build/omp; same results)
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
`--count N` (distinct items, generated as one batch: item k is exactly
what `--seed` + k gives alone), `--data FILE --new-only` (skip names that are
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
vocab --data FILE --seq-len 100 --carry --clip 1 --patience 5 --val 0.05`;
`train FILE --val-file OTHER` validates on a separate file instead of the
last part of FILE.

### Trace a sequence: recurrence and BPTT, step by step

`trace --text TEXT` (or `--sample N --data FILE`) runs one sequence through
the net and prints it unrolled through time — for one hidden unit
(`--unit J`; by default the one that changes most) every term of
h_t = tanh(x_t·W_xh + h_{t−1}·W_hh + b), and the next-character
probabilities:

```
$ ./build/neuromz -m names.nmz trace --text "Christopher"
RNN UNROLLED THROUGH TIME   the same weights at every step; h carries the past forward
  h_t = tanh(x_t·W_xh + h_{t-1}·W_hh + b)      shown for hidden unit j = 6 of 128 (the unit that changes most here; ...)
  x_t = the layer's input at step t, h_{-1} = 0 (or the carried state)

  t=0   input C    h_t[6] = tanh(-1.4276 + 0.0000 + -0.1032) = tanh(-1.5307) = -0.9105
        x_t·W_xh[:,j]   h_{t-1}·W_hh[:,j]   b[j]
        next: h     33.4% ██████████   <- target
              a     22.9% ███████
              o     10.5% ███
              l     10.3% ███
              i      3.3% █
        target h: p = 0.3338, loss -log p = 1.0971

  t=1   input h    h_t[6] = tanh(-1.0076 + -0.1262 + -0.1032) = tanh(-1.2370) = -0.8446
  ...
HIDDEN STATE OVER TIME   rows = hidden units, columns = time steps
  ' ' = -1 ... '+' = 0 ... '@' = +1   (tanh range)
        Christophe
      1 *%@: .+##+
      2 #-   :% +:
      3 #   =:=@#:
  ...
```

The backward part shows backpropagation through time — how the gradient for
h_t is assembled from the output at t and from step t+1, shrunk by tanh′, and
passed back — and then the key picture: the gradient of the **last**
character's loss alone, followed back through time:

```
VANISHING / EXPLODING GRADIENTS   the loss of the LAST character only, followed back in time
  step   char  |dL/dh_t|   x/step  (log scale)
  t=9    e        0.1494           ██████████████████████████████
  t=8    h        0.1246  x0.834   ████████████████████████████
  t=7    p        0.0975  x0.782   █████████████████████████
  t=6    o        0.0655  x0.672   ████████████████████
  ...
  t=1    h        0.0155  x0.661   ██
  t=0    C        0.0127  x0.822

  bars: log scale over 1.1 decade(s). Each step back multiplies |dL/dh| by 0.761 on average;
  after 9 steps 0.085 of the h-gradient is left.
  That is a VANISHING gradient: a vanilla RNN learns short-range patterns easily and long-range
  ones hardly at all - the problem the LSTM's additive cell state fixes (try `compare`).
```

On a 64-character window of Alice only about 1e-6 of it is left after 63
steps (see `compare` below). The trace then shows the gradient clipping
decision (`global gradient norm 4.53 <= clip 5 -> unchanged`) and the Adam
update for every weight. `step` applies it.

## RNN vs LSTM: why gates matter

A vanilla RNN squeezes the whole past into h_t = tanh(x_t·W_xh + h_{t−1}·W_hh
+ b). Learning "what happened 40 characters ago" needs a gradient to travel
40 steps back, and every step multiplies it by W_hhᵀ and by tanh′ ≤ 1: it
shrinks geometrically (the trace above: ×0.76 per step). An **LSTM** adds a
second state, the *cell* c, that is updated by addition and controlled by
gates — numbers between 0 and 1 computed from x_t and h_{t−1}:

```
i, f, o = σ(x_t·W_x + h_{t−1}·W_h + b)   input, forget, output gates
g       = tanh(x_t·W_x + h_{t−1}·W_h + b) candidate
c_t     = f ⊙ c_{t−1} + i ⊙ g             keep part of the old memory, write some new
h_t     = o ⊙ tanh(c_t)                   show part of it
```

Backwards, ∂L/∂c_{t−1} = ∂L/∂c_t ⊙ f: along the cell the gradient is only
multiplied by the forget gate — no weight matrix, no tanh′. A cell whose
forget gate stays near 1 passes the gradient back almost untouched: the
"gradient highway". A **GRU** gets a similar path with two gates:
h_t = z ⊙ h_{t−1} + (1 − z) ⊙ h~, where the update gate z decides how much
of the old state to keep (NeuroMZ uses the original formulation of Cho et
al. 2014, DESIGN decision 73).

```bash
./build/neuromz new --preset char-lstm -o lstm.nmz      # embed:32 lstm:64 vocab: the weights of char-rnn's rnn:128
./build/neuromz -m lstm.nmz train data/alice.txt -o lstm.nmz
./build/neuromz new text embed:32 gru:75 vocab --data data/alice.txt --lr 0.01 --batch 16 -o gru.nmz
./build/neuromz new text embed:32 lstm:64 dropout:0.2 lstm:64 vocab --data data/alice.txt   # stacked
```

`lstm:64` has about as many weights as `rnn:128` (four gates per cell), so
the comparisons below are fair: every model has 29–33 k parameters.
Recurrent matrices start orthogonal (all singular values 1), and the forget
gate's bias starts at 1, so a new cell keeps ~73 % of its memory per step
(`lstm:64:forget_bias=2` to change it; `new` prints what each choice means).

### Every gate, with real numbers

`trace` on an LSTM shows, for one cell (`--unit J`; by default the one whose
output changes most), every gate at every step: the three terms of its
input, the sigmoid, and a bar for how open it is; then the cell update and
the output:

```
$ ./build/neuromz -m lstm.nmz trace --text "the Queen said"
LSTM UNROLLED THROUGH TIME   gates decide what the cell forgets (f), writes (i, g) and shows (o)
  for cell j = 5 of 64 (the cell whose output changes most here; choose with --unit J)
  ...
  t=1   input h
     i  input gate  =    σ(   2.970 +    4.565 +   0.586) =    σ(  8.121) =  1.000 ████████████████████
     f  forget gate =    σ(   8.458 +  -10.514 +   1.174) =    σ( -0.882) =  0.293 ██████··············
     g  candidate   = tanh(   3.988 +   -1.007 +  -0.075) = tanh(  2.906) =  0.994           |██████████
     o  output gate =    σ(  -2.410 +    5.328 +   0.113) =    σ(  3.032) =  0.954 ███████████████████·
     c_t = f·c_{t-1} + i·g = 0.293·1.65e-04 + 1.000·0.994 = 4.82e-05 + 0.994 = 0.994             |███
     h_t = o·tanh(c_t) = 0.954·0.759 = 0.724
        next: e     76.9% ███████████████████████   <- target
              o      8.2% ██
  ...
GATE ACTIVITY OVER TIME   cell 3: one row per gate, columns = time steps
                 “Off␣with␣her␣head!”␣the␣Queen␣shouted␣at␣the␣top␣of␣her␣voic
  i input gate   @@@@: @@@@:@@@.@@@@@@@@@@@@@@@@ @@@@@@@@@@%@@@@@@@@@@ @@@ @@@
  f forget gate    @@ @ @@ @@# @@ @ + %@@ @@@%= @@%*@.-  @ @@% @-@ .@ @@@ @@-.
  o output gate  %+@@: @#@@-@@@.@@@@@@#@@@:@@@@@ =@@=@@@@ @..@@ @@@@@@=@@@ @@
  c cell         -*%@===----+*--+*%-+-: :-: :=*--=*#+*#-*=-: :-:++-*#--+*--++*
```

(The three numbers in each gate are x_t·W_x, h_{t−1}·W_h and the bias. In
the heatmap ' ' is 0 and '@' is 1; cell 3's forget gate closes at most
spaces, so it keeps something about the current word and drops it at the
next word boundary.)

followed by a heatmap of that cell's gates over time (and of every cell's
forget gate), the backward pass with ∂L/∂h *and* ∂L/∂c per step, and the
vanishing-gradient view along both paths. GRU models show z, r, h~ and h_t
the same way. `--layer N` picks a layer of a stacked model.

### Gradient survival, side by side

`compare` takes trained models and one text and shows how much of the **last
character's** loss gradient reaches each earlier step — for the RNN along h,
for the LSTM along h and along c, plus the product of the forget gates (what
the cell path alone would keep), and for the GRU along h and through its
update gates. Three models trained on Alice with the same number of weights,
on 65 characters of the validation part:

```
$ ./build/neuromz compare alice-rnn.nmz alice-lstm.nmz alice-gru.nmz --data data/alice.txt --from 140000 --chars 65
text (64 steps; the models predict the last character from all before it):
  re!” said the King triumphantly, pointing to the
tarts on the tab

models (recurrent layer 1 of each):
  A  alice-rnn.nmz        rnn 128, 32683 parameters            loss of the last prediction 1.943
  B  alice-lstm.nmz       lstm 64, 32107 parameters            loss of the last prediction 2.327
  C  alice-gru.nmz        gru 75, 32400 parameters             loss of the last prediction 1.161

  back  char     A rnn h   B lstm h   B lstm c      B Π f    C gru h      C Π z
     0  a          1.000      1.000      1.000      1.000      1.000      1.000
     1  t          0.841      0.460      1.015      0.777      0.835      0.159
     2  ␣          0.680      0.315      0.929      0.591      0.444      0.013
     4  h          0.311      0.072      0.411      0.578      0.198    4.4e-03
     8  o          0.091    8.1e-03      0.020      0.515      0.030    9.8e-05
    16  e          0.011    5.0e-04    1.2e-03      0.428    6.7e-03    9.6e-06
    24  n        1.1e-03    4.7e-05    1.5e-04      0.397    4.7e-04    2.3e-07
    32  ,        1.4e-04    2.0e-05    1.1e-04      0.352    7.4e-05    3.2e-10
    40  m        6.3e-05    6.7e-05    1.4e-04      0.322    1.6e-05    3.2e-10
    48  i        1.5e-05    6.8e-06    8.8e-05      0.300    9.2e-06    1.6e-11
    56  i        3.2e-06    7.9e-06    9.1e-05      0.271    9.5e-07    2.8e-15
    63  r        9.6e-07    2.0e-05    7.1e-05      0.183    4.6e-07    9.4e-21

the same on a log scale, one column per step back (0 at the left; '█' = 1, empty = below 1e-06):
  A rnn h    ███████▇▇▇▇▇▇▇▆▆▆▆▆▅▅▅▅▅▅▄▄▄▄▄▄▄▃▃▃▃▃▃▃▃▃▃▃▃▃▂▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁
  B lstm h   ███▇▇▇▆▆▆▆▅▅▅▅▄▅▄▄▄▄▃▃▃▃▃▃▃▂▂▂▂▃▂▄▃▃▃▃▃▃▃▃▂▂▂▂▂▂▂▂▃▂▂▂▂▂▂▁▂▂▂▂▁▂
  B lstm c   █████▇▇▆▆▆▆▅▆▅▅▅▅▄▅▄▄▄▄▃▃▃▃▃▃▃▃▃▃▄▄▄▄▄▃▄▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃
  B Π f      ████████████████████████████████████████████████████████████████
  C gru h    █████▇▇▇▆▆▆▆▆▆▆▆▆▅▅▅▅▄▄▄▄▄▄▄▄▄▄▃▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▁▁▁▁
  C Π z      █▇▆▆▅▅▃▃▃▃▃▃▃▃▃▂▂▂▂

summary
               x per step    x/step k>=       left at       left at    < 1% after
                    (all)            16          k=10         k=end
  A rnn h          x0.803        x0.820         0.062       9.6e-07      17 steps
  B lstm h         x0.842        x0.934        0.0036         2e-05       8 steps
  B lstm c         x0.859        x0.942          0.01       7.1e-05      11 steps
  B Π f            x0.973        x0.982          0.47          0.18   never (>63)
  C gru h          x0.793        x0.815         0.019       4.6e-07      15 steps
  C Π z            x0.481        x0.479       5.9e-05       9.4e-21       3 steps
  B: 12 of 64 cells keep at least half along the cell-state path over all 63 steps (Π f >= 0.5)
  C: 0 of 75 units keep at least half along the direct path over all 63 steps (Π z >= 0.5)
```

How to read it — and it is not quite the textbook picture:

* **Far back, the RNN's gradient vanishes geometrically; the LSTM's does
  not.** From 16 steps back on, every step costs the RNN ×0.82 and the
  LSTM's cell state only ×0.94; after 63 steps the RNN keeps 1e-6 of the
  gradient, the cell state 7e-5 (70× more), and its sparkline goes flat
  from about 30 steps back while the RNN's keeps falling. That flat tail is
  the gradient highway: 12 of the 64 cells keep at least half of their
  state (Π f ≥ 0.5) over all 63 steps — no RNN unit can do that, because
  its only path back goes through W_hhᵀ and tanh′ at every step.
* **Close to the last character the LSTM's gradient falls *faster*** (0.4 %
  left after 10 steps, against 6 % for the RNN). The trained LSTM has
  learned to make its prediction mostly from the last few characters and
  to keep only a few long-term cells; a gate can close as well as open.
* **The highway is open, but little enters it.** Π f stays at 0.18 after 63
  steps, yet |∂L/∂c| falls to 7e-5: the cells with f ≈ 1 are often
  saturated (|c| large, so tanh′(c) ≈ 0) and receive almost no gradient at
  the last step in the first place.
* **The GRU barely uses its direct path here** (Π z collapses within a few
  steps: its update gates mostly replace the state), and its gradient
  vanishes about as fast as the RNN's — yet it has the best validation loss
  of the three (table below). For predicting the next character of Alice,
  most of the information is in the last few characters; what the gates
  buy is mainly a better use of them.

On Kalila wa-Dimna (`benchmark_gated.sh` prints the same table) the order
changes: after 63 steps the RNN keeps 9e-10 of the gradient, the LSTM's cell
5e-7 (500× more, 11 cells with Π f ≥ 0.5) and the GRU 2e-5 — there the GRU's
gradient survives best, through its candidate path rather than z. Which path
a trained model uses is something it learns from the data, which is why
`compare` measures it rather than assuming it.

The same three models on English names (22 steps) show no highway at all:
names are short and the LSTM learns to forget (only 9 of 64 cells have a
mean forget gate above 0.9 there, against 36 of 64 on Alice — see below).

### What the gates do: `show gates`

```
$ ./build/neuromz -m alice-lstm.nmz show gates --data data/alice.txt --from 130000 --chars 14000
layer 1: lstm 32->64 (gates i f g o, forget bias 1, orthogonal init)
  gate                mean      < 0.1  0.1-0.5  0.5-0.9    > 0.9    (share of all 896000 values)
  i  input gate      0.755      16.4%     7.4%     9.4%    66.8%
  f  forget gate     0.789      16.4%     4.3%     5.5%    73.8%
  o  output gate     0.530      39.3%     7.7%     7.7%    45.3%

  mean forget gate f per cell (64 cells): how much of its state each one keeps per step
    0.0-0.1    0
    0.1-0.2    3 ███
    ...
    0.8-0.9    8 █████████
    0.9-1.0   36 ████████████████████████████████████████  <- long memory
  mostly open (mean > 0.9): 36 of 64; mostly closed (mean < 0.1): 0 of 64
  longest memory (a state that is only multiplied by f halves after ln 0.5 / ln f steps):
    cell   55   mean f = 0.999   half-life ~ 574.4 steps
    cell   62   mean f = 0.998   half-life ~ 360.4 steps
```

On running text more than half of the cells keep their memory (f > 0.9 on
average); the same architecture trained on the names list has only 9 such
cells — the model's gates adapt to how far back the data is worth
remembering. The longest-memory cell, 55, has both gates wide open and adds
about −1 to c at every character: its c saturates within a few characters
of a reset, so it signals "near the start of a stream" rather than storing
anything specific — `trace --unit 55` shows it. Not every long memory is
an interesting one.

### Arabic prose, and normalisation

`make data-text` also fetches *Kalila wa-Dimna* (كليلة ودمنة), Ibn
al-Muqaffaʿ's 8th-century book of fables and a classic of Arabic prose, from
Arabic Wikisource (240 k characters; source and licence in
[`examples/DATASETS.md`](examples/DATASETS.md)). `--preset arabic-text` is
two stacked LSTMs (96 cells each, dropout between them) for it.

The text is *partly* vocalised: about 1 % of its characters are tashkeel
(mostly tanwin ً ٌ ٍ and some shadda ّ), placed wherever an editor happened
to add them. `new ... --strip-diacritics` removes the harakat and the
tatweel ـ; `--unify-alef` also writes أ إ آ ٱ as ا. The choice is stored in
the model and applied to everything it reads — training text, `--prime`,
`trace --text` — so `generate --prime "قالَ"` works on a stripped model.

The trade-off:

* **Stripping** gives the model a smaller alphabet (53 → 45 characters for Kalila wa-Dimna) and a
  more predictable stream — it no longer has to guess where an editor put a
  tanwin — but it can no longer *write* the marks, and words that differ
  only in their vowels become the same string. The loss is not comparable
  one-to-one either: the stripped model predicts fewer, easier characters
  per word.
* **Unifying alef** goes further: همزة spelling (أ/إ/آ) is information in
  Arabic (أن vs إن), so a unified model cannot learn it. It helps only
  when the source is inconsistent about hamza, which matters more for
  informal text than for an edited classic.

Off is the default: the model learns the text as it is written.

---

## A small Transformer: what attention does

A recurrent network squeezes everything it has read into one state vector
and carries it forward one character at a time. A Transformer keeps a
vector for *every* character it has read and lets each position **look
back** at all of them, choosing with learned weights which ones matter.
NeuroMZ has a decoder-only Transformer, GPT-style:

```
embedding → position encoding → block → block → LayerNorm → output (tied to the embedding) → softmax
block:  h = x + attention(LayerNorm(x))      y = h + feed-forward(LayerNorm(h))
```

```bash
./build/neuromz new --preset char-gpt -o gpt.nmz     # 32 043 weights: the size of char-lstm
./build/neuromz -m gpt.nmz train data/alice.txt -o gpt.nmz
./build/neuromz -m gpt.nmz generate --prime Alice --temperature 0.7
```

```
Preset char-gpt: small Transformer for running text: the `char-lstm` comparison (same number of weights)
  embed 75 chars->32                       [preset]  each character gets a learned 32-number vector (also the output matrix: vocab:tied)
  pos Tx32->Tx32           learned 128     [preset]  context 128 characters (~25 words): attention can look back that far in one step
  block Tx32->Tx32         4 heads x 8     [preset]  4 heads of 8: each can follow a different relation (previous letter, word start, ...)
    feed-forward           128 gelu        [default] feed-forward x*Phi(x) (a smooth relu); hidden size 4D by default; 12704 weights in this block
  block Tx32->Tx32         4 heads x 8     [preset]  a second block: 32k weights in all, the same as char-lstm (lstm:64)
  ...
  layernorm Tx32                           [preset]  a final LayerNorm, as in GPT-2
  output Tx32->Tx75        tied softmax    [preset]  next-character scores = h * E^T: the embedding matrix read backwards
  optimizer                adamw           [preset]  AdamW: Adam's per-weight steps plus decoupled weight decay (0.1) on the weight matrices
  weight decay             0.1             [default] w <- w - lr*lambda*w for weight matrices only; not biases, LayerNorm, embeddings, positions
  schedule                 warmup-cosine   [preset]  warm-up (5% of the updates), then a cosine down to a tenth of the rate, per batch
    warm-up batches        192             [default] about 5% of all updates: early Adam steps are noisy, and big early updates can stall attention
Transformer: context 128 characters, 2 block(s) x 4 heads of 8, 32043 parameters (the output shares the embedding)
```

The layers are ordinary NeuroMZ layers — `pos:learned:128`, `block:heads=4`,
`ln`, `vocab:tied` — so you can build your own (`block:heads=8:head_dim=16:ff=512:relu:dropout=0.1`,
`pos:sin:256`, or the parts `attn:heads=4`, `ffn:256`, `ln` on their own).
`show params` breaks the weights down: in char-gpt 52% are feed-forward,
26% attention, 13% the position table, 7.5% the shared character table
and 1% LayerNorm.

### How attention works — step by step

`trace` on a Transformer follows one position through one head with every
number. This is the char-gpt trained above, reading "the Queen said to the "
(`trace --text "the Queen said to the Q" --layer 1 --head 2`; the input is
the text without its last character, the target the text shifted by one):

**1. Each character becomes a vector: what it is plus where it is.**

```
  e_t = W_emb[id_t] (32 numbers);  p_t = learned row t of P
  t=0   t  e = [0.604, 0.237, 0.450, -0.336, ...]   p = [0.561, 0.145, -0.029, -0.281, ...]
  t=1   h  e = [-0.463, -0.937, 0.370, -0.737, ...]   p = [0.222, -0.136, -0.052, 0.078, ...]
  t=2   e  e = [-0.246, -0.446, -0.427, -0.176, ...]   p = [0.283, 0.009, 0.031, 0.380, ...]
```

The same `e` at t = 2, 6, 7 and 20 has the same character vector; only `p`
tells them apart. Without it attention could not know what is *previous*:
a weighted sum does not care where a value came from.

**2. The query asks, every key answers.** Each block LayerNorms the vector
and projects it three ways: a query q ("what am I looking for?"), a key k
("what do I offer?") and a value v ("what do I pass on?"), split into 4
heads of 8 numbers. At the last position, the space after "the":

```
  query position 21 = '␣' (the model has read up to here and predicts the next character)
  q = x·W_q + b_q   (8 numbers per head; head 2's part):  q = [0.674, -2.313, 1.011, 3.759, 1.647, -0.461, -0.506, 1.222]
```

**3. Scores, the causal mask and the softmax.** The score of position j is
`q·k_j/√8`. Only j ≤ 21 exist for this query — later positions are masked
(here there are none; at position 5 the 16 positions after it would be
listed as masked, weight 0). The softmax turns scores into weights that
sum to 1:

```
     j  char  k_j (first 3)                     score s_j   weight a_j
     0  t     [-4.697, 3.565, -4.052, ...]       -17.265   6.21e-06
     ...
    13  d     [-2.491, 2.330, -2.078, ...]        -9.192      0.020  █
    15  t     [-2.483, 3.096, -3.071, ...]        -8.417      0.043  █
    19  h     [-2.840, 1.318, -1.999, ...]        -7.917      0.071  ██
    20  e     [-3.347, 2.315, -1.742, ...]        -5.943      0.513  ███████████████  <- largest
    21  ␣     [-2.116, 3.289, -1.992, ...]        -6.562      0.276  ████████

  why 'e' at 20: q·k = Σ q_d·k_d = -16.810; the largest terms: d=2: -2.313·2.315 = -5.355, d=8: 1.222·-2.110 = -2.579, d=4: 3.759·-0.625 = -2.351
```

Every score is negative, and that does not matter: the softmax only sees
*differences*. They fall steadily with distance — −17.3 for the first
character, about −9 for the last two words, −7.9 for 'h', −5.9 for the
previous character 'e' — so this head has learned a positional rule, *look
at the character just before me*, and spends 51% of its attention on 'e'.

**4. The weighted sum.** The head's output is the values mixed by those
weights; mostly the value of 'e':

```
  context c = Σ_j a_j·v_j  (the values, mixed by the weights) = [-1.380, 0.323, 0.364, 0.368, -0.564, 0.051, -0.994, -0.004]
  the largest contributions a_j·v_j:
    0.513 · v_20 = [-0.887, 0.463, 0.527, 0.930, ...]   (e)
    0.276 · v_21 = [-0.471, -0.194, -0.150, -0.474, ...]   (␣)
    0.071 · v_19 = [-0.012, -0.057, 0.108, -0.023, ...]   (h)
```

**5. Four heads look at different things.** The four contexts are put side
by side, projected back to 32 numbers (`W_o`) and *added* to the vector:

```
  head 1:   'e'@20 0.330   '␣'@21 0.228   'h'@19 0.115
  head 2:   'e'@20 0.513   '␣'@21 0.276   'h'@19 0.071
  head 3:   'h'@19 0.374   '␣'@21 0.323   'e'@20 0.086
  head 4:   '␣'@21 0.419   'h'@19 0.276   '␣'@17 0.082
```

Together, layer 1 sees "the␣" — enough to know a noun is coming.

**6. The whole picture: a heatmap.** Rows are query positions, columns the
positions they may look at; right of the diagonal is the masked future.
Head 2 is one bright stripe just left of the diagonal — the previous
character, at every position:

```
           0         1         2
           the␣Queen␣said␣to␣the␣
     0 t   @
     1 h   +=
     2 e   -+:
     3 ␣    :+:
     4 Q    .--:
     5 u      .=:
     6 e       .*.
     ...
    18 t                   .*:
    19 h                    --.
    20 e                    ..::
    21 ␣                      .+:
```

**7. The residual stream.** A block never replaces the vector, it adds to
it. At position 21 the first block's attention adds a vector as long as
the input, the feed-forward net more; in the second block the
feed-forward net dominates (86% of the change):

```
  block         |x|    +|attn|      |h|     +|ffn|      |y|   cos(x,y)   attn share
  1           3.056      3.447    4.036      5.256    6.128      0.474          40%
  2           6.128      2.935    6.235     17.675   18.902      0.437          14%
```

**8. The prediction.** After the final LayerNorm, `z = h·Eᵀ + b` scores each
character by how well the vector matches its *embedding* (weight tying);
the softmax gives the next character:

```
  'M'  0.101  ███
  'D'  0.079  ██
  'c'  0.074  ██
  'C'  0.072  ██
  'w'  0.054  ██
  the actual next character 'Q': p = 0.045 (rank 7 of 75)
```

"the M…" (March Hare, Mock Turtle), "the D…" (Duchess, Dormouse): capital
letters after "the ", as in the book. A 32 k-weight model has not narrowed
it down to the Queen. The trace then shows the gradient of this loss
reaching every earlier position directly, and the AdamW update.

### What the heads learned

`show attention` runs any text through the model in windows of the
context length, draws a heatmap for one head and, for every head,
measures where its attention goes: on itself, the previous character, two
back, earlier copies of the same character, the character after such a
copy (what came next last time), the start of the current word, the last
space or punctuation, the first position, or more than 16 positions back —
each against what uniform attention would give. The label is inferred
from these numbers, not looked up (DESIGN decision 101).

**char-gpt on Alice** (8 000 characters of the validation part,
`show attention --data data/alice.txt --from 130000 --chars 8000`):

```
  layer 1 head 1  the character two back: 23% (uniform 3%);  then the previous character 23%
  layer 1 head 2  the previous character: 50% (uniform 3%);  then itself 23%
                  span 3.5 positions, top weight 0.56 on average
  layer 1 head 3  the previous character: 28% (uniform 3%);  then the character two back 21%
  layer 1 head 4  the previous character: 29% (uniform 3%);  then the character two back 19%
  layer 2 head 1  spread out: no single kind of position stands out (best: the start of the current word 6%, uniform 2%)
                  span 13.6 positions, top weight 0.31 on average
  layer 2 head 3  spread out: no single kind of position stands out (best: the last space or punctuation 9%, uniform 3%)
  ...
```

The first layer has become a set of **local n-gram detectors**: every head
looks at the previous one or two characters (and itself), one of them
with half of its weight on the previous character alone — the clean
diagonal of the heatmap above. The second layer's heads spread over ~14
positions: they mix the word so far rather than point at one place.

**The best model (213 k weights, 4 blocks of 64, below)** has the same
first layer — a previous-character head (60%, span 3.4), two "two back"
heads and a head on *itself and the start of the current word* (37% and
17%) — and three more layers whose heads all spread over 15–22 positions.
None of them is a "copy" head: the pattern *the character after an earlier
copy of this one* (an induction head, which lets a model repeat a name it
has seen) reaches 8% against 6% uniform at best. At this size and data
those heads did not form.

**arabic-gpt on Kalila wa-Dimna** (3 blocks of 64):

```
  layer 1 head 1  the previous character: 58% (uniform 3%);  then the last space or punctuation 19%
  layer 1 head 2  the character two back: 32% (uniform 3%);  then the last space or punctuation 20%
  layer 1 head 3  itself: 30% (uniform 3%);  then the start of the current word 10%
  layer 1 head 4  itself: 35% (uniform 3%);  then the previous character 20%
  layer 2 head 4  the start of the current word: 17% (uniform 2%);  then the character two back 10%
  (the other 7 heads of layers 2 and 3: spread out)
```

The same division of labour in Arabic, plus a layer-2 head that watches
**the start of the current word** — where Arabic puts its prefixes (و، ف،
ب، ال), which constrain what may follow. Its heatmap with Arabic on both
axes (one column per letter, `show attention --text "قال دمنة: زعموا أن"
--layer 1 --head 1`; the previous-character diagonal again):

```
‎           0         1       
‎           ق‌‎ا‌‎ل‌‎␣د‌‎م‌‎ن‌‎ة‌‎:␣ز‌‎ع‌‎م‌‎و‌‎ا‌‎␣أ‌‎ن‌‎
‎     0 ق‌‎   @                 
‎     1 ا‌‎   @                 
‎     2 ل‌‎   *:.               
‎     3 ␣    .*.              
‎     4 د‌‎     .#.             
‎     5 م‌‎      --:            
‎     6 ن‌‎       -+.           
‎     7 ة‌‎        .==          
‎     8 :        .:=:         
‎     9 ␣          :#         
‎    10 ز‌‎           .=:       
‎    11 ع‌‎            .*.      
‎    12 م‌‎             :+.     
‎    13 و‌‎             ..+.    
‎    14 ا‌‎               :*.   
‎    15 ␣                --:  
‎    16 أ‌‎                 .+. 
‎    17 ن‌‎                  :+.
```

### Attention vs recurrence: the gradient path

`compare` accepts Transformers next to RNNs, LSTMs and GRUs. The same
experiment as in phase 4 — only the last character's loss, followed back —
measured for a Transformer at the input of its first block:

```
$ neuromz compare alice-lstm.nmz alice-gru.nmz alice-gpt.nmz --data data/alice.txt --from 140000 --chars 65

text (64 steps; the models predict the last character from all before it):
  re!” said the King triumphantly, pointing to the
tarts on the tab

  A  alice-lstm5.nmz      lstm 64, 32107 parameters            loss of the last prediction 2.327
  B  alice-gru5.nmz       gru 75, 32400 parameters             loss of the last prediction 1.161
  C  alice-gpt.nmz        transformer, 2 blocks, 32043 parameters loss of the last prediction 1.384

  back  char    A lstm h   A lstm c      A Π f    B gru h      B Π z    C gpt x
     0  a          1.000      1.000      1.000      1.000      1.000      1.000
     1  t          0.460      1.015      0.777      0.835      0.159      0.542
     2  ␣          0.315      0.929      0.591      0.444      0.013      0.284
     3  e          0.073      0.476      0.580      0.290    6.6e-03      0.129
     4  h          0.072      0.411      0.578      0.198    4.4e-03      0.466
     5  t          0.037      0.072      0.539      0.125    3.7e-03      0.090
    10  s        3.6e-03      0.010      0.466      0.019    5.9e-05      0.022
    16  e        5.0e-04    1.2e-03      0.428    6.7e-03    9.6e-06      0.072
    24  n        4.7e-05    1.5e-04      0.397    4.7e-04    2.3e-07      0.021
    32  ,        2.0e-05    1.1e-04      0.352    7.4e-05    3.2e-10      0.062
    48  i        6.8e-06    8.8e-05      0.300    9.2e-06    1.6e-11    4.8e-03
    63  r        2.0e-05    7.1e-05      0.183    4.6e-07    9.4e-21    8.6e-03

the same on a log scale, one column per step back (0 at the left; '█' = 1, empty = below 1e-06):
  A lstm h   ███▇▇▇▆▆▆▆▅▅▅▅▄▅▄▄▄▄▃▃▃▃▃▃▃▂▂▂▂▃▂▄▃▃▃▃▃▃▃▃▂▂▂▂▂▂▂▂▃▂▂▂▂▂▂▁▂▂▂▂▁▂
  A lstm c   █████▇▇▆▆▆▆▅▆▅▅▅▅▄▅▄▄▄▄▃▃▃▃▃▃▃▃▃▃▄▄▄▄▄▃▄▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃
  B gru h    █████▇▇▇▆▆▆▆▆▆▆▆▆▅▅▅▅▄▄▄▄▄▄▄▄▄▄▃▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▁▁▁▁
  C gpt x    ███▇█▇▇▇▇▇▆▇▇▆▇▇▇▇▆▇▆▆▆▇▆▆▆▆▆▇▇▇▇▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▅▇▅▆▆▆▆▆▅▆▅▆▆▆▆▆

               x per step    x/step k>=       left at       left at    < 1% after
                    (all)            16          k=10         k=end
  A lstm h         x0.842        x0.934        0.0036         2e-05       8 steps
  A lstm c         x0.859        x0.942          0.01       7.1e-05      11 steps
  B gru h          x0.793        x0.815         0.019       4.6e-07      15 steps
  C gpt x          x0.927        x0.956         0.022        0.0086      27 steps
```

How to read it:

* **Same size, very different profiles.** 63 characters back the
  Transformer still receives 0.86% of the last prediction's gradient —
  120× more than the LSTM's cell state, 400× more than its h and 18 000×
  more than the GRU. Its sparkline stays at the same height while the
  recurrent ones slope down: there is no chain of steps to shrink along.
* **It is not monotonic.** 4 characters back the Transformer's gradient
  *rises* to 0.47 — that is the 'h' of "the", which the previous-character
  and two-back heads point at; 3 back ('e') gets 0.13. What reaches a
  position is decided by where the heads look, not by how far away it is.
* **It is still small far away.** "Directly connected" does not mean
  "used": at this size the layer-2 heads spread their weight over ~14
  positions and the model leans on the last few characters, like the
  recurrent ones (decision 90). The larger model (213 k, four blocks; see
  Results) keeps 7.2% of the gradient 63 characters back and never drops
  below 1% — its deeper heads spread over 15–22 positions. The difference is that a Transformer
  *could* use a character 100 positions back at the same cost as the
  previous one; an RNN would first have to keep it alive for 100 steps.
* The "x per step" columns are only a summary for the Transformer (its
  gradient is not multiplied step by step). The context window is the
  hard limit: past 128 characters a Transformer sees nothing at all, a
  recurrent model a faint trace.

### Generating with a KV cache

Generation feeds one new character per step. A Transformer would
recompute the keys and values of the whole window every time; with a **KV
cache** each attention layer keeps them (its carried state, like an RNN's
h) and computes only the new row. The text is bit-identical either way —
`generate --no-cache` recomputes everything and is the reference the tests
compare against. When the text outgrows the context window, the newest
half is kept and re-encoded from position 0 (positions are absolute, so a
cached key cannot be moved to a new position), and single steps continue.

On the trained char-gpt, 20 texts of 300 characters (`benchmark_transformer.sh`):

| | time | text |
|---|---|---|
| with the KV cache | **228 ms** | — |
| recomputing the window every step (`--no-cache`) | 8 469 ms (37×) | identical |

Each text outgrows the 128-position window after 127 characters and then
every 64, so it slides three times; a slide costs one full pass over the
64 characters it keeps.

### Training a Transformer: AdamW and warm-up

* **AdamW** (`--optimizer adamw --weight-decay 0.1`) adds `λ·w` to the Adam
  step instead of to the gradient, so every weight decays at the same
  relative rate whatever its gradient history. Only weight matrices decay;
  biases, LayerNorm gains, the character table and the position table do
  not (`--decay-all` changes that; `show params` lists who decays).
* **Warm-up + cosine, per batch** (`--schedule warmup-cosine --warmup N`):
  the rate rises linearly over the first ~5% of the updates, then follows a
  cosine down to a tenth. `trace`/`step` show the AdamW formula and the λ
  of each parameter.
* Tiny Transformers want **large** steps. On Alice, char-gpt's peak rate
  0.001 / 0.002 / 0.003 / 0.005 / 0.01 / 0.02 gave validation loss 2.30 /
  2.02 / 1.82 / 1.72 / 1.67 / 1.66 after 30 epochs (batch 16); batch 8 (twice the
  updates) 1.633; sinusoidal positions 1.637; dropout 0.1 1.70 (worse: at
  32 k weights the model underfits rather than overfits).

### Speed: the matrix product

Attention and the feed-forward net are matrix products; `nmz_matmul`
computes them cache-blocked, keeping a 1 × 32 tile of the result in
registers. It gives **bit-identical** results to the three-loop
reference (same sum order; tested), which is also what makes the KV cache
exact. `build/bench_matmul`, one core of the i7-7500U:

| product (M × N × K), as in a D = 64 model | naive loops | blocked, 1 thread | speedup | GFLOP/s | 2 threads (`OMP=1`) |
|---|---|---|---|---|---|
| `x·W_qkv` 4096 × 192 × 64 | 40.5 ms | 5.75 ms | 7.0× | 17.5 | 3.20 ms (31.5 GFLOP/s) |
| `a·W2` 4096 × 64 × 256 | 71.7 ms | 6.80 ms | 10.5× | 19.8 | 5.97 ms (22.5) |
| `x·W1` 4096 × 256 × 64 | 59.9 ms | 8.98 ms | 6.7× | 14.9 | 4.71 ms (28.5) |
| `dW1 = xᵀ·dz` 64 × 256 × 4096 | 488 ms | 13.5 ms | 36× | 9.9 | 7.01 ms (19.1) |
| `dx = dz·W1ᵀ` 4096 × 64 × 256 | 64.5 ms | 6.93 ms | 9.3× | 19.4 | 3.45 ms (38.9) |

(The naive weight-gradient loop reads A by columns; the blocked version
copies Aᵀ once.) A 4 × 8 register tile, the textbook choice, came out at
4–6 GFLOP/s — GCC kept it on the stack — and the plain row loop at 11–12.

`make OMP=1` builds with OpenMP (off by default; the library stays
dependency-free): the row blocks of each product and the (row, head) pairs
of attention are shared among threads, each number is still computed by
one thread in the same order, so results stay bit-identical (all tests
pass in both builds). One training epoch of a 113 k-weight model: 16.2 s
on one thread, 11.2 s on 2 threads; 4 threads (hyper-threads) 14.6 s.

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
| activations | `linear`, `sigmoid`, `tanh`, `relu`, `leaky_relu[=ALPHA]`, `softmax`, `gelu` |
| init | `auto` (He before relu and for conv, Xavier otherwise; orthogonal for recurrent layers), `xavier`, `he`, `orthogonal` (recurrent layers only) |
| loss | `mse` (½·squared error), `cross_entropy`/`ce` (fused with a softmax output) |
| optimizer | `--optimizer sgd` (`--momentum`), `adam` (`--beta1 --beta2 --eps`) or `adamw` (`--weight-decay X`, `--decay-all`), `--lr` |
| schedule | `--schedule constant\|step\|cosine\|warmup-cosine`, `--step-size N --gamma X`, `--min-lr X`, `--warmup N` (batches) |
| data | `--normalize none\|minmax\|zscore` (fitted at the first `train`, saved) |
| text | input `text` with `--data FILE` (vocabulary), `embed:D`, `rnn:H`, `lstm:H[:forget_bias=F]`, `gru:H`, `vocab[:ACT]` (one output per character); stack recurrent layers with `dropout:R` between them; `--items`/`--no-items`; `--strip-diacritics`, `--unify-alef` (Arabic) |
| training | `--batch`, `--epochs`, `--target-loss`, `--no-shuffle`, `--seed` |
| validation, stopping | `--val FRACTION` (the last part of the data is held out) or, for `train`, `--val-file FILE` (a separate set), `--patience N` (early stopping, keeps the best weights) |
| recurrent | `--seq-len N` (BPTT window), `--carry`/`--no-carry` (hidden state carried between batches), `--clip NORM` |
| transformer | `pos:learned:CTX`, `pos:sin:CTX` (position encoding; CTX = context window), `block[:heads=H][:head_dim=K][:ff=F][:gelu\|relu][:dropout=R]` (pre-LN block), the parts `attn[:heads=H][:head_dim=K]`, `ffn[:F][:gelu\|relu]`, `ln`; `vocab:tied` (output = embedding matrix); presets `names-gpt`, `char-gpt`, `arabic-gpt` |

* `train DATA [options]` overrides the stored settings for one run;
  `set --optimizer adam --lr 0.001 ...` changes the settings saved with the model.
  `DATA` is a CSV file, `mnist` (training set for `train`, test set for
  `test`), `mnist:train`/`mnist:test`, or an IDX images file.
* `show` / `show weights` / `show filters` / `show vocab` / `show params` /
  `show gates --data FILE` / `show attention --data FILE [--layer N --head H]` /
  `show sample N [--data SPEC]`, `compare A.nmz B.nmz ... --text TEXT`,
  `generate [--context N] [--no-cache]`,
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

**Recurrent text models: RNN vs LSTM vs GRU** (`sh examples/benchmark_gated.sh`,
part of `make bench`): the same parameter budget for the three
architectures (`rnn:128`, `lstm:64`, `gru:75`: ±3 %), seed 42, the same
machine, one thread. The validation split is the last 10 % of each text;
early stopping with patience 3. Learning rates are the best of a small sweep
per architecture: RNN 0.003 (names; 0.006 gave 2.108) and 0.004 (books;
0.008 was worse), LSTM and GRU 0.01 (better than 0.005 on names and than
0.004/0.006 on Alice). Validation loss is in nats per character; lower is
better.

| data | RNN (rnn:128) | LSTM (lstm:64) | GRU (gru:75) |
|---|---|---|---|
| English names (5163, 36 k chars) | 2.118 · ppl 8.31 · 26 s | **2.028** · ppl 7.60 · 38 s | 2.056 · ppl 7.81 · 19 s |
| Arabic names (504, 2.7 k chars) | 2.660 · ppl 14.30 · 2 s | **2.460** · ppl 11.70 · 3 s | 2.464 · ppl 11.75 · 2 s |
| Alice (145 k chars, state carried) | 1.638 · ppl 5.15 · 146 s | 1.614 · ppl 5.03 · 270 s | **1.580** · ppl 4.86 · 187 s |
| Kalila wa-Dimna (240 k chars, Arabic) | 1.965 · ppl 7.14 · 235 s | 1.951 · ppl 7.03 · 421 s | **1.938** · ppl 6.95 · 160 s |
| parameters (names / Alice / Kalila) | 26.2 k / 32.7 k / 29.1 k | 25.0 k / 32.1 k / 30.0 k | 25.6 k / 32.4 k / 30.0 k |

Stopping epochs (stopped / best): names 15/12, 17/14, 10/7; Arabic names
16/13, 19/16, 15/12; Alice 17/14, 30/30 (the limit), 21/18; Kalila 18/15,
26/23, 12/9. Also on Kalila: `--preset char-lstm --strip-diacritics` 1.935
(ppl 6.92, 45 instead of 53 characters; a slightly different task, see
"Arabic prose" above), and `--preset arabic-text` (two stacked LSTMs of 96,
130 k parameters, 4.3× the others) **1.863** (ppl 6.44, 15 epochs, 865 s): on
the largest text, more capacity helps more than the choice of cell.

Honest summary: **the gated models win everywhere, but by small margins on
the long texts** (0.01–0.06 nats per character on the books, 0.06–0.09 on
English names) and by a lot on the small Arabic list (0.2: the gates seem
to help most when data is scarce). The GRU is the best on both books and the faster of the
two gated models; the LSTM learns more slowly (Alice: still improving at
the 30-epoch limit). Orthogonal initialisation changed nothing measurable for
the RNN (names 2.118 vs 2.112 with the phase-3 Glorot init, Alice 1.638 vs
1.634). Character-level prediction is dominated by the last few characters,
which is exactly where a vanilla RNN is fine; the long-range gradient that
the LSTM keeps (`compare` above) is worth a few hundredths of a nat here.

Generated with `--seed 1` (unedited; names marked "(in the training data)"
were not invented):

* **English names** — RNN: Ebron, Corinetta, Shannna, Samarda, Fantta,
  Karlene *(in the data)*, Sasfita, Jeish, Tamina, Dantor. LSTM: Darrice,
  Corine *(in the data)*, Shannury, Samard, Elizonda, Janice *(in the
  data)*, Sara *(in the data)*, Helora, Tamila, Damrus. GRU: Ebmie, Doshing,
  Thanniah, Sambie, Hantonda, Latiche, Tathawana, Jebria, Tamika *(in the
  data)*, Eghris.
* **Arabic names** — RNN: رامم، حنية، نشاد، ماسة، رادية، ضالف، نان، سطام، هؤار، ذارية.
  LSTM: رالي، حهيد، نفاد، مجرال، زجديل، طبوف، نان، شعال، هؤاف، ذجديفن.
  GRU: ذالي، حوناء، نصار *(in the data)*، مجراف، زحبيد، ضاقر، نان، ششام، هادر، دجحيرة.
* **Alice** (`--prime Alice --temperature 0.7`) — GRU:

  > Alice another of cerion, he _chourn_ timid. “Bith on of the botting of
  > her know put it back—” “I? They a crose to say who been to be all very
  > enpleeg,” said the Kings weind down all the seating and not to the
  > Lorybe, said to play would be in a poon, and party, cried the Caterpillar.

  LSTM: *“…the King, and the Queen was nothing of the Mock Turtle side it
  would be little miss near-for could her a tall pallent.”* RNN: *“…The
  March Hare had not they down as you can a little both are no at the
  dinning the Gryphon.”*
* **Kalila wa-Dimna** (`--prime قال --temperature 0.7`) — GRU:

  > قال الملك وأنا الصديق إلى أشراوز في الناس في القول، والفيل الملك لهم إلى
  > الملك قيل البراه يواعد في أمهاً التي التيت لا أنا بزوجه ونعافه ما يعلم بين
  > يده، ولا يتم تنطعم وأنا على القتال والحسن في ذلك في الملوك.

  LSTM, tashkeel stripped: *قال الملك وأنا أمر ما أعد أمرهم عنده من حتى إلى
  ما كان في الملك وفي دمنة من الأسواد من أهلهم وبين العقل؟ قال دمنة: …*

Judged honestly: the **English names** of all three are pronounceable and
name-like; the LSTM's are the most convincing (Darrice, Elizonda, Helora,
Tamila) but it also repeats more training names. The **Arabic names** are
still weak with 504 examples: one real name (ماسة; نصار was in the data),
a few plausible-looking forms (رادية, هادر, سطام), several non-words that
break Arabic patterns (ذجديفن, زجديل, مجرال), and نان from all three. **Alice**: book
style, dialogue punctuation, the right characters (the King, the Queen, the
Mock Turtle, the Caterpillar), real words, no sense — no visible difference
in kind between the three. **Kalila**: the most striking output: mostly real
Arabic words (قال الملك، الصديق، الناس، القول، العقل، العدو، الملوك), the book's
names (دمنة, الفيل) and its rhetorical rhythm ("قال دمنة: …", "وأنا …"),
correctly attached prefixes (و، ف، ب، ال), but invented words in between
(أشراوز، تنطعم) and no syntax beyond a few words. A character model of this
size learns spelling and the shape of phrases, not meaning.

**Transformer vs LSTM vs GRU** (`sh examples/benchmark_transformer.sh`,
part of `make bench`): the phase-4 LSTM and GRU against a Transformer with
the same number of weights (char-gpt: 2 blocks of 32, 4 heads, context
128, tied output), seed 42, one core of the same i7-7500U. The LSTM and
GRU numbers reproduce phase 4 exactly (training is deterministic; the
times are this run's). Validation loss in nats per character:

| data | LSTM (lstm:64) | GRU (gru:75) | Transformer (char-gpt / names-gpt) |
|---|---|---|---|
| English names | **2.028** · ppl 7.60 · 28 s | 2.056 · ppl 7.81 · 19 s | 2.097 · ppl 8.14 · 33 s |
| Arabic names | **2.460** · ppl 11.70 · 3 s | 2.464 · ppl 11.75 · 2 s | 2.770 · ppl 15.96 · 1 s |
| Alice | 1.614 · ppl 5.03 · 243 s | **1.580** · ppl 4.86 · 174 s | 1.633 · ppl 5.12 · 285 s |
| Kalila wa-Dimna | 1.951 · ppl 7.03 · 333 s | **1.938** · ppl 6.95 · 149 s | 1.940 · ppl 6.96 · 588 s |
| parameters (names / Alice / Kalila) | 25.0 k / 32.1 k / 30.0 k | 25.6 k / 32.4 k / 30.0 k | 24.1 k / 32.0 k / 31.3 k |

Larger models, each trained once, within about 20–25 minutes of one core:

| model | parameters | validation | time |
|---|---|---|---|
| Alice: 4 blocks of 64, 4 heads, dropout 0.2, context 128 | 213 k | **1.564** · ppl 4.78 | 1 346 s (22 min, 30 epochs) |
| Kalila: `--preset arabic-gpt` (3 blocks of 64, dropout 0.1) | 162 k | **1.883** · ppl 6.57 | 979 s (20 epochs) |
| Kalila: `--preset arabic-text` (two LSTMs of 96, phase 4) | 130 k | **1.863** · ppl 6.44 | 865 s |

The "best Transformer in reasonable CPU time" budget was about half an
hour of one core for Alice. Three larger configurations were tried; the
one above won. 4 blocks of 96 (467 k weights, dropout 0.1, 22 epochs, 27
min) reached 1.627 but overfitted hard (training loss 1.07 against 1.63
on validation): Alice's 130 k training characters are too few for it; the
same size with dropout 0.3 was behind (1.688 after 18 epochs) and was
stopped.

Honest summary:

* **At equal size the Transformer does not win.** On the books it lands
  next to the recurrent models — on Alice behind both (1.633 vs 1.614 /
  1.580), on Kalila level with the GRU (1.940 vs 1.938) — and it takes
  1.2–4× longer to train (every position attends to every earlier one, and
  each epoch also re-evaluates the training set). On the names it is
  behind, and on the 504 Arabic names clearly worse (2.77 vs 2.46): with
  2.7 k characters it overfits after 9 epochs, before attention has learned
  much. A recurrent model's built-in bias — read in order, remember
  recent things — is worth a lot when data is small.
* **With more capacity it wins on Alice**: 213 k weights with dropout give
  1.564, better than every 32 k model. On Kalila the larger recurrent model
  is still slightly ahead (phase 4's two stacked LSTMs, 130 k: 1.863;
  arabic-gpt, 162 k: 1.883).
* **Tiny Transformers need large learning rates**, ten times Adam's usual
  0.001 (decision 106), and a warm-up.
* What the comparison does show clearly is the *gradient path*
  ("Attention vs recurrence" above): the Transformer's gradient 63
  characters back is 120× the LSTM cell's and 18 000× the GRU's. At this
  scale, next-character prediction does not need that reach.

Generated with `--seed 1`, `--temperature 0.7` for the books (unedited):

* **English names**, names-gpt: Darria, Corinette, Telenink, Sandie
  *(in the data)*, Fantter, Lathalin, Tateny, Jeissa, Tamima, Danusta.
* **Arabic names**, names-gpt: د، جنى *(in the data)*، نسد، ناسراسسسن، زة،
  فالك، ن، سغامم، دياماكة، لمسعكياريه — one-letter "names" and runs of س:
  the weak model of the table above.
* **Alice**, char-gpt (32 k):

  > Alice broud the had grearer again, so not the Queen short the went once
  > looked of you nead. “Well that way be with tick I urong, this a lef at
  > sterying in good-assoon to as all course.

  The 213 k model:

  > Alice angoing the exectly, and to little for the hedgehog to dready to
  > childen they dish as woment had the baby a down a but, and story see at
  > so sitting to of as something about the could for thing minute were as is
  > the fancy, and this the was she was she cament her sing had could
* **Kalila wa-Dimna**, the matched Transformer (31 k):

  > قال الملك وأنا ألا يجد على القيام من التي عن العمري. فلما الملك من غير من
  > ذلك إليك تعاص، فيه من الأحدية. فأخرج بذلك فلا يدنو من الملك في أحسن من سوء
  > الصيادانة والملك. … قال كليلة: فإنما ولا على الملك …

  arabic-gpt (162 k):

  > قال إلى الماء والصديق الذي لا يستعمل لي من كان له من الصور به لا يقول له في
  > الأسود في المخيل ولا يجاءه ولا ينبغي من قد أن يدعه لهم لا يحتاجه لي تكون
  > على السمك … فلما جماعلم أن يحشر عليه عند سرار الصياده من قد أحسن على
  > المنزلة، وقال: أيها ال

Judged honestly: the English names are as name-like as the LSTM's
(Corinette, Lathalin, Jeissa, Tamima) — the loss difference is not
visible in ten samples. The Arabic names are the worst output of the
project. The 32 k Alice text has the book's shape (dialogue, "the Queen")
but more broken words than the GRU's; the 213 k model has fewer broken words and the book's
vocabulary (the hedgehog, the baby, "sitting", "something about") but
still no sentence that means anything — 0.07 nats are not a different kind
of text. On Kalila the
matched Transformer writes mostly real words (الملك، القيام، فلما، أحسن،
كليلة), the book's frame ("قال كليلة:") and correct prefixes, like the
recurrent models — and arabic-gpt's is the most fluent Arabic of the project: long runs of
real words and phrases (الذي لا يستعمل، ولا ينبغي، أن يدعه، أحسن على المنزلة،
"وقال: أيها"), the fable's water and fish (الماء، السمك، الصياد), few invented
words (جماعلم), and still no meaning beyond a phrase.

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

`make test` runs 86 tests (~106 000 checks) in a few seconds:

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
* **gated layers**: the LSTM and GRU forward against hand-computed values;
  gradient checks over 6 steps with per-step heads, with the loss on the
  last step only, with a carried state (h and c ≠ 0), stacked (lstm→lstm,
  gru→gru, lstm→dropout→gru→dropout→rnn in evaluation mode, lstm→dropout→
  lstm in training mode); **the gradient into c_{t−1} and h_{t−1}** from
  the trace against central differences of the loss in the carried state;
  carrying the state over two halves = one pass over the whole; orthogonal
  initialisation (WᵀW = I for every gate block); forget-gate bias, save and
  load of both layers;
* **phase 4 text**: Arabic normalisation (tashkeel, tatweel, alef forms;
  the vocabulary and the saved flags), batch generation equals one-at-a-time
  generation row by row, a separate validation set, and **a character RNN
  written by the last phase-3 commit still loads and generates exactly the
  same names**;
* **Transformer**: the blocked matrix product bit-identical to the naive
  loops (36 shapes × transposition/accumulation flags); LayerNorm, the
  sinusoidal position table and one attention head against hand-computed
  values; gradient checks of every building block (LayerNorm, learned and
  sinusoidal positions, attention with E = D and E ≠ D, GELU and ReLU
  feed-forward, the block with dropout in evaluation and training mode),
  of whole two-block models (tied and untied output, learned and
  sinusoidal positions, a loss on the last position only); **the causal
  mask** (the output at t is bit-identical when later inputs change, and
  ∂L/∂x_j = 0 exactly for j > t); **weight tying** (one matrix, both
  gradients, save/load keeps the tie); **the KV cache equals recomputing
  the window** for small windows, slides, long primes, items and batches;
  **AdamW and warm-up + cosine against hand-computed steps** (and which
  parameters decay); save/load of a Transformer; the attention trace
  events; and **a model written by the last phase-4 commit still loads and
  generates exactly the same names**;
* CSV and IDX parsing errors; every API entry point with `NULL`/bad input.

`make sanitize` runs all of that plus `tests/cli_smoke.sh` (212 CLI cases,
including image layers on a generated dataset, English and Arabic name
models, generation determinism, traces in UTF-8 and ASCII, invalid UTF-8,
LSTM/GRU traces, `show gates`, `compare`, stacked models, Arabic
normalisation, Transformer specs and presets, attention traces in UTF-8 and
ASCII (Arabic included), `show attention`, `show params`, KV-cache and
no-cache generation, no-network commands, malformed input, a 300 KB line and random
binary input, and MNIST, Alice and Kalila wa-Dimna when present) under
ASan + UBSan with leak detection. `make test OMP=1` and `make sanitize
OMP=1` run the same in the OpenMP build. The slow benchmarks are in `make bench`.

## Layout

```
include/neuromz.h        public API          src/        libneuromz (portable C11)
include/neuromz_layer.h  layer-author API    cli/        the neuromz program
docs/DESIGN.md           design + decisions  tests/      test suite + CLI smoke test
examples/                data, C demo, benchmarks        scripts/    MNIST / text download
legacy/                  v1 (2017–2020)      data/       MNIST, Alice, Kalila (downloaded, gitignored)
```

Dataset sources and licenses: [`examples/DATASETS.md`](examples/DATASETS.md).

Model files (`.nmz`) use a portable little-endian format with a magic number,
version, optional sections (normalization, optimizer state, vocabulary,
text normalisation, text-training settings) and CRC-32 (see DESIGN §7).

## Roadmap

Done: dense networks (phase 1); convolution, pooling, dropout, Adam,
schedules, normalization and MNIST (phase 2); text, embeddings, a vanilla RNN
with truncated BPTT, clipping, validation/early stopping and generation
(phase 3); LSTM, GRU, orthogonal initialisation, stacking, gate traces, the
gradient-survival comparison, Arabic prose and batch generation (phase 4 —
added with no change to the engine, as DESIGN §3.1 planned); a decoder-only
Transformer with AdamW, warm-up, a blocked matrix product, a KV cache and
attention traces (phase 5 — DESIGN §3.2). Next: a GUI that consumes the
trace events.
