# Omega 07: comparisons

**Prerequisites:** [page 03](./03-train.md) and [page 04](./04-infer-an-llm.md): this page compares what those pages ran, and every command here is one of theirs.

**You will:** see, for each of the five targets, either a measured row with the exact command, the hardware and the variance, or a named gap with the artifact that is missing.

**New concepts (in order):** what a comparison row must carry · the machine · omega decode measured · llama.cpp on the same checkpoint · burn's own example, and why it is not the same workload · PyTorch · whisper.cpp · stable-diffusion.cpp · reading the table.

## 1. What a row must carry

A row needs the command, the build profile, the hardware, the number of runs, the mean and the spread. Without those you cannot tell whether a difference is the code or the afternoon, so where one is missing the row says `gap` and names what would complete it.

The machine for every row: Apple M1 Max, 10 cores, macOS on `arm64`, one cargo job at a time. The build profile is named per row, because a debug wall clock is not comparable to a release one, and so is the load the row was taken under when it was not a quiet machine.

The five targets, and the answer each one gets today:

| target | row | number, or the missing piece |
|---|---|---|
| llama.cpp | measured, same checkpoint | omega 21.08 ms/token warm vs llama.cpp 9.59 ms/token; **omega is 2.2x slower** (N=3 / 5) |
| burn | rate measured, accuracy not comparable | 2.25 s per 256-image iteration; its example is a CNN over 20 epochs, page 03 is an MLP over 4 |
| PyTorch | gap | not installed, and this workspace installs no Python tooling |
| whisper.cpp | gap | source only, no `ggml-*.bin`, and the converter is Python |
| stable-diffusion.cpp | gap | `sd-cli` exists, but omega has no image path to run against it |

## 2. llama.cpp: decode on the same checkpoint

Both sides load `~/models/Llama-3.2-1B-Instruct/Llama-3.2-1B-Instruct-f16.gguf`, a 16-bit GGUF of Llama-3.2-1B-Instruct, and generate 24 tokens greedily.

omega, release build, through `decode_loop_wall_clock` (page 04):

```sh
PROXIMA_OPENCHAT_GGUF=$HOME/models/Llama-3.2-1B-Instruct/Llama-3.2-1B-Instruct-f16.gguf \
PROXIMA_MAX_TOKENS=24 \
PROXIMA_PROMPT="The capital of France is" \
cargo run --release -p proxima-model-interop --features std,metal --example decode_loop_wall_clock
```

```text
decode_summary tokens_generated=24 stopped_by_eos=false total_wall_clock_ms=501.920 mean_ms_per_token=20.913 generated_text=" Paris. The Eiffel Tower is located in Paris. The Louvre Museum is also located in Paris. The famous" ids=[12366, 13, 578, 469, 3168, 301, 22703, 374, 7559, 304, 12366, 13, 578, 9928, 49606, 16730, 374, 1101, 7559, 304, 12366, 13, 578, 11495]
```

Run three times warm, the same binary printed `mean_ms_per_token` of 20.913, 20.920 and 21.416 — mean 21.083 ms, sample coefficient of variation 1.4% at N=3. The first run after a rebuild was 116.152 ms per token on the same command: Metal compiles its pipelines on first use and caches them.

llama.cpp, the `llama-bench` binary at `~/repos/others/llama.cpp/bin/llama-bench` (build `b2534622`, release), same checkpoint, 24 tokens generated, 5 repetitions:

```sh
~/repos/others/llama.cpp/bin/llama-bench \
  -m $HOME/models/Llama-3.2-1B-Instruct/Llama-3.2-1B-Instruct-f16.gguf -p 0 -n 24 -r 5
```

```text
| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| llama 1B F16                   |   2.30 GiB |     1.24 B | Metal,BLAS |       8 |            tg24 |        104.30 ± 0.98 |
```

`104.30 ± 0.98` tokens per second over 5 repetitions is **9.59 ms per token** (CoV 0.94%). Against omega's warm 21.08 ms, the measured ratio is **2.20x slower** on this prompt, this checkpoint and this machine. The cold omega number, 116 ms, is 12x slower and is not the steady-state ratio — report it as first-request cost, never as decode throughput.

Both sides of this row were measured back to back, with no other build running. That condition is not decoration: a later window on the same machine under load average 78 put omega between 24 and 64 ms per token for the same command. A decode number without the load it was taken under is not reusable.

That is a 1B f16 decode, one prompt, 24 tokens, in a release build on both sides.

## 3. burn: training on MNIST

burn 0.19.1 is source only at `~/repos/others/burn`. Its MNIST example builds and runs here, and the first thing it prints is its own schedule:

```sh
cd ~/repos/others/burn/examples/mnist
cargo run --release --no-default-features --features ndarray --example mnist -j 2
```

```text
    Finished `release` profile [optimized] target(s) in 2m 21s
     Running `/tmp/cargo_target/release/examples/mnist`
TrainingProgress { progress: Progress { items_processed: 256, items_total: 320000 }, epoch: 1, epoch_total: 20, iteration: 1 }
TrainingProgress { progress: Progress { items_processed: 512, items_total: 320000 }, epoch: 1, epoch_total: 20, iteration: 2 }
TrainingProgress { progress: Progress { items_processed: 768, items_total: 320000 }, epoch: 1, epoch_total: 20, iteration: 3 }
```

It keeps printing one such line per iteration; the run was stopped after the numbers below were taken. Read the schedule before any number: `items_total: 320000`, `epoch_total: 20`, batch 256 (`examples/mnist/src/training.rs:43`), an AdamW optimizer with cautious weight decay (`:64-65`), and a two-convolution model with dropout before its linear layers (`examples/mnist/src/model.rs:15-23`, `:34-40`). Page 03 trains a 784-128-10 MLP over 8,000 images for 4 epochs. **These are not the same workload**, so their wall clocks are not comparable and there is no accuracy number here to set beside page 03.

What the run does give is a rate, taken from the two timestamps the learner writes for itself:

```sh
grep -m1 "Iteration 1$" /tmp/burn-example-mnist/experiment.log
grep -m1 "Iteration 50$" /tmp/burn-example-mnist/experiment.log
```

```text
2026-11-21T14:39:45.965060Z  INFO burn_train::learner::strategies::single::epoch: Iteration 1
2026-11-21T14:41:36.236671Z  INFO burn_train::learner::strategies::single::epoch: Iteration 50
```

49 iterations in 110.3 s is **2.25 s per 256-image iteration**. Other work ran on this machine while it trained, so that number is an upper bound on the time per step, not a lower bound. At 1250 iterations its own schedule is about 47 minutes.

**Gap.** A comparable training row needs burn configured to page 03's network and schedule — 784-128-10, 8,000 examples, batch 32, 4 epochs — which is an edit to `examples/mnist/src/model.rs` and `examples/mnist/src/training.rs:39-45`, not a flag. Until that exists, burn has a measured rate and no accuracy number that could sit beside `0.9180`.

## 4. PyTorch

**Gap.** PyTorch is not installed on this machine and this workspace does not install Python tooling, so there is no PyTorch row. The artifact that would complete it is a PyTorch build on the same M1 Max running an equivalent 784-128-10 MLP for four epochs on the same idx files (`~/.cache/burn-dataset/mnist`), reporting the same three numbers page 03 reports: first-epoch mean loss, last-epoch mean loss, and test accuracy. Page 03's numbers are therefore comparable to burn only, not to PyTorch.

## 5. whisper.cpp

**Gap.** No `ggml-*.bin` exists on this machine and whisper.cpp has never been built here; converting the tiny HF checkpoint needs its Python converter, which this workspace does not run. Page 05 covers the rest: even with a ggml model there is no proxima whisper path to run against it.

## 6. stable-diffusion.cpp

**Gap.** `sd-cli` exists (`master-1052-f5df5f8`), but omega has no image path, so wall time per image has nothing to compare. Page 06 says what is missing; the row needs that path plus a checkpoint both sides load.

## 7. What omega does not do yet, and how to read the table

- **Decode** is the one row where both sides run the same weights and the same token count, so the milliseconds are comparable. One prompt on one machine says nothing about long context, batching, or another model family.
- **Training** is comparable only if both sides use the same data files and the same network; page 03 fixes omega's side, section 3 of this page shows that burn's shipped example does not, and names what would have to change.
- **A gap is not a loss.** Three of five targets have nothing to compare because an artifact is missing, not because omega lost. A benchmark nobody ran tells you about the evidence, not about the code.
