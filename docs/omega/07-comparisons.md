# Omega 07: comparisons

**Prerequisites:** [page 03](./03-train.md) and [page 04](./04-infer-an-llm.md): this page compares what those pages ran, and every command here is one of theirs.

**You will:** see, for each of the five targets, either a measured row with the exact command, the hardware and the variance, or a named gap with the artifact that is missing. There is no third category on this page.

**New concepts (in order):** what a comparison row must carry · the machine · omega decode measured · llama.cpp on the same checkpoint · burn on the same dataset · PyTorch · whisper.cpp · stable-diffusion.cpp · reading the table.

## 1. What a row must carry

A row is admissible only with all of: the exact command, the build profile, the hardware, the number of runs, the mean, and the spread. A number without those is a measurement, not a result — it cannot tell you whether a difference is the code or the afternoon. Where any of those is missing, the row below says `gap` and names the artifact that would complete it.

The machine for every row: Apple M1 Max, 10 cores, macOS on `arm64`, load average under 30 before each run, one job at a time. The build profile is named per row, because a debug wall clock is not comparable to a release one.

The five targets at a glance:

| target | row | number, or the missing piece |
|---|---|---|
| llama.cpp | measured, same checkpoint | omega 21.08 ms/token warm vs llama.cpp 9.59 ms/token; **omega is 2.20x slower** (N=3 / 5) |
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

Read that as what it is: a 1B f16 decode, one prompt, 24 tokens, on a shared laptop, in a release build on both sides. It says nothing about long context, batching, or another family.

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
... (1250 iterations over 20 epochs)
```

Read the schedule before any number: `items_total: 320000`, `epoch_total: 20`, batch 256 (`examples/mnist/src/training.rs:43`), an AdamW optimizer with cautious weight decay (`:64-65`), and a two-convolution model with dropout before its linear layers (`examples/mnist/src/model.rs:15-23`, `:34-40`). Page 03 trains a 784-128-10 MLP over 8,000 images for 4 epochs. **These are not the same workload**, so their wall clocks are not comparable and this is not the row the brief asked for.

What the run does give is a rate. From its own log (`/tmp/burn-example-mnist/experiment.log`): iteration 1 at `14:39:45.965`, iteration 50 at `14:41:36.237` — 49 iterations in 110.3 s, **2.25 s per 256-image iteration**, while other work ran on this machine, so it is an upper bound on burn's own speed. At 1250 iterations its schedule is about 47 minutes.

**Gap.** A comparable training row needs burn configured to page 03's network and schedule — 784-128-10, 8,000 examples, batch 32, 4 epochs — which is an edit to `examples/mnist/src/model.rs` and `examples/mnist/src/training.rs:39-45`, not a flag. Until that exists, burn has a measured rate and no accuracy number that could sit beside `0.9180`.

## 4. PyTorch

**Gap.** PyTorch is not installed on this machine and this workspace does not install Python tooling, so there is no PyTorch row. The artifact that would complete it is a PyTorch build on the same M1 Max running an equivalent 784-128-10 MLP for four epochs on the same idx files (`~/.cache/burn-dataset/mnist`), reporting the same three numbers page 03 reports: first-epoch mean loss, last-epoch mean loss, and test accuracy. Page 03's numbers are therefore comparable to burn only, not to PyTorch.

## 5. whisper.cpp

**Gap.** `~/repos/others/whisper.cpp` is source only (`build/bin` is empty) and there is no `ggml-*.bin` checkpoint on this machine. Converting `~/models/asr/whisper-tiny` (HF safetensors) to ggml needs whisper.cpp's Python converter, which this workspace does not run. The artifact that would complete the row is a `ggml-tiny.bin`, plus a proxima whisper path to compare against — page 05 explains why the second does not exist yet.

## 6. stable-diffusion.cpp

**Gap.** `~/repos/others/stable-diffusion.cpp/build/bin/sd-cli` exists (`master-1052-f5df5f8`), but there is no image path in omega to run against it, so wall time per image has nothing to compare. The checkpoints on disk (`~/models/image_gen/flux2-klein-4b`, `~/models/diffusion_pixel/ddpm-cifar10-32`) are diffusers layouts, and whether `sd-cli` loads them was not verified. The artifact that would complete the row is a proxima image path (`proxima-model-interop/specs/flux2-klein/SPEC.md`, draft) plus a checkpoint both sides load.

## 7. Reading the table

- **Decode** is the one row where both sides run the same weights and the same token count, so the milliseconds are comparable. It is still one prompt on one machine: it says nothing about long context, batching, or another model family.
- **Training** is comparable only if both sides use the same data files and the same network; page 03 fixes omega's side, section 3 of this page shows that burn's shipped example does not, and names what would have to change.
- **A gap is not a loss.** Three of five targets are gaps for missing artifacts, not for missing performance. Saying so is the point of the page: a benchmark you did not run is a fact about your evidence, not about your code.
