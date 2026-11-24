# Omega 07: comparisons

**Prerequisites:** [page 03](./03-train.md) and [page 04](./04-infer-an-llm.md): this page compares what those pages ran, plus the whisper (page 05) and FLUX (page 06) paths. The llama.cpp, whisper.cpp and sd.cpp rows cite raw logs and scoreboard sections in place of a command.

**You will:** see, for each of the five targets, either a measured row with the exact command, the hardware and the variance, or a named gap with the artifact that is missing.

**New concepts (in order):** what a comparison row must carry · the machine · llama.cpp, by model · burn's own example, and why it is not the same workload · PyTorch · whisper.cpp · stable-diffusion.cpp · reading the table.

## 1. What a row must carry

A row needs the command, the build profile, the hardware, the number of runs, the mean and the spread. Without those you cannot tell whether a difference is the code or the afternoon, so where one is missing the row says `gap` and names what would complete it.

The machine for every row: Apple M1 Max, 10 cores, macOS on `arm64`, one cargo job at a time. The build profile is named per row, because a debug wall clock is not comparable to a release one, and so is the load the row was taken under when it was not a quiet machine.

The five targets, and the answer each one gets today:

| target | row | number, or the missing piece |
|---|---|---|
| llama.cpp | measured, depends on the model | flint (MoE) on Metal, 2026-11-22: TTFT 1.002x, decode 0.904x, peak RAM 0.81x of llama.cpp b12110. Other models range from 0.43x (gemma4 e2b warm TTFT, 2,024 tokens) to 1.48x decode and 7.9x TTFT (granite on CUDA), section 2 |
| burn | rate measured, accuracy not comparable | 2.25 s per 256-image iteration; its example is a CNN over 20 epochs, page 03 is an MLP over 4 |
| PyTorch | gap | not installed, and this workspace installs no Python tooling |
| whisper.cpp | measured, exact token parity | Metal, tiny, 2026-11-24 same window: 0.46-0.47 s vs 0.30-0.32 s against whisper.cpp greedy, the like-for-like run (about 1.5x slower); large-v3-turbo: 62.14 s vs 1.318 s (47.2x slower, not re-measured). Section 5 |
| stable-diffusion.cpp | measured, FLUX.2 klein 4B | Metal, 2026-11-24 same window: 48.87 s vs 59.80 s median wall (0.82x), n=3. Model load is the stage omega loses, section 6 |

Ratios are omega over incumbent, so below 1.0x omega takes less time or memory. The numbers on this page come from the slot-0 `kv-epic/SCOREBOARD.md` (llama.cpp, whisper.cpp, sd.cpp) and from page 06 section 4; each section names its date and run count.

## 2. llama.cpp: the result depends on the model

The earlier row on this page, omega 21.08 vs 9.59 ms/token on Llama-3.2-1B f16 (2026-11-21, 2.2x slower), is superseded. omega's current path is flint, served through ragd, and the gap to llama.cpp is small or negative on one model and positive on another. Source for every row below: slot-0 `kv-epic/SCOREBOARD.md`, sections "flint (llm, MoE) vs llama.cpp b12110" and "other models and backends", read 2026-11-24.

**flint, MoE, 4,069-token prompt, Metal, llama.cpp b12110.** The llama arm is `/opt/homebrew/bin/llama-server`, which reports `version: 0.9.0 (build 12110, commit cd959bae9)` on `MTL0: Apple M1 Max`. Launch line: `~/.cache/kvgate/proximaomega62/attn-fa/sb679/block10/run.sh:27`. Flash attention on both sides. Runs admitted only at load below 4.5, 2026-11-22 13:47-14:28 CST, n=1 per arm per run, 8 runs in all.

| metric, block 10 | omega | llama.cpp | ratio |
|---|---|---|---|
| first-request TTFT | 4,964.0 ms (main `ca546929e`); 4,975.1 ms (636 `676c5aca7`) | 4,953.2 ms | 1.002x; 1.004x |
| decode | 18.76 ms/token (main); 18.83 (636) | 20.76 ms/token | 0.904x; 0.907x |
| peak RAM (`time -l` footprint) | 439.2 MB (main); 460.3 MB (636) | 540.3 MB | 0.81x; 0.85x |

Omega's decode is `(total - ttft) / 63`; llama.cpp's is its own `predicted_per_token_ms`. All 8 runs keep the order: decode omega 18.76-19.37 vs llama.cpp 20.48-20.76 ms/token, RAM omega 424-477 vs llama.cpp 540-554 MB. The llama Metal claim rests on the launch line and the binary's identity; `llama_server_r1.log` has no device banner. One prompt, n=1 per run: a direction, not a fit.

**Other models** (ratio is omega over llama.cpp; a decode row gives tok/s where the board does):

A decode ratio marked (derived) is omega's time per token over llama.cpp's, computed from the board's tok/s as llama.cpp tok/s over omega tok/s, so below 1.0x omega is faster, as everywhere on this page.

| model | backend | metric | omega | llama.cpp | ratio | date and caveat |
|---|---|---|---|---|---|---|
| gemma4:e2b-it-qat, short prompt | Metal | warm TTFT; decode | 60.8 / 50.4 ms; 75-83 tok/s | 68.2 ms; 49.1 tok/s | 0.89x TTFT; 0.59-0.65x decode (derived) | 2026-11-21, b11671, load 43, quiet gate off; I4 (token ids) passes. Cold 1,195 ms vs 133 ms (9.0x), a first-use compile |
| gemma4:e2b-it-qat, 2,024 tokens | Metal | warm TTFT; decode | 558.8 / 577.6 ms; 76 tok/s | 1,298.4 ms; 56.1 tok/s | 0.43x TTFT; 0.74x decode (derived) | same run; cold 1,180.8 vs 1,399.2 ms (0.84x) |
| granite (ollama blob), 2,024 tokens, f32 | Metal | warm TTFT; decode; peak RAM | 581.7-613.8 ms; 51.9 tok/s; 553.3 MB | 500.3 ms; 44.7 tok/s; 387.5 MB | 1.16-1.23x TTFT; 1.43x RAM; 0.86x decode (derived) | 2026-11-21, b11671, load 47, quiet gate off; I4 (token ids) passes |
| granite (ollama blob), 2,024 tokens, f16 | Metal | warm TTFT; decode | 558-572 ms; 5.91 ms/token | about 426 ms; 5.34 ms/token | 1.31x TTFT; 1.11x decode | 2026-11-21, b12110, load 32; ids diverge, so not a like-for-like row |
| granite-4.0-350m Q8_0 | Metal | decode | 4.03 ms/token | 3.61 ms/token | 1.12x | mid-November probe; the owner reports about 2x now, unreproduced |
| granite | CUDA, RTX 2070 | decode; TTFT | 5.35 ms/token; 1,189 ms | 3.62 ms/token; 151 ms | 1.48x; 7.9x | 2026-11-13 |
| qwen3.6:35b-a3b, short prompt | Metal | TTFT; decode | 5,486-6,024 ms; 152-155 ms/token | no run: llama-server never became healthy in 180 s | - | 2026-11-21; omega only. 24.6 GB steady RSS is the full GGUF mapping |

Not measured: granite at 5,005 tokens (the 2026-11-13 1.24-1.44x decode and 1.6-1.8x TTFT rows are stale), gemma4-26b, qwen3:8b and 4b.

Two things move these numbers more than the code does. The first request after a build pays a Metal pipeline compile; on granite it moved from 3.0 s of first-request TTFT into load-time prewarm. And load average: the rows above carry theirs, and a window under load 78 put an earlier omega decode between 24 and 64 ms/token for a command that ran 21 ms/token quiet.

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

49 iterations in 110.3 s is **2.25 s per 256-image iteration**. Other work ran on this machine while it trained, so that number is an upper bound on the time per step. At 1250 iterations its own schedule is about 47 minutes.

**Gap.** A comparable training row needs burn configured to page 03's network and schedule — 784-128-10, 8,000 examples, batch 32, 4 epochs — which means editing `examples/mnist/src/model.rs` and `examples/mnist/src/training.rs:39-45`. Until that exists, burn has a measured rate and no accuracy number that could sit beside `0.9180`.

## 4. PyTorch

**Gap.** PyTorch is not installed on this machine and this workspace installs no Python tooling, so there is no PyTorch row. The artifact that would complete it is a PyTorch build on the same M1 Max running an equivalent 784-128-10 MLP for four epochs on the same idx files (`~/.cache/burn-dataset/mnist`), reporting the same three numbers page 03 reports: first-epoch mean loss, last-epoch mean loss, and test accuracy. Page 03's numbers are therefore comparable to burn only.

## 5. whisper.cpp

omega transcribes jfk.wav (11 s) end to end (page 05) and is slower than whisper.cpp v1.9.5 on every row. Source: slot-0 `kv-epic/SCOREBOARD.md`, section "whisper (speech) vs whisper.cpp v1.9.5", read 2026-11-24. M1 Max. Token ids match whisper.cpp exactly on every row (24/24 tiny, 27/27 large-v3-turbo).

| model | device | omega | whisper.cpp | ratio | commit |
|---|---|---|---|---|---|
| tiny | Metal | 0.46-0.47 s warm (2.02 s cold first run, shader compile) | 0.30-0.32 s greedy (`-bs 1 -bo 1`); 0.34-0.35 s defaults (beam 5, best-of-5) | about 1.5x slower than greedy, about 1.36x than defaults | `fad2fd1c6db` |
| tiny | CPU | 20.92 s | 0.506 s | 41.3x | `ca546929e`, `evaluate_named` |
| large-v3-turbo | Metal | 62.14 s | 1.318 s | 47.2x | `ca546929e` plus the Metal backend, load 3.8-14.5 |
| large-v3-turbo | CPU | 150.92 s | 1.318 s | 114.5x | `ca546929e` |

The tiny Metal row is one window, 2026-11-24 about 17:0x: 3 rounds alternating omega on GPU, whisper.cpp defaults, whisper.cpp greedy and omega on CPU, each quiet-gated, jfk.wav, ids 24/24 in all 6 omega runs; logs are under `~/.cache/kvgate/proxnum683/stack/runs/`. The like-for-like incumbent is whisper.cpp greedy, because omega's decode is greedy; the defaults run does more decode work (beam 5, best-of-5). The owner's bar is about 300 ms for tiny; the 5% bar against greedy is 0.325 s (1.05 x 0.31 s), and omega is about 1.43x over. The turbo row was not re-measured; its 5% bar is 1.384 s, and omega is 44.9x over.

Where the time goes, tiny on Metal (omega vs whisper.cpp greedy): mel 133-137 ms vs 4.5-4.6 ms (about 29x), encode 94-95 ms vs 36-38 ms (about 2.6x), decode 150-155 ms (6.3-6.5 ms/token) vs 43 ms (about 3.5x), footprint 379-380 MB vs 232-233 MB (about 1.6x; omega's RSS is 440 MB). Mel carries most of the gap, then decode. Omega on CPU in the same window took 6.62 s, of which encode was 6.28 s, so the 20.92 s CPU row above is older.

For turbo the only stage breakdown is the CPU one from before PROXNUM-681 (n=1, `2a0bfa33d`): 123.38 s wall, of which encode 104.4 s and decode 17.58 s (651 ms/token, 27 tokens), against whisper.cpp's 378.4 ms encode and 1.91 ms/token. `user` 117.27 s against `real` 123.38 s, so about one core. Peak RAM was 22.79 GB against 1.84 GB; on the unlanded `card/proxnum-681` it is 4.81 GB (about 2.6x) with 27/27 ids, and wall is unchanged because the work still runs on the CPU.

## 6. stable-diffusion.cpp

FLUX.2 klein 4B, 1024x1024, 4 Euler steps, seed 42, Metal, against `sd-cli` `master-1052-f5df5f8`. Medians of 3 rounds, interleaved omega then sd.cpp, 2026-11-24 11:42-11:53, quiet 0/0/0, wall from `/usr/bin/time -l`. Raw logs: `~/.cache/kvgate/proxnum699/rebench/{u1,t1,sd}-r{1,2,3}.log`, tables in page 06 section 4 and slot-0 `kv-epic/SCOREBOARD.md`.

| stage | omega | sd.cpp | ratio |
|---|---|---|---|
| wall, same window | 48.87 s (51.89 / 48.87 / 44.43) | 59.80 s (76.71 / 59.66 / 59.80) | 0.82x |
| wall, cold first run | 51.89 s | 76.71 s | 0.68x |
| wall, warm, back to back | 40.82 s (40.33 / 40.85 / 40.82), `5d5167b864` | 59.80 s (the same-window median) | 0.68x |
| text encode | 4.69 s | 5.09 s | 0.92x |
| DiT, 4 steps | 25.59 s | 41.79 s | 0.61x |
| VAE decode | 3.66 s | 12.28 s | 0.30x |
| peak footprint | 12.32 GB | 14.80 GB | 0.83x |

The time cost omega carries is DiT load. omega takes 6.3 s warm and 9.2-16.1 s in the same window; sd.cpp takes 1.74-1.87 s for the DiT tensors (3.4x warm, 4.9-8.6x same window). sd.cpp's separate 4.14-5.22 s is its qwen text-encoder load, which falls on omega's text stage. omega decodes the 7.75 GB bf16 DiT to f32, re-encodes to f16 and copies; sd.cpp reads the bf16 into place. Planned fix: PROXNUM-699 U2-U7, views of the mapping with no re-encode.

The load tracks the file-backed page count at start (567,751 to 851,597 pages for today's code, against 856,404 to 1,123,457 for the control binary, which has the same compute). Rounds 2 and 3 of today's code ran straight after an sd.cpp run; its 14.8 GB footprint evicts about 300k file pages. Three rounds, one prompt: a direction. The sd.cpp round 1 wall of 76.71 s is unexplained. The image scored 32.58 dB against the reference on 2026-11-22 (49.16 dB with reference text) and was not re-scored on 2026-11-24. The scoreboard contract asks for 20 runs per arm; this is n=3.

## 7. What omega does not do yet, and how to read the table

- The llama.cpp, whisper.cpp and sd.cpp rows run the same weights on both sides, so the times are comparable. One prompt on one machine says nothing about another model: section 2 has omega ahead on flint decode and behind on granite.
- Training is comparable only if both sides use the same data files and the same network; page 03 fixes omega's side, section 3 of this page shows that burn's shipped example does not, and names what would have to change.
- A gap is missing evidence. One of five targets, PyTorch, has nothing to compare because an artifact is missing. A benchmark nobody ran says which artifact is missing; it does not say how fast the code is.
