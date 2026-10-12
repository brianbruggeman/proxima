# decode performance lab

This lab uses compiled Rust and the existing `gguf_generate` CLI. Its inputs are the installed Granite 3.1 1B-A400M MoE and Gemma 4 E2B QAT checkpoints. Model math is unchanged. The CLI now exposes the existing speculative statistics API.

## captured evidence

The initial default-configuration campaign is archived in `results/`. It contains 32 attempted arms and 16 OFF/ON pairs. Every pair retained matching token IDs and text; none emitted a request-level speculative draft counter. These observations are recorded individually in `results/pairs.jsonl` and the adjacent raw stdout/stderr files. Matching these arms does not establish model fidelity.

The default ngram-simple configuration cannot draft on these short inputs: `proxima-tokenizer/src/draft/ngram_simple.rs:135` returns while history length is at most `size_n + size_m + 1`. Default sizes are 12 and 48; the longest prompt in the pilot has 13 tokens and the output budget is 24. The second campaign explicitly set sizes 3 and 3 in both arms. The current runner uses sizes 1 and 3. These change experiment configuration without changing the drafter implementation.

Granite emitted `ed` and EOS for the France prefix, and a mixed string for the repeated-color prefix. Its generic routed-MoE binding also omits scaling values present in its checkpoint. Those source differences are documented in `references/inventory.jsonl`. No toggle experiment attributes the emitted tokens to an individual omitted scale. Granite has no speculative verifier in the inspected registry path.

The explicit 3/3 campaign is archived in `results-ngram-3x3/`: 32 captured arms, 16 payload-matching pairs, and no recorded positive draft counter. Every timing difference remains observational. On the M1 Max, Gemma E2B QAT's repeated-color prefix produced the same 24 token IDs in both arms. Across the three non-warmup fresh-process requests per arm, reported generation throughput (including prefill) was:

| configured arm | mean tokens/s | sample CoV% |
|---|---:|---:|
| ngram-simple ON, no recorded drafts | 36.359 | 3.4204 |
| speculation OFF | 36.8487 | 0.5007 |

Sources: `results-ngram-3x3/summary.json:496` and `:367`; per-pair wall differences, including both signs, are in `pairs.jsonl:14-16`. These observations do not establish acceleration.

For the emitted period-three sequence `[2604,3730,3826]`, the drafter's newest matching three-token sequence leaves two following tokens. Its source rejects that continuation because two is below `size_n=3` (`proxima-tokenizer/src/draft/ngram_simple.rs:156-177`). It does not search an older match after rejecting the newest one. With `size_n=1`, a most-recent match of sampled token `2604` can copy subsequent `[3730,3826]`, since two meets the minimum of one. This source-level reconstruction motivates the 1/3 experiment; request counters and exact payloads determine what actually executed.

The 1/3 campaign is archived in `results-ngram-1x3/`: 32 captured arms and 16 matching-payload pairs. All four Gemma repeated-color ON requests recorded 15 drafted and 15 accepted tokens. Their output IDs and text matched the paired OFF requests. Resolved-setting telemetry records the requested 1/3 values. No other model/prompt combination recorded positive drafts.

For the same M1 Max, greedy Gemma E2B QAT repeated-color prefix, 24 output tokens and three fresh-process requests per arm after warmup exclusion:

| arm | mean reported tokens/s, including prefill | sample CoV% |
|---|---:|---:|
| speculation OFF | 36.3963 | 4.4438 |
| ngram-simple ON, 1/3 | 41.2943 | 1.7941 |

Sources: `results-ngram-1x3/summary.json`, `pairs.jsonl:14-16`, and `gemma-4-e2b-qat-repeated-colors-pair-01-on.stdout:88-136`. Each non-warmup repetition pair had lower ON process wall time; other model/prompt pairs include both signs and stay outside executed-draft comparisons. This does not establish general acceleration or model fidelity.

Token-emission interval CoV was 161.0037% ON and 63.7427% OFF across 69 intervals per arm. ON emits accepted batches at identical millisecond timestamps, producing zero intervals between tokens interleaved with verifier waits. Its interval p50 was zero and p99 was 90 ms; OFF p50 was 16 ms and p99 was 78 ms. This is a burstier emission record despite the request-level throughput observations. Per-token emission spacing and per-request throughput CoV measure different variation.

## variance and provenance

`summary.json` reports `cov_percent = 100 * sample_standard_deviation / mean`, using denominator `n - 1`. Fewer than two samples, nonpositive means, and signed wall-time deltas produce null CoV. It also includes sample count, standard deviation, range, and nearest-rank percentiles.

Observational distributions contain captured non-warmup arms. Comparisons attributed to an executed speculative method additionally require matching payloads, an eligible verifier, and positive recorded draft counts. Pair zero is excluded as warmup. Every raw arm and every pair remains archived, including slower observations and unusable comparisons.

Requests start a fresh process on an Apple M1 Max. Generation throughput includes prefill; reciprocal mean token-event intervals exclude the first token. Token-event intervals measure emission spacing, and accepted draft batches can emit several tokens at the same timestamp. These measurements are not cached-server latency or an independent quality evaluation. Campaigns run sequentially without exclusion of other host applications.

Raw records preserve the actual temporary execution paths and commands. Archived files use the same basenames, so a recorded `.stdout`, `.stderr`, or `.time` basename resolves within its campaign archive.

## public references

- [Gemma MTP drafters](https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/): the E2B assistant checkpoint is downloaded and its digest/tensor header recorded in `references/inventory.jsonl`; Proxima assistant binding is absent.
- [Gemma QAT](https://blog.google/innovation-and-ai/technology/developers-tools/quantization-aware-training-gemma-4/): the local Gemma target is already a QAT conversion. No comparison against PTQ was performed.
- [DiffusionGemma](https://blog.google/innovation-and-ai/technology/developers-tools/diffusion-gemma-faster-text-generation/): its configuration is pinned. Its block decoder and weights were not run locally. Google reports lower output quality than standard Gemma 4 and says Apple Silicon may not get the dedicated-GPU acceleration.
- [LoopCD manuscript](https://arxiv.org/html/2610.02185v1): the formula contrasts complete recurrent iterations. Ordinary Granite and Gemma layers are not those iterations. The pinned Ouro configuration and unexecuted author source are architecture references; Proxima recurrent execution and LoopCD were not run.

## native commands

Run from the Proxima repository root. A measured Metal run requires device access.

```text
cargo build --release --no-default-features --target-dir /private/tmp/proxima-decode-lab-target --example gguf_generate --example decode_performance_lab
cargo test --no-default-features --target-dir /private/tmp/proxima-decode-lab-target --example decode_performance_lab
/private/tmp/proxima-decode-lab-target/release/examples/decode_performance_lab --manifest proxima-tensor/specs/decode-performance-lab/manifest.json --inventory --check-references --check-fixtures
/private/tmp/proxima-decode-lab-target/release/examples/decode_performance_lab --manifest proxima-tensor/specs/decode-performance-lab/manifest.json --binary /private/tmp/proxima-decode-lab-target/release/examples/gguf_generate --output /private/tmp/proxima-decode-lab-new-campaign
```

Use a new output directory for each campaign to preserve its evidence.
