# Card 23: isolate per-head SIMD topology on Granite prefill

## Question

At sized Legacy simdgroup count and the F16-MMA/F32-KV baseline, does changing only `simd_topology` to `PerHead` preserve captured output/IDs and lower cached-attention GPU timestamps at actual 256- and 972-token prompt sizes across three serial processes?

## Pre-registered hypothesis and kill criterion

Hypothesis: per-head mapping of row/head ownership lowers the median cached-attention timestamp at both established prompt sizes in each of three serial processes. Kill this candidate if any one of the six shape/process cells has selected p50 greater than or equal to Legacy p50, or if output/ID equality fails. A killed candidate remains recorded and does not alter serving defaults.

This is a schedule experiment on the existing Granite replay inputs, not evidence for other prompts, models, or complete request latency. The output equality gate is required before timing is accepted.

## Source admission and one-variable contract

`validate_simd_topology_selection` admits PerHead only for row-tiled attention with at least two grouped query heads and when `head_dim` is divisible by `16 * resolved_simdgroup_count`. Granite's captured head dimension is 64 and the sized Legacy topology resolves to two simdgroups, so the source admission expression evaluates to `64 % 32 == 0`. The test asserts that resetting `simd_topology` to Legacy returns the selected `AttentionVariant` to the exact baseline.

Both arms use F16 MMA, F32 K/V, Legacy K/V reuse, Legacy tile height, Legacy query parallelism, Legacy simdgroup count, and prefetch off. Only `simd_topology` changes. MSL maps the per-head vector blocks and block ownership through distinct `simd_per_head` formulas (`omega/src/msl/cached_attention_row_tiled.rs:480-501`); this establishes a source mapping change, not a timing mechanism.

## Commands

Build the focused integration binary with one Cargo job, copy it from Cargo's `deps` directory, and run three Metal processes serially. Each process writes full stdout/stderr and a distinct report.

```sh
nice -n 20 env CARGO_BUILD_JOBS=1 CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo test -p proxima-model-interop --features std,metal,instrument,metal-attn-split-rows,metal-attn-variants --test granite_attention_variant_prefill --no-run
cp /tmp/cargo_target/debug/deps/granite_attention_variant_prefill-4c29247f309c5b09 /tmp/card23-granite-attention-tests
for run in 1 2 3; do
  PROXIMA_TEST_TIMEOUT_MS=900000 PROXIMA_CAPTURE_LIVE=1 PROXIMA_CAPTURE_NODES=all PROXIMA_CAPTURE_STEPS=0 PROXIMA_GRANITE_AB_REPORT="/private/tmp/proxima-granite-main/proxima-tensor/specs/granite-attention-variant-performance/evidence/card23-per-head-run${run}-report.json" /tmp/card23-granite-attention-tests perf_granite_per_head_topology_warmup_control_two_shapes_against_f16_legacy --nocapture
done
```

Each report must pass the positive checker with `--expected-shapes 2 --selected-arm topology_per_head`, then the same invocation with `--negative-controls`. Preserve all report samples, output identities, entries/source hashes, grids/resources, and host context.

## Acceptance

- Each process selects one test, passes one, and captures both actual tokenizer prompt counts with two alternating warmup pairs and 20 measured pairs per arm.
- Each report records the expected four arm-shape observations, equal complete output hashes and generated IDs, and a changed selected entry/source identity.
- The positive checker prints the two-shape, 20-sample counts with zero errors; the negative checker rejects all 13 mutations.
- Report all six cell distributions and paired signs. Apply the pre-registered kill criterion without hiding counter-evidence.

## Result

Focused integration target compiled with one Cargo job in 6.07 seconds. Three serial processes each selected one test, passed one, and filtered 34. Each process captured nominal prompts 256/971 at actual counts 256/972, two warmup pairs, and 20 measured pairs per arm. All six pairs had equal 1 MiB/3,981,312-byte attention outputs and generated ID `[322]`; output SHAs were `454ca3f4483db67f72f42baa6f7625c46853a9be768e4bbcdd35116789f72ba6` and `970fa9d575423441d23b639a5831938d9e9c46aa8c3c6e84fb2d4f7fe0dfbcae` respectively. Every report passed the positive checker and rejected all 13 negative controls (`evidence/card23-report-checks-2026-10-10.log`).

Each cell lists Legacy/PerHead p50, p90, p99 GPU nanoseconds; CoV percent; and PerHead-minus-Legacy signs (+/−/0):

| Run/report SHA256 | Actual tokens | p50 ns | p90 ns | p99 ns | CoV % | signs +/−/0 |
|---|---:|---:|---:|---:|---:|---:|
| 1 `4a5be7b33a52…a72b1f792` | 256 | 182,625.0 / 182,250.0 | 182,875.1 / 182,875.1 | 187,000.0 / 183,000.0 | 0.59 / 0.37 | 5/14/1 |
| 1 | 972 | 1,099,083.3 / 1,092,291.7 | 1,116,625.0 / 1,111,375.1 | 1,124,000.1 / 1,130,000.1 | 2.17 / 1.78 | 10/10/0 |
| 2 `9dec4807e84a…ab1afedd9` | 256 | 182,458.3 / 182,041.7 | 186,416.7 / 182,916.6 | 186,999.9 / 183,083.3 | 0.91 / 0.33 | 7/13/0 |
| 2 | 972 | 1,101,125.0 / 1,092,625.1 | 1,121,375.0 / 1,130,125.0 | 1,143,374.9 / 1,153,375.0 | 1.55 / 2.65 | 7/13/0 |
| 3 `fd455f0f8d02…13830b6cd` | 256 | 181,750.0 / 182,041.7 | 182,500.0 / 183,375.0 | 184,791.6 / 183,749.9 | 0.48 / 0.46 | 12/8/0 |
| 3 | 972 | 1,096,125.0 / 1,097,374.9 | 1,116,125.0 / 1,119,250.0 | 1,203,333.3 / 1,166,041.7 | 2.43 / 2.41 | 11/9/0 |

The pre-registered p50 criterion is killed: in process 3, PerHead p50 is higher at both prompt lengths. Paired timestamp signs are mixed in every cell, including both positive and negative rounds in each. Resource metadata and launch grids are identical between arms: threadgroup width 64, threads 16,384/62,464, `max_threads=448`, `tg_static_bytes=9600`, `exec_width=32`. The entry adds `_simd_per_head`; its source SHA is `2e7a0539cdc125d698c00f01dbfdb5bb6d533ffc732c03ea44f97bc46d3313e8`, versus Legacy `b85db41a81b3503641fbdb453cff0b7dd32e442085614eef5ab20fe1b1e0ba5d`. The six reports preserve all raw paired samples and process context. This replay does not support selecting PerHead as a Granite timing optimization; the device cause remains unexplained.

Report SHA256 values: run 1 `4a5be7b33a52de286c84c524c469d975b99038186fbef00f439dea0a72b1f792`; run 2 `9dec4807e84ab0bceb8258ed3677f3546a041633b628d142a410f55ab1afedd9`; run 3 `fd455f0f8d022e35189ac1112d67dd2fc2dd72eb906ae063145827513830b6cd`.

## Residual

Pipeline metadata and the MSL ownership formulas cannot identify actual device transactions or occupancy. If timestamps move, that mechanism remains unexplained without hardware counters or labeled shader intervals.
