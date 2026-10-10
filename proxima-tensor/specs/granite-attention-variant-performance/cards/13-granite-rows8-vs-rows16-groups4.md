# Card 13: compare Granite Rows8 and Rows16 with Groups4 fixed

**Dependency:** Card 12 simdgroup-count dispatch and replay captures.

**Purpose:** measure whether changing only `tile_height` moves the captured Granite prefill dispatch when the Groups4 topology is held fixed. The control is F16 MMA, F32 K/V, Legacy K/V reuse, Legacy query parallelism, Legacy SIMD topology, and prefetch off.

**Abandoned design:** no new kernel, selector enum, or replay loop. Reuse the existing `AttentionVariant` axes and `run_variant_prefill_probe_shapes_with_grid_requirement` helper. No pipe applies: this is a sequential test-owned capture/replay with no producer/consumer flow.

## Slice

Add `perf_granite_rows8_vs_rows16_groups4` to `proxima-model-interop/tests/granite_attention_variant_prefill.rs`. Build the baseline from F16 MMA with `tile_height=Rows8` and `simdgroup_count=Groups4`; build the selected variant by changing only `tile_height=Rows16`. Assert every `AttentionVariant` field for both values so default changes cannot add another axis. Run both tokenizer-driven prompt shapes (nominal 256 and 971 tokens) through the existing full-output and replay helper with `require_equal_output=true`; grid equality is not required because the tile height changes dispatch coverage.

Keep the existing production default and kernel source unchanged. The tested axis is runtime opt-in through `metal-attn-variants`; numeric axes are tile height and simdgroup count, structural axes are row ownership, number of row tiles, and dispatch grid. No new allocation or dataflow primitive is introduced.

## Acceptance

Build once with one Cargo job, copy the completed test binary to `/tmp` to avoid the observed Cargo-target-path loader stall, and run three serial processes. Do not overlap Cargo or Metal work. Use the same binary for all three runs.

```sh
CARGO_BUILD_JOBS=1 CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo test -p proxima-model-interop --features std,metal,instrument,metal-attn-split-rows,metal-attn-variants --test granite_attention_variant_prefill --no-run
cp /tmp/cargo_target/debug/deps/granite_attention_variant_prefill-4c29247f309c5b09 /tmp/card13-granite-rows8-rows16-groups4
PROXIMA_TEST_TIMEOUT_MS=900000 PROXIMA_CAPTURE_LIVE=1 PROXIMA_CAPTURE_NODES=all PROXIMA_CAPTURE_STEPS=0 /tmp/card13-granite-rows8-rows16-groups4 perf_granite_rows8_vs_rows16_groups4 --nocapture
```

Repeat the final command two more times serially; save full stdout/stderr and exit status to `evidence/card13-rows8-rows16-groups4-run{1,2,3}-2026-10-10.log`.

| Gate | Required evidence per process |
|---|---|
| Test selection | Exactly 1 named test passes; remaining test count is recorded; process exits 0. |
| Axis isolation | Baseline/selected differ only in `tile_height` (`Rows8`/`Rows16`); both have F16 MMA, F32 K/V, Legacy reuse, Groups4, Legacy query parallelism/topology, and prefetch off. |
| Shape and capture | Exactly two records, tokenizer-confirmed actual token counts for nominal 256/971, matched attention node/extents per pair, distinct entry/source SHA, no fault bindings, replayable captures, nonempty grids. |
| Output and IDs | Full output span compared before timing; `changed_bits=0`, `non_finite_pairs=0`, finite max/RMS differences, and identical generated ID arrays. Preserve observed output element counts and IDs. |
| Timing | 20 finite positive GPU samples per arm per shape in alternating order; retain all raw samples and 20 signed selected-minus-baseline deltas. Record positive/negative/zero counts, p50s, CoVs, min/max, and both arm orders without dropping outliers. |
| Dispatch | Record each arm's entry, MSL SHA256, grid, and `(tg_static_bytes,max_threads,exec_width)`. Grid equality is not required. |
| Across processes | For each shape, preserve all three runs and p50 ranges. CoV above 5% remains labeled noisy; no timing-only cause or whole-request claim. |

If capture, output equality, IDs, or timing fails, preserve the failure payload and do not summarize that cell as a timing comparison. This card establishes only per-dispatch observations; any performance decision remains with the owner.

## Captured records

The three serial runs each selected one test, passed one, and skipped 27. Both actual prompt shapes were emitted per run. All six arm pairs had `changed_bits=0`, `non_finite_pairs=0`, and matching IDs `[322]`; output spans were 262,144 F32 elements at 256 tokens and 994,304 at 971 tokens. Every pair matched node 89 and extents within its shape, had a distinct entry/MSL SHA, and had no fault binding. Each arm retained 20 raw replay times and 20 signed paired differences.

| Actual tokens | Run | Rows8 p50 ns | Rows16 p50 ns | Rows8/Rows16 CoV | Selected − baseline signs (+/−/0) | Output equality |
|---:|---:|---:|---:|---:|---:|---|
| 256 | 1 | 142,666.744 | 171,708.292 | 2.343% / 0.652% | 20/0/0 | 262,144 elements; IDs `[322]` |
| 256 | 2 | 142,791.658 | 171,624.939 | 2.174% / 0.803% | 20/0/0 | 262,144 elements; IDs `[322]` |
| 256 | 3 | 143,375.015 | 172,375.003 | 1.523% / 0.608% | 20/0/0 | 262,144 elements; IDs `[322]` |
| 971 | 1 | 1,043,791.650 | 1,101,250.061 | 3.005% / 1.500% | 19/1/0 | 994,304 elements; IDs `[322]` |
| 971 | 2 | 1,058,833.324 | 1,091,125.072 | 3.212% / 1.907% | 16/4/0 | 994,304 elements; IDs `[322]` |
| 971 | 3 | 1,056,291.629 | 1,091,166.749 | 3.043% / 1.342% | 16/4/0 | 994,304 elements; IDs `[322]` |

The p50 and CoV values are derived by the harness from the 20 measured samples in each linked raw log. Across these captured cells, Rows16's p50 is higher at both shapes in all three runs; status for that per-dispatch timing relationship is **plausible**, with the timing cause unexplained. This does not assign a device-level cause or describe whole-request latency.

The selector changes `tile_height` only. `row_tiled_form` resolves explicit rows into `rows_per_threadgroup`, then computes tile count and split count from that height (`omega/src/msl/signature_tokens_prelude.rs:2800-2845`). The MSL template substitutes `tile_rows`; the kernel uses it in `tile_blocks` and grid-index row-tile calculations (`omega/src/msl/cached_attention_row_tiled.rs:440-457`). Groups4 stays fixed, so captured threadgroup width is 128 in both arms. The captured total grid threads and pipeline resources differ as listed; those observations describe the generated dispatch and do not identify why its timing changed.

All runs captured these same entries and source identities: Rows8 `omega_cached_attention_h8_g2_d64_s3c800000_ln9223372036854775808_up0_r8_n4_b64_rt_mma_f16_tile_rows8_simdgroups4`, MSL SHA256 `e339acf417451e74304c75ca247f45cae137d845597b5fa219c1765fe91d413a`; Rows16 `omega_cached_attention_h8_g2_d64_s3c800000_ln9223372036854775808_up0_r16_n4_b64_rt_mma_f16_tile_rows16_simdgroups4`, MSL SHA256 `f0cb8c7e89e5ba446beb4ca8a6ceee827dff4e58d53a4c58459610539288753a`. Threadgroup width stayed 128. Total grid threads were 32,768→16,384 at 256 tokens and 124,928→62,464 at 971; pipeline resources `(tg_static_bytes,max_threads,exec_width)` were `(9600,576,32)`→`(8960,384,32)`. These are captured dispatch/resource values; they do not explain the timing difference. Full arrays are in `evidence/card13-rows8-rows16-groups4-run{1,2,3}-2026-10-10.log`.

The first sandboxed launch failed before dispatch with `Metal(NoDevice)` and is preserved in `evidence/card13-sandbox-no-device-2026-10-10.log`. `system_profiler SPDisplaysDataType` reported Apple M1 Max with Metal support, while the sandbox process's `MTLCopyAllDevices()` returned an empty list. The three captured runs were launched outside the sandbox and completed on the Mac's Metal device.
