# Card 04: probe shared K/V with simdgroup row parallelism

**Owner:** GPT-6 Luna
**Dependency:** Card 03
**Commit:** `perf(interop): probe shared kv row parallelism`
**Budget:** at most 30 minutes front-to-back, including acceptance and task update.

## Purpose

Test whether the existing SharedKv staging path preserves K/V reuse when each simdgroup owns query rows.

## Scope

Add one captured-dispatch probe to `granite_attention_variant_prefill.rs`. Keep F16 MMA and F32 K/V in both arms. Compare the F16-MMA/F32-KV baseline with a selected variant that changes only `kv_reuse` to SharedKv and `query_parallelism` to SimdgroupRows; leave tile height, SIMD topology, and prefetch at Legacy/Off. Run actual tokenizer prompt lengths 256 and 971. Record the captured source identity and grid, output difference, equal generated IDs checked before timing, 20 alternating replay samples per arm, signed paired deltas, and summaries. Do not change kernel code, defaults, or dispatch selection.

This is a two-axis interaction probe. Card 03 already measured the axes separately and found higher selected replay times; this card isolates their combination against the same F16-MMA baseline so shared staging's effect can be read with row parallelism active.

## Acceptance

```sh
PROXIMA_TEST_TIMEOUT_MS=900000 PROXIMA_CAPTURE_LIVE=1 PROXIMA_CAPTURE_NODES=all PROXIMA_CAPTURE_STEPS=0 CARGO_BUILD_JOBS=1 CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo nextest run -p proxima-model-interop --features std,metal,instrument,metal-attn-split-rows,metal-attn-variants --test granite_attention_variant_prefill -E 'test(~perf_granite_shared_kv_simdgroup_rows)' -j 1 --success-output immediate
```

Require exactly 1 selected test, 1 pass, and 20 filtered out. It emits exactly 2 shape records for actual counts 256 and 971; each has 20 samples per arm, 20 signed deltas, captured entry/grid, output difference summary, equal token IDs, and no fault bindings. Save full output to `evidence/card04-shared-kv-parallel-2026-10-10.log`.

This card reports only isolated dispatch replay. The interaction result does not establish whole-request latency or semantic quality; the four Granite viability requests remain the user-facing generation checks.


## Observation

| actual tokens | baseline p50 ns | SharedKv + SimdgroupRows p50 ns | output / generated ID | record |
|---:|---:|---:|---|---|
| 256 | 182,999.997 | 398,583.361 | bit-identical; `[322]` both | `evidence/card04-shared-kv-parallel-2026-10-10.log:8` |
| 971 | 1,071,374.980 | 2,371,958.340 | bit-identical; `[322]` both | `evidence/card04-shared-kv-parallel-2026-10-10.log:9` |

The combined selector's captured replay p50 was higher at both prompt lengths. SharedKv did not recover the row-parallel slowdown in this two-axis interaction. This does not allocate the slowdown among threadgroup memory traffic, barriers, occupancy, or cache behavior; the log has no GPU hardware counters. The source emits `threadgroup_barrier` around staged K/V fragments and reloads those fragments (`omega/src/msl/cached_attention_row_tiled.rs:578-600,767-805`); the exact hardware contribution remains unmeasured.

No new API or kernel primitive was introduced. The test reuses the existing `AttentionVariant` config and captured-dispatch replay helper; the feature gate remains `metal-attn-variants`.
