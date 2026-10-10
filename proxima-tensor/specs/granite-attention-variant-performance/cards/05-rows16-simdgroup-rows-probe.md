# Card 05: probe Rows16 with simdgroup row parallelism

**Owner:** GPT-6 Luna
**Dependency:** Card 03
**Commit:** `perf(interop): probe rows16 simdgroup parallelism`
**Budget:** at most 30 minutes front-to-back, including acceptance and task update.

## Purpose

Test whether carrying twice as many query rows per threadgroup amortizes each simdgroup's repeated K/V reads while retaining query parallelism, without SharedKv staging.

## Scope

Add one captured-dispatch probe to the existing Granite integration test. Keep F16 MMA and F32 K/V in both arms. Compare the F16-MMA/F32-KV baseline with a selected variant that changes only `tile_height` to Rows16 and `query_parallelism` to SimdgroupRows. Keep K/V reuse at Legacy, SIMD topology at Legacy, and prefetch Off. Run actual tokenizer prompt lengths 256 and 971. Capture source identity/grid, output difference, exact IDs before timing, 20 alternating replay samples per arm, paired signed deltas, and summaries. Do not change kernel implementation or defaults.

This tests the interaction left open by Cards 03–04: a taller tile may increase rows served per simdgroup's K/V loads, while query-parallel ownership can still duplicate those loads across simdgroups. Card 04 showed SharedKv staging did not erase the replay cost.

## Acceptance

```sh
PROXIMA_TEST_TIMEOUT_MS=900000 PROXIMA_CAPTURE_LIVE=1 PROXIMA_CAPTURE_NODES=all PROXIMA_CAPTURE_STEPS=0 CARGO_BUILD_JOBS=1 CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo nextest run -p proxima-model-interop --features std,metal,instrument,metal-attn-split-rows,metal-attn-variants --test granite_attention_variant_prefill -E 'test(~perf_granite_rows16_simdgroup_rows)' -j 1 --success-output immediate
```

Require exactly 1 selected test, 1 pass, and 21 filtered out. It emits exactly 2 records for actual tokenizer counts 256 and 971; each record has 20 samples per arm and signed deltas, equal IDs, output differences, captured entry identities/grids, and no fault bindings. Save full output to `evidence/card05-row16-simdgroup-2026-10-10.log`.

This is isolated replay evidence. It does not establish whole-request latency or semantic quality.


## Observation

| actual tokens | baseline p50 ns | Rows16 + SimdgroupRows p50 ns | grid threads baseline/selected | output / generated ID | record |
|---:|---:|---:|---:|---|---|
| 256 | 179,625 | 489,292 | 16,384 / 8,192 | bit-identical; `[322]` both | `evidence/card05-row16-simdgroup-2026-10-10.log:8` |
| 971 | 1,109,125 | 6,411,500 | 62,464 / 31,232 | bit-identical; `[322]` both | `evidence/card05-row16-simdgroup-2026-10-10.log:9` |

This variant recorded a higher p50 at both prompt lengths. The captured grid carries half as many threads, while its output bytes and generated token ID match the baseline. Those source/grid observations do not explain the full replay delta; no spill, occupancy, or hardware-counter trace was collected.

No new API or kernel primitive was introduced; the probe composes existing `AttentionVariant` axes behind the current feature gate.
