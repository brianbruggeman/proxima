# Card 06: probe Rows16 with shared K

**Owner:** GPT-6 Luna

**Dependency:** Cards 03–05

**Commit:** `perf(interop): probe rows16 shared key reuse`

**Budget:** at most 30 minutes front-to-back, including acceptance and task update.

## Purpose

Test whether Rows16 amortizes the existing SharedK staging path while leaving query parallelism disabled. Cards 04 and 05 tested each axis separately and saw higher captured replay p50; this interaction tests the code path where one threadgroup owns the query rows and stages K once.

## Scope

Use the existing `AttentionVariant` axes only. Compare F16 MMA/F32 K/V baseline against a selected variant changing only `tile_height` to Rows16 and `kv_reuse` to SharedK. Keep query parallelism and SIMD topology at Legacy; keep prefetch Off. Capture the real Granite dispatch at actual tokenizer prompt lengths 256 and 971. Retain kernel entry names, grid, output differences, exact generated IDs, and 20 alternating replay samples per arm. The existing probe helper does not emit MSL hashes or pipeline resource counts; this card reports only the fields in its captured log. Do not change kernel source or defaults.

The design abandoned is adding a specialized kernel or a Granite-specific selector. Existing generated axes already express this combination, so adding a primitive would duplicate the current composable selector.

## Acceptance

```sh
PROXIMA_TEST_TIMEOUT_MS=900000 PROXIMA_CAPTURE_LIVE=1 PROXIMA_CAPTURE_NODES=all PROXIMA_CAPTURE_STEPS=0 CARGO_BUILD_JOBS=1 CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo nextest run -p proxima-model-interop --features std,metal,instrument,metal-attn-split-rows,metal-attn-variants --test granite_attention_variant_prefill -E 'test(~perf_granite_rows16_shared_k)' -j 1 --success-output immediate
```

Require exactly 1 selected test, 1 pass, and 22 filtered out. It emits two records for actual tokenizer counts 256 and 971; each has 20 paired samples, output comparison, equal generated IDs before timing, kernel entry names/grid, and zero replay/binding errors. Save full output to `evidence/card06-rows16-shared-k-2026-10-10.log`.

This isolated replay does not establish whole-request latency or semantic quality. A non-improving result remains in the log as a negative observation.

## Observation

| actual tokens | baseline p50 ns | Rows16 + SharedK p50 ns | grid threads baseline/selected | output / generated ID | record |
|---:|---:|---:|---:|---|---|
| 256 | 180,541.654 | 244,375.085 | 16,384 / 8,192 | bit-identical; `[322]` both | `evidence/card06-rows16-shared-k-2026-10-10.log:8` |
| 971 | 1,085,374.970 | 2,004,625.043 | 62,464 / 31,232 | bit-identical; `[322]` both | `evidence/card06-rows16-shared-k-2026-10-10.log:9` |

These observations are Granite 3.1 1B A400M Instruct Q8_0, local Metal, F16 MMA/F32 K/V, with actual prompt lengths 256 and 971, compared against F16-MMA/F32-KV legacy scheduling. The selected p50 is higher in both captured replay cells. The selected grid has half as many threads; the output is identical across all 262,144 and 994,304 f32 elements respectively, and each generated ID is `[322]`.

The selected source takes the SharedK path that stores key fragments to threadgroup memory and reloads them for query vectors (`omega/src/msl/cached_attention_row_tiled.rs:578-600`). The observed replay delta does not isolate the cost of those stores, barriers, or reloads; no hardware counter or GPU frame trace was collected. The grid reduction and exact output match do not explain the full timing difference.

## Discipline

This is a probe of an existing GPU kernel variant, not a new primitive. The meaningful axis is captured attention replay time by prompt size, with output bytes and generated IDs checked for each arm. No external library implements this internal generated-MSL selector. The fixture is the existing real Granite checkpoint and tokenizer. Runtime configuration and kernel source remain unchanged.
