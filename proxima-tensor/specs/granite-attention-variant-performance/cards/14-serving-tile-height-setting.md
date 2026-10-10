# Card 14: expose Granite tile height through serving settings

## Scope

Expose only the already implemented Rows8/Rows16 cached-attention tile-height axis in macOS Metal serving settings. Preserve Legacy as the default, lower the setting into the existing `AttentionVariant`, and replay the actual Granite prefill dispatch with Groups4 fixed. Do not change kernel arithmetic or another variant axis.

## Commands

```sh
CARGO_BUILD_JOBS=1 CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo test -p proxima-model-interop --lib --features std,metal,instrument,metal-attn-split-rows,metal-attn-variants serving_settings::tests::attention_tile_height_setting_round_trips_and_reaches_serving_config --no-run
cp /tmp/cargo_target/debug/deps/proxima_model_interop-a7d123d9af6be267 /tmp/card13-attention-tile-settings-tests
/tmp/card13-attention-tile-settings-tests serving_settings::tests::attention_tile_height_setting_round_trips_and_reaches_serving_config --exact --nocapture
CARGO_BUILD_JOBS=1 CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo test -p proxima-model-interop --features std,metal,instrument,metal-attn-split-rows,metal-attn-variants --test granite_attention_variant_prefill --no-run
cp /tmp/cargo_target/debug/deps/granite_attention_variant_prefill-4c29247f309c5b09 /tmp/card14-granite-serving-tile-height
PROXIMA_TEST_TIMEOUT_MS=900000 PROXIMA_CAPTURE_LIVE=1 PROXIMA_CAPTURE_NODES=all PROXIMA_CAPTURE_STEPS=0 /tmp/card14-granite-serving-tile-height perf_granite_serving_tile_height_rows8_vs_rows16_groups4 --nocapture
```

Run the final command three times sequentially. Retain full stdout/stderr and exit status as `evidence/card14-serving-tile-height-run{1,2,3}-2026-10-10.log`.

## Acceptance

- The conflag test passes builder/TOML/environment parity for Rows8 and Rows16 with Groups4, confirms default Legacy and rejects `rows32`.
- Each GPU run selects one test, passes one, and skips the rest; captures actual 256 and 971 token prompts with 20 paired samples per arm.
- Each pair has equal complete output bytes and generated IDs. Keep all raw samples, both-sign deltas, source identities, grids, and pipeline resources.
- Captured Rows8 and Rows16 settings lower to variants with equal fields except `tile_height`; both use Groups4.
- Do not infer a timing cause from latency, grid, or resource differences alone.

## Implementation

`ServingSettings::attention_tile_height` accepts `legacy`, `rows8`, and `rows16`; Legacy plus a Legacy simdgroup count still lowers to `None`, preserving the prior sized dispatch. Explicit tile height composes with the existing simdgroup count and leaves every other `AttentionVariant` field at its existing default. The focused integration replay creates both arms from `ServingSettings` and asserts the fields are identical except the requested row height.

## Evidence

The focused conflaguration test passed: 1 passed, 735 filtered. Builder, TOML, and environment values agreed for Rows8/Rows16 with Groups4; default Legacy lowered to `None`; `rows32` was rejected. The focused Granite integration target compiled. Each copied-binary replay selected 1 test, passed 1, and filtered 28.

All three serial replay logs contain actual prompt lengths 256 and 971, 20 samples per arm, full output equality (`changed_bits=0`, no non-finite pairs), and IDs `[322]` for each shape. Independently parsed arrays agree with each recorded paired-delta array:

| Actual tokens | Run | Rows8 p50 ns | Rows16 p50 ns | Rows16 − Rows8 paired signs (+/−/0) |
|---:|---:|---:|---:|---:|
| 256 | 1 | 137,958.326 | 180,937.583 | 20/0/0 |
| 256 | 2 | 138,354.197 | 182,833.348 | 20/0/0 |
| 256 | 3 | 138,208.328 | 182,937.481 | 20/0/0 |
| 971 | 1 | 1,038,187.474 | 1,144,791.604 | 20/0/0 |
| 971 | 2 | 1,038,895.804 | 1,137,229.207 | 20/0/0 |
| 971 | 3 | 1,012,145.774 | 1,120,437.519 | 20/0/0 |

The serving path captured MMA `Legacy`, F32 K/V, Legacy reuse, Groups4, Legacy query parallelism/topology, and prefetch off for both arms. Rows8 and Rows16 emitted entries `omega_cached_attention_h8_g2_d64_s3c800000_ln9223372036854775808_up0_r8_n4_b64_rt_tile_rows8_simdgroups4` and `omega_cached_attention_h8_g2_d64_s3c800000_ln9223372036854775808_up0_r16_n4_b64_rt_tile_rows16_simdgroups4`; source SHA256 values were `893d85bd4474ff6a81c512be8baf15a9484f47b248b2ade3dcbd2e3debe42856` and `7a781d42dcfa6c9121550994c660fcff5ac2e5a8101bfa489c8e699c074c95fb`. Threadgroup width remained 128. Grid threads changed 32,768→16,384 at 256 tokens and 124,928→62,464 at 971. Pipeline resource tuples `(static_threadgroup_bytes,max_threads,execution_width)` changed `(9600,512,32)`→`(8960,384,32)`. These dispatch/resource observations do not explain timing.

Rows16's per-dispatch p50 was higher in all six cells and each cell's 20 paired deltas was positive. This repeated timing relation is plausible; its cause is unexplained. No default was changed and the capture does not establish a whole-request latency effect. Full raw samples and payloads are in `evidence/card14-serving-tile-height-run{1,2,3}-2026-10-10.log`. The earlier F16-MMA direct-variant comparison is in Card 13 and remains a separate configuration.
