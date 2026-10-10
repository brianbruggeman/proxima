# Card 21: verify configured Groups4 on the main checkout

## Question

Does the serving count setting compile and reach the real Granite cached-attention dispatch on the main checkout, and does the current Groups4 replay retain the same output payload as sized Legacy?

## Build and setting checks

The focused integration target was built with:

```sh
nice -n 20 env CARGO_BUILD_JOBS=1 CARGO_INCREMENTAL=0 RUSTC_WRAPPER= \
  cargo test -p proxima-model-interop \
  --features std,metal,instrument,metal-attn-split-rows,metal-attn-variants \
  --test granite_attention_variant_prefill --no-run
```

That build first found a missing cfg-gated `AttentionTileHeightSetting` definition/re-export and a missing `ShortConv` arm in instrumented cache-upload accounting. The definition and re-export are in `proxima-model-interop/src/serving_settings.rs` and `src/lib.rs`; the counter match includes `history` and `roll_indices` in `src/generate/decode.rs`. After those source changes the integration target compiled and the Granite test executable was copied out of Cargo's `deps` directory.
The final focused build was repeated after the fixes; its captured output is `evidence/card21-granite-integration-build-2026-10-10.log` and reports the Granite integration executable at `/tmp/cargo_target/debug/deps/granite_attention_variant_prefill-4c29247f309c5b09`.

Two exact unit tests ran from the copied library-test binary:

```sh
/tmp/card20-interop-tests serving_settings::tests::simdgroup_count_setting_round_trips_and_reaches_serving_config --exact --nocapture
/tmp/card20-interop-tests serving_settings::tests::attention_tile_height_setting_round_trips_and_reaches_serving_config --exact --nocapture
```

Each selected 1 test and passed 1. The simdgroup test checks builder/TOML/environment round-trips for Groups2/4/8, Legacy default behavior, rejection of `groups16`, and lowering to `ServingConfig.attention_variant`; the tile-height test checks Rows8/Rows16 parity and invalid-value rejection. Output is `evidence/card21-serving-setting-tests-2026-10-10.log`.

## Replay

One serial main-checkout process ran the existing warmup-controlled two-shape test with the serving environment set to Groups4:

```sh
PROXIMA_SERVING_ATTENTION_SIMDGROUP_COUNT=groups4 \
PROXIMA_TEST_TIMEOUT_MS=900000 PROXIMA_CAPTURE_LIVE=1 \
PROXIMA_CAPTURE_NODES=all PROXIMA_CAPTURE_STEPS=0 \
PROXIMA_GRANITE_AB_REPORT=/private/tmp/proxima-granite-main/proxima-tensor/specs/granite-attention-variant-performance/evidence/card21-groups4-main-report.json \
/tmp/card21-granite-tests \
perf_granite_simdgroups4_warmup_control_two_shapes_against_f16_legacy --nocapture
```

The process selected 1 test, passed 1, and filtered 32. The positive report check printed `prompt_shapes=2 arms=4 samples_per_arm=20 resource_cells=4 matching_pairs=2 errors=0`; the negative-control check rejected 13/13 mutations. Full output is `evidence/card21-groups4-main-2026-10-10.log`; raw paired samples are in `evidence/card21-groups4-main-report.json` (SHA256 `5c96b3490a6c4fba3ba7ea151c83883ab0780c833885ab91ab583b99c9a09248`).

The report identifies Granite 3.1 1B A400M Instruct Q8_0 on the local M1 Max, F16 MMA/F32 K/V, nominal prompt lengths 256/971 and actual lengths 256/972. Both selected dispatches use the configured `omega_cached_attention_h8_g2_d64_s3c800000_ln9223372036854775808_up0_r8_n4_b64_rt_mma_f16_simdgroups4` entry. Output SHA256 and generated ID `[322]` match Legacy at both shapes: `454ca3f4483db67f72f42baa6f7625c46853a9be768e4bbcdd35116789f72ba6` at 256 and `970fa9d575423441d23b639a5831938d9e9c46aa8c3c6e84fb2d4f7fe0dfbcae` at 972.

For this one process, paired selected-minus-Legacy signs were 0 positive/20 negative at 256 and 1 positive/19 negative at 972. GPU timestamp p50/p90/p99 values (Legacy → Groups4) were `182083.3/182750.1/183666.6 → 144166.6/145416.7/145625.0 ns` at 256, and `1096541.6/1118333.4/1139666.7 → 1049875.0/1079166.7/1122500.0 ns` at 972. These are one added process; Card 17 remains the three-process record and contains counter-evidence in other tail cells.

The captured dispatch changes threadgroup width from 64 to 128 and total grid threads from 16,384 to 32,768 at 256 tokens and 62,464 to 124,928 at 972. Pipeline resources `(max_threads, static_threadgroup_bytes, exec_width)` change from `(448,9600,32)` to `(576,9600,32)`. In the emitted row-tiled source, `threads = simdgroups * 32`, `dims_per_group = head_dim / 8 / simdgroups`, and `key_tiles_per_group = (block / 8) / simdgroups` (`omega/src/msl/cached_attention_row_tiled.rs:444-449`); for Granite's 64-wide head and 64-key block this changes depth/key fragments from 4/4 at Legacy's resolved count 2 to 2/2 at Groups4. This records the changed work partition and dispatch; it does not identify which GPU effect produced the timestamps.

## Discovery-path observation

The copied Granite test binary returned 33 names from `--list --format terse` under a 5-second bound. A later `cargo nextest list` against the already-built target also returned all 33 names; `/usr/bin/time -p` recorded `real 0.72` seconds. The warmed `cargo nextest run` compiled the focused library test target in 42.18 seconds and then selected 1 setting test, passed 1, and skipped 727. A sample of the paired `--list --format terse` process taken 30 seconds after launch contained 780 main-thread samples; 742 were in `_dyld_start`, and 23 were in `dyld4::RemoteNotificationResponder::blockOnSynchronousEvent`/`mach_msg2_trap`. The same sample reached Rust `list_tests_console` and a stdout write in one sample. This locates the wait in dyld's synchronous debugger notification path during discovery; the receiving process and reason it did not respond are not identified. The saved sample is `evidence/card21-nextest-dyld-sample-2026-10-10.txt`; the successful setting run and test listing are in `evidence/card21-nextest-setting-test-2026-10-10.log` and `evidence/card21-nextest-list-2026-10-10.log`.

## Boundary

This validates the setting-to-dispatch path and adds one matched Groups4 replay with full output evidence. It does not change the global Legacy default, establish a timing cause, or replace Card 17's multi-process record. No automatic Granite policy was added.
