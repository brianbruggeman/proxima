# Card 17: replay warmup-controlled Groups4 at both Granite prompt sizes

## Question

Card 16 repeats the noisy Groups4 count cell at 972 actual prompt tokens. This card checks the same single-axis count selection and two-pair warmup protocol at both the 256- and 972-token Granite prefill dispatches. The setting is read from `ServingSettings::from_env`, lowered into `ServingConfig.attention_variant`, and asserted to be Groups4 before the captured request. The baseline remains F16 MMA, F32 K/V, and sized Legacy count.

## Commands

Run one Cargo job and one Metal process at a time. Build and copy the integration binary, then execute the test three serial times with `PROXIMA_SERVING_ATTENTION_SIMDGROUP_COUNT=groups4`. Use unique report and log paths per run. Check each report with `--expected-shapes 2 --selected-arm simdgroups4`, then repeat with `--negative-controls`.

```sh
nice -n 20 env CARGO_BUILD_JOBS=1 CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo test -p proxima-model-interop --features std,metal,instrument,metal-attn-split-rows,metal-attn-variants --test granite_attention_variant_prefill --no-run
cp /tmp/cargo_target/debug/deps/granite_attention_variant_prefill-4c29247f309c5b09 /tmp/card17-granite-attention-tests
PROXIMA_SERVING_ATTENTION_SIMDGROUP_COUNT=groups4 PROXIMA_TEST_TIMEOUT_MS=900000 PROXIMA_CAPTURE_LIVE=1 PROXIMA_CAPTURE_NODES=all PROXIMA_CAPTURE_STEPS=0 PROXIMA_GRANITE_AB_REPORT=/private/tmp/proxima-model-matrix/proxima-tensor/specs/granite-attention-variant-performance/evidence/card17-groups4-run1-report.json /tmp/card17-granite-attention-tests perf_granite_simdgroups4_warmup_control_two_shapes_against_f16_legacy --nocapture 2>&1 | tee proxima-tensor/specs/granite-attention-variant-performance/evidence/card17-groups4-run1-2026-10-10.log
PROXIMA_SERVING_ATTENTION_SIMDGROUP_COUNT=groups4 PROXIMA_TEST_TIMEOUT_MS=900000 PROXIMA_CAPTURE_LIVE=1 PROXIMA_CAPTURE_NODES=all PROXIMA_CAPTURE_STEPS=0 PROXIMA_GRANITE_AB_REPORT=/private/tmp/proxima-model-matrix/proxima-tensor/specs/granite-attention-variant-performance/evidence/card17-groups4-run2-report.json /tmp/card17-granite-attention-tests perf_granite_simdgroups4_warmup_control_two_shapes_against_f16_legacy --nocapture 2>&1 | tee proxima-tensor/specs/granite-attention-variant-performance/evidence/card17-groups4-run2-2026-10-10.log
PROXIMA_SERVING_ATTENTION_SIMDGROUP_COUNT=groups4 PROXIMA_TEST_TIMEOUT_MS=900000 PROXIMA_CAPTURE_LIVE=1 PROXIMA_CAPTURE_NODES=all PROXIMA_CAPTURE_STEPS=0 PROXIMA_GRANITE_AB_REPORT=/private/tmp/proxima-model-matrix/proxima-tensor/specs/granite-attention-variant-performance/evidence/card17-groups4-run3-report.json /tmp/card17-granite-attention-tests perf_granite_simdgroups4_warmup_control_two_shapes_against_f16_legacy --nocapture 2>&1 | tee proxima-tensor/specs/granite-attention-variant-performance/evidence/card17-groups4-run3-2026-10-10.log
```

## Capture requirements

Retain both actual tokenizer counts, two warmup pairs and 20 measured pairs per shape/arm, every raw GPU time, signed deltas, complete output bytes/hash, generated IDs, captured entry/source SHA, grid and pipeline resources, and the five-replay process resource context. Require the full output/ID gate for both shapes. Keep p50, p90, p99, CoV and both signs per run and shape; no timing number alone explains its cause or establishes whole-request latency.

## Results

The focused integration binary compiled with one Cargo job in 27.57s. Three serial Metal processes each selected one test, passed one, and filtered 31. Each report contains both actual prompt lengths (256 and 972), two alternating warmup rounds and 20 measured rounds per arm/shape. All six shape-pairs have equal full output hashes, equal IDs `[322]`, no fault bindings, and zero replay errors. The 256-token output SHA256 is `454ca3f4483db67f72f42baa6f7625c46853a9be768e4bbcdd35116789f72ba6` (1,048,576 bytes); the 972-token output SHA256 is `970fa9d575423441d23b639a5831938d9e9c46aa8c3c6e84fb2d4f7fe0dfbcae` (3,981,312 bytes).

| Run | Actual tokens | Legacy / Groups4 p50 ns | Legacy / Groups4 p90 ns | Legacy / Groups4 p99 ns | CoV % | Groups4 − Legacy signs (+/−/0) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 256 | 181,917 / 144,750 | 182,875 / 145,958 | 184,958 / 146,583 | 0.46 / 0.66 | 0/20/0 |
| 2 | 256 | 182,250 / 144,583 | 183,292 / 146,167 | 183,833 / 146,792 | 0.38 / 0.69 | 0/20/0 |
| 3 | 256 | 182,625 / 145,167 | 183,625 / 147,500 | 185,875 / 148,875 | 0.62 / 1.04 | 0/20/0 |
| 1 | 972 | 1,094,250 / 1,057,792 | 1,112,625 / 1,083,000 | 1,126,792 / 1,142,750 | 2.30 / 2.47 | 3/17/0 |
| 2 | 972 | 1,089,125 / 1,044,167 | 1,109,750 / 1,079,125 | 1,180,250 / 1,110,333 | 2.52 / 2.32 | 2/18/0 |
| 3 | 972 | 1,079,375 / 1,048,583 | 1,100,750 / 1,090,250 | 1,116,958 / 1,104,542 | 1.41 / 2.59 | 4/16/0 |

Each report's checker printed `prompt_shapes=2 arms=4 samples_per_arm=20 resource_cells=4 matching_pairs=2 errors=0`; its negative-control pass rejected all 13 mutations. The report SHA256 values for runs 1–3 are `c9f6104fb9929aa9e7a4f0787916e76148655a9675478d64627659bf0715aa4a`, `65bec55d5786b3eb7ccf8425c6dad8e82f04590059e474192fa3ccc92da02f40`, and `6c377cd714ac438f1cd46e8f25dcb2a129eefd774b4021609efa010d82fd1b1a`. Raw warmup and measured samples, per-round deltas, all output records, and resources are in `evidence/card17-groups4-run{1,2,3}-report.json`; complete process output is in the matching `...-run{1,2,3}-2026-10-10.log` files. Checker output is `evidence/card17-report-checks-2026-10-10.log`.

The reports capture Legacy entry `omega_cached_attention_h8_g2_d64_s3c800000_ln9223372036854775808_up0_r8_n2_b64_rt_mma_f16` and Groups4 entry `omega_cached_attention_h8_g2_d64_s3c800000_ln9223372036854775808_up0_r8_n4_b64_rt_mma_f16_simdgroups4` with source SHA256 values `b85db41a81b3503641fbdb453cff0b7dd32e442085614eef5ab20fe1b1e0ba5d` and `637d62efe5ffb3c07f0df1b406cd5a4171c337a73177b80e3737b4eecbcd956f`. At 256 tokens, threadgroup width/grid threads are 64/16,384 versus 128/32,768; at 972 they are 64/62,464 versus 128/124,928. Pipeline tuples `(max_threads, static_threadgroup_bytes, execution_width)` are `(448,9600,32)` versus `(576,9600,32)` at both shapes. The count formulas redistribute head-dimension and key-fragment ownership across the four simdgroups; the captures keep tile height 8 and other variant fields fixed. Those source and dispatch changes do not establish why the measured GPU times differ.

The lower Groups4 p50 and paired samples are observed for both prompt lengths across these three processes. The long-shape p99 exceeds Legacy in run 1, and both signs occur in every long-shape run. Preserve those counter-observations; this is isolated dispatch timing, not whole-request latency or a performance verdict. Groups4 remains explicit opt-in.
