# Card 18: replay warmup-controlled Groups2 and Groups8 at both Granite prompt sizes

## Question

Card 17 records three warmup-controlled Groups4 runs at actual 256 and 972 tokens. This card applies the same two-pair warmup protocol to the remaining supported explicit counts, Groups2 and Groups8, with the F16-MMA/F32-KV sized Legacy baseline. Only `simdgroup_count` changes. Counts are read through `ServingSettings::from_env` and lowered through `ServingConfig.attention_variant`; Groups4 is rejected by this test because its comparable evidence is Card 17.

## Commands

Build one integration binary with one Cargo job. Run one Metal process at a time, three times for each selected count. Use a unique report and full output log per process. The copied executable lives outside Cargo's flat `deps` directory because direct execution from that directory stalls during loader startup.

```sh
set -o pipefail
nice -n 20 env CARGO_BUILD_JOBS=1 CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo test -p proxima-model-interop --features std,metal,instrument,metal-attn-split-rows,metal-attn-variants --test granite_attention_variant_prefill --no-run
cp /tmp/cargo_target/debug/deps/granite_attention_variant_prefill-4c29247f309c5b09 /tmp/card18-granite-attention-tests
for count in groups2 groups8; do
  for run in 1 2 3; do
    PROXIMA_SERVING_ATTENTION_SIMDGROUP_COUNT="$count" PROXIMA_TEST_TIMEOUT_MS=900000 PROXIMA_CAPTURE_LIVE=1 PROXIMA_CAPTURE_NODES=all PROXIMA_CAPTURE_STEPS=0 PROXIMA_GRANITE_AB_REPORT="/private/tmp/proxima-model-matrix/proxima-tensor/specs/granite-attention-variant-performance/evidence/card18-${count}-run${run}-report.json" /tmp/card18-granite-attention-tests perf_granite_simdgroup_count_warmup_control_two_shapes_against_f16_legacy --nocapture 2>&1 | tee "proxima-tensor/specs/granite-attention-variant-performance/evidence/card18-${count}-run${run}-2026-10-10.log"
  done
done
```

Run `check_report.py --expected-shapes 2 --selected-arm simdgroupsN` and repeat with `--negative-controls` against each of the six reports, substituting the selected label and report path.

## Acceptance

- Each of six serial test processes selects one named test, passes one, filters 32, exits zero, and writes exactly two actual prompt shapes.
- Every shape has two alternating warmup pairs and 20 measured pairs per arm; each report retains the raw GPU samples, signed differences, both arm orders, and five-replay process resource observations.
- The baseline is F16 MMA/F32 K/V with sized Legacy count. The selected arm is the configured count. Each pair holds checkpoint, tokenizer prompt, serving options, node, and extents fixed; output bytes and generated IDs match, there are zero fault bindings and replay errors, and both grids have positive active dimensions.
- Each positive report check prints `prompt_shapes=2 arms=4 samples_per_arm=20 resource_cells=4 matching_pairs=2 errors=0`; each negative-control check prints `negative_controls=13 rejected=13`.
- Preserve p50/p90/p99, CoV, all paired signs, source SHA, entry, grid, pipeline resources, and host/load context for both shapes and all runs. Keep any tail reversal or mixed signs visible. Timing cause remains unclaimed unless source and payload evidence identify it.

## Evidence

### Captured MSL: groups2
- `256` `legacy`: `evidence/card18-msl/groups2/pipeline_b85db41a81b35036.metal`; SHA256 `b85db41a81b3503641fbdb453cff0b7dd32e442085614eef5ab20fe1b1e0ba5d`; entry `omega_cached_attention_h8_g2_d64_s3c800000_ln9223372036854775808_up0_r8_n2_b64_rt_mma_f16`.
- `256` `simdgroups2`: `evidence/card18-msl/groups2/pipeline_92146f5f93379827.metal`; SHA256 `92146f5f93379827b243b5593f7520c3da8e50fe359b0897b4d4eee4d115f0da`; entry `omega_cached_attention_h8_g2_d64_s3c800000_ln9223372036854775808_up0_r8_n2_b64_rt_mma_f16_simdgroups2`.
- `972` `legacy`: `evidence/card18-msl/groups2/pipeline_b85db41a81b35036.metal`; SHA256 `b85db41a81b3503641fbdb453cff0b7dd32e442085614eef5ab20fe1b1e0ba5d`; entry `omega_cached_attention_h8_g2_d64_s3c800000_ln9223372036854775808_up0_r8_n2_b64_rt_mma_f16`.
- `972` `simdgroups2`: `evidence/card18-msl/groups2/pipeline_92146f5f93379827.metal`; SHA256 `92146f5f93379827b243b5593f7520c3da8e50fe359b0897b4d4eee4d115f0da`; entry `omega_cached_attention_h8_g2_d64_s3c800000_ln9223372036854775808_up0_r8_n2_b64_rt_mma_f16_simdgroups2`.
- MSL comparison: the only source delta from Legacy is the entry symbol suffix `simdgroups2`; the kernel body and `simdgroups = 2` constant are identical.
### Captured MSL: groups4
- `256` `legacy`: `evidence/card18-msl/groups4/pipeline_b85db41a81b35036.metal`; SHA256 `b85db41a81b3503641fbdb453cff0b7dd32e442085614eef5ab20fe1b1e0ba5d`; entry `omega_cached_attention_h8_g2_d64_s3c800000_ln9223372036854775808_up0_r8_n2_b64_rt_mma_f16`.
- `256` `simdgroups4`: `evidence/card18-msl/groups4/pipeline_637d62efe5ffb3c0.metal`; SHA256 `637d62efe5ffb3c07f0df1b406cd5a4171c337a73177b80e3737b4eecbcd956f`; entry `omega_cached_attention_h8_g2_d64_s3c800000_ln9223372036854775808_up0_r8_n4_b64_rt_mma_f16_simdgroups4`.
- `972` `legacy`: `evidence/card18-msl/groups4/pipeline_b85db41a81b35036.metal`; SHA256 `b85db41a81b3503641fbdb453cff0b7dd32e442085614eef5ab20fe1b1e0ba5d`; entry `omega_cached_attention_h8_g2_d64_s3c800000_ln9223372036854775808_up0_r8_n2_b64_rt_mma_f16`.
- `972` `simdgroups4`: `evidence/card18-msl/groups4/pipeline_637d62efe5ffb3c0.metal`; SHA256 `637d62efe5ffb3c07f0df1b406cd5a4171c337a73177b80e3737b4eecbcd956f`; entry `omega_cached_attention_h8_g2_d64_s3c800000_ln9223372036854775808_up0_r8_n4_b64_rt_mma_f16_simdgroups4`.
- MSL comparison: the entry symbol suffix changes to `simdgroups4`, and `constexpr long simdgroups` changes from 2 to 4; all other emitted source lines match Legacy.
### Captured MSL: groups8
- `256` `legacy`: `evidence/card18-msl/groups8/pipeline_b85db41a81b35036.metal`; SHA256 `b85db41a81b3503641fbdb453cff0b7dd32e442085614eef5ab20fe1b1e0ba5d`; entry `omega_cached_attention_h8_g2_d64_s3c800000_ln9223372036854775808_up0_r8_n2_b64_rt_mma_f16`.
- `256` `simdgroups8`: `evidence/card18-msl/groups8/pipeline_c01f2194431c9f42.metal`; SHA256 `c01f2194431c9f428feee1ea9439be7ef505cef18195491405ebf31f7a6537b7`; entry `omega_cached_attention_h8_g2_d64_s3c800000_ln9223372036854775808_up0_r8_n8_b64_rt_mma_f16_simdgroups8`.
- `972` `legacy`: `evidence/card18-msl/groups8/pipeline_b85db41a81b35036.metal`; SHA256 `b85db41a81b3503641fbdb453cff0b7dd32e442085614eef5ab20fe1b1e0ba5d`; entry `omega_cached_attention_h8_g2_d64_s3c800000_ln9223372036854775808_up0_r8_n2_b64_rt_mma_f16`.
- `972` `simdgroups8`: `evidence/card18-msl/groups8/pipeline_c01f2194431c9f42.metal`; SHA256 `c01f2194431c9f428feee1ea9439be7ef505cef18195491405ebf31f7a6537b7`; entry `omega_cached_attention_h8_g2_d64_s3c800000_ln9223372036854775808_up0_r8_n8_b64_rt_mma_f16_simdgroups8`.
- MSL comparison: the entry symbol suffix changes to `simdgroups8`, and `constexpr long simdgroups` changes from 2 to 8; all other emitted source lines match Legacy.

### Replay report ledger
| count | run | shape | Legacy → selected p50/p90/p99 (ns) | CoV % | paired selected−Legacy (+/−/0) | grid threads Legacy → selected | pipeline max threads Legacy → selected |
|---|---:|---:|---|---:|---:|---:|---:|
| Groups2 | 1 | 256 | 182,000/182,792/183,167 → 182,083/182,875/182,958 | 0.36/0.32 | 10/10/0 | 64/16384 → 64/16384 | 448 → 448 |
| Groups2 | 1 | 972 | 1,087,583/1,110,875/1,128,875 → 1,087,875/1,114,000/1,114,917 | 1.52/2.09 | 10/10/0 | 64/62464 → 64/62464 | 448 → 448 |
| Groups2 | 2 | 256 | 182,000/182,750/182,750 → 181,833/183,000/183,250 | 0.34/0.43 | 9/11/0 | 64/16384 → 64/16384 | 448 → 448 |
| Groups2 | 2 | 972 | 1,082,958/1,109,208/1,123,625 → 1,092,792/1,121,125/1,129,583 | 2.41/2.08 | 10/10/0 | 64/62464 → 64/62464 | 448 → 448 |
| Groups2 | 3 | 256 | 182,500/183,417/185,667 → 182,250/183,083/186,125 | 0.66/0.68 | 8/12/0 | 64/16384 → 64/16384 | 448 → 448 |
| Groups2 | 3 | 972 | 1,099,792/1,110,833/1,122,125 → 1,089,500/1,120,792/1,125,125 | 1.70/2.27 | 10/10/0 | 64/62464 → 64/62464 | 448 → 448 |
| Groups8 | 1 | 256 | 181,875/182,625/184,500 → 185,000/185,833/188,417 | 0.48/0.59 | 20/0/0 | 64/16384 → 256/65536 | 448 → 640 |
| Groups8 | 1 | 972 | 1,092,500/1,111,500/1,123,500 → 1,371,583/1,399,375/1,436,625 | 1.24/1.87 | 20/0/0 | 64/62464 → 256/249856 | 448 → 640 |
| Groups8 | 2 | 256 | 181,958/182,375/183,292 → 185,000/186,000/186,500 | 0.25/0.37 | 20/0/0 | 64/16384 → 256/65536 | 448 → 640 |
| Groups8 | 2 | 972 | 1,088,250/1,129,167/1,148,000 → 1,360,625/1,407,458/1,427,625 | 2.23/1.66 | 20/0/0 | 64/62464 → 256/249856 | 448 → 640 |
| Groups8 | 3 | 256 | 182,292/185,125/186,125 → 184,333/185,750/187,750 | 0.74/0.60 | 18/2/0 | 64/16384 → 256/65536 | 448 → 640 |
| Groups8 | 3 | 972 | 1,097,833/1,119,125/1,126,500 → 1,367,417/1,385,750/1,422,208 | 1.91/1.39 | 20/0/0 | 64/62464 → 256/249856 | 448 → 640 |

### Output, identity, and report hashes
- `card18-groups2-run1-report.json` report SHA256 `fcd674bdd3b868be14e0d76abe396afa5284d0f9d82f31e5a3c51fba8b543a92`; selected `simdgroups2`.
  - 256 tokens: output SHA `454ca3f4483db67f72f42baa6f7625c46853a9be768e4bbcdd35116789f72ba6` on both arms, `1048576` bytes, IDs `[322]`; zero fault bindings/replay errors; baseline source `b85db41a81b3503641fbdb453cff0b7dd32e442085614eef5ab20fe1b1e0ba5d`, selected source `92146f5f93379827b243b5593f7520c3da8e50fe359b0897b4d4eee4d115f0da`.
  - 972 tokens: output SHA `970fa9d575423441d23b639a5831938d9e9c46aa8c3c6e84fb2d4f7fe0dfbcae` on both arms, `3981312` bytes, IDs `[322]`; zero fault bindings/replay errors; baseline source `b85db41a81b3503641fbdb453cff0b7dd32e442085614eef5ab20fe1b1e0ba5d`, selected source `92146f5f93379827b243b5593f7520c3da8e50fe359b0897b4d4eee4d115f0da`.
- `card18-groups2-run2-report.json` report SHA256 `9b2a5a22fc4e8d3a00d32229da133e392459565b02e4ff7621e6ebbf21c4088c`; selected `simdgroups2`.
  - 256 tokens: output SHA `454ca3f4483db67f72f42baa6f7625c46853a9be768e4bbcdd35116789f72ba6` on both arms, `1048576` bytes, IDs `[322]`; zero fault bindings/replay errors; baseline source `b85db41a81b3503641fbdb453cff0b7dd32e442085614eef5ab20fe1b1e0ba5d`, selected source `92146f5f93379827b243b5593f7520c3da8e50fe359b0897b4d4eee4d115f0da`.
  - 972 tokens: output SHA `970fa9d575423441d23b639a5831938d9e9c46aa8c3c6e84fb2d4f7fe0dfbcae` on both arms, `3981312` bytes, IDs `[322]`; zero fault bindings/replay errors; baseline source `b85db41a81b3503641fbdb453cff0b7dd32e442085614eef5ab20fe1b1e0ba5d`, selected source `92146f5f93379827b243b5593f7520c3da8e50fe359b0897b4d4eee4d115f0da`.
- `card18-groups2-run3-report.json` report SHA256 `3280ccb49f76e00fc001fb6a5388dbdd8ff8a7ffa5e4bd59c9f0d962a7dd3260`; selected `simdgroups2`.
  - 256 tokens: output SHA `454ca3f4483db67f72f42baa6f7625c46853a9be768e4bbcdd35116789f72ba6` on both arms, `1048576` bytes, IDs `[322]`; zero fault bindings/replay errors; baseline source `b85db41a81b3503641fbdb453cff0b7dd32e442085614eef5ab20fe1b1e0ba5d`, selected source `92146f5f93379827b243b5593f7520c3da8e50fe359b0897b4d4eee4d115f0da`.
  - 972 tokens: output SHA `970fa9d575423441d23b639a5831938d9e9c46aa8c3c6e84fb2d4f7fe0dfbcae` on both arms, `3981312` bytes, IDs `[322]`; zero fault bindings/replay errors; baseline source `b85db41a81b3503641fbdb453cff0b7dd32e442085614eef5ab20fe1b1e0ba5d`, selected source `92146f5f93379827b243b5593f7520c3da8e50fe359b0897b4d4eee4d115f0da`.
- `card18-groups8-run1-report.json` report SHA256 `a7a760971a3b84a9b1e2cf8799054a787fcfc04709da52f88de2fa40ab3be1b3`; selected `simdgroups8`.
  - 256 tokens: output SHA `454ca3f4483db67f72f42baa6f7625c46853a9be768e4bbcdd35116789f72ba6` on both arms, `1048576` bytes, IDs `[322]`; zero fault bindings/replay errors; baseline source `b85db41a81b3503641fbdb453cff0b7dd32e442085614eef5ab20fe1b1e0ba5d`, selected source `c01f2194431c9f428feee1ea9439be7ef505cef18195491405ebf31f7a6537b7`.
  - 972 tokens: output SHA `970fa9d575423441d23b639a5831938d9e9c46aa8c3c6e84fb2d4f7fe0dfbcae` on both arms, `3981312` bytes, IDs `[322]`; zero fault bindings/replay errors; baseline source `b85db41a81b3503641fbdb453cff0b7dd32e442085614eef5ab20fe1b1e0ba5d`, selected source `c01f2194431c9f428feee1ea9439be7ef505cef18195491405ebf31f7a6537b7`.
- `card18-groups8-run2-report.json` report SHA256 `5b840c4836740fb29b3e271a34e06d021c1ac8cdb677b922b0f017778d7f0f61`; selected `simdgroups8`.
  - 256 tokens: output SHA `454ca3f4483db67f72f42baa6f7625c46853a9be768e4bbcdd35116789f72ba6` on both arms, `1048576` bytes, IDs `[322]`; zero fault bindings/replay errors; baseline source `b85db41a81b3503641fbdb453cff0b7dd32e442085614eef5ab20fe1b1e0ba5d`, selected source `c01f2194431c9f428feee1ea9439be7ef505cef18195491405ebf31f7a6537b7`.
  - 972 tokens: output SHA `970fa9d575423441d23b639a5831938d9e9c46aa8c3c6e84fb2d4f7fe0dfbcae` on both arms, `3981312` bytes, IDs `[322]`; zero fault bindings/replay errors; baseline source `b85db41a81b3503641fbdb453cff0b7dd32e442085614eef5ab20fe1b1e0ba5d`, selected source `c01f2194431c9f428feee1ea9439be7ef505cef18195491405ebf31f7a6537b7`.
- `card18-groups8-run3-report.json` report SHA256 `b7995749d337b2c9a490d47ea654701e2f84aeaef06e8539d9a8411e1d056f8c`; selected `simdgroups8`.
  - 256 tokens: output SHA `454ca3f4483db67f72f42baa6f7625c46853a9be768e4bbcdd35116789f72ba6` on both arms, `1048576` bytes, IDs `[322]`; zero fault bindings/replay errors; baseline source `b85db41a81b3503641fbdb453cff0b7dd32e442085614eef5ab20fe1b1e0ba5d`, selected source `c01f2194431c9f428feee1ea9439be7ef505cef18195491405ebf31f7a6537b7`.
  - 972 tokens: output SHA `970fa9d575423441d23b639a5831938d9e9c46aa8c3c6e84fb2d4f7fe0dfbcae` on both arms, `3981312` bytes, IDs `[322]`; zero fault bindings/replay errors; baseline source `b85db41a81b3503641fbdb453cff0b7dd32e442085614eef5ab20fe1b1e0ba5d`, selected source `c01f2194431c9f428feee1ea9439be7ef505cef18195491405ebf31f7a6537b7`.

### Checker records
- `card18-report-checks-2026-10-10.log`:
  - `prompt_shapes=2 arms=4 samples_per_arm=20 resource_cells=4 matching_pairs=2 errors=0`
  - `round_differences_positive=20 negative=20 zero=0`
  - `negative_controls=13 rejected=13`
  - `prompt_shapes=2 arms=4 samples_per_arm=20 resource_cells=4 matching_pairs=2 errors=0`
  - `round_differences_positive=19 negative=21 zero=0`
  - `negative_controls=13 rejected=13`
  - `prompt_shapes=2 arms=4 samples_per_arm=20 resource_cells=4 matching_pairs=2 errors=0`
  - `round_differences_positive=18 negative=22 zero=0`
  - `negative_controls=13 rejected=13`
  - `prompt_shapes=2 arms=4 samples_per_arm=20 resource_cells=4 matching_pairs=2 errors=0`
  - `round_differences_positive=40 negative=0 zero=0`
  - `negative_controls=13 rejected=13`
  - `prompt_shapes=2 arms=4 samples_per_arm=20 resource_cells=4 matching_pairs=2 errors=0`
  - `round_differences_positive=40 negative=0 zero=0`
  - `negative_controls=13 rejected=13`
  - `prompt_shapes=2 arms=4 samples_per_arm=20 resource_cells=4 matching_pairs=2 errors=0`
  - `round_differences_positive=38 negative=2 zero=0`
  - `negative_controls=13 rejected=13`
- `card18-source-report-checks-2026-10-10.log`:
  - `prompt_shapes=2 arms=4 samples_per_arm=20 resource_cells=4 matching_pairs=2 errors=0`
  - `round_differences_positive=16 negative=24 zero=0`
  - `negative_controls=13 rejected=13`
  - `prompt_shapes=2 arms=4 samples_per_arm=20 resource_cells=4 matching_pairs=2 errors=0`
  - `round_differences_positive=3 negative=37 zero=0`
  - `negative_controls=13 rejected=13`
  - `prompt_shapes=2 arms=4 samples_per_arm=20 resource_cells=4 matching_pairs=2 errors=0`
  - `round_differences_positive=40 negative=0 zero=0`
  - `negative_controls=13 rejected=13`

### Mechanism read from source
- In `omega/src/msl/cached_attention_row_tiled.rs:438-450`, MSL sets `threads = simdgroups * 32`, `dims_per_group = head_dim / 8 / simdgroups`, and `key_tiles_per_group = (block / 8) / simdgroups`; fixed values are head dimension 64, block 64, tile rows 8, F16 MMA, F32 K/V, and the existing K/V reuse flags.
- For this shape, Legacy/Groups2 use 2 groups, 64 threads, 4 depth fragments and 4 key fragments per group; Groups4 uses 128 threads, 2 and 2; Groups8 uses 256 threads, 1 and 1. Each selected capture has the same source as its counterpart for both prompt sizes; only entry name changes for Groups2 (the emitted constant is already 2), while Groups4/8 also substitute the simdgroup constant. Groups2 pipeline resources match Legacy `(9600,448,32)`; Groups8 holds threadgroup bytes/exec width at `(9600,32)` but max threads is 640 versus Legacy 448. The MSL explains the dispatch specialization; it does not explain the timing behavior.
- Groups2 signs are mixed (each cell 8–12 positive and 8–12 negative over 20 pairs), and p50 moves in both directions across runs. Groups8 has higher p50 in all six cells and 20/20 positive paired rounds in five cells; the remaining 256-token cell has 18/2. The 972-token selected p50 values are about 1.36M ns in all three Groups8 runs versus about 1.09M ns baseline. The execution records do not establish why.
- Count support comes from `omega/src/msl/signature_tokens_prelude.rs:2800-2838`: head dimension and block fragment divisibility, accumulator-fragment cap, and threadgroup memory admission constrain the choices. This card does not broaden legality.

### Interpretation boundary
The six matched executions and source captures are observations for the local Mac Metal Granite prefill replay, not a whole-request benchmark. Groups2 has mixed paired signs and run-to-run directions. Groups8 shifts the kernel to 256 threads while reducing each group to one depth/key fragment; its p50 is higher in each recorded cell, but the cause is unexplained. Groups4 evidence remains in Card 17 and is not combined with this count sweep to claim an optimization. No default changed.

Card 17 supplies the matched Groups4 comparison under the same warmup protocol; Card 12 retains the earlier cold Groups2/4/8 count sweep.
