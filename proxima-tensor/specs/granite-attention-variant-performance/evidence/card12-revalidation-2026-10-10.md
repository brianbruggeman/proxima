# Card 12 revalidation — 2026-10-10

## Serving setting test

The single-job command compiled the test binary:

```sh
nice -n 20 env CARGO_BUILD_JOBS=1 CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo test -p proxima-model-interop --features std,metal,metal-attn-split-rows,metal-attn-variants --lib simdgroup_count_setting_round_trips_and_reaches_serving_config -- --nocapture
```

`nice` reported `setpriority: Operation not permitted`. Cargo finished compiling and printed `Running unittests`, but the binary at its target path produced no harness output for 30 seconds; that invocation was interrupted (exit 130). The test binary was copied to `/tmp/card12-serving-settings-tests`. Its SHA256 equals the Cargo-target binary SHA256 `02d8e2bba61f2798744269b28764b015b86fd6adcfbffbeb6be3ae980144ce80`.

The copied binary's `--list` output gave the exact test name:

```text
serving_settings::tests::simdgroup_count_setting_round_trips_and_reaches_serving_config: test
```

This selector was run from the copy:

```sh
/tmp/card12-serving-settings-tests serving_settings::tests::simdgroup_count_setting_round_trips_and_reaches_serving_config --exact --nocapture
```

Output: `1 passed; 0 failed; 0 ignored; 0 measured; 727 filtered out`. The assertion compares builder, TOML, and environment for Groups2/4/8, checks Legacy remains the default with no explicit `attention_variant`, checks each explicit count lowers into `ServingConfig.attention_variant` with Legacy MMA precision, and rejects `groups16` (`proxima-model-interop/src/serving_settings.rs:833-875`). An initial selector without the module path matched 0 tests and is not counted as execution evidence.

## Binary discovery path probe

The Granite integration binary at `/private/tmp/cargo_target/debug/deps/granite_attention_variant_prefill-4c29247f309c5b09` was run with `--list` under a six-second subprocess timeout. It emitted zero stdout and zero stderr bytes before timing out. The byte-identical copy `/tmp/card12-granite-attention-tests --list` exited 0 in 0.013 seconds and listed 27 tests. `shasum`, `file`, `xattr -l`, and `otool -L` showed matching bytes, arm64 Mach-O type, provenance attribute, and framework dependencies. The observed failure is executable-path dependent; its loader/runtime cause is unexplained. This reproduces the nextest discovery symptom without running a test body.

## Report revalidation

For each retained report, the independent positive checker emitted `prompt_shapes=1 arms=2 samples_per_arm=20 resource_cells=2 matching_pairs=1 errors=0`. With `--negative-controls`, it emitted `negative_controls=13 rejected=13` for each report:

- `card12-simdgroups2-report.json` with `--selected-arm simdgroups2`
- `card12-simdgroups4-run1-report.json`, `card12-simdgroups4-run2-report.json`, and `card12-simdgroups4-run3-report.json` with `--selected-arm simdgroups4`
- `card12-simdgroups8-report.json` with `--selected-arm simdgroups8`

The ten invocations were:

```sh
python3 proxima-tensor/specs/granite-attention-variant-performance/check_report.py --expected-shapes 1 --selected-arm simdgroups2 proxima-tensor/specs/granite-attention-variant-performance/evidence/card12-simdgroups2-report.json
python3 proxima-tensor/specs/granite-attention-variant-performance/check_report.py --expected-shapes 1 --selected-arm simdgroups2 --negative-controls proxima-tensor/specs/granite-attention-variant-performance/evidence/card12-simdgroups2-report.json
python3 proxima-tensor/specs/granite-attention-variant-performance/check_report.py --expected-shapes 1 --selected-arm simdgroups4 proxima-tensor/specs/granite-attention-variant-performance/evidence/card12-simdgroups4-run1-report.json
python3 proxima-tensor/specs/granite-attention-variant-performance/check_report.py --expected-shapes 1 --selected-arm simdgroups4 --negative-controls proxima-tensor/specs/granite-attention-variant-performance/evidence/card12-simdgroups4-run1-report.json
python3 proxima-tensor/specs/granite-attention-variant-performance/check_report.py --expected-shapes 1 --selected-arm simdgroups4 proxima-tensor/specs/granite-attention-variant-performance/evidence/card12-simdgroups4-run2-report.json
python3 proxima-tensor/specs/granite-attention-variant-performance/check_report.py --expected-shapes 1 --selected-arm simdgroups4 --negative-controls proxima-tensor/specs/granite-attention-variant-performance/evidence/card12-simdgroups4-run2-report.json
python3 proxima-tensor/specs/granite-attention-variant-performance/check_report.py --expected-shapes 1 --selected-arm simdgroups4 proxima-tensor/specs/granite-attention-variant-performance/evidence/card12-simdgroups4-run3-report.json
python3 proxima-tensor/specs/granite-attention-variant-performance/check_report.py --expected-shapes 1 --selected-arm simdgroups4 --negative-controls proxima-tensor/specs/granite-attention-variant-performance/evidence/card12-simdgroups4-run3-report.json
python3 proxima-tensor/specs/granite-attention-variant-performance/check_report.py --expected-shapes 1 --selected-arm simdgroups8 proxima-tensor/specs/granite-attention-variant-performance/evidence/card12-simdgroups8-report.json
python3 proxima-tensor/specs/granite-attention-variant-performance/check_report.py --expected-shapes 1 --selected-arm simdgroups8 --negative-controls proxima-tensor/specs/granite-attention-variant-performance/evidence/card12-simdgroups8-report.json
```

Each checked report has equal full output hashes (`970fa9d575423441d23b639a5831938d9e9c46aa8c3c6e84fb2d4f7fe0dfbcae`), 3,981,312 output bytes, equal generated IDs `[322]`, equal dispatch extents `[972,8,2,64]`, no fault binding, and zero replay errors. The report SHA256 values and raw 20-sample arrays per arm are in Card 12.

## Dispatch evidence read

The selector in `proxima-model-interop/tests/granite_attention_variant_prefill.rs:1924-1968` starts from the same F16 MMA/F32 K/V Legacy variant and changes only `simdgroup_count` from `PROXIMA_SERVING_ATTENTION_SIMDGROUP_COUNT`. The row-tiled kernel template in `omega/src/msl/cached_attention_row_tiled.rs:444-447` resolves `threads = simdgroups * 32`, `dims_per_group = head_dim / 8 / simdgroups`, and `key_tiles_per_group = (block / 8) / simdgroups`. For this 64-wide head and 64-key block, Groups2/4/8 map to widths 64/128/256 and divide each simdgroup's depth and key-fragment assignments accordingly. The selected count does not alter the captured `r8` tile height or Legacy K/V reuse. Reports retain the changed entry and MSL SHA, full grid, and pipeline resource tuple for every arm.

In the three Groups4 processes, the captured paired sign counts were 4/16/0, 7/13/0, and 6/14/0 (selected-minus-Legacy positive/negative/zero); the p50 comparison changes sign across processes. Two runs have per-arm CoV near 40%, and their raw arrays contain early samples above 3.4 ms followed by samples near 1.0–1.1 ms. This timing pattern is observed; its cause is unexplained. The single Groups8 run has 20/20 positive paired deltas. No kernel source was changed based on these count cells.

## Bounded discovery-stall diagnosis

The current Cargo-target Granite test binary timed out during `--list` with zero stdout/stderr; a two-second `sample` captured `_dyld_start` and a 96 KiB physical footprint. The byte-identical `/tmp/card16-granite-attention-tests` copy listed 31 tests. Follow-up bounded path probes isolated the launch condition: both the Cargo binary and an identical temporary executable placed directly in `/private/tmp/cargo_target/debug/deps` timed out at five seconds; an identical executable in a fresh child directory `/private/tmp/cargo_target/debug/deps/card17-probe` listed 32 tests in 0.427 seconds, and one in `/private/tmp/card17-probe` listed 32 in 0.455 seconds. Both child-directory copies also passed `codesign --verify` in 0.061–0.063 seconds, while verification on executables directly in `deps` timed out at three and five seconds. `/tmp` resolves to `/private/tmp`; the target and copies are on the same device. `deps` contained 65,535 entries when inspected. This identifies direct execution from the flat Cargo `deps` directory as the trigger for the loader/discovery stall. The dyld or macOS service sub-operation that reacts to this directory context remains unidentified. The raw stack is `evidence/card16-cargo-target-dyld-sample-2026-10-10.txt`; probes were removed after each run.

The focused copied-binary conflaguration assertion was rerun: `serving_settings::tests::simdgroup_count_setting_round_trips_and_reaches_serving_config` passed 1 test with 727 filtered. The test compares builder/TOML/environment settings, Legacy default behavior, invalid-value rejection, and lowering into `ServingConfig.attention_variant`.
