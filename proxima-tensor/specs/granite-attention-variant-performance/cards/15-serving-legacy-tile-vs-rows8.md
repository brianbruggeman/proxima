# Card 15: compare Legacy tile height with Rows8 in serving dispatch

## Scope

Determine whether the Rows8 direction measured against Rows16 also differs from the current Legacy tile-height selection. Keep Groups4, MMA precision, K/V storage and reuse, query parallelism, topology, and prefetch fixed. This is an opt-in Granite prefill dispatch comparison; it does not change the serving default.

## Commands

```sh
CARGO_BUILD_JOBS=1 CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo test -p proxima-model-interop --features std,metal,instrument,metal-attn-split-rows,metal-attn-variants --test granite_attention_variant_prefill --no-run
cp /tmp/cargo_target/debug/deps/granite_attention_variant_prefill-4c29247f309c5b09 /tmp/card15-granite-legacy-tile-rows8-groups4
PROXIMA_TEST_TIMEOUT_MS=900000 PROXIMA_CAPTURE_LIVE=1 PROXIMA_CAPTURE_NODES=all PROXIMA_CAPTURE_STEPS=0 /tmp/card15-granite-legacy-tile-rows8-groups4 perf_granite_serving_legacy_tile_vs_rows8_groups4 --nocapture
```

Run the final command three times sequentially. Retain complete output and exit status in `evidence/card15-serving-legacy-tile-rows8-run{1,2,3}-2026-10-10.log`.

## Acceptance

- Both arms are created through `ServingSettings` and lower to Groups4 variants identical except `tile_height` (`Legacy` or `Rows8`).
- Each run selects one test, passes one, filters the others, and captures actual 256- and 971-token prompts with 20 paired samples per arm.
- Every matched pair has equal full outputs and IDs. Retain raw times, signed deltas, entry/source hashes, grids, and resources.
- Report each cell's raw-derived p50 and positive/negative/zero delta counts. A timing relation does not explain its cause or establish whole-request latency.

## Implementation

The test uses the Card 14 `attention_tile_height` setting. The baseline specifies `legacy` while also selecting Groups4, so the comparison isolates tile height without changing simdgroup count. No kernel or default selector is modified.

## Evidence

Each of the three serial runs selected one test, passed one, and filtered 29. All six matched pairs were bit-identical (`changed_bits=0`, no non-finite pairs) and generated IDs `[322]`. Each arm retained 20 samples and 20 signed paired deltas. Independent parsing confirmed every saved delta equals the corresponding selected-minus-baseline sample difference.

| Actual tokens | Run | Legacy tile p50 ns | Rows8 p50 ns | Rows8 − Legacy signs (+/−/0) |
|---:|---:|---:|---:|---:|
| 256 | 1 | 139,750.075 | 139,937.503 | 11/9/0 |
| 256 | 2 | 140,625.052 | 140,562.479 | 11/9/0 |
| 256 | 3 | 139,874.988 | 140,833.319 | 15/5/0 |
| 971 | 1 | 1,026,874.990 | 1,025,937.498 | 11/9/0 |
| 971 | 2 | 1,006,145.787 | 1,004,083.315 | 10/10/0 |
| 971 | 3 | 1,006,249.979 | 1,014,479.145 | 12/8/0 |

The captured Legacy arm emitted an `r8_n4_b64` entry; the explicit Rows8 arm emitted `r8_n4_b64` with the `_tile_rows8` selection suffix. Both arms had identical grids and pipeline resource tuples: threadgroup width 128; 32,768 threads at 256 tokens and 124,928 at 971; `(9600,512,32)` in both arms. The captured `tile_rows` constant was 8 in both generated kernels. The source texts differ on only the kernel entry symbol: the Legacy name ends `..._r8_n4_b64_rt_simdgroups4`, and the explicit name ends `..._r8_n4_b64_rt_tile_rows8_simdgroups4`; the remaining MSL lines are identical. The exact files are `evidence/card15-msl/pipeline_59887ba9bd5908c3.metal` and `evidence/card15-msl/pipeline_893d85bd4474ff6a.metal`. Thus this setting selects the same row-tiled kernel body already selected by Legacy for these shapes, with a different pipeline identity.

Rows8 p50 is lower in three cells and higher in three; paired signs are shown above and in the raw logs. The timing cause remains unexplained despite identical kernel bodies, grids, and resource tuples. The source-capture replay is retained in `evidence/card15-source-inspection-2026-10-10.log`; it is not included in the three-run timing table. Full raw output is in `evidence/card15-serving-legacy-tile-rows8-run{1,2,3}-2026-10-10.log`.
