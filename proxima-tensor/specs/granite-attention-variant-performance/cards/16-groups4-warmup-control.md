# Card 16: warmup-controlled Groups4 Granite prefill replay

## Question

Card 12's three Groups4 processes had mixed paired signs and two high-CoV runs. Repeat the same 971-nominal-token F16-MMA/F32-KV Legacy-count baseline against Groups4 after two alternating warmup pairs. Keep the twenty measured alternating pairs and retain warmup samples separately. This changes only measurement protocol; it does not change the kernel selector, precision, reuse, tile height, or parallelism.

## Execution

The test `perf_granite_simdgroups4_warmup_control_against_f16_legacy` captures the real Granite request, enforces complete attention output and generated-ID equality, executes two warmup pairs (`legacy, Groups4`, then `Groups4, legacy`), and records every warmup and measured GPU time in the report. Resource replays remain after measured samples. The normal Card 12 test path continues to use zero warmup rounds.

```sh
nice -n 20 env CARGO_BUILD_JOBS=1 CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo test -p proxima-model-interop --features std,metal,instrument,metal-attn-split-rows,metal-attn-variants --test granite_attention_variant_prefill --no-run
cp /tmp/cargo_target/debug/deps/granite_attention_variant_prefill-4c29247f309c5b09 /tmp/card16-granite-attention-tests
PROXIMA_TEST_TIMEOUT_MS=900000 PROXIMA_CAPTURE_LIVE=1 PROXIMA_CAPTURE_NODES=all PROXIMA_CAPTURE_STEPS=0 PROXIMA_GRANITE_AB_REPORT=/private/tmp/proxima-model-matrix/proxima-tensor/specs/granite-attention-variant-performance/evidence/card16-groups4-warmup-run1-report.json /tmp/card16-granite-attention-tests perf_granite_simdgroups4_warmup_control_against_f16_legacy --nocapture 2>&1 | tee proxima-tensor/specs/granite-attention-variant-performance/evidence/card16-groups4-warmup-run1-2026-10-10.log
PROXIMA_TEST_TIMEOUT_MS=900000 PROXIMA_CAPTURE_LIVE=1 PROXIMA_CAPTURE_NODES=all PROXIMA_CAPTURE_STEPS=0 PROXIMA_GRANITE_AB_REPORT=/private/tmp/proxima-model-matrix/proxima-tensor/specs/granite-attention-variant-performance/evidence/card16-groups4-warmup-run2-report.json /tmp/card16-granite-attention-tests perf_granite_simdgroups4_warmup_control_against_f16_legacy --nocapture 2>&1 | tee proxima-tensor/specs/granite-attention-variant-performance/evidence/card16-groups4-warmup-run2-2026-10-10.log
PROXIMA_TEST_TIMEOUT_MS=900000 PROXIMA_CAPTURE_LIVE=1 PROXIMA_CAPTURE_NODES=all PROXIMA_CAPTURE_STEPS=0 PROXIMA_GRANITE_AB_REPORT=/private/tmp/proxima-model-matrix/proxima-tensor/specs/granite-attention-variant-performance/evidence/card16-groups4-warmup-run3-report.json /tmp/card16-granite-attention-tests perf_granite_simdgroups4_warmup_control_against_f16_legacy --nocapture 2>&1 | tee proxima-tensor/specs/granite-attention-variant-performance/evidence/card16-groups4-warmup-run3-2026-10-10.log
python3 proxima-tensor/specs/granite-attention-variant-performance/check_report.py --expected-shapes 1 --selected-arm simdgroups4 proxima-tensor/specs/granite-attention-variant-performance/evidence/card16-groups4-warmup-run1-report.json
python3 proxima-tensor/specs/granite-attention-variant-performance/check_report.py --expected-shapes 1 --selected-arm simdgroups4 --negative-controls proxima-tensor/specs/granite-attention-variant-performance/evidence/card16-groups4-warmup-run1-report.json
python3 proxima-tensor/specs/granite-attention-variant-performance/check_report.py --expected-shapes 1 --selected-arm simdgroups4 proxima-tensor/specs/granite-attention-variant-performance/evidence/card16-groups4-warmup-run2-report.json
python3 proxima-tensor/specs/granite-attention-variant-performance/check_report.py --expected-shapes 1 --selected-arm simdgroups4 --negative-controls proxima-tensor/specs/granite-attention-variant-performance/evidence/card16-groups4-warmup-run2-report.json
python3 proxima-tensor/specs/granite-attention-variant-performance/check_report.py --expected-shapes 1 --selected-arm simdgroups4 proxima-tensor/specs/granite-attention-variant-performance/evidence/card16-groups4-warmup-run3-report.json
python3 proxima-tensor/specs/granite-attention-variant-performance/check_report.py --expected-shapes 1 --selected-arm simdgroups4 --negative-controls proxima-tensor/specs/granite-attention-variant-performance/evidence/card16-groups4-warmup-run3-report.json
```

Three serial processes each selected one test, passed one, filtered thirty, and retained one shape with 20 measured samples per arm and two warmup rounds. The actual token count was 972. All pairs had `output_equal=true`, `ids_equal=true`, output SHA256 `970fa9d575423441d23b639a5831938d9e9c46aa8c3c6e84fb2d4f7fe0dfbcae` on both arms, and IDs `[322]` on both arms. Each positive report check printed `prompt_shapes=1 arms=2 samples_per_arm=20 resource_cells=2 matching_pairs=1 errors=0`; all three 13-control checks rejected all 13 mutations.

| Run | Legacy / Groups4 p50 ns | Legacy / Groups4 p90 ns | Legacy / Groups4 p99 ns | CoV % | Groups4 − Legacy signs (+/−/0) | Report SHA256 |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 1,101,750 / 1,042,208 | 1,116,333 / 1,086,958 | 1,138,042 / 1,164,208 | 1.71 / 3.27 | 2/18/0 | `4b20d316d4e7822b733b2ee1dfd309c0a0b23652f3403afb227a204ec28ab672` |
| 2 | 1,081,375 / 1,049,167 | 1,110,458 / 1,085,542 | 1,121,750 / 1,128,750 | 2.20 / 2.67 | 2/18/0 | `94e6b88ac3f1da16ce7e7a378c52f7e5435096d6a38bb941f1d2f149a8cfb783` |
| 3 | 1,086,375 / 1,045,917 | 1,109,125 / 1,098,625 | 1,136,500 / 1,111,292 | 2.72 / 2.66 | 6/14/0 | `03342c971223c00f3818278cd58d381c30782703ba4dd54e0a8d791253cc0333` |

Raw timed and warmup samples, output records, settings, resource cells, source SHA256, and grids are in `evidence/card16-groups4-warmup-run{1,2,3}-report.json`; full replay output is in matching `...-run{1,2,3}-2026-10-10.log` files. Checker output is retained in `evidence/card16-report-checks-2026-10-10.log`. The selected-minus-Legacy p50 relation was lower for Groups4 in all three processes, but the third run has six positive paired samples and the Groups4 p99 crosses above Legacy in run 1 and run 2. Those observations do not explain the timing difference.

The captured entries are `..._r8_n2_b64_rt_mma_f16` and `..._r8_n4_b64_rt_mma_f16_simdgroups4`; source hashes are respectively `b85db41a81b3503641fbdb453cff0b7dd32e442085614eef5ab20fe1b1e0ba5d` and `637d62efe5ffb3c07f0df1b406cd5a4171c337a73177b80e3737b4eecbcd956f`. Width/grid threads are 64/62,464 and 128/124,928. Pipeline tuples `(max_threads, static_threadgroup_bytes, execution_width)` are `(448,9600,32)` and `(576,9600,32)`. The MSL count formulas assign each simdgroup fewer head-dimension fragments and key fragments as the count rises (`omega/src/msl/cached_attention_row_tiled.rs:438-450`); the selected count doubles threadgroup width and total grid threads here. That describes the changed work partition, not why the GPU times differ.

This is isolated cached-attention dispatch timing on the Mac Metal device. It does not establish whole-request latency or a performance verdict. Groups4 remains an explicit opt-in count; the default remains Legacy.
