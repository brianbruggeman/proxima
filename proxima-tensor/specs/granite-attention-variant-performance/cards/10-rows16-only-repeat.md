# Card 10: repeat the Rows16-only Granite prefill comparison

**Owner:** GPT-6 Luna

**Dependency:** Cards 03 and 09

**Commit:** `perf(interop): repeat rows16-only granite prefill probe`

**Budget:** at most 30 minutes front-to-back, including the three serial captures, evidence review, and task update.

## Purpose

Resolve the one-process Rows16-only control from Card 03. At actual 971 prompt tokens, its F16-MMA/F32-K/V baseline and Rows16 selected arm had replay p50s of 1,086,708.391 and 1,080,291.695 ns. The selected arm's CoV was 5.710%, and the 20 paired differences had 9 positive and 11 negative signs (`evidence/card03-acceptance-2026-10-10.log:11`). That record does not establish a repeatable tile-height effect. Repeat the same one-axis comparison at actual 256 and 971 tokens in three independent processes.

`AttentionVariant` already separates tile height from K/V representation, MMA precision, K/V reuse, query parallelism, SIMD topology, and prefetch (`omega/src/msl/signature_tokens_prelude.rs:1696-1730,1761-1770`). The abandoned design is a new kernel or Granite selector: the existing tile-height axis expresses this experiment. The captured row-tiled source uses `tile_rows` to compute tile count, query blocks, score vectors, and accumulator vectors (`omega/src/msl/cached_attention_row_tiled.rs:440-456,504-505`). Those are source-level changes, not a measured device-level cause.

## Scope

Add one `#[proxima::test]` named `perf_granite_rows16_only_repeat` to `proxima-model-interop/tests/granite_attention_variant_prefill.rs`. Its baseline is `f16_variant()` and its selected arm is `f16_rows16_variant()` (`proxima-model-interop/tests/granite_attention_variant_prefill.rs:1074-1106`). Assert the complete baseline config is `kv_storage=F32`, `mma_precision=F16`, `kv_reuse=Legacy`, `tile_height=Legacy`, `query_parallelism=Legacy`, `simd_topology=Legacy`, `prefetch=Off`; assert the selected config has the same six other fields and `tile_height=Rows16`. Compare those values directly so a future default change cannot silently add another axis. Call the existing capture/probe helper with `Some(baseline)`, the selected variant, and `[SHORT_PROMPT_TOKENS, PROMPT_TOKENS]` (nominal 256 and 971); record the tokenizer's actual counts (`proxima-model-interop/tests/granite_attention_variant_prefill.rs:32-38,225-242,1197-1210`). The serving config binds both K/V caches as F32 (`proxima-model-interop/tests/granite_attention_variant_prefill.rs:124-135`).

The helper already requires matching node/extents, distinct entry/MSL hashes, no fault binding, complete finite output spans, equal generated IDs before timing, 20 alternating paired replay samples per shape, and prints samples, signed deltas, grid, resources, and summaries (`proxima-model-interop/tests/granite_attention_variant_prefill.rs:272-339,1191-1322`). Add an output-equality requirement to that helper's existing optional requirement path: when enabled, assert the full-span output difference has `changed_bits=0` before timing. Keep it disabled for the F16/F32 precision comparison, whose finite output bits legitimately differ. This card enables output equality and leaves equal-grid checking disabled because Rows16 changes the grid. Assert each replay time is finite and positive before retaining it, so both arm arrays and their differences satisfy the sample contract. Do not duplicate the capture or replay loop.

Do not change the row-tiled kernel, sized rule, production default, model configuration, or any other `AttentionVariant` axis. The existing `metal-attn-variants` gate continues to keep this caller-selected experiment opt-in.

## Acceptance

Run this exact command three times in separate, serial processes; retain complete stdout/stderr and process exit status as `evidence/card10-rows16-only-run{1,2,3}-2026-10-10.log`. Run only one GPU test process at a time.

```sh
PROXIMA_TEST_TIMEOUT_MS=900000 PROXIMA_CAPTURE_LIVE=1 PROXIMA_CAPTURE_NODES=all PROXIMA_CAPTURE_STEPS=0 CARGO_BUILD_JOBS=1 CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo nextest run -p proxima-model-interop --features std,metal,instrument,metal-attn-split-rows,metal-attn-variants --test granite_attention_variant_prefill -E 'test(~perf_granite_rows16_only_repeat)' -j 1 --success-output immediate
```

| criterion | required record or count |
|---|---|
| Selection | Each of 3 processes selects exactly 1 test, passes 1, skips 25, and exits 0. The test binary has 26 tests after this addition. |
| Shape and isolation | Each process prints exactly 2 `granite_variant_probe` records, for tokenizer-verified actual 256 and 971 tokens. Each shape has 2 captured F32-cache arms, equal node/extents, explicit baseline/selected config assertions, distinct entry names and MSL SHA256 values, no fault binding or replay error. |
| Output and request completion | For each shape, compare the entire output span: 262,144 F32 elements at 256 tokens and 994,304 at 971 tokens. Require `changed_bits=0`, `non_finite_pairs=0`, finite maximum/RMS differences, and exact equal generated-ID arrays before timing. Record the observed arrays; Card 03 observed `[322]` in both arms (`evidence/card03-acceptance-2026-10-10.log:11`). |
| Timing payload | Each shape in each process has 20 finite positive raw GPU replay times per arm and 20 finite signed selected-minus-baseline differences from alternating arm order. Report positive, negative, and zero difference counts; their sum must be 20. Report both p50 values, selected-minus-baseline p50, and both CoVs, without filtering outliers or preselecting a sign. |
| Dispatch and resources | Report each arm's entry, MSL SHA256, grid, and pipeline resource tuple `(static threadgroup bytes, max threads per threadgroup, execution width)`. Record the grid difference; equal grid is not an acceptance condition for a tile-height change. |
| Across-process report | For each actual prompt length, report the baseline and selected p50 ranges across all 3 processes and each process's CoV. Retain every positive and negative paired difference in the raw logs. If any process has CoV above 5%, report its samples and sign distribution as noisy evidence; do not replace it with a clean average or drop the run. |

If binding, compilation, replay, output equality, or ID equality fails, preserve the failing payload and do not convert that process into a timing comparison. A p50 movement in this isolated captured dispatch does not establish whole-layer or request latency, semantic model quality, or a device-level cause. The owner decides any performance verdict after reading both directions of the paired records.

## Evidence to fill after the runs

| actual tokens | run | baseline p50 ns | Rows16 p50 ns | selected − baseline p50 ns | baseline CoV | selected CoV | signed +/−/0 | output/IDs | entry/hash/grid/resources |
|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 256 | 1 | 182,999.996 | 165,791.600 | −17,208.396 | 2.484% | 1.033% | 0/20/0 | `changed_bits=0`, 262,144 elements, IDs `[322]` / `[322]` | `evidence/card10-rows16-only-run1-2026-10-10.log:10` |
| 256 | 2 | 183,125.027 | 165,541.656 | −17,583.370 | 2.255% | 1.539% | 0/20/0 | `changed_bits=0`, 262,144 elements, IDs `[322]` / `[322]` | `evidence/card10-rows16-only-run2-2026-10-10.log:10` |
| 256 | 3 | 183,291.617 | 165,458.303 | −17,833.314 | 1.582% | 0.726% | 0/20/0 | `changed_bits=0`, 262,144 elements, IDs `[322]` / `[322]` | `evidence/card10-rows16-only-run3-2026-10-10.log:9` |
| 971 | 1 | 1,082,749.921 | 1,093,583.298 | +10,833.377 | 2.066% | 1.783% | 11/9/0 | `changed_bits=0`, 994,304 elements, IDs `[322]` / `[322]` | `evidence/card10-rows16-only-run1-2026-10-10.log:11` |
| 971 | 2 | 1,089,249.970 | 1,096,124.994 | +6,875.023 | 1.506% | 1.361% | 13/7/0 | `changed_bits=0`, 994,304 elements, IDs `[322]` / `[322]` | `evidence/card10-rows16-only-run2-2026-10-10.log:11` |
| 971 | 3 | 1,080,583.315 | 1,108,875.033 | +28,291.717 | 2.313% | 2.751% | 17/3/0 | `changed_bits=0`, 994,304 elements, IDs `[322]` / `[322]` | `evidence/card10-rows16-only-run3-2026-10-10.log:10` |

Each row is a captured Metal replay on Granite 3.1 1B A400M Instruct Q8_0, with F32 K/V, F16 MMA, and only tile height changing from legacy to Rows16. Status for every captured timing/output record: **proven**; source lines are the corresponding run-log lines above. All six rows share the following dispatch identity and resource records from those same source lines: baseline entry `omega_cached_attention_h8_g2_d64_s3c800000_ln9223372036854775808_up0_r8_n2_b64_rt_mma_f16`, selected entry `omega_cached_attention_h8_g2_d64_s3c800000_ln9223372036854775808_up0_r16_n2_b64_rt_mma_f16_tile_rows16`; baseline/selected MSL SHA256 `b85db41a81b3503641fbdb453cff0b7dd32e442085614eef5ab20fe1b1e0ba5d` / `4d5ca689147051823fbc85665a2275c4236b9d193523b7c4a1ec363c25fc166e`; grid threads `16384→8192` at 256 and `62464→31232` at 971 (width 64, depth 1); pipeline resources `(9600,448,32)` / `(8960,384,32)`. Full sample arrays and every signed delta remain in the linked logs.

Across the three processes, baseline/selected p50 ranges were `182,999.996–183,291.617` / `165,458.303–165,791.600 ns` at actual 256 tokens, and `1,080,583.315–1,089,249.970` / `1,093,583.298–1,108,875.033 ns` at actual 971 tokens. These range calculations use the six per-process p50 values in this table; they describe this captured dispatch experiment only.

The reproducible unit is this command plus the three complete logs. Card 03's earlier 971-token cell remains a separate historical process, not a fourth sample of this card's controlled three-process capture.
