# Card 09: compare F16 and F32 MMA on the same Rows16+SharedK schedule

**Owner:** GPT-6 Luna

**Dependency:** Card 08

**Commit:** `perf(interop): probe f32 mma on rows16 shared key`

**Budget:** at most 30 minutes front-to-back, including acceptance and task update.

## Purpose

Measure the existing coupled MMA precision axis on the captured Granite prefill dispatch. Card 08's Rows16+SharedK+SimdgroupRows arm had higher isolated replay p50 than Rows16+SharedK in all six paired cells while producing identical output bits (`cards/08-parallel-rows-with-shared-k.md`). Keep the Rows16+SharedK schedule and change only explicit `AttentionMmaPrecision::F16` to explicit `AttentionMmaPrecision::F32`.

The local llama.cpp F32 K/V classic flash-attention entry selects `FA_TYPES_F32`: its Q matrix fragment is half, K and V matrix fragments are float, QK score and output accumulators are float (`/Users/brianbruggeman/repos/others/llama.cpp/ggml/src/ggml-metal/kernels/fa_f32.metal:23-29,37`; type positions in `fa_common.metal:18-46`). Proxima's current `AttentionMmaPrecision` is one selector for Q, K, and V matrix operands; F16 narrows all three loaded fragments to half, while F32 uses float fragments for all three, with float score/output accumulation in both (`omega/src/msl/signature_tokens_prelude.rs:1702-1706,1743-1750`; `omega/src/msl/cached_attention_row_tiled.rs:68-90,265-323,421-425,504-505,541-542`). Thus this is a Proxima F16-versus-F32 axis measurement on F32 cache storage. Neither arm reproduces llama.cpp's mixed half-Q/float-K/V operand path, and no exact dtype or kernel parity claim follows.

For the captured `h8_g2_d64_r16_n2_b64` row tile, `tile_blocks=(16/8)*2=4`, `score_key_tiles=64/8=8`, and `query_parallelism=Legacy` leaves `scores[8][4]` and `accumulated[4][4]` in both arms (`cached_attention_row_tiled.rs:439-452,504-505,541-542`). The SharedK stage has `2 * block * 8 * depth_unroll * operand_bytes` bytes; with block 64 and depth unroll 2, this is 4,096 bytes for half operands and 8,192 for float operands (`cached_attention_row_tiled.rs:170-205`). Record compiled resources because those source formulas do not establish occupancy or spill behavior.

## Scope

Add one `#[proxima::test]` in `proxima-model-interop/tests/granite_attention_variant_prefill.rs`, named `perf_granite_rows16_shared_k_f16_vs_f32_mma`. Build the baseline from `f16_rows16_shared_k_variant()`. Build the selected variant by copying it and assigning only `mma_precision = AttentionMmaPrecision::F32`. Assert both configs explicitly retain `kv_storage=F32`, `kv_reuse=SharedK`, `tile_height=Rows16`, `query_parallelism=Legacy`, `simd_topology=Legacy`, and `prefetch=Off`; the baseline must explicitly be F16. Call `run_variant_prefill_probe_shapes_with_grid_requirement` for nominal `[SHORT_PROMPT_TOKENS, PROMPT_TOKENS]`, recording actual tokenizer counts from the real checkpoint (currently 256 and 971). The serving config binds both K/V cache tensors as F32 (`granite_attention_variant_prefill.rs:124-129`); do not change cache storage or serving configuration.

The capture helper matches node and extents, rejects fault bindings, requires distinct entry and MSL SHA256, replays the complete output span, compares generated IDs before timing, alternates arm order for 20 paired GPU samples per shape, and prints entry, source SHA256, grid, pipeline resources, output difference, IDs, raw samples, signed round differences, and summaries (`granite_attention_variant_prefill.rs:1190-1308`). The helper now rejects non-finite output pairs or difference summaries. It retains complete F32 output spans with 262,144 elements at 256 tokens and 994,304 at 971. The optional grid check is enabled for this MMA-only comparison and left disabled for scheduling probes, whose grids can differ. Record `elements`, `changed_bits`, maximum absolute difference, RMS difference, and non-finite pairs for each shape; differing finite output bits are allowed. Require generated IDs equal between arms and retain the observed ID arrays (the captured control value is `[322]` at both lengths). Retain the actual grid and resource tuple `(static threadgroup bytes, max threads per threadgroup, execution width)` for each arm. No resource value or timing direction is preselected.

Keep the feature-gated `AttentionVariant` toggle default-off: no production default, sized rule, public API, or Granite dispatch rule changes. The design abandoned for this card is adding a mixed half-Q/float-K/V precision enum to imitate llama.cpp. The current coupled selector answers the narrower measured question without changing a foundational precision contract; a mixed-operand design needs its own spec and evidence.

## Acceptance

Run this exact validation command three times, in separate sequential processes, saving complete stdout/stderr and exit status as `evidence/card09-f16-vs-f32-mma-run{1,2,3}-2026-10-10.log`:

```sh
PROXIMA_TEST_TIMEOUT_MS=900000 PROXIMA_CAPTURE_LIVE=1 PROXIMA_CAPTURE_NODES=all PROXIMA_CAPTURE_STEPS=0 CARGO_BUILD_JOBS=1 CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo nextest run -p proxima-model-interop --features std,metal,instrument,metal-attn-split-rows,metal-attn-variants --test granite_attention_variant_prefill -E 'test(~perf_granite_rows16_shared_k_f16_vs_f32_mma)' -j 1 --success-output immediate
```

The test binary now has 25 tests (`granite_attention_variant_prefill.rs`); the filtered command selects 1 test, passes 1, and skips 24 per process. Each process prints 2 actual prompt lengths, 2 matched F32-cache arms per length, 20 finite positive paired replay samples and 20 signed deltas per length, zero fault bindings or replay errors, complete finite output-difference records, equal generated IDs, distinct entry names and MSL hashes, equal grid geometry, and both pipeline resource tuples. If the F32 arm cannot bind, compile, replay, or preserve generated IDs, retain the failure payload and do not turn it into a timing comparison.

For each process and shape, report both p50 values, selected-minus-baseline p50, the counts of positive/negative/zero signed rounds, the full output-difference record and IDs, entry/hash pair, grid pair, and resource pair. Across the three processes report each arm's p50 range for each actual prompt length. These are isolated cached-attention replay observations; neither a p50 movement nor a resource change alone identifies its device-level cause or a whole-request effect.

## Observations

All three serial processes selected 1 test, passed 1, skipped 24, and exited 0. Each emitted both actual prompt lengths and 20 paired samples per arm. Values below are ns; every run had 20 positive, 0 negative, and 0 zero selected-minus-baseline rounds.

| actual tokens | run | F16 p50 | F32 p50 | F32 − F16 p50 | F16 CoV | F32 CoV |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 1 | 244,583.236 | 407,624.990 | +163,041.754 | 3.238% | 0.776% |
| 256 | 2 | 244,249.939 | 407,666.666 | +163,416.727 | 2.938% | 0.746% |
| 256 | 3 | 245,625.037 | 407,499.960 | +161,874.923 | 2.893% | 0.755% |
| 971 | 1 | 2,053,375.007 | 2,872,041.659 | +818,666.653 | 3.157% | 1.701% |
| 971 | 2 | 2,000,374.952 | 2,865,499.933 | +865,124.981 | 2.884% | 0.255% |
| 971 | 3 | 2,000,541.659 | 2,863,250.091 | +862,708.432 | 1.188% | 0.318% |

Across the three processes, F16/F32 p50 ranges were `244,249.939–245,625.037 / 407,499.960–407,666.666 ns` at 256 tokens and `2,000,374.952–2,053,375.007 / 2,863,250.091–2,872,041.659 ns` at 971 tokens. The full output-difference records were identical across runs: at 256, `changed_bits=262131`, `elements=262144`, `max_absolute_difference=2.457291603088379`, `rms_difference=0.016535265166861546`, `non_finite_pairs=0`; at 971, `changed_bits=994276`, `elements=994304`, `max_absolute_difference=11.756719589233398`, `rms_difference=0.05444857119052462`, `non_finite_pairs=0`. Both arms generated `[322]` at both lengths.

Every pair used the same captured entry pair: `omega_cached_attention_h8_g2_d64_s3c800000_ln9223372036854775808_up0_r16_n2_b64_rt_mma_f16_kv_shared_k_tile_rows16` and `omega_cached_attention_h8_g2_d64_s3c800000_ln9223372036854775808_up0_r16_n2_b64_rt_mma_f32_kv_shared_k_tile_rows16`. MSL SHA256 was `0ae8a952ec5eb3109ee26495c399adf53528f719d4b57fb4be113fdb672bab9e` for F16 and `aff0b0ef03b9f898b40106f5a0ed4c1b511a4866f67cb28c6cef1a1492ce91ff` for F32. The grids matched within each pair: 8,192 threads at 256 tokens and 31,232 at 971, both with width 64 and depth 1. Pipeline resources were F16 `(13056,384,32)` and F32 `(17152,384,32)` at both shapes.

The source path sets `half_operands=false` for F32 and uses `simdgroup_float8x8` operands; F16 loads float cache fragments then narrows them into `simdgroup_half8x8`, with F32 accumulation in both (`omega/src/msl/cached_attention_row_tiled.rs:68-90,269-323,410-431`). The SharedK staging-size formula multiplies operand bytes by 2 for F16 and 4 for F32 (`:201-205`), matching the 4,096-byte difference in the captured static threadgroup resource tuple. The captures show higher F32 replay p50s in all six cells and changed output bits across most output elements, while IDs remained equal; they do not isolate device instruction throughput, occupancy, or spill causes and do not establish semantic parity.

Raw paired samples and signed rounds: `evidence/card09-f16-vs-f32-mma-run1-2026-10-10.log`, `...run2-2026-10-10.log`, and `...run3-2026-10-10.log`.

## Discipline

The real Granite 3.1 1B A400M Instruct Q8_0 checkpoint and tokenizer supply the workload. The only changed experiment variable is the existing MMA operand precision; Rows16+SharedK, F32 K/V storage, Legacy query parallelism/topology, and prefetch Off are held fixed. The full captured MSL identity, grid, resources, output data, IDs, and both directions of timing deltas accompany every measured cell. If a run fails, record the specific payload and repair the card implementation before interpreting timing.
