# Card 08: add simdgroup row parallelism while retaining SharedK

**Owner:** GPT-6 Luna

**Dependency:** Cards 06–07

**Commit:** `perf(interop): probe parallel rows with shared key`

**Budget:** at most 30 minutes front-to-back, including acceptance and task update.

## Purpose

Test the composition that assigns query vectors to simdgroups while keeping K staged once for reuse. The baseline is the current Rows16+SharedK variant from Card 06/07; the selected arm changes only `query_parallelism` to `SimdgroupRows`.

For the captured `h8_g2_d64_r16_n2_b64` shape, the existing MSL formulas imply `tile_blocks=4`. With query parallelism Legacy, SharedK uses `score_vectors_per_simdgroup=4`, so `scores` is `[8][4]`. With SimdgroupRows, query ownership is enabled and `score_vectors_per_simdgroup=ceil(4/2)=2`, so `scores` is `[8][2]`. K reuse stays enabled; this tests whether reducing per-simdgroup score-matrix state can preserve parallel query ownership without restoring the legacy schedule's K loads.

## Scope

Use `AttentionVariant` only. Keep F16 MMA, F32 K/V, Rows16, SharedK, Legacy SIMD topology, and prefetch Off in both arms. Compare actual tokenizer prompt lengths 256 and 971. Capture output bits, generated IDs before timing, both entry names and MSL hashes, grids, pipeline resources, 20 paired GPU replay samples, and summaries. Run three independent test processes sequentially.

The design abandoned is treating query parallelism and K reuse as mutually exclusive modes. Both axes already compose in the selector, so the test keeps SharedK enabled while adding the existing simdgroup-row mode.

## Acceptance

```sh
PROXIMA_TEST_TIMEOUT_MS=900000 PROXIMA_CAPTURE_LIVE=1 PROXIMA_CAPTURE_NODES=all PROXIMA_CAPTURE_STEPS=0 CARGO_BUILD_JOBS=1 CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo nextest run -p proxima-model-interop --features std,metal,instrument,metal-attn-split-rows,metal-attn-variants --test granite_attention_variant_prefill -E 'test(~perf_granite_rows16_shared_k_simdgroup_rows)' -j 1 --success-output immediate
```

Run the exact command three times, serially. Each process selects 1 test, passes 1, and skips 23. Each emits both actual prompt lengths with 20 paired samples, equal output bytes and IDs before timing, distinct MSL hashes between arms, captured grids/resources, and zero replay/binding errors. Save outputs to `evidence/card08-parallel-shared-k-run{1,2,3}-2026-10-10.log`.

Report each process's baseline and selected p50, all three run ranges, the signed per-run deltas, exact IDs/output differences, entries/hashes, and resource tuples. The baseline is Rows16+SharedK (not Legacy); Card 06 supplies the previous baseline-to-Rows16+SharedK comparison. Do not infer a timing cause from a p50 movement alone.

## Observation

The corrected test uses `f16_rows16_shared_k_variant()` as its baseline. The first three runs initially used the F16 default baseline; their raw logs are retained as `evidence/card08-wrong-baseline-run{1,2,3}-2026-10-10.log` and do not satisfy this card.

Corrected runs 1–3 each selected one test, passed one, and skipped 23. All three logged both actual prompt lengths and 20 paired replay samples per arm. Baseline and selected outputs had `changed_bits=0` (262,144 elements at 256 tokens; 994,304 at 971), and both arms generated `[322]`.

| Run | Actual tokens | Rows16+SharedK p50 (ns) | Plus SimdgroupRows p50 (ns) | Selected − baseline (ns) |
|---|---:|---:|---:|---:|
| 1 | 256 | 239,333.371 | 481,249.997 | +241,916.627 |
| 2 | 256 | 239,750.021 | 481,125.084 | +241,375.063 |
| 3 | 256 | 239,624.991 | 481,541.734 | +241,916.743 |
| 1 | 971 | 1,988,958.335 | 4,397,958.284 | +2,408,999.950 |
| 2 | 971 | 2,024,499.932 | 4,439,875.018 | +2,415,375.086 |
| 3 | 971 | 1,996,999.956 | 4,477,791.605 | +2,480,791.649 |

Across runs, the p50 ranges were 239,333.371–239,750.021 / 481,125.084–481,541.734 ns at 256 tokens, and 1,988,958.335–2,024,499.932 / 4,397,958.284–4,477,791.605 ns at 971 tokens (baseline / selected). Both arms used 8,192 threads at 256 tokens and 31,232 at 971, with width 64. Captured pipeline resources were `(13056, 384, 32)` for Rows16+SharedK and `(13056, 512, 32)` with SimdgroupRows. The compiled MSL hashes were constant across runs: `0ae8a952ec5eb3109ee26495c399adf53528f719d4b57fb4be113fdb672bab9e` and `072123d02dead29525e413f91f521f9ac62b593b654a28b87c5572e39bc61016`.

In the SharedK path, the existing loop already assigns query blocks to simdgroups by `vector_block = simdgroup_slot; vector_block < tile_blocks; vector_block += simdgroups` (`cached_attention_row_tiled.rs:600`). The SimdgroupRows selector also changes score and accumulator ownership extents (`:450-452`, `:541-542`, `:736-752`). Thus the toggle changes more than which arithmetic instruction runs: it changes live matrix ownership and row scaling. The captured replay p50s moved higher in all six paired cells while K sharing remained enabled. Static threadgroup bytes stayed 13,056; the pipeline's reported maximum threads changed from 384 to 512. These metadata do not expose occupancy, register spills, or the device-level cause of the latency movement.

The source comparison with llama.cpp's local Metal path is documented in the task record: for F32 K/V and head dimension 64, llama's classic flash-attention entry is `fa_f32` with `Q=8`, `C=64`, and `NSG=4`; each simdgroup owns `Q/NSG=2` query rows and two 8-key chunks per cache iteration. Proxima's captured entry has 8 KV heads, 2 query groups, head dimension 64, and 2 simdgroups. Llama's `FA_TYPES_F32` keeps K/V and QK accumulation in float while storing Q as half; Proxima's F16 MMA path narrows Q/K/V operands. These code paths are not dtype-parity comparisons. See `fa_f32.metal:23-29`, `fa_common.metal:89-91, 185-190, 295-343`, `ggml-metal-ops.cpp:3515-3573`, and `cached_attention_row_tiled.rs:443-452, 541-542, 600-615, 736-752`.

## Discipline

This composes the existing numeric and scheduling axes; it adds no primitive or Granite-specific dispatch. The relevant cost axis is isolated cached-attention GPU replay time by prompt length. Real checkpoint/tokenizer inputs and the captured Metal dispatch provide the workload. MSL hashes, grids, resources, output bytes, and IDs identify each arm. No hardware occupancy or spill counter is collected.
