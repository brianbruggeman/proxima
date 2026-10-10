# Card 07: remove the adjacent SharedK barrier

**Owner:** GPT-6 Luna

**Dependency:** Card 06

**Commit:** `perf(omega): remove adjacent shared-k barrier`

**Budget:** at most 30 minutes front-to-back, including acceptance and task update.

## Purpose

The SharedK path has two consecutive `threadgroup_barrier(mem_flags::mem_threadgroup)` calls: one immediately after simdgroup stores to shared K, and another before shared K loads (`omega/src/msl/cached_attention_row_tiled.rs`, before this change at lines 589 and 592). No shared-memory write occurs between them. Keep the first barrier, which publishes all stores, and remove the second barrier.

This edits the existing SharedK path. It adds no API, enum, kernel, or Granite-specific behavior. `AttentionVariant::kv_reuse` remains the runtime experiment toggle; the default Legacy path does not enter this branch.

## Scope

Remove only the second adjacent barrier. Extend the existing captured probe output with each arm's generated MSL SHA256 and pipeline resource tuple `(static threadgroup bytes, max threads per threadgroup, execution width)` so the selected pipeline identity/resources are recorded with its replay time. Run the existing Rows16+SharedK probe for actual prompt lengths 256 and 971 in three separate, sequential test processes; do not run GPU tests concurrently.

The approach abandoned is adding a new barrier-policy axis or a specialized kernel. The existing SharedK variant already gates this path, and the source has no intervening shared write requiring the second barrier.

## Acceptance

```sh
PROXIMA_TEST_TIMEOUT_MS=900000 PROXIMA_CAPTURE_LIVE=1 PROXIMA_CAPTURE_NODES=all PROXIMA_CAPTURE_STEPS=0 CARGO_BUILD_JOBS=1 CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo nextest run -p proxima-model-interop --features std,metal,instrument,metal-attn-split-rows,metal-attn-variants --test granite_attention_variant_prefill -E 'test(~perf_granite_rows16_shared_k)' -j 1 --success-output immediate
```

Run the command three times, serially, saving complete outputs to `evidence/card07-shared-k-barrier-run{1,2,3}-2026-10-10.log`. Each process must select 1 test, pass 1, and skip 22. Each output must include both actual prompt lengths, 20 paired samples per shape, entry names, different MSL hashes for Legacy versus SharedK, pipeline resource tuples, equal output bytes, equal generated IDs before timing, and zero replay/binding errors.

Compare each new run's baseline and selected p50 against Card 06's recorded pair for the same shape. Report all three new pairs and their range; do not attribute any timing change to barrier removal if the paired deltas do not move consistently across runs. Output bytes and generated IDs must remain equal within every pair.

## Observation

All three serial nextest processes passed: 1 selected, 1 passed, 22 filtered. Each emitted 2 shapes × 20 paired samples. For every shape/run, `changed_bits=0`, all output values were finite, and both arms generated ID `[322]` before replay timing.

| actual tokens | run 1 baseline/selected p50 ns | run 2 baseline/selected p50 ns | run 3 baseline/selected p50 ns | run 3 baseline/selected range ns |
|---:|---:|---:|---:|---:|
| 256 | 181,749.929 / 243,374.961 | 182,291.726 / 244,374.969 | 180,916.628 / 240,791.589 | baseline 180,916.628–182,291.726; selected 240,791.589–244,374.969 |
| 971 | 1,079,624.984 / 1,996,291.569 | 1,083,041.658 / 2,006,749.972 | 1,079,874.928 / 2,020,125.045 | baseline 1,079,624.984–1,083,041.658; selected 1,996,291.569–2,020,125.045 |

These are Granite 3.1 1B A400M Instruct Q8_0 on local Metal, F16 MMA/F32 K/V, using the 256-token and 971-token prompt captures. In each run, baseline/selected pipeline resources were `(9600, 448, 32)` / `(13056, 384, 32)` in `(static threadgroup bytes, max threads per threadgroup, execution width)` order. The baseline/selected grids were 16,384/8,192 threads at 256 tokens and 62,464/31,232 at 971 tokens. All runs emitted the same entry names. Their MSL SHA256 values were baseline `b85db41a81b3503641fbdb453cff0b7dd32e442085614eef5ab20fe1b1e0ba5d` and selected `0ae8a952ec5eb3109ee26495c399adf53528f719d4b57fb4be113fdb672bab9e`.

Against Card 06's logged selected p50, the 256-token results moved down by 1,000.124 ns, 0.116 ns, and 3,583.496 ns across the three runs. At 971 tokens they moved down by 8,333.474 ns, up by 2,124.929 ns, and up by 15,500.002 ns. The direction is inconsistent at 971 tokens; these observations do not attribute a timing change to barrier removal. In all six Card 07 paired cells, SharedK's selected p50 remained above the paired baseline. The source removed one adjacent synchronization point; the collected pipeline resources do not measure occupancy or explain the remaining replay gap.

The captured source/resource pair does expose a larger per-simdgroup matrix footprint in the selected row schedule. For this `h8_g2_d64_r8_n2_b64` baseline, `tile_blocks=2`, `key_tiles_per_group=4`, and the `scores` array at `omega/src/msl/cached_attention_row_tiled.rs:541-542` is `[4][2]` (8 `simdgroup_float8x8` matrices). For the selected `h8_g2_d64_r16_n2_b64` SharedK schedule, `tile_blocks=4`, `score_key_tiles=8`, and `scores` is `[8][4]` (32 matrices). The `accumulated` array is `[4][2]` versus `[4][4]` (8 versus 16 matrices; declaration at `:504`). These dimensions follow the formulas at `:443-452` and the captured entry names. The compiled pipeline also reports lower `maxTotalThreadsPerThreadgroup` (448→384) and more static threadgroup memory (9,600→13,056 bytes). This is concrete source and compiler-resource evidence consistent with higher register/threadgroup pressure; it does not report actual occupancy, register spills, or attribute the full time difference.

Logs: `evidence/card07-shared-k-barrier-run1-2026-10-10.log:11-12`, `...run2...log:9-10`, and `...run3...log:9-10`.

## Discipline

This is a gated optimization of an existing kernel path, not a new primitive. The relevant cost axis is isolated cached-attention GPU replay time by prompt length; output bytes, generated token IDs, MSL SHA256, grid, and pipeline resources identify what ran. The real Granite checkpoint and tokenizer remain the workload. No hardware instruction, register spill, or occupancy counters are captured by this card; do not claim that the barrier or resource footprint alone caused the measured timing.
