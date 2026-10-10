# Card 22: warmup-controlled query-row parallelism replay

## Question

Does changing only `query_parallelism` from Legacy to `SimdgroupRows` retain the same complete Granite attention output and generated IDs, and what paired GPU timestamps and dispatch source are captured at both established prompt sizes?

## Source-grounded change

Card03 measured one cold 971-token query-parallelism-only cell. Its captured Legacy baseline and selected variant both used F16 MMA/F32 K/V, Legacy K/V reuse, Legacy tile height, Legacy SIMD topology, and prefetch off; its selected p50 was 1,993,750 ns versus 1,085,750 ns Legacy, and all 20 selected-minus-Legacy samples were positive (`cards/03-prefill-variant-probes.md`, raw `evidence/card03-acceptance-2026-10-10.log`). This card repeats that single-axis comparison using the established two-warmup-pair, two-prompt, three-process protocol.

For the admitted Granite shape, Omega's generated MSL sets `key_tiles_per_simdgroup` to `block / 8` for `SimdgroupRows`, while Legacy uses `(block / 8) / simdgroups`; the key-tile loop selects `group` for the row-parallel form and `simdgroup_slot + group * simdgroups` for Legacy (`omega/src/msl/cached_attention_row_tiled.rs:448-449,546`). This is source ownership, not a counter of device loads or a timing cause.

## One-variable contract

- Baseline: F16 MMA, F32 K/V, sized Legacy simdgroup count, Legacy K/V reuse, Legacy tile height, Legacy query parallelism, Legacy topology, prefetch off.
- Selected: the same `AttentionVariant`, with only `query_parallelism=SimdgroupRows`.
- Inputs: the existing 256-target and 971-target Granite word-boundary Sherlock prompts; retain actual token counts from the report.
- Replay: two alternating warmup pairs, then 20 alternating measured pairs per shape, with the existing five-replay resource observations.
- Output gates: complete output bytes/hash and generated IDs equal before timing; require captured entry/source identity changes and matched node/extents.

## Commands

Build one focused integration binary with one Cargo job, copy it out of Cargo's `deps` directory, and run three Metal processes serially. Each process writes its complete stdout/stderr and a unique report. Use the copied binary to avoid the observed Cargo `deps` dyld discovery stall.

```sh
nice -n 20 env CARGO_BUILD_JOBS=1 CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo test -p proxima-model-interop --features std,metal,instrument,metal-attn-split-rows,metal-attn-variants --test granite_attention_variant_prefill --no-run
cp /tmp/cargo_target/debug/deps/granite_attention_variant_prefill-4c29247f309c5b09 /tmp/card22-granite-attention-tests
for run in 1 2 3; do
  PROXIMA_TEST_TIMEOUT_MS=900000 PROXIMA_CAPTURE_LIVE=1 PROXIMA_CAPTURE_NODES=all PROXIMA_CAPTURE_STEPS=0 PROXIMA_GRANITE_AB_REPORT="/private/tmp/proxima-granite-main/proxima-tensor/specs/granite-attention-variant-performance/evidence/card22-query-rows-run${run}-report.json" /tmp/card22-granite-attention-tests perf_granite_query_rows_warmup_control_two_shapes_against_f16_legacy --nocapture
  # Preserve each process output in evidence/card22-query-rows-run${run}-2026-10-10.log.
done
```

For each report, run `check_report.py --expected-shapes 2 --selected-arm query_simdgroup_rows`, then repeat with `--negative-controls`. Preserve raw samples, paired signs, p50/p90/p99, CoV, source hashes, entries, grids, pipeline resources, output hashes, and resource/load records.

## Acceptance

- Each serial process selects one named test, passes one, and records both prompt shapes, two warmup pairs and 20 measured pairs per arm.
- Each report's positive checker prints `prompt_shapes=2 arms=4 samples_per_arm=20 resource_cells=4 matching_pairs=2 errors=0`; the negative checker rejects all 13 controls and does not mutate the source report.
- Both actual-shape output hashes and generated ID arrays match between F16-MMA/F32-KV Legacy and the selected query-row schedule.
- Each pair differs only in `query_parallelism`; all timing deltas and cross-process variation remain visible. This card makes no whole-request, performance, or causal claim.

## Residual

## Captured records

Environment: Apple M1 Max, macOS 15.8 (24H20), Metal 3, aarch64; Granite 3.1 1B A400M Instruct Q8_0, F32 K/V, F16 MMA. Reports identify node 89 and actual prompt extents `[256,8,2,64]` and `[972,8,2,64]`. Each process recorded two warmup pairs and 20 alternating measured pairs per shape. The same 1 MiB output SHA and generated ID `[322]` matched within each pair in all three processes: 256-token SHA `454ca3f4483db67f72f42baa6f7625c46853a9be768e4bbcdd35116789f72ba6`; 972-token SHA `970fa9d575423441d23b639a5831938d9e9c46aa8c3c6e84fb2d4f7fe0dfbcae`.

The table gives p50/p90/p99 GPU timestamp nanoseconds and sample CoV. Signs are paired `query_simdgroup_rows - legacy` counts from the 20 rounds at each shape.

| Process/report SHA256 | Actual tokens | Legacy p50/p90/p99; CoV | Query rows p50/p90/p99; CoV | signs +/−/0 |
|---|---:|---:|---:|---:|
| run1 `b79b1846a1cc…c53c2d5a` | 256 | 181,916.6 / 182,958.3 / 185,791.6; 0.54% | 335,375.0 / 336,708.4 / 339,750.0; 0.38% | 20/0/0 |
| run1 | 972 | 1,085,958.3 / 1,122,875.0 / 1,139,499.9; 2.78% | 2,030,291.6 / 2,224,500.1 / 2,355,583.4; 5.20% | 20/0/0 |
| run2 `4a3ee96c97c8…b5c341a4` | 256 | 182,458.3 / 184,625.0 / 186,625.1; 0.77% | 336,583.4 / 338,375.0 / 342,916.7; 0.61% | 20/0/0 |
| run2 | 972 | 1,100,166.7 / 1,122,874.9 / 1,129,708.4; 1.31% | 2,003,875.0 / 2,102,583.4 / 2,140,875.0; 3.08% | 20/0/0 |
| run3 `aa31e62b9341…d580e09a4` | 256 | 181,875.0 / 182,625.0 / 187,166.7; 1.01% | 338,375.0 / 341,666.7 / 353,083.4; 1.26% | 20/0/0 |
| run3 | 972 | 1,093,250.0 / 1,113,416.7 / 1,123,000.1; 2.19% | 1,976,166.6 / 2,026,666.7 / 2,053,291.7; 2.09% | 20/0/0 |

Run report files retain full samples, both arm records, warmup records, source SHA, grids, resource tuples, output identity, and process context. The report SHA256 values are independently checkable with `shasum -a 256` against each file.

## Dispatch/source observation

Legacy and selected use the same grid for each shape: 16,384 threads at 256 tokens and 62,464 at 972 tokens, threadgroup width 64, depth 1. Both report `exec_width=32` and `tg_static_bytes=9600`; Legacy reports `max_threads=448`, and query rows reports `max_threads=384`. Legacy entry/source SHA are `omega_cached_attention_h8_g2_d64_s3c800000_ln9223372036854775808_up0_r8_n2_b64_rt_mma_f16` / `b85db41a81b3503641fbdb453cff0b7dd32e442085614eef5ab20fe1b1e0ba5d`. The query-row entry adds `_query_simdgroup_rows`, with source SHA `3f29d84803567a8d6806cd9f7e56b3a9a9754d1a16f502975c164518600be94a`.

In `omega/src/msl/cached_attention_row_tiled.rs:447-449,546`, query-row mode sets the key-tile loop bound to `block / 8` and uses `group` as `key_tile`; Legacy divides that bound by simdgroup count and uses `simdgroup_slot + group * simdgroups`. Thus the source assigns each simdgroup the same key-tile index sequence in query-row mode, versus partitioned indices in Legacy. The records do not include emitted MSL text for this axis. Neither source ownership nor pipeline resource metadata establishes actual device loads, occupancy, or the cause of the timestamp difference.

## Residual

Across all six shape/process cells, all 120 measured pairs have positive selected-minus-Legacy timestamp deltas, with equal captured outputs and IDs. The timestamp behavior is observed; why it occurs on this device remains unexplained. Metal System Trace and counter probes in Cards 19–20 did not expose per-dispatch shader counters or labels. No follow-on kernel change is justified by this replay alone. Keep query-row parallelism opt-in; this card does not change serving defaults or establish whole-request latency.
