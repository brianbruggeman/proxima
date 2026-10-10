# Card 12: replay the Granite simdgroup-count axis on Metal

**Owner:** GPT-6 Luna

**Dependency:** Card 11 landed; Cards 00–10 supply the capture, replay, resource, and report-checking paths.

**Commit:** `perf(interop): probe granite simdgroup count on metal`

**Budget:** at most 30 minutes front-to-back, including serial Metal runs for all supported explicit counts, report checks, evidence review, and the TASKS update. If compilation or the loader stalls, retain that process evidence and leave this card open; do not substitute a compile result for a replay.

## Purpose

Observe the one-axis effect of changing row-tiled simdgroup count from the sized Legacy count to each supported explicit count (`Groups2`, `Groups4`, `Groups8`) on the captured Granite prefill dispatch. The selection is `PROXIMA_SERVING_ATTENTION_SIMDGROUP_COUNT` (or `attention_simdgroup_count` in ServingSettings TOML); `legacy` remains the default. For Granite's 64-wide head, the sized count is two, and explicit counts produce 64-, 128-, and 256-thread groups. Card 11's CPU dispatch test holds query-row and key-block sizes fixed (`omega/src/msl/attn_rows_tests.rs:2375-2445`). The count changes dimension ownership and split sizing, **not** query rows per threadgroup or K/V reuse across row tiles (`omega/src/msl/signature_tokens_prelude.rs:2775-2839`). This replay cannot establish a K/V traffic mechanism.

## Kernel and dispatch effect for the 64-wide Granite head

`row_tiled_form` admits an explicit count only when `head_dim` is divisible by `8 * simdgroups`, the 8-wide key block divides evenly across the count, the accumulator-fragment ceiling is respected, and the row-tile threadgroup-memory budget fits (`omega/src/msl/signature_tokens_prelude.rs:2800-2838`). For Granite `head_dim=64` and `block=64`, all three explicit counts satisfy those shape divisions; no other count is inferred as legal. The generated kernel sets `threads = simdgroups * 32`, `dims_per_group = head_dim / 8 / simdgroups`, and `key_tiles_per_group = (block / 8) / simdgroups` (`omega/src/msl/cached_attention_row_tiled.rs:438-450`).

| Variant | Resolved simdgroups | Threads per group | 8-wide head fragments per group | 8-key fragments per group | Observed dispatch width at 972 tokens |
|---|---:|---:|---:|---:|---:|
| sized Legacy | 2 | 64 | 4 | 4 | 64 |
| Groups2 | 2 | 64 | 4 | 4 | 64 |
| Groups4 | 4 | 128 | 2 | 2 | 128 |
| Groups8 | 8 | 256 | 1 | 1 | 256 |

The count is a compile-time kernel selector, so each explicit count has a distinct captured entry/source SHA. The attention row tile stays fixed; the kernel redistributes dimension and key-fragment ownership across the simdgroups, and dispatch threadgroup width scales with the count. The replay records these structural changes, but they do not by themselves explain the GPU timing differences.

Use the real Granite 3.1 1B A400M Instruct Q8_0 checkpoint and the nominal 971-token Sherlock prefix already used by Cards 03–10. The tokenizer produced 972 actual tokens in these runs, so the captured dispatch extents were `[972,8,2,64]` and the complete attention output was 3,981,312 bytes. `[1000,8,2,64]` was Card 11's CPU sizing fixture, not a captured checkpoint request (`proxima-model-interop/tests/granite_attention_variant_prefill.rs:32-38,1179-1236`; `omega/src/msl/attn_rows_tests.rs:2375-2396`). If the real tokenizer produces a different count, retain the actual count and refuse a comparison whose two arms have different extents.

## Scope

1. Add one `#[proxima::test]` named `perf_granite_simdgroup_count_against_f16_legacy`. It reads `PROXIMA_SERVING_ATTENTION_SIMDGROUP_COUNT` through `ServingSettings::from_env`; invoke it with `groups2`, `groups4`, and `groups8`. Compose the selected count onto `f16_variant()` so both arms hold `kv_storage=F32`, `mma_precision=F16`, `kv_reuse=Legacy`, `tile_height=Legacy`, `query_parallelism=Legacy`, `simd_topology=Legacy`, and `prefetch=Off`. The production sized rule and default stay unchanged. A ServingSettings unit test must show builder/TOML/environment parity for all three values, lower them into `ServingConfig.attention_variant`, retain Legacy by default, and reject `groups16`.
2. Reuse and narrowly parameterize the Card 00/02 capture and report path; do not duplicate capture/replay loops. Use nominal `[971]`, encode the real prompt with the checkpoint tokenizer, and retain the actual token count and prompt-ID hash. Match captured `CachedAttention` records by node and all extents. Before timing, require no fault binding, positive grid dimensions, distinct captured entry and MSL SHA256, **equal complete replay-output bytes** (3,977,216 bytes for the previously observed 971-token shape), and equal nonempty generated-ID arrays. Require selected suffix and widths: `groups2`→`_simdgroups2`/64, `groups4`→`_simdgroups4`/128, `groups8`→`_simdgroups8`/256; baseline width is 64. Include the selected label in report metadata. A failed equality gate leaves no usable timing report. Retire each report destination at test start using the existing checkpoint-alias guard.
3. Use the existing 20-round alternating replay loop and five-replay `Cell` resource observation per arm. Labels are `legacy` and selected `simdgroupsN`; preserve all raw positive GPU times, signed deltas, p50/p90/p99/min/max/CoV, entry, MSL SHA256, grid, pipeline resources, bound bytes, host/load context, checkpoint provenance, and serving options. Record both signs and zeros. Use a separate report path for each count.
4. Extend the existing `check_report.py` with `--selected-arm shared_k|simdgroups2|simdgroups4|simdgroups8`; default remains `shared_k`. Thread the label through arm, resource, alternating-order, and mutation checks. Retain all twelve negative controls and add a selected-entry-suffix mutation; all thirteen must be rejected for each simdgroup report. Recompute summaries and signed deltas from raw samples. Validate suffix, width, and selected label against the report.

The abandoned design is a test-local hardcoded Groups4 toggle, a parallel count enum, a second Granite-only replay harness, or a new attention kernel. The existing `AttentionSimdgroupCount`, `ServingSettings`, and report path express the experiment. The helper remains a synchronous test-owned capture/replay sequence; it introduces no async producer, queue, channel, or pipe primitive. `metal-attn-variants` keeps explicit selection opt-in.

## Acceptance

Run one GPU test process at a time. This host's Cargo-target test binary stalled at the macOS loader; the copied binary entered the harness, as recorded in `evidence/card11-loader-stall-2026-10-10.md`. Save each complete stdout/stderr and exit status in the matching `evidence/card12-simdgroups*-metal-2026-10-10.log`, and retain the full reports in `evidence/card12-simdgroups*-report.json` with their SHA256 values in TASKS. Groups4 was replayed in three separate processes; Groups2 and Groups8 were each replayed once.

```sh
nice -n 20 env CARGO_BUILD_JOBS=1 CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo test -p proxima-model-interop --features std,metal,metal-attn-split-rows,metal-attn-variants --lib simdgroup_count_setting_round_trips_and_reaches_serving_config -- --nocapture
nice -n 20 env CARGO_BUILD_JOBS=1 CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo test -p proxima-model-interop --features std,metal,instrument,metal-attn-split-rows,metal-attn-variants --test granite_attention_variant_prefill --no-run
cp /tmp/cargo_target/debug/deps/granite_attention_variant_prefill-4c29247f309c5b09 /tmp/card12-granite-attention-tests
PROXIMA_SERVING_ATTENTION_SIMDGROUP_COUNT=groups2 PROXIMA_TEST_TIMEOUT_MS=900000 PROXIMA_CAPTURE_LIVE=1 PROXIMA_CAPTURE_NODES=all PROXIMA_CAPTURE_STEPS=0 PROXIMA_GRANITE_AB_REPORT=/private/tmp/granite-attention-simdgroups2.json /tmp/card12-granite-attention-tests perf_granite_simdgroup_count_against_f16_legacy --nocapture
# Repeat the command above with groups4 and groups8; repeat groups4 twice more.
python3 proxima-tensor/specs/granite-attention-variant-performance/check_report.py --expected-shapes 1 --selected-arm simdgroups4 proxima-tensor/specs/granite-attention-variant-performance/evidence/card12-simdgroups4-run1-report.json
python3 proxima-tensor/specs/granite-attention-variant-performance/check_report.py --expected-shapes 1 --selected-arm simdgroups4 --negative-controls proxima-tensor/specs/granite-attention-variant-performance/evidence/card12-simdgroups4-run1-report.json
```

## Captured records

All rows use Granite 3.1 1B A400M Instruct Q8_0, the same nominal 971-token Sherlock prompt (972 tokenizer IDs), F32 K/V, F16 MMA, and a same-process Legacy-count baseline. Each pair has equal complete output bytes and IDs `[322]`; the report contains the prompt-ID hash, all 40 raw dispatch samples, paired rounds, pipeline resources, and host/load context. `paired signs` count `selected GPU ns - Legacy GPU ns`, in positive/negative/zero order. These are isolated Metal dispatch cells; cross-process ranking remains **plausible**, not a verdict.

| Count/run | Legacy p50 / CoV | Selected p50 / CoV | Paired signs | Width; total grid threads; max pipeline threads | Report SHA256 |
|---|---:|---:|---:|---|---|
| Groups2 / 1 | 1,108,042 ns / 39.24% | 1,125,042 ns / 22.58% | 11/9/0 | 64; 62,464/62,464; 448/448 | `02c08ca78c25004a462034115ae1682e8156d22cb6e8da3e36ac91aca2bdec8c` |
| Groups4 / 1 | 1,087,625 ns / 2.06% | 1,047,042 ns / 2.01% | 4/16/0 | 64/128; 62,464/124,928; 448/576 | `c83c9b2c89892c88ebe2f0bf77affbe4d263c22a4e489b5e474c9dd38d2e031e` |
| Groups4 / 2 | 1,111,958 ns / 40.60% | 1,122,500 ns / 40.46% | 7/13/0 | 64/128; 62,464/124,928; 448/576 | `e4739586edea6cf1defc92706765a1c43ec99e0f50664ac66a55bb141c6ba597` |
| Groups4 / 3 | 1,100,625 ns / 43.38% | 1,073,625 ns / 42.92% | 6/14/0 | 64/128; 62,464/124,928; 448/576 | `165b3b20f4ec19f044ea875ac51da26f64a46227e0ca1451a36abb8089ae31a6` |
| Groups8 / 1 | 1,091,792 ns / 1.55% | 1,369,917 ns / 1.84% | 20/0/0 | 64/256; 62,464/249,856; 448/640 | `177a201e017b7c0f97916b3909249a94f7a331ebcc3865e08958fc1a0c74e93a` |

The per-run record is the report file with the matching name beside this card. `Groups2` and sized Legacy both resolve to two simdgroups, width 64, and 62,464 total threads; the explicit choice changes the entry/source identity but not the selected count. In the row-tiled MSL, `threads = simdgroups * 32`, `dims_per_group = head_dim / 8 / simdgroups`, and `key_tiles_per_group = (block / 8) / simdgroups` (`omega/src/msl/cached_attention_row_tiled.rs:438-450`). Every captured entry stayed at `r8`, and the tested variant leaves K/V reuse at Legacy. Thus Groups4/8 change work ownership and threadgroup width while keeping the eight-row tile and its reuse policy fixed; these records do not show additional K/V reuse or identify the cause of Groups8's longer cell time.

The first Groups2 attempt failed before dispatch because `validate_simdgroup_count_selection` also rejected non-cached-attention operations in the full model plan. The validator now returns for non-cached-attention operations; `explicit_simdgroup_count_ignores_non_cached_attention_operations` passes. The captured retry completed the real request. This is why the selector must be tested through a model request, not only with a one-op attention fixture.


| criterion | required count or record |
|---|---|
| Test selection | Each of three serial invocations selects 1 named test, passes 1, skips 26, and exits 0. Each emits exactly 1 shape record. |
| Config | Builder, TOML, and environment agree for `groups2`, `groups4`, and `groups8`; default is `legacy`; `groups16` is rejected; each explicit value reaches `ServingConfig.attention_variant`. |
| One variable | Each pair has equal node/extents, same checkpoint/prompt/serving settings, F32 K/V and F16 MMA. `AttentionVariant` differs only in Legacy versus the requested simdgroup count. Captured suffix and widths match 64/64, 64/128, and 64/256 respectively; MSL SHA differs. Record both full grids and resource tuples; do not demand equal total thread count or a particular resource value. |
| Complete output and request | The pair has 2 complete output hashes that match byte for byte, 2 nonempty generated-ID arrays that match exactly, 0 fault bindings, 0 replay errors, and dispatch first extents equal the actual tokenizer count. For the nominal 971-token prompt, the captured tokenizer count is 972 and each report records 3,981,312 output bytes; the report checker derives the required length from captured extents. |
| Raw replay and resources | The shape has 2 arms × 20 finite positive samples and 20 signed paired deltas, with exactly 10 rounds in each arm order; 1 `Cell` resource line per arm from 5 additional replays, including CPU, RSS, footprint, GPU allocation, and host-load fields. Preserve all samples and both positive and negative deltas. |
| Independent positive check | Each checker run prints `prompt_shapes=1 arms=2 samples_per_arm=20 resource_cells=2 matching_pairs=1 errors=0`; positive, negative, and zero round counts sum to 20. |
| Degenerate control | Each checker run prints `negative_controls=13 rejected=13`; each original report SHA256 is unchanged. The thirteenth mutation removes the selected entry suffix and must be rejected. |
| SharedK compatibility | Run `python3 proxima-tensor/specs/granite-attention-variant-performance/check_report.py --expected-shapes 2 /private/tmp/granite-attention-ab.json` against the retained Card 02 artifact; it still prints `prompt_shapes=2 arms=4 samples_per_arm=20 resource_cells=4 matching_pairs=2 errors=0`. If that external artifact is unavailable, use the saved Card 02 report bytes identified by TASKS; do not manufacture a replacement. |

Read the raw report and record both p50s, both CoVs, the positive/negative/zero signed-round counts, full output/ID evidence, entries/SHA, grids, and resource tuples. Retain unexpected signs and noisy cells. If binding, compilation, capture, output, IDs, or replay fails, preserve its exact payload; do not call an incomplete pair a timing result. Isolated dispatch timing supports no whole-layer, whole-request, semantic-quality, llama.cpp-parity, or device-level causal conclusion. The owner decides any performance verdict.

The focused config test and copied-binary discovery probe were rerun with their command output recorded in `evidence/card12-revalidation-2026-10-10.md`. The Groups4 raw samples show high variation in two processes and the p50 comparison changes sign across the three processes; that variation's cause remains unexplained.
