# granite-attention-variant-performance

status: draft
owner: brian
created: 2026-10-09

## problem

On the real Granite 3.1 1B A400M Instruct GGUF Q8_0 checkpoint and local Metal device, Proxima has no saved raw A/B replay cells for a captured multi-row attention dispatch under all-legacy versus `SharedK`, at two verified prefill prompt lengths, with output agreement and resource provenance attached to each cell.

## refutation condition

The proposed measurement path is refused if either arm is only a requested selector or synthetic manifest, if either captured dispatch has a fault binding or is unreplayable, if the pair's full replay output bytes or generated IDs differ, if a timed sample is missing/nonfinite, or if the second prompt is not a distinct tokenizer-verified prefix length of the same passage.

## requirements

| id | requirement | testable in isolation |
|---|---|---|
| R1 | each arm is an executed, matched, multi-row `CachedAttention` capture from the real Granite checkpoint, admitted by numeric-matrix Card 26's output/fault gate | yes |
| R2 | each arm retains 20 raw, alternating-order, single-dispatch GPU replay times with a defined summary and zero dropped replay errors | yes |
| R3 | each arm retains the existing process CPU, RSS, footprint, Metal-allocation, load, and wall-clock resource observation with its limits stated | yes |
| R4 | an independent checker validates the saved schema, raw-to-summary calculations, pair identity/output/ID constraints, alternating round order, and six degenerate mutations | yes |
| R5 | the report contains two distinct tokenizer-verified prompt lengths from the same Sherlock passage, with one legacy/SharedK pair at each length | yes |

## architecture

The test target `proxima-model-interop/tests/granite_attention_variant_prefill.rs` loads the model and invokes the public `ServingConfig` path. Card 26 supplies a matched real capture pair and checks full output bytes; this spec's private measurement helper must retain and return ownership of both selected `CapturedDispatch` handles along with their output bytes and generated IDs. If Card 26's helper returns only metadata, Card 00 refactors that private helper to return the owning pair without changing Card 26 behavior. The measurement function calls `time_gpu_ns(1)` on references to those retained handles in alternating order for 20 rounds per shape. `Cell` from `proxima-model-interop/examples/cell_resources/cell.rs` supplies one labeled five-replay process resource observation per arm. A JSON report at `PROXIMA_GRANITE_AB_REPORT` carries version, checkpoint path and byte length, host/OS and nullable device description, serving config, and a `shapes` array. Each shape carries nominal and actual tokenizer counts, prompt-ID SHA256, `output_equal=true`, `ids_equal=true`, and exactly two arm records. Each arm carries observed node/extents/entry/source SHA/grid/pipeline resources/bound bytes, `fault_binding_present=false` from captured bindings, full replay-output byte count and SHA256, generated IDs, 20 ordered raw nanosecond samples, summary statistics, resource string, `timing_attempts=20`, `resource_replay_attempts=5`, and `replay_errors=0`. Each shape also carries 20 signed selected-minus-legacy round differences. No field is inferred from the requested selector where a captured value exists.

The first measurement slice writes one shape from the existing nominal 971-token Sherlock prompt. A separate checker slice validates that artifact and accepts an explicit `--expected-shapes` count. The final slice parameterizes the same prompt builder to add a nominal 256-token prefix and re-runs both shapes into one report. Actual checkpoint-tokenizer counts, rather than nominal targets, identify the two cells. Prompt-ID SHA256 hashes the concatenation of each token ID as a little-endian `u32`; output SHA256 hashes the exact `replay_output_elements` byte span. `pipeline_resources` names `tg_static_bytes`, `max_threads`, and `exec_width`; the resource string retains `wall_ms`, `cpu_pct`, `rss_peak_mb`, `footprint_mb`, `gpu_alloc_mb`, `load_before`, and `load_after` labels.

### sample statistics contract

Each arm has exactly 20 finite, positive `f64` nanosecond values. Sort a copy ascending. `min` and `max` are the first and last values. Percentile `p` uses nearest rank `sorted[ceil(p * 20) - 1]`, so p50/p90/p99 use zero-based positions 9/17/19. The mean is `sum/20`; `cov_percent = 100 * sqrt(sum((sample - mean)^2)/20) / mean` (population standard deviation). The checker recomputes these from raw values and accepts an absolute difference of at most `1e-6` ns for min/max/percentiles and `1e-8` percentage points for CoV. `time_gpu_ns(1)` already includes its command-buffer floor; no empty-buffer sample or subtraction is specified.

For each shape, `rounds` is an array of exactly 20 objects in order with `round` equal to integers `0..19`. Each object has `arm_order`, an ordered two-element array of exact strings: `["legacy","shared_k"]` on even rounds and `["shared_k","legacy"]` on odd rounds; and `selected_minus_legacy_ns`, the signed selected time minus legacy time for that round. Each arm's `samples` array has exactly one `{round, position, gpu_ns}` object for each `round` in `0..19`, in ascending round order. `position` is integer 0 or 1 and must equal that arm's index in the round's `arm_order`; the two arms occupy different positions. The checker requires these bijections, finite positive `gpu_ns`, and each signed difference within `1e-6` ns of the two raw sample values. Preserve both signs and count gains/losses separately; neither sign is required.

### decisions

| decision | chosen | reason |
|---|---|---|
| measurement seam | two actual `ServingConfig` executions and their captured pipelines | `norm_variant_ab` currently loads Gemma4, refuses typed selectors outside describe mode, and times external `.metal` body files; describe metadata is not a timed Granite dispatch (`proxima-model-interop/examples/norm_variant_ab.rs:355-402,419-487,654-669,802-812`). |
| replay unit | one matched multi-row attention dispatch per shape | directly tests the selected kernel; isolated replay is explicitly labeled and cannot stand in for layer/request latency. |
| resources | reuse existing `Cell` formatted observation | keeps the established macOS sampler; a printed zero footprint can mean unavailable, so it is never interpreted as measured zero memory (`proxima-model-interop/examples/cell_resources/cell.rs:21-50`). |
| scope | one shape, checker, then second prefix shape | each card remains a coherent commit within 30 minutes and the checker can validate either cardinality without a rewrite. |

## acceptance criteria

| id | discharges | command | expected |
|---|---|---|---|
| AC0 | R1,R2,R3 | `PROXIMA_TEST_TIMEOUT_MS=900000 PROXIMA_CAPTURE_LIVE=1 PROXIMA_CAPTURE_NODES=all PROXIMA_CAPTURE_STEPS=0 PROXIMA_GRANITE_AB_REPORT=/private/tmp/granite-attention-ab.json cargo nextest run -p proxima-model-interop --features std,metal,instrument,metal-attn-split-rows,metal-attn-variants --test granite_attention_variant_prefill -E 'test(~perf_card_00_granite_attention_replay_cell)' -j 1 --success-output immediate` | filter selects 1 test, 1 passed; saved report has 1 verified prompt shape, 2 captured arms, 20 raw samples and 1 process resource observation per arm, 1 exact output/ID pair, 0 replay errors |
| AC1a | R4 | `python3 proxima-tensor/specs/granite-attention-variant-performance/check_report.py --expected-shapes 1 /private/tmp/granite-attention-ab.json` | prints `prompt_shapes=1 arms=2 samples_per_arm=20 resource_cells=2 matching_pairs=1 errors=0` |
| AC1b | R4 | `python3 proxima-tensor/specs/granite-attention-variant-performance/check_report.py --expected-shapes 1 --negative-controls /private/tmp/granite-attention-ab.json` | prints `negative_controls=6 rejected=6`; report bytes unchanged |
| AC2a | R1,R2,R3,R5 | `PROXIMA_TEST_TIMEOUT_MS=900000 PROXIMA_CAPTURE_LIVE=1 PROXIMA_CAPTURE_NODES=all PROXIMA_CAPTURE_STEPS=0 PROXIMA_GRANITE_AB_REPORT=/private/tmp/granite-attention-ab.json cargo nextest run -p proxima-model-interop --features std,metal,instrument,metal-attn-split-rows,metal-attn-variants --test granite_attention_variant_prefill -E 'test(~perf_card_02_granite_two_prefill_shapes)' -j 1 --success-output immediate` | filter selects 1 test, 1 passed; report has 2 distinct actual prompt counts, 4 captured arms, 20 raw samples and 1 process resource observation per arm, 2 exact output/ID pairs, 0 replay errors |
| AC2b | R4,R5 | `python3 proxima-tensor/specs/granite-attention-variant-performance/check_report.py --expected-shapes 2 /private/tmp/granite-attention-ab.json` | prints `prompt_shapes=2 arms=4 samples_per_arm=20 resource_cells=4 matching_pairs=2 errors=0` |

## out of scope

- Whole-layer, whole-step, or public-request latency/throughput claims; llama.cpp comparisons; BF16/BF8 or other attention axes; model-quality conclusions.
- Timing work inside `../granite-attention-numeric-matrix/`. Its Card 26 capture/output gate is a prerequisite, not a timed cell.
- An empty-command-buffer floor measurement: the chosen `time_gpu_ns(1)` observation includes that floor by contract.

## risks

| risk | likelihood | cost | response |
|---|---|---|---|
| retained live buffers change after capture | medium | replay output is not the live kernel input/output | Card 26 checks complete bytes immediately after each request and refuses missing or fault-bound records; retain the mismatch payload rather than calling it arithmetic divergence. |
| sampled dispatch differs by shape | medium | false A/B comparison | pair by node/extents within each prompt length, require different captured entry/SHA between arms, and record both grids. |
| resource sampler reports unavailable values as zero | medium | false memory claim | retain opaque `Cell` output; never interpret a zero footprint as measured zero. |
| GPU/background load varies | high | apparent gain or loss from noise | alternate arm order, retain all samples and signed per-round deltas, include host/load context, and render no speed verdict. |

## context

- `../granite-attention-numeric-matrix/SPEC.md`, `../granite-attention-numeric-matrix/cards/26-granite-attention-replay-pair.md`, and that spec's `TASKS.md` provide the real capture prerequisite.
- `proxima-model-interop/tests/granite_attention_variant_prefill.rs:23-26,99-154,427-464` identifies the checkpoint, F32 K/V serving configuration, tokenizer-driven Sherlock prompt, and live capture path.
- `omega/src/metal/arena_encode_dispatch_finish.rs:1278-1355,1555-1601,2283-2390` defines capture ownership, replay output behavior, and isolated replay timing.
- `proxima-model-interop/examples/cell_resources/cell.rs:1-85` supplies process resource observations.
