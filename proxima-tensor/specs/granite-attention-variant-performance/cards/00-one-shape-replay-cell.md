# Card 00: record one real Granite attention replay pair

**Owner:** GPT-6 Luna
**Dependency:** `../../granite-attention-numeric-matrix/cards/26-granite-attention-replay-pair.md` admitted and landed
**Commit:** `perf(interop): record one granite attention replay pair`
**Budget:** at most 30 minutes front-to-back, including AC0, review, and TASKS.md update.

## Purpose

Write one raw, two-arm Metal replay report for the existing nominal 971-token Sherlock prompt. The benchmark consumes the real legacy/SharedK `CachedAttention` captures and full-output gate from numeric-matrix Card 26; it does not substitute `norm_variant_ab` describe metadata or external `.metal` body files.

## Read

- `../SPEC.md` for the exact JSON fields and sample-statistics contract.
- `../../granite-attention-numeric-matrix/cards/26-granite-attention-replay-pair.md` and its implemented capture helper.
- `omega/src/metal/arena_encode_dispatch_finish.rs:2283-2390` for replay output and `time_gpu_ns(1)` semantics, floor included.
- `proxima-model-interop/examples/cell_resources/cell.rs:1-85` for the existing process resource sampler.
- `proxima-model-interop/tests/fixtures/llama-parity/checkpoints.toml:77-85` and `proxima-model-interop/tests/fixtures/llama-parity/granite_moe/gguf_kv.txt:4-10,26` for the exact checkpoint SHA, size, architecture, model name, and Q8_0 file type.

## Edit

- `proxima-model-interop/tests/granite_attention_variant_prefill.rs`: add the named Card 00 measurement test and a small reusable `ShapeReport` writer using existing `serde_json`; retain the two owning `CapturedDispatch` handles from Card 26's pair helper (refactor its private return type if it only returns metadata); import `Cell` with `#[path = "../examples/cell_resources/cell.rs"]`.
- `proxima-tensor/specs/granite-attention-variant-performance/TASKS.md`: tick Card 00 and record the report path/counts after AC0.

## Steps

1. Resolve the checkpoint and report destination; refuse report destinations that resolve to or alias the checkpoint, then retire a previous report. Verify the mapped checkpoint's full SHA256, `general.architecture=granitemoe`, `general.name=Granite 3.1 1b A400M Instruct`, and `general.file_type=7` before loading. Derive checkpoint byte length from the mapped bytes. Reuse Card 26's real paired capture on that checkpoint and its existing Sherlock prompt. The private pair helper must return an owning pair containing the legacy and selected `CapturedDispatch` handles, their full replay-output bytes, and generated IDs; do not discard the handles after projecting metadata. Require both matched records to have no `Binding::Fault`, positive active grid dimensions, distinct captured entry/SHA, full replay output bytes equal, and nonempty generated IDs equal before any timed sample. Retain actual tokenizer count, prompt-ID SHA256, checkpoint path, byte length, content SHA, GGUF identity fields, host/OS, serving options, and nullable device description.
2. Call `time_gpu_ns(1)` on references to those retained handles for exactly 20 rounds numbered `0..19`. Record `rounds[i].arm_order=["legacy","shared_k"]` for even `i` and `["shared_k","legacy"]` for odd `i`; execute in that order. Each arm gets one `samples` entry `{round:i,position:0|1,gpu_ns}` per round, with `position` equal to its index in `arm_order`. Fail on any replay error or nonfinite/nonpositive time. Compute p50/p90/p99, min/max, population CoV, and 20 signed selected-minus-legacy deltas exactly as `SPEC.md` defines. Do not sample or subtract an empty command buffer.
3. Wrap five additional replays of each arm with existing `Cell::begin`/`end`. Store each complete resource line as process context. Serialize `version=1` and a one-element `shapes` array to the required `PROXIMA_GRANITE_AB_REPORT` path. The shape records `output_equal=true` and `ids_equal=true`; its two observed captured arm records carry `fault_binding_present=false`, exact output-byte count/hash and generated IDs, raw times, summaries, `timing_attempts=20`, `resource_replay_attempts=5`, `replay_errors=0`, and resource string. Print exactly two `granite ab arm` lines plus the report path.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC0 | `PROXIMA_TEST_TIMEOUT_MS=900000 PROXIMA_CAPTURE_LIVE=1 PROXIMA_CAPTURE_NODES=all PROXIMA_CAPTURE_STEPS=0 PROXIMA_GRANITE_AB_REPORT=/private/tmp/granite-attention-ab.json cargo nextest run -p proxima-model-interop --features std,metal,instrument,metal-attn-split-rows,metal-attn-variants --test granite_attention_variant_prefill -E 'test(~perf_card_00_granite_attention_replay_cell)' -j 1 --success-output immediate` | filter selects exactly 1 test and reports 1 passed; output prints exactly 2 `granite ab arm` lines; one JSON report has 1 tokenizer-verified prompt shape, 2 captured arms with distinct entry/SHA, 20 raw samples and 1 resource line per arm, 1 equal output/ID pair, 0 replay errors |

Read the raw report after AC0. Record both signs of round differences in TASKS.md; do not summarize the pair as a speed verdict.

## Residual

One prompt length supplies no length trend. Isolated replay includes command-buffer floor and does not measure a layer or public request. The `Cell` line is process context and may encode unavailable footprint as zero.
