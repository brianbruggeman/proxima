# Card 22: benchmark entrypoint selector

**Owner:** GPT-6 Luna
**Dependency:** 12,19,20,21
**Commit:** `feat(interop): select attention ab variant by config`
**Budget:** at most 30 minutes front-to-back, from first source read through AC, review, and TASKS.md update. Split the card before implementation if it cannot fit; do not commit a partial selector.

## Purpose

Make the existing `norm_variant_ab` Metal A/B entrypoint accept the seven-axis `AttentionVariant` record and emit the selected dispatch identity, so future measurements can label an actual kernel rather than an inferred configuration.

## Scope

In scope: one parser/forwarding path in the existing example and a describe-only mode that uses the same selector but performs no timed replay. Out of scope: a new benchmark framework, running measurements, CUDA, changing default selection, and editing unrelated dirty files. All-legacy remains selectable.

## Read

- `proxima-model-interop/examples/norm_variant_ab.rs:1-23,369-429` for the existing captured-dispatch A/B entrypoint and capture keys.
- `proxima-model-interop/Cargo.toml:624-626` for the existing example feature gate.
- `omega/src/msl/signature_tokens_prelude.rs:1630-1674,1798-1831` and card 20's dispatch manifest API.
- `proxima-tensor/specs/decode-prefill-parity/SPEC.md:4336-4344` for the saved benchmark invocation and evidence boundary.

## Edit

- `proxima-model-interop/examples/norm_variant_ab.rs`
- `proxima-model-interop/Cargo.toml` (feature pass-through only; no dependency changes)
- `proxima-tensor/specs/granite-attention-numeric-matrix/TASKS.md` (row and resume only)

## Steps

Add an off-by-default `metal-attn-variants` interop feature forwarding to omega's feature. Parse `AB_ATTENTION_VARIANT` as exactly seven named `key=value` fields with no duplicates and no omitted field; use the same typed `AttentionVariant` constructor as card 20. Pass that value into the existing attention capture/selection call before `decode(step)`. Print one `ab variant` record containing all seven values plus selected form, entry, MSL SHA, grid threads and threadgroup width. The record must come from the captured selected dispatch, not an echo of the requested string. `AB_VARIANT_DESCRIBE_ONLY=1` selects the fixed Granite fixture `[1000,8,2,64]` through the same dispatch selection API, prints one record and exits before model load or timing. Keep existing A/B replay behavior when the describe-only key is absent. Test parser and dispatch inspection for all-legacy and the valid multi-axis selection from card 21, plus a duplicate or missing-field refusal.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC22a | `cargo nextest run -p proxima-model-interop --features std,metal,instrument,metal-attn-split-rows,metal-attn-variants --all-targets -E 'test(~card_22_bench_entry)'` | filter selects 2 tests; 2 passed; one valid seven-field selector and one malformed/missing-field refusal |
| AC22b | `AB_ATTENTION_VARIANT='kv_storage=f32,mma_precision=legacy,kv_reuse=legacy,tile_height=legacy,query_parallelism=legacy,simd_topology=legacy,prefetch=off' AB_VARIANT_DESCRIBE_ONLY=1 cargo run -p proxima-model-interop --example norm_variant_ab --features std,metal,instrument,metal-attn-split-rows,metal-attn-variants --` | exactly 1 `ab variant` line with all 7 fields, selected entry/source hash/grid, and 0 `ab res` timed-cell lines |
| AC22c | `AB_ATTENTION_VARIANT='kv_storage=bf16,mma_precision=f16,kv_reuse=shared_k,tile_height=rows_8,query_parallelism=simdgroup_rows,simd_topology=per_head,prefetch=off' AB_VARIANT_DESCRIBE_ONLY=1 cargo run -p proxima-model-interop --example norm_variant_ab --features std,metal,instrument,metal-attn-split-rows,metal-attn-variants --` | exactly 1 `ab variant` line with all 7 selected values, non-legacy entry/source hash/grid distinct from AC22b, and 0 `ab res` timed-cell lines |

The parser tests must inspect the selected dispatch fields; a returned config object alone is insufficient. AC22b and AC22c are dry describe paths, not timing runs. All three commands assert their counts, including zero timed cells.

## Residual

Future benchmark results need real model payloads, matched all-off and one-factor cells, resource counters and per-output numeric comparisons. This card establishes the runnable selector and identity only.
