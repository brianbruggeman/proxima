# Card 15: v-reuse

**Owner:** GPT-6 Luna
**Dependency:** 14
**Commit:** `feat(omega): share cached values across queries`
**Budget:** at most 30 minutes front-to-back, from first source read through AC, review, and TASKS.md update. If the work cannot fit, split this card before implementation; do not ship a partial commit.

## Purpose

Share cached V fragment without sharing softmax weights.

## Scope

In scope: one behavior, the two acceptance tests (or two checker gates for card 00), and the TASKS.md row/resume update in the same coherent commit. Out of scope: CUDA, benchmarks, default activation, unrelated source and later cards. Keep the all-legacy baseline selectable.

## Read

- `omega/src/msl/cached_attention_row_tiled.rs:315-356`

## Edit

- `omega/src/msl/signature_tokens_prelude.rs` (selectable `SharedKv` dispatch identity)
- `omega/src/msl/cached_attention_render.rs` (row-tiled admission)
- `omega/src/msl/cached_attention_row_tiled.rs` (bounded V slabs, query-owned P·V, F32/F16 operands, resource admission)
- `omega/src/msl/attn_rows_tests.rs` (source and resource acceptance)
- `omega/tests/cached_attention_row_tiled_parity.rs` (Metal execution against the CPU reference and padded-row control)
- `proxima-tensor/specs/granite-attention-numeric-matrix/TASKS.md` (row and resume only)

## Steps

Implement `kv_reuse=shared_kv` by decoding each cached V fragment, converting it to the selected MMA operand representation, and staging bounded slabs in threadgroup memory for all simdgroups of the row tile, together with the K staging from card 14. Give each simdgroup disjoint query blocks and all output-dimension fragments; index persistent F32 accumulators by `(dimension_block, local_query_block)`. Each query row keeps independent softmax weights and state. Charge K plus the selected V slab bytes together at the selected two- or four-byte operand width; typed-decline when the combined threadgroup budget or accumulator-fragment budget is exceeded. Assert unique V producers, cross-simdgroup consumers using their own row weights, distinct output rows against the CPU reference, and invariance when the padded final V row is changed to `+Inf`; `legacy` and `shared_k` are controls. The slab keeps threadgroup staging bounded; this card makes no timing claim.

Use the existing Proxima numeric and Metal abstractions. Do not introduce num-traits, model-name special cases, or a silent fallback. Storage width, MMA operand width and f32 accumulator are different contracts. The two AC checks must cover a normal case and a refusing or degenerate case. Inspect assertions and emitted payloads, not merely the runner summary.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC15a | `cargo nextest run -p omega --features metal-attn-split-rows,metal-attn-variants --lib -E 'test(~card_15_v_reuse)'` | filter selects 2 tests; 2 passed; F32/F16 and BF16/BF8 source paths are inspected, and the accumulator-budget decline is typed |
| AC15b | `cargo nextest run -p omega --features metal,metal-attn-split-rows,metal-attn-variants --test cached_attention_row_tiled_parity -E 'test(~card_15_v_reuse_device)'` | filter selects 1 Metal test; 1 passed; Legacy, SharedK, and SharedKv outputs agree with the CPU reference, two rows differ, and padded-V `+Inf` leaves output bits unchanged |

A zero-test match or a checker that does not open its fixture is failure. Run this local command after the edit. The expected count is part of the criterion.

## Residual

The card establishes behavior and structural or numeric payload evidence only; timing, throughput and model-quality claims require separate measured cells.
