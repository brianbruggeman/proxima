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

- `omega/src/msl/cached_attention_row_tiled.rs`
- `proxima-tensor/specs/granite-attention-numeric-matrix/TASKS.md` (row and resume only)

## Steps

Implement `kv_reuse=shared_kv` by decoding each cached V block, converting it to the selected MMA operand representation, and staging it once in threadgroup memory for all simdgroups of the row tile, together with the K staging from card 14. Each query row keeps independent softmax weights and f32 accumulators. Charge K plus V staged bytes together at the selected two- or four-byte operand width, with typed budget decline. Assert one staged V fetch per block and different weighted outputs for two query rows; `legacy` and `shared_k` are controls.

Use the existing Proxima numeric and Metal abstractions. Do not introduce num-traits, model-name special cases, or a silent fallback. Storage width, MMA operand width and f32 accumulator are different contracts. The two AC checks must cover a normal case and a refusing or degenerate case. Inspect assertions and emitted payloads, not merely the runner summary.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC15 | `cargo nextest run -p omega --features metal-attn-split-rows,metal-attn-variants --lib -E 'test(~card_15_v_reuse)'` | filter selects 2 tests; 2 passed |

A zero-test match or a checker that does not open its fixture is failure. Run this local command after the edit. The expected count is part of the criterion.

## Residual

The card establishes behavior and structural or numeric payload evidence only; timing, throughput and model-quality claims require separate measured cells.
