# Card 14: k-reuse

**Owner:** GPT-6 Luna
**Dependency:** 12,13
**Commit:** `feat(omega): share cached keys across queries`
**Budget:** at most 30 minutes front-to-back, from first source read through AC, review, and TASKS.md update. If the work cannot fit, split this card before implementation; do not ship a partial commit.

## Purpose

Implement the K-only shared reuse mode as an independently selectable intermediate.

## Scope

In scope: one behavior, the two acceptance tests (or two checker gates for card 00), and the TASKS.md row/resume update in the same coherent commit. Out of scope: CUDA, benchmarks, default activation, unrelated source and later cards. Keep the all-legacy baseline selectable.

## Read

- `omega/src/msl/cached_attention_row_tiled.rs:4-16,235-260`
- `/Users/brianbruggeman/repos/others/llama.cpp/ggml/src/ggml-metal/kernels/fa_common.metal:300-338`

## Edit

- `omega/src/msl/cached_attention_row_tiled.rs`
- `proxima-tensor/specs/granite-attention-numeric-matrix/TASKS.md` (row and resume only)

## Steps

The current row renderer already reuses a K fragment across query vectors inside one simdgroup (`cached_attention_row_tiled.rs:235-260`). Implement `kv_reuse=shared_k`: decode each cached K block, convert it to the selected MMA operand representation, stage it once in threadgroup memory for all simdgroups of a row tile, then consume it under a barrier; V remains on the legacy path. `kv_reuse=shared_kv` is reserved for card 15. Charge the actual two- or four-byte staged operand width to the existing threadgroup budget. Assert one staged K fetch per block, unchanged per-row causal masks, distinct emitted source/entry, and a budget-refusal control.

Use the existing Proxima numeric and Metal abstractions. Do not introduce num-traits, model-name special cases, or a silent fallback. Storage width, MMA operand width and f32 accumulator are different contracts. The two AC checks must cover a normal case and a refusing or degenerate case. Inspect assertions and emitted payloads, not merely the runner summary.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC14 | `cargo nextest run -p omega --features metal-attn-split-rows,metal-attn-variants --lib -E 'test(~card_14_k_reuse)'` | filter selects 2 tests; 2 passed |

A zero-test match or a checker that does not open its fixture is failure. Run this local command after the edit. The expected count is part of the criterion.

## Residual

The card establishes behavior and structural or numeric payload evidence only; timing, throughput and model-quality claims require separate measured cells.
