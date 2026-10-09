# Card 19: simd-topology

**Owner:** GPT-6 Luna
**Dependency:** 12
**Commit:** `feat(omega): select attention simd topology`
**Budget:** at most 30 minutes front-to-back, from first source read through AC, review, and TASKS.md update. If the work cannot fit, split this card before implementation; do not ship a partial commit.

## Purpose

Select per-head or grouped-query SIMD lane topology.

## Scope

In scope: one behavior, the two acceptance tests (or two checker gates for card 00), and the TASKS.md row/resume update in the same coherent commit. Out of scope: CUDA, benchmarks, default activation, unrelated source and later cards. Keep the all-legacy baseline selectable.

## Read

- `omega/src/msl/signature_tokens_prelude.rs:2131-2153`
- `omega/src/msl/cached_attention_row_tiled.rs:142-150,235-260`
- `/Users/brianbruggeman/repos/others/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp:3515-3558`

## Edit

- `omega/src/msl/signature_tokens_prelude.rs`
- `omega/src/msl/cached_attention_row_tiled.rs`
- `proxima-tensor/specs/granite-attention-numeric-matrix/TASKS.md` (row and resume only)

## Steps

Add explicit lane-to-head map and width for simd_topology values; leave query_parallelism and tile_height fixed. Test exactly-once lane ownership in Granite GQA and invalid head_dim decline.

Use the existing Proxima numeric and Metal abstractions. Do not introduce num-traits, model-name special cases, or a silent fallback. Storage width, MMA operand width and f32 accumulator are different contracts. The two AC checks must cover a normal case and a refusing or degenerate case. Inspect assertions and emitted payloads, not merely the runner summary.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC19 | `cargo nextest run -p omega --features metal-attn-split-rows,metal-attn-variants --lib -E 'test(~card_19_simd_topology)'` | filter selects 2 tests; 2 passed |

A zero-test match or a checker that does not open its fixture is failure. Run this local command after the edit. The expected count is part of the criterion.

## Residual

The card establishes behavior and structural or numeric payload evidence only; timing, throughput and model-quality claims require separate measured cells.
