# Card 10: bf16-row

**Owner:** GPT-6 Luna
**Dependency:** 06
**Commit:** `feat(omega): read bf16 row cache`
**Budget:** at most 30 minutes front-to-back, from first source read through AC, review, and TASKS.md update. If the work cannot fit, split this card before implementation; do not ship a partial commit.

## Purpose

Read BF16 cache in the row tiled renderer.

## Scope

In scope: one behavior, the two acceptance tests (or two checker gates for card 00), and the TASKS.md row/resume update in the same coherent commit. Out of scope: CUDA, benchmarks, default activation, unrelated source and later cards. Keep the all-legacy baseline selectable.

## Read

- `omega/src/msl/cached_attention_row_tiled.rs:37-42,142-149,235-260,346-356`
- `omega/src/msl/cached_attention_render.rs:160-193`

## Edit

- `omega/src/msl/cached_attention_row_tiled.rs`
- `omega/src/msl/cached_attention_render.rs`
- `proxima-tensor/specs/granite-attention-numeric-matrix/TASKS.md` (row and resume only)

## Steps

Thread BF16 codec to renderer and widen cached K/V at loads. New-range buffers, softmax and output accumulator stay f32. With F32 MMA selected, use BF16-exact K/V fixture values and compare output bytes to the F32-storage row-tiled baseline at the same shape. Test an admitted Granite GQA shape and mixed-codec decline.

Use the existing Proxima numeric and Metal abstractions. Do not introduce num-traits, model-name special cases, or a silent fallback. Storage width, MMA operand width and f32 accumulator are different contracts. The two AC checks must cover a normal case and a refusing or degenerate case. Inspect assertions and emitted payloads, not merely the runner summary.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC10 | `cargo nextest run -p omega --features metal-attn-split-rows --test cached_attention_row_tiled_parity -E 'test(~card_10_bf16_row)'` | filter selects 2 tests; 2 passed; BF16-exact Metal attention output bits match F32-cache output, the plans have distinct kernel keys, and mixed cache codecs decline |

A zero-test match or a checker that does not open its fixture is failure. Run this local command after the edit. The expected count is part of the criterion.

## Residual

The card establishes behavior and structural or numeric payload evidence only; timing, throughput and model-quality claims require separate measured cells.
