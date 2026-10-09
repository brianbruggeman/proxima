# Card 03: bf8-element

**Owner:** GPT-6 Luna
**Dependency:** 01,02
**Commit:** `feat(tensor): evaluate bf8 typed buffers`
**Budget:** at most 30 minutes front-to-back, from first source read through AC, review, and TASKS.md update. If the work cannot fit, split this card before implementation; do not ship a partial commit.

## Purpose

Add BF8 to the CPU Element dispatch and one f32 widened reduction.

## Scope

In scope: one behavior, the two acceptance tests (or two checker gates for card 00), and the TASKS.md row/resume update in the same coherent commit. Out of scope: CUDA, benchmarks, default activation, unrelated source and later cards. Keep the all-legacy baseline selectable.

## Read

- `proxima-tensor/src/cpu/typed_eval.rs:48-83,322-366,368-417,595-679`

## Edit

- `proxima-tensor/src/cpu/typed_eval.rs`
- `proxima-tensor/specs/granite-attention-numeric-matrix/TASKS.md` (row and resume only)

## Steps

Add TypedBuffer::BFloat8 and Element that widens for each scalar op then narrows by contract. Add only the BFloat8/Float32 widened pair; no implicit BF8 accumulation. Test arithmetic result and a reduction whose f32 accumulator differs from BF8 per-step rounding.

Use the existing Proxima numeric and Metal abstractions. Do not introduce num-traits, model-name special cases, or a silent fallback. Storage width, MMA operand width and f32 accumulator are different contracts. The two AC checks must cover a normal case and a refusing or degenerate case. Inspect assertions and emitted payloads, not merely the runner summary.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC03 | `cargo nextest run -p proxima-tensor --lib -E 'test(~card_03_bf8_element)'` | filter selects 2 tests; 2 passed |

A zero-test match or a checker that does not open its fixture is failure. Run this local command after the edit. The expected count is part of the criterion.

## Residual

The card establishes behavior and structural or numeric payload evidence only; timing, throughput and model-quality claims require separate measured cells.
