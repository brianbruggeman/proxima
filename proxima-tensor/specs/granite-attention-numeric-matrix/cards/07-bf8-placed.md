# Card 07: bf8-placed

**Owner:** GPT-6 Luna
**Dependency:** 01
**Commit:** `feat(omega): add bf8 placed conversion`
**Budget:** at most 30 minutes front-to-back, from first source read through AC, review, and TASKS.md update. If the work cannot fit, split this card before implementation; do not ship a partial commit.

## Purpose

Expose one-byte BF8 placed-buffer seed/read/narrow functions.

## Scope

In scope: one behavior, the two acceptance tests (or two checker gates for card 00), and the TASKS.md row/resume update in the same coherent commit. Out of scope: CUDA, benchmarks, default activation, unrelated source and later cards. Keep the all-legacy baseline selectable.

## Read

- `omega/src/metal/execute_and_hazards.rs:1192-1240`
- `omega/src/lib.rs:92-93`

## Edit

- `omega/src/metal/execute_and_hazards.rs`
- `omega/src/lib.rs`
- `proxima-tensor/specs/granite-attention-numeric-matrix/TASKS.md` (row and resume only)

## Steps

Use the committed Proxima BF8 conversion. Test the 16 golden vectors after storage and adjacent sentinel bytes at a nonzero offset.

Use the existing Proxima numeric and Metal abstractions. Do not introduce num-traits, model-name special cases, or a silent fallback. Storage width, MMA operand width and f32 accumulator are different contracts. The two AC checks must cover a normal case and a refusing or degenerate case. Inspect assertions and emitted payloads, not merely the runner summary.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC07 | `cargo nextest run -p omega --lib -E 'test(~card_07_bf8_placed)'` | filter selects 2 tests; 2 passed |

A zero-test match or a checker that does not open its fixture is failure. Run this local command after the edit. The expected count is part of the criterion.

## Residual

The card establishes behavior and structural or numeric payload evidence only; timing, throughput and model-quality claims require separate measured cells.
