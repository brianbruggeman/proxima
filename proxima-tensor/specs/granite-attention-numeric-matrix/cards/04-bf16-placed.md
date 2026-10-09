# Card 04: bf16-placed

**Owner:** GPT-6 Luna
**Dependency:** none
**Commit:** `feat(omega): add bf16 placed conversion`
**Budget:** at most 30 minutes front-to-back, from first source read through AC, review, and TASKS.md update. If the work cannot fit, split this card before implementation; do not ship a partial commit.

## Purpose

Expose placed-buffer BF16 seed/read/narrow functions.

## Scope

In scope: one behavior, the two acceptance tests (or two checker gates for card 00), and the TASKS.md row/resume update in the same coherent commit. Out of scope: CUDA, benchmarks, default activation, unrelated source and later cards. Keep the all-legacy baseline selectable.

## Read

- `omega/src/metal/execute_and_hazards.rs:1192-1240`
- `omega/src/lib.rs:92-93`

## Edit

- `omega/src/metal/execute_and_hazards.rs` (helpers and the two named acceptance tests)
- `omega/src/lib.rs`
- `proxima-tensor/specs/granite-attention-numeric-matrix/TASKS.md` (row and resume only)

## Steps

Mirror existing F16 helpers using half::bf16. Keep byte offsets explicit. Test signed zero, finite, inf/NaN and a nonzero offset with adjacent sentinel untouched.

Use the existing Proxima numeric and Metal abstractions. Do not introduce num-traits, model-name special cases, or a silent fallback. Storage width, MMA operand width and f32 accumulator are different contracts. The two AC checks must cover a normal case and a refusing or degenerate case. Inspect assertions and emitted payloads, not merely the runner summary.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC04 | `cargo nextest run -p omega --lib -E 'test(~card_04_bf16_placed)'` | filter selects 2 tests; 2 passed |

A zero-test match or a checker that does not open its fixture is failure. Run this local command after the edit. The expected count is part of the criterion.

## Residual

The card establishes behavior and structural or numeric payload evidence only; timing, throughput and model-quality claims require separate measured cells.
