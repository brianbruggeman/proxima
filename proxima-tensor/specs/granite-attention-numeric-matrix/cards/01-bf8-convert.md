# Card 01: bf8-convert

**Owner:** GPT-6 Luna
**Dependency:** 00
**Commit:** `feat(tensor): add scalar bf8 conversion`
**Budget:** at most 30 minutes front-to-back, from first source read through AC, review, and TASKS.md update. If the work cannot fit, split this card before implementation; do not ship a partial commit.

## Purpose

Add a repr-transparent BFloat8 scalar and two Pipe conversions.

## Scope

In scope: one behavior, the two acceptance tests (or two checker gates for card 00), and the TASKS.md row/resume update in the same coherent commit. Out of scope: CUDA, benchmarks, default activation, unrelated source and later cards. Keep the all-legacy baseline selectable.

## Read

- `proxima-tensor/src/convert.rs:1-12,196-212,461-464`
- `proxima-tensor/specs/granite-attention-numeric-matrix/BF8_CONTRACT.md`
- `proxima-tensor/specs/granite-attention-numeric-matrix/bf8_vectors.csv`

## Edit

- `proxima-tensor/src/bfloat8.rs`
- `proxima-tensor/src/lib.rs`
- `proxima-tensor/src/convert.rs`
- `proxima-tensor/specs/granite-attention-numeric-matrix/TASKS.md` (row and resume only)

## Steps

Use from_bits/to_bits and integer rounding, not num-traits. Add Convert<f32,BFloat8> and inverse beside BF16. Tests cover all 256 byte decodes and 16 golden conversion vectors, including tie and NaN normalization.

Use the existing Proxima numeric and Metal abstractions. Do not introduce num-traits, model-name special cases, or a silent fallback. Storage width, MMA operand width and f32 accumulator are different contracts. The two AC checks must cover a normal case and a refusing or degenerate case. Inspect assertions and emitted payloads, not merely the runner summary.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC01 | `cargo nextest run -p proxima-tensor --lib -E 'test(~card_01_bf8_convert)'` | filter selects 2 tests; 2 passed |

A zero-test match or a checker that does not open its fixture is failure. Run this local command after the edit. The expected count is part of the criterion.

## Residual

The card establishes behavior and structural or numeric payload evidence only; timing, throughput and model-quality claims require separate measured cells.
