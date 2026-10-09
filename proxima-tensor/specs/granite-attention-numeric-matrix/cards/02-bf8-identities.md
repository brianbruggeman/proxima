# Card 02: bf8-identities

**Owner:** GPT-6 Luna
**Dependency:** 01
**Commit:** `feat(primitives): identify bf8 storage codec`
**Budget:** at most 30 minutes front-to-back, from first source read through AC, review, and TASKS.md update. If the work cannot fit, split this card before implementation; do not ship a partial commit.

## Purpose

Add BF8 scalar and codec identities without renumbering wire tags.

## Scope

In scope: one behavior, the two acceptance tests (or two checker gates for card 00), and the TASKS.md row/resume update in the same coherent commit. Out of scope: CUDA, benchmarks, default activation, unrelated source and later cards. Keep the all-legacy baseline selectable.

## Read

- `proxima-tensor/src/dtype.rs:14-52,85-105`
- `proxima-primitives/src/codec.rs:13-45,47-103`

## Edit

- `proxima-tensor/src/dtype.rs`
- `proxima-primitives/src/codec.rs`
- `proxima-tensor/specs/granite-attention-numeric-matrix/TASKS.md` (row and resume only)

## Steps

Add DType::BFloat8 as one-byte float with explicit non-self-accumulating status. Append Codec::BFloat8 at unused tag 29; retain tags 0..28. Test all 30 round trips and legacy tag stability; reject unknown tag 30.

Use the existing Proxima numeric and Metal abstractions. Do not introduce num-traits, model-name special cases, or a silent fallback. Storage width, MMA operand width and f32 accumulator are different contracts. The two AC checks must cover a normal case and a refusing or degenerate case. Inspect assertions and emitted payloads, not merely the runner summary.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC02a | `cargo nextest run -p proxima-primitives --lib -E 'test(~card_02_bf8_codec)'` | filter selects 2 tests; 2 passed; checks codec tags and unknown-tag refusal |
| AC02b | `cargo nextest run -p proxima-tensor --lib -E 'test(~card_02_bf8_dtype)'` | filter selects 2 tests; 2 passed; checks BF8 width/float identity and existing DType identities |

A zero-test match or a checker that does not open its fixture is failure. Run this local command after the edit. The expected count is part of the criterion.

## Residual

The card establishes behavior and structural or numeric payload evidence only; timing, throughput and model-quality claims require separate measured cells.
