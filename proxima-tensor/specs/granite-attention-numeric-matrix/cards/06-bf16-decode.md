# Card 06: bf16-decode

**Owner:** GPT-6 Luna
**Dependency:** 05
**Commit:** `feat(omega): read bf16 decode cache`
**Budget:** at most 30 minutes front-to-back, from first source read through AC, review, and TASKS.md update. If the work cannot fit, split this card before implementation; do not ship a partial commit.

## Purpose

Read cached BF16 K/V in decode split with f32 scores.

## Scope

In scope: one behavior, the two acceptance tests (or two checker gates for card 00), and the TASKS.md row/resume update in the same coherent commit. Out of scope: CUDA, benchmarks, default activation, unrelated source and later cards. Keep the all-legacy baseline selectable.

## Read

- `omega/src/msl/cached_attention_render.rs:61-103,160-193`
- `omega/src/msl/cached_attention_decode_split.rs:54-104,130-168`

## Edit

- `omega/src/msl/cached_attention_render.rs`
- `omega/src/msl/cached_attention_decode_split.rs`
- `omega/tests/cached_attention_decode_split_parity.rs` (two named acceptance tests)
- `proxima-tensor/specs/granite-attention-numeric-matrix/TASKS.md` (row and resume only)

## Steps

Admit only all-BF16 cached operands in decode split. Widen each BF16 lane before Q.K and weighted V; leave query, softmax and accumulator f32. For the accepted output fixture, use K/V values exactly representable in BF16 and compare output bytes against the same decode dispatch with those values in F32 storage and F32 MMA. Also assert source selection and mixed-codec decline.

Use the existing Proxima numeric and Metal abstractions. Do not introduce num-traits, model-name special cases, or a silent fallback. Storage width, MMA operand width and f32 accumulator are different contracts. The two AC checks must cover a normal case and a refusing or degenerate case. Inspect assertions and emitted payloads, not merely the runner summary.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC06 | `cargo nextest run -p omega --features metal-attn-split-decode --test cached_attention_decode_split_parity -E 'test(~card_06_bf16_decode)'` | filter selects 2 tests; 2 passed; the Metal output bits match the same F32-cache decode dispatch |

A zero-test match or a checker that does not open its fixture is failure. Run this local command after the edit. The expected count is part of the criterion.

## Residual

The card establishes behavior and structural or numeric payload evidence only; timing, throughput and model-quality claims require separate measured cells.
