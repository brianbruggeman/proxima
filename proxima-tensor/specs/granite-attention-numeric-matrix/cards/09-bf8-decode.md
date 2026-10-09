# Card 09: bf8-decode

**Owner:** GPT-6 Luna
**Dependency:** 08
**Commit:** `feat(omega): read bf8 decode cache`
**Budget:** at most 30 minutes front-to-back, from first source read through AC, review, and TASKS.md update. If the work cannot fit, split this card before implementation; do not ship a partial commit.

## Purpose

Decode scalar E5M2 cached K/V in split attention.

## Scope

In scope: one behavior, the two acceptance tests (or two checker gates for card 00), and the TASKS.md row/resume update in the same coherent commit. Out of scope: CUDA, benchmarks, default activation, unrelated source and later cards. Keep the all-legacy baseline selectable.

## Read

- `omega/src/msl/cached_attention_decode_split.rs:54-104,130-168`
- `omega/src/msl/cached_attention_render.rs:61-103`

## Edit

- `omega/src/msl/cached_attention_decode_split.rs`
- `omega/src/msl/cached_attention_render.rs`
- `omega/src/msl/kernel_types_identity.rs`
- `omega/src/msl/emit_and_classify.rs`
- `omega/src/error.rs`
- `proxima-tensor/src/cpu/epilogue.rs` (accept scalar BF8 payload lengths)
- `omega/tests/cached_attention_decode_split_parity.rs`
- `proxima-tensor/specs/granite-attention-numeric-matrix/TASKS.md` (row and resume only)

## Steps

Admit only an all-BF8 cached triple. Decode bytes before f32 score and value math. Compare all 16 scalar encode/decode vectors with the golden values and inspect the emitted vector widening helper. For the accepted output fixture, use K/V values exactly representable in BF8 and compare output bytes against the same decode dispatch with those values in F32 storage and F32 MMA. Reject mixed cached codecs.

Use the existing Proxima numeric and Metal abstractions. Do not introduce num-traits, model-name special cases, or a silent fallback. Storage width, MMA operand width and f32 accumulator are different contracts. The two AC checks must cover a normal case and a refusing or degenerate case. Inspect assertions and emitted payloads, not merely the runner summary.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC09 | `cargo nextest run -p omega --features metal-attn-split-decode --test cached_attention_decode_split_parity -E 'test(~card_09_bf8_decode)'` | filter selects 2 tests; 2 passed; Metal output bits match F32-cache decode |

A zero-test match or a checker that does not open its fixture is failure. Run this local command after the edit. The expected count is part of the criterion.

## Residual

The card establishes behavior and structural or numeric payload evidence only; timing, throughput and model-quality claims require separate measured cells.
