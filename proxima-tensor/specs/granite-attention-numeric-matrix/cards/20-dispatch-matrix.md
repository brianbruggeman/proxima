# Card 20: dispatch-matrix

**Owner:** GPT-6 Luna
**Dependency:** 06,09,11,13,14,15,16,17,18,19
**Commit:** `test(omega): expose attention dispatch matrix`
**Budget:** at most 30 minutes front-to-back, from first source read through AC, review, and TASKS.md update. If the work cannot fit, split this card before implementation; do not ship a partial commit.

## Purpose

Expose selected dispatch identity and verify the seven independent one-factor flips.

## Scope

In scope: one behavior, the two acceptance tests (or two checker gates for card 00), and the TASKS.md row/resume update in the same coherent commit. Out of scope: CUDA, benchmarks, default activation, unrelated source and later cards. Keep the all-legacy baseline selectable.

## Read

- `omega/src/msl/signature_tokens_prelude.rs:1630-1674,1798-1831`
- `omega/src/msl/attn_split_tests.rs:4-58`
- `omega/src/metal/arena_encode_dispatch_finish.rs:1181-1197`
- `proxima-tensor/specs/decode-prefill-parity/evidence/attn5/raw/probe/granite.out:12` for the captured Granite shape

## Edit

- `omega/src/msl/signature_tokens_prelude.rs`
- `omega/src/msl/attn_rows_tests.rs`
- `omega/src/msl/emit_and_classify.rs`
- `omega/src/msl/mod.rs`
- `omega/src/lib.rs`
- `proxima-tensor/specs/granite-attention-numeric-matrix/TASKS.md` (row and resume only)

## Steps

Expose the typed selector/manifest to the existing dispatch inspection path. Emit form, cache codec, K/V storage, MMA operand, f32 accumulator, reuse mode, tile rows, query schedule, SIMD topology, prefetch, entry, grid and dispatch identity. At the captured `[1000,8,2,64]` shape, compare all-legacy source and grid byte-for-byte with the same build's pre-variant dispatch. Then use the admitted F16/shared-KV/rows-8/simdgroup-rows control and make seven one-factor flips: each selected field changes while the other six stay fixed. This control allows the standalone prefetch flip within the existing threadgroup budget. Keep this card to manifest construction and inspection; cross-axis resource admission is card 21. Include a deliberately false selector control and assert that the manifest comparison rejects it.

Use the existing Proxima numeric and Metal abstractions. Do not introduce num-traits, model-name special cases, or a silent fallback. Storage width, MMA operand width and f32 accumulator are different contracts. The two AC checks must cover a normal case and a refusing or degenerate case. Inspect assertions and emitted payloads, not merely the runner summary.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC20 | `cargo nextest run -p omega --features metal-attn-split-rows,metal-attn-variants --lib -E 'test(~card_20_dispatch_matrix)'` | filter selects 2 tests; 2 passed; legacy dispatch matches base emitter; manifest exposes all axes, form, codec, F32 accumulator, entry, grid and identity; seven flips from admitted control change one typed axis each; false selector is rejected and cache mismatch refuses |

A zero-test match or a checker that does not open its fixture is failure. Run this local command after the edit. The expected count is part of the criterion.

## Residual

The card establishes behavior and structural or numeric payload evidence only; timing, throughput and model-quality claims require separate measured cells.
