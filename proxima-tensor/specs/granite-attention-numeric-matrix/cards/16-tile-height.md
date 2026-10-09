# Card 16: tile-height

**Owner:** GPT-6 Luna
**Dependency:** 12
**Commit:** `feat(omega): select attention tile height`
**Budget:** at most 30 minutes front-to-back, from first source read through AC, review, and TASKS.md update. If the work cannot fit, split this card before implementation; do not ship a partial commit.

## Purpose

Select explicit row tile height independently of other axes.

## Scope

In scope: one behavior, the two acceptance tests (or two checker gates for card 00), and the TASKS.md row/resume update in the same coherent commit. Out of scope: CUDA, benchmarks, default activation, unrelated source and later cards. Keep the all-legacy baseline selectable.

## Read

- `omega/src/msl/signature_tokens_prelude.rs:1957-1998,2029-2129`
- `omega/omega-runtime.toml:518-539,579`
- `proxima-tensor/specs/decode-prefill-parity/evidence/attn5/raw/probe/granite.out:12` for captured extents `[1000,8,2,64]`

## Edit

- `omega/src/msl/signature_tokens_prelude.rs`
- `proxima-tensor/specs/granite-attention-numeric-matrix/TASKS.md` (row and resume only)

## Steps

Route tile_height through existing tile-unit, register and threadgroup-memory admission. An explicit height must be exactly representable as whole tile units or decline. For the captured Granite shape `[1000,8,2,64]`, two query groups require an eight-row tile unit: assert `rows_16` changes the grid, and explicit `rows_4` returns a shape-specific decline. For `[16,1,8,64]`, `rows_16` requires 35,840 score/state bytes against the 32,768-byte threadgroup limit; assert the exact budget decline.

Use the existing Proxima numeric and Metal abstractions. Do not introduce num-traits, model-name special cases, or a silent fallback. Storage width, MMA operand width and f32 accumulator are different contracts. The two AC checks must cover a normal case and a refusing or degenerate case. Inspect assertions and emitted payloads, not merely the runner summary.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC16 | `cargo nextest run -p omega --features metal-attn-split-rows,metal-attn-variants --lib -E 'test(~card_16_tile_height)'` | filter selects 2 tests; 2 passed; captured Granite rows_16 admitted, rows_4 shape decline, and `[16,1,8,64]` 35,840-vs-32,768-byte budget decline |

A zero-test match or a checker that does not open its fixture is failure. Run this local command after the edit. The expected count is part of the criterion.

## Residual

The card establishes behavior and structural or numeric payload evidence only; timing, throughput and model-quality claims require separate measured cells.
