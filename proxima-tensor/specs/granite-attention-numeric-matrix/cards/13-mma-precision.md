# Card 13: MMA operand precision

**Owner:** GPT-6 Luna
**Dependency:** 10,11,12
**Commit:** `feat(omega): select attention mma precision`
**Budget:** at most 30 minutes front-to-back, from first source read through AC, review, and TASKS.md update. This card changes only the row-tiled MMA operand choice; split decode arithmetic and storage formats remain as implemented.

## Purpose

Make F32 versus F16 MMA operand precision selectable per dispatch, independently of K/V storage.

## Scope

In scope: one operand-precision selector through row-tiled binding/rendering, assertions for operand/storage/accumulator separation, and the TASKS.md row/resume update in the same coherent commit. Out of scope: changing cache byte formats, softmax precision, accumulator precision, other dispatch axes, benchmarks, and default selection.

## Read

- `omega/src/msl/cached_attention_row_tiled.rs:37-64,101-115,191-260,315-356`
- `omega/src/msl/signature_tokens_prelude.rs:1943-1998`
- `omega/src/sized.rs:79-84` and `omega/omega-runtime.toml:549-557` for the existing sized MMA default

## Edit

- `omega/src/msl/cached_attention_row_tiled.rs`
- `omega/src/msl/cached_attention_render.rs`
- `omega/src/msl/signature_tokens_prelude.rs`
- `proxima-tensor/specs/granite-attention-numeric-matrix/TASKS.md` (row and resume only)

## Steps

Route `mma_precision=legacy|f32|f16` to the row-tiled renderer at dispatch time. `legacy` reads the existing sized default; explicit F16 narrows loaded Q/K/V operands to `simdgroup_half8x8`, and explicit F32 keeps `simdgroup_float8x8`. Both use F32 accumulators and F32 softmax. K/V storage remains independent: decode F32, BF16, or BF8 bytes to F32, then convert to the selected MMA operand at the load boundary. Reject explicit F16 MMA on forms without the row-tiled MMA implementation. Test both explicit operand tokens for each admitted cache codec and assert the output accumulator/softmax remain F32. With an F16-exact fixture, compare F32-MMA and F16-MMA output bytes; retain the sized default as the legacy selection.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC13 | `cargo nextest run -p omega --features metal-attn-split-rows,metal-attn-variants --lib -E 'test(~card_13_mma_precision)'` | filter selects 2 tests; 2 passed; F32/F16 operand choice flips independently of F32/BF16/BF8 cache storage, output bytes agree on the F16-exact fixture, and accumulation/softmax stay F32 |

A zero-test match is failure. Inspect rendered MSL for each selected operand and accumulator type.

## Residual

This proves selected operand and storage paths structurally. Numerical-quality and performance comparisons require matched execution cells.
