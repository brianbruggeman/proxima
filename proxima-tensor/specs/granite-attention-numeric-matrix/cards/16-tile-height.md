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
- `omega/src/msl/emit_and_classify.rs` (variant-aware identity and launch grid)
- `omega/src/msl/cached_attention_render.rs` (selected form reaches the renderer)
- `omega/src/metal/device_buffers_arena_plan.rs` and `omega/src/metal/arena_encode_dispatch_finish.rs` (plan key, invalidation and resolved launch shape)
- `omega/omega-runtime.toml:518-539,579`
- `proxima-tensor/specs/decode-prefill-parity/evidence/attn5/raw/probe/granite.out:12` for captured extents `[1000,8,2,64]`

## Edit

- `omega/src/msl/signature_tokens_prelude.rs`
- `omega/src/msl/emit_and_classify.rs`
- `omega/src/msl/cached_attention_render.rs`
- `omega/src/msl/cached_attention_row_tiled.rs`
- `omega/src/msl/attn_rows_tests.rs`
- `omega/tests/cached_attention_row_tiled_parity.rs` (plan variant and executed payload)
- `omega/src/error.rs`
- `omega/src/metal/{arena_encode_dispatch_finish,device_buffers_arena_plan,dispatch_timed_and_classify,execute_and_hazards,mod,pipeline_buffers_upload}.rs`
- `proxima-tensor/specs/granite-attention-numeric-matrix/TASKS.md` (row and resume only)

## Steps

Route tile_height through variant identity, plan invalidation, resolved dispatch geometry, and existing tile-unit, register and threadgroup-memory admission. An explicit height must be exactly representable as whole tile units or decline; it must never fall through to Legacy. For the captured Granite shape `[1000,8,2,64]`, two query groups require an eight-row tile unit: assert `rows_16` changes both the kernel identity and grid, and explicit `rows_4` returns the whole-unit shape decline. For `[16,1,8,64]`, `rows_16` requires exactly 35,840 score/state bytes against the 32,768-byte threadgroup limit; assert the typed budget decline.

Use the existing Proxima numeric and Metal abstractions. Do not introduce num-traits, model-name special cases, or a silent fallback. Storage width, MMA operand width and f32 accumulator are different contracts. The two AC checks must cover a normal case and a refusing or degenerate case. Inspect assertions and emitted payloads, not merely the runner summary.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC16 | `cargo nextest run -p omega --features metal-attn-split-rows,metal-attn-variants --lib -E 'test(~card_16_tile_height)'` | filter selects 2 tests; 2 passed; captured Granite rows_16 changes kernel identity and grid, rows_4 returns the exact whole-unit decline, and `[16,1,8,64]` returns the typed 35,840-vs-32,768-byte budget decline |
| AC16b | `cargo nextest run -p omega --features metal,metal-attn-split-rows,metal-attn-variants --test cached_attention_row_tiled_parity -E 'test(~card_16_tile_height_plan)'` | filter selects 1 Metal test; 1 passed; Plan kernel identity contains the selected tile token and executed rows16 output is compared against the CPU payload |

A zero-test match or a checker that does not open its fixture is failure. Run this local command after the edit. The expected count is part of the criterion.

## Residual

The card establishes behavior and structural or numeric payload evidence only; timing, throughput and model-quality claims require separate measured cells.
