# Card 21: cross-axis admission

**Owner:** GPT-6 Luna
**Dependency:** 06,09,11,13,14,15,16,17,18,19,20
**Commit:** `test(omega): verify attention variant composition`
**Budget:** at most 30 minutes front-to-back, from first source read through AC, review, and TASKS.md update. This card is limited to two dispatch cases; split it if either case requires changing an axis implementation.

## Purpose

Prove that supported attention axes compose independently and unsupported resource combinations decline with an explicit reason.

## Scope

In scope: one valid multi-axis dispatch, one over-budget dispatch decline, their assertions, and the TASKS.md row/resume update in the same coherent commit. Out of scope: changes to per-axis kernel behavior, benchmark execution, CUDA, default activation, and unrelated dirty files.

## Read

- `omega/src/msl/signature_tokens_prelude.rs:1630-1674,1798-1831`
- `omega/src/msl/signature_tokens_prelude.rs:2029-2092` for tile and staged-buffer byte formulas
- `omega/src/msl/cached_attention_row_tiled.rs:191-215,323-356`
- `omega/omega-runtime.toml:579` for the 32,768-byte threadgroup limit
- `omega/src/msl/attn_split_tests.rs:4-58`
- `proxima-tensor/specs/granite-attention-numeric-matrix/SPEC.md` cross-product rule

## Edit

- `omega/src/msl/attn_split_tests.rs`
- `proxima-tensor/specs/granite-attention-numeric-matrix/TASKS.md` (row and resume only)

## Steps

Using the public selector and manifest from card 20, use the captured Granite shape `query_rows=1000, kv_heads=8, query_groups=2, head_dim=64`. Select `kv_storage=bf16,mma_precision=f16,kv_reuse=shared_k,tile_height=rows_8,query_parallelism=simdgroup_rows,simd_topology=per_head,prefetch=off`; assert all seven selected values plus its emitted grid and source identity. Then use the same shape with `kv_reuse=shared_kv,prefetch=next_block` and the remaining axes unchanged; assert a typed `prefetch` threadgroup-budget decline with required 42,368 bytes and available 32,768 bytes. The required value is derived as 4,480 row score/state bytes + 5,120 staged query bytes + 32,768 bytes for double-buffered BF16 K/V blocks. Assert no legacy fallback in either case. Do not alter kernel selection or resource formulas here.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC21 | `cargo nextest run -p omega --features metal-attn-split-rows,metal-attn-variants --lib -E 'test(~card_21_cross_axis)'` | filter selects 2 tests; 2 passed; one supported multi-axis manifest and one `prefetch` resource decline |

A zero-test match is failure. Inspect the selected manifest and decline payload, not merely the runner summary.

## Residual

This card verifies configuration composition and admission evidence. Timing and numeric output comparisons belong to matched measured cells.
