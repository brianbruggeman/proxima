# Card 21: cross-axis admission

**Owner:** GPT-6 Luna
**Dependency:** 06,09,11,13,14,15,16,17,18,19,20
**Commit:** `test(omega): verify attention variant composition`
**Budget:** at most 30 minutes front-to-back, from first source read through AC, review, and TASKS.md update. This card is limited to two dispatch cases plus exposing the computed memory requirement in the typed refusal.

## Purpose

Prove that supported attention axes compose independently and unsupported resource combinations decline with an explicit reason.

## Scope

In scope: one valid multi-axis dispatch, one over-budget dispatch decline, exact required/available bytes on that typed refusal, their assertions, and the TASKS.md row/resume update in the same coherent commit. Out of scope: changes to per-axis kernel behavior, benchmark execution, CUDA, default activation, and unrelated dirty files.

## Read

- `omega/src/msl/signature_tokens_prelude.rs:1630-1674,1798-1831`
- `omega/src/msl/signature_tokens_prelude.rs:2029-2092` for tile and staged-buffer byte formulas
- `omega/src/msl/cached_attention_row_tiled.rs:191-215,323-356`
- `omega/omega-runtime.toml:579` for the 32,768-byte threadgroup limit
- `omega/src/msl/attn_split_tests.rs:4-58`
- `proxima-tensor/specs/granite-attention-numeric-matrix/SPEC.md` cross-product rule

## Edit

- `omega/src/msl/attn_rows_tests.rs`
- `omega/src/msl/cached_attention_row_tiled.rs`
- `omega/src/error.rs`
- `proxima-tensor/specs/granite-attention-numeric-matrix/TASKS.md` (row and resume only)

## Steps

Using the public selector and manifest from card 20, use the captured Granite shape `query_rows=1000, kv_heads=8, query_groups=2, head_dim=64`. Select `kv_storage=bf16,mma_precision=f16,kv_reuse=shared_k,tile_height=rows_8,query_parallelism=simdgroup_rows,simd_topology=per_head,prefetch=off`; assert all seven selected values plus its emitted grid and source identity. Then use the same shape with `kv_storage=bf16,mma_precision=f32,kv_reuse=shared_kv,tile_height=rows_8,query_parallelism=simdgroup_rows,simd_topology=per_head,prefetch=next_block`; assert a typed `prefetch` threadgroup-budget decline with required 52,608 bytes and available 32,768 bytes. The current F16 version of this second selector is admitted: its source allocates a 16,384-byte next-block tile and the shared-K/V storage totals 5,120 bytes, so adding the 9,600-byte row/query base totals 31,104 bytes. The F32 decline is 9,600 row/query base + 8,192 shared K + 2,048 shared V + 32,768 next-block tile. Assert no legacy fallback in either case. Report the bytes from the same formula that controls admission.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC21 | `cargo nextest run -p omega --features metal-attn-split-rows,metal-attn-variants --lib -E 'test(~card_21_cross_axis)'` | filter selects 2 tests; 2 passed; selected multi-axis manifest, and typed F32 `prefetch` decline reports exactly 52,608 required / 32,768 available bytes |

A zero-test match is failure. Inspect the selected manifest and decline payload, not merely the runner summary.

## Residual

This card verifies configuration composition and admission evidence. Timing and numeric output comparisons belong to matched measured cells.
