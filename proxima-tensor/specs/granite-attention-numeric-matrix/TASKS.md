# granite-attention-numeric-matrix -- slices

Each card is one coherent commit, at most 30 minutes front-to-back. Update the row and resume lines in that same commit.

| # | card | dependency | discharges | validation command | expected | done | note |
|---|---|---|---|---|---|---|---|
| 00 | [card](cards/00-bf8-contract.md) | none | R1 | `python3 proxima-tensor/specs/granite-attention-numeric-matrix/check_bf8_contract.py` | checks=2 vectors=16 | [x] | `checks=2 vectors=16` |
| 01 | [card](cards/01-bf8-convert.md) | 00 | R1,R2 | `cargo nextest run -p proxima-tensor --lib -E 'test(~card_01_bf8_convert)'` | 2 tests run: 2 passed, 0 skipped | [ ] | |
| 02 | [card](cards/02-bf8-identities.md) | 01 | R2 | `cargo nextest run -p proxima-primitives --lib -E 'test(~card_02_bf8_codec)' && cargo nextest run -p proxima-tensor --lib -E 'test(~card_02_bf8_dtype)'` | 4 tests total: each command runs 2 passed, 0 skipped | [ ] | |
| 03 | [card](cards/03-bf8-element.md) | 01,02 | R2 | `cargo nextest run -p proxima-tensor --lib -E 'test(~card_03_bf8_element)'` | 2 tests run: 2 passed, 0 skipped | [ ] | |
| 04 | [card](cards/04-bf16-placed.md) | none | R3 | `cargo nextest run -p omega --lib -E 'test(~card_04_bf16_placed)'` | 2 tests run: 2 passed, 0 skipped | [ ] | |
| 05 | [card](cards/05-bf16-device-kv.md) | 04 | R3 | `cargo nextest run -p proxima-model-interop --features std,metal,metal-attn-split-rows --lib -E 'test(~card_05_bf16_device)'` | 2 tests run: 2 passed, 0 skipped | [ ] | |
| 06 | [card](cards/06-bf16-decode.md) | 05 | R4 | `cargo nextest run -p omega --features metal-attn-split-decode --lib -E 'test(~card_06_bf16_decode)'` | 2 tests run: 2 passed, 0 skipped | [ ] | |
| 07 | [card](cards/07-bf8-placed.md) | 01 | R3 | `cargo nextest run -p omega --lib -E 'test(~card_07_bf8_placed)'` | 2 tests run: 2 passed, 0 skipped | [ ] | |
| 08 | [card](cards/08-bf8-device-kv.md) | 02,07 | R3 | `cargo nextest run -p proxima-model-interop --features std,metal,metal-attn-split-rows --lib -E 'test(~card_08_bf8_device)'` | 2 tests run: 2 passed, 0 skipped | [ ] | |
| 09 | [card](cards/09-bf8-decode.md) | 08 | R4 | `cargo nextest run -p omega --features metal-attn-split-decode --lib -E 'test(~card_09_bf8_decode)'` | 2 tests run: 2 passed, 0 skipped | [ ] | |
| 10 | [card](cards/10-bf16-row.md) | 06 | R4 | `cargo nextest run -p omega --features metal-attn-split-rows --lib -E 'test(~card_10_bf16_row)'` | 2 tests run: 2 passed, 0 skipped | [ ] | |
| 11 | [card](cards/11-bf8-row.md) | 09,10 | R4 | `cargo nextest run -p omega --features metal-attn-split-rows --lib -E 'test(~card_11_bf8_row)'` | 2 tests run: 2 passed, 0 skipped | [ ] | |
| 12 | [card](cards/12-variant-config.md) | none | R5,R6 | `cargo nextest run -p omega --features metal-attn-split-rows,metal-attn-variants --lib -E 'test(~card_12_variant_config)'` | 2 tests run: 2 passed, 0 skipped; seven fields and explicit defaults | [ ] | |
| 13 | [card](cards/13-mma-precision.md) | 10,11,12 | R5,R6 | `cargo nextest run -p omega --features metal-attn-split-rows,metal-attn-variants --lib -E 'test(~card_13_mma_precision)'` | 2 tests run: 2 passed, 0 skipped; operand choice independent of storage | [ ] | |
| 14 | [card](cards/14-k-reuse.md) | 12,13 | R5 | `cargo nextest run -p omega --features metal-attn-split-rows,metal-attn-variants --lib -E 'test(~card_14_k_reuse)'` | 2 tests run: 2 passed, 0 skipped | [ ] | |
| 15 | [card](cards/15-v-reuse.md) | 14 | R5 | `cargo nextest run -p omega --features metal-attn-split-rows,metal-attn-variants --lib -E 'test(~card_15_v_reuse)'` | 2 tests run: 2 passed, 0 skipped | [ ] | |
| 16 | [card](cards/16-tile-height.md) | 12 | R5 | `cargo nextest run -p omega --features metal-attn-split-rows,metal-attn-variants --lib -E 'test(~card_16_tile_height)'` | 2 tests run: 2 passed, 0 skipped; rows_16 admitted and both exact declines asserted | [ ] | |
| 17 | [card](cards/17-query-parallelism.md) | 12,14,15,16 | R5 | `cargo nextest run -p omega --features metal-attn-split-decode,metal-attn-split-rows,metal-attn-variants --lib -E 'test(~card_17_query_parallelism)'` | 2 tests run: 2 passed, 0 skipped; explicit one-row decline and legacy decode-split control | [ ] | |
| 18 | [card](cards/18-prefetch.md) | 12,14,15 | R5 | `cargo nextest run -p omega --features metal-attn-split-rows,metal-attn-variants --lib -E 'test(~card_18_prefetch)'` | 2 tests run: 2 passed, 0 skipped | [ ] | |
| 19 | [card](cards/19-simd-topology.md) | 12 | R5 | `cargo nextest run -p omega --features metal-attn-split-rows,metal-attn-variants --lib -E 'test(~card_19_simd_topology)'` | 2 tests run: 2 passed, 0 skipped | [ ] | |
| 20 | [card](cards/20-dispatch-matrix.md) | 06,09,11,13,14,15,16,17,18,19 | R5,R6 | `cargo nextest run -p omega --features metal-attn-split-rows,metal-attn-variants --lib -E 'test(~card_20_dispatch_matrix)'` | 2 tests run: 2 passed, 0 skipped; seven one-factor flips | [ ] | |
| 21 | [card](cards/21-cross-axis-admission.md) | 06,09,11,13,14,15,16,17,18,19,20 | R5,R6 | `cargo nextest run -p omega --features metal-attn-split-rows,metal-attn-variants --lib -E 'test(~card_21_cross_axis)'` | 2 tests run: 2 passed, 0 skipped; valid composition and exact `prefetch` budget decline | [ ] | |
| 22 | [card](cards/22-bench-entrypoint.md) | 12,19,20,21 | R7 | `AB_ATTENTION_VARIANT='kv_storage=f32,mma_precision=legacy,kv_reuse=legacy,tile_height=legacy,query_parallelism=legacy,simd_topology=legacy,prefetch=off' AB_VARIANT_DESCRIBE_ONLY=1 cargo run -p proxima-model-interop --example norm_variant_ab --features std,metal,instrument,metal-attn-split-rows,metal-attn-variants -- && AB_ATTENTION_VARIANT='kv_storage=bf16,mma_precision=f16,kv_reuse=shared_k,tile_height=rows_8,query_parallelism=simdgroup_rows,simd_topology=per_head,prefetch=off' AB_VARIANT_DESCRIBE_ONLY=1 cargo run -p proxima-model-interop --example norm_variant_ab --features std,metal,instrument,metal-attn-split-rows,metal-attn-variants --` | each run emits 1 `ab variant` and 0 `ab res` lines; selected source/grid differ | [ ] | |

## resume

Last landed slice: card 00 implementation and AC00
Next action: card 01, implement BF8 scalar conversion using the card 00 byte contract
Open question, if any: none; E5M2 is bound as Proxima-owned in SPEC.md

## struck

- none
