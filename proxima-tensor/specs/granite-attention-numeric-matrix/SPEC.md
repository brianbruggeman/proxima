# granite-attention-numeric-matrix

status: draft
owner: brian
created: 2026-10-09

## problem

Proxima's typed numeric path and Granite Metal cached attention cannot select and independently inspect BF16/BF8 cache storage, F32/F16 MMA precision, K/V reuse, query parallelism, tile height, SIMD topology, and prefetch; this spec supplies a bit-defined scalar format and one bounded, independently selectable implementation slice per axis.

## refutation condition

The proposed variant is rejected if its selected dispatch cannot be distinguished from the baseline in emitted source and grid evidence, if its output violates its declared numeric contract against the incumbent on a real Granite shaped fixture, or if disabling its one axis changes any other axis's selected value.

## requirements

| id | requirement | testable in isolation |
|---|---|---|
| R1 | BF8 is a Proxima owned, scalar brain float with exact bit encoding and conversion; BF16 remains separately identified | yes |
| R2 | `DType`, `Codec`, scalar conversion, and CPU `Element` route the numeric family without confusing storage and accumulator types | yes |
| R3 | device K/V seeding, append and flush support selectable F32/BF16/BF8 storage with f32 computation where declared | yes |
| R4 | cached attention decode and row tiled kernels read the selected K/V representation and preserve admitted shapes | yes |
| R5 | seven named axes are independently selectable: K/V storage precision, MMA operand precision, K/V reuse mode, tile height, query parallelism, SIMD topology, prefetch | yes |
| R6 | every dispatch reports the selected form, storage, MMA precision, tile, query schedule, topology, reuse, and prefetch; all legacy defaults remain selectable | yes |
| R7 | the existing Metal attention A/B entrypoint accepts the structured seven-axis selector and emits selected config plus dispatch identity before timing | yes |

## architecture

`BFloat8` is a one-byte Proxima scalar. `Convert<f32, BFloat8>` and its inverse are `Pipe` impls beside the existing BF16 pair. `DType::BFloat8` describes a scalar; `Codec::BFloat8` describes a stored byte layout. CPU `Element` chooses the scalar implementation once per buffer. Metal attention's query and softmax arithmetic remain f32 unless a separate operand precision choice is explicitly made; selecting byte width does not claim BF8 arithmetic or BF8 accumulation.

The BF8 format decision is **local to Proxima**: `s eeeee mm`, bias 15, normal value `(-1)^s × 2^(e−15) × (1+m/4)` for `1≤e≤30`, subnormal value `(-1)^s × 2^(-14) × (m/4)` for `e=0`, signed zero for `e=m=0`, infinity for `e=31,m=0`, and NaN for `e=31,m≠0`. Encoding rounds to nearest with ties to even, preserves sign of zero and NaN, canonicalizes NaN mantissa to binary `10`, overflows to infinity, and gradually underflows into subnormals. The byte order is irrelevant for a one-byte scalar. Card 00 copies this golden table into a CSV before conversion code is written:

| case | f32 bits | BF8 bits |
|---|---|---|
| positive_zero | `0x00000000` | `0x00` |
| negative_zero | `0x80000000` | `0x80` |
| one | `0x3f800000` | `0x3c` |
| negative_one | `0xbf800000` | `0xbc` |
| one_and_quarter | `0x3fa00000` | `0x3d` |
| one_and_half | `0x3fc00000` | `0x3e` |
| tie_down_even | `0x3f900000` | `0x3c` |
| tie_up_even | `0x3fb00000` | `0x3e` |
| min_normal | `0x38800000` | `0x04` |
| min_subnormal | `0x37800000` | `0x01` |
| max_finite | `0x47600000` | `0x7b` |
| overflow | `0x47700000` | `0x7c` |
| positive_infinity | `0x7f800000` | `0x7c` |
| negative_infinity | `0xff800000` | `0xfc` |
| positive_nan | `0x7fc00000` | `0x7e` |
| negative_nan | `0xffc00000` | `0xfe` |

The Metal selection surface is a compile-time default-off feature and a named configuration record with seven independent fields: `kv_storage={f32,bf16,bf8}`, `mma_precision={legacy,f32,f16}`, `kv_reuse={legacy,shared_k,shared_kv}`, `tile_height={legacy,rows_2,rows_4,rows_8,rows_16}`, `query_parallelism={legacy,simdgroup_rows}`, `simd_topology={legacy,per_head,grouped_queries}`, `prefetch={off,next_block}`. `mma_precision=legacy` retains the current sized-rule choice; explicit `f32` and `f16` override it. `tile_height` is the number of query rows owned by one threadgroup; `query_parallelism` chooses whether those rows are assigned concurrently to simdgroups or retain the legacy row schedule; `simd_topology` maps lanes within each simdgroup across a head or grouped query heads. `shared_k` stages K for the row tile while V remains on the legacy path; `shared_kv` stages both K and V. Each axis card implements its own behavior and requires one-factor dispatch evidence. Unsupported combinations return a typed admission decline, never silent fallback. Preserve the existing `NumericPolicy` gate and its form classifier.

The cross-product rule is constructive: all seven axes compose when each field's own admission passes; no model name or preselected combination table may alter a field. `shared_k`, `shared_kv`, and `next_block` require a row-tiled form; the bytes staged by the selected reuse and prefetch values must fit the threadgroup budget. Shared-memory values use the selected MMA operand representation after cache decode: F16 operands consume two bytes per staged value and F32 operands consume four; global cache traffic still follows `kv_storage`. Explicit tile rows must satisfy existing tile-unit, register, and memory limits. `simdgroup_rows` needs at least two query rows; `grouped_queries` needs at least two query heads per KV head; BF16/BF8 storage needs its corresponding reader; F16 MMA operands are available only on forms with the row-tiled MMA implementation. A rejected combination reports the failed field and exact shape/budget, and never silently selects a legacy value. Card 20 verifies the seven one-factor flips; card 21 verifies valid multi-axis selection and combined-memory rejection.

The attention output fixture uses Q/K/V values from `{0, ±0.5, ±1, ±2}`; these are exactly representable in F16, BF16, and this E5M2 BF8 contract. It runs the F32-storage baseline and selected cache storage through the same f32 softmax/accumulator path and compares output bytes. The MMA precision card uses the same fixture to compare explicit F32 and F16 operands. These checks isolate conversion/dispatch correctness without claiming model-quality parity for lossy quantization. BF8 conversion vectors separately cover rounding and exceptional values.

### decisions

| decision | chosen | reason and source |
|---|---|---|
| BF8 identity | **Proxima BF8 = signed E5M2 scalar**, bits `s eeeee mm`, exponent bias 15, gradual subnormals, signed zero, exponent 31/mantissa 0 infinity, exponent 31/nonzero NaN, round to nearest ties to even, overflow to infinity, canonical quiet NaN `0x7e` (sign preserved on conversion); decode accepts all NaN payloads | This is a Proxima format decision, not a claim that a standard names E5M2 “BF8”. ONNX has several distinct float8 types (`proxima-onnx/src/types.rs:33-40`), so the name alone is insufficient. E5M2 keeps explicit infinities/NaNs and a wider exponent range than E4M3, at the cost of mantissa resolution. Commit card 00's golden byte table before coding. No shared scale or block metadata exists in this scalar. |
| existing BF16 | retain `half::bf16` and its round-to-nearest-even conversions | `proxima-tensor/src/convert.rs:22-31,196-212`; `proxima-tensor/src/cpu/typed_eval.rs:322-366` |
| typed numeric family | extend Proxima's `Element`, `Convert`, `DType`, `Codec` ownership | `proxima-tensor/src/cpu/typed_eval.rs:48-61`, `proxima-tensor/src/convert.rs:1-12`, `proxima-primitives/src/codec.rs:1-11`; no `num-traits` dependency |
| storage vs arithmetic | K/V cache width is independent of MMA operand width and f32 softmax/accumulators | `omega/src/msl/cached_attention_row_tiled.rs:37-42,191-192,323-356`; llama.cpp may dequantize K/V to F16 before attention (`/Users/brianbruggeman/repos/others/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp:2959-2977,3331-3367`) |
| baseline and variant | preserve F32 cached K/V, sized-rule MMA precision, and current form/grid as explicit legacy values | `omega/src/msl/signature_tokens_prelude.rs:1798-1831,1957-1998`; `omega/omega-runtime.toml:495-557` |
| reuse and topology | use existing split and row tiled Metal architecture; select variants at form/grid construction and render without model-name branches | `omega/src/msl/cached_attention_decode_split.rs:130-168`, `omega/src/msl/cached_attention_row_tiled.rs:4-41`; llama.cpp dispatch/kernel are reference shapes, not a copied implementation (`/Users/brianbruggeman/repos/others/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp:2949-2956,3515-3558`; `.../kernels/fa_common.metal:300-338`) |

## acceptance criteria

The per-card ACs in `cards/` are the executable requirements. Each nextest filter is a uniquely named new test pair and must report **2 tests run: 2 passed, 0 skipped**; a zero-match result is failure. Card 00 has a checker with two independent gates. The dispatch cards additionally assert the selected manifest and baseline source/grid comparison inside those two tests. Tests inspect structure and payload, not timing. Run no benchmark or model measurement during card authoring.

| id | discharges | command | expected |
|---|---|---|---|
| AC1 | R1, R2 | `cargo nextest run -p proxima-tensor --lib -E 'test(~card_03_bf8_element)'` | 2 tests run: 2 passed, 0 skipped; after cards 00-03 |
| AC2 | R3 | `cargo nextest run -p proxima-model-interop --features std,metal,metal-attn-split-rows --lib -E 'test(~card_08_bf8_device)'` | 2 tests run: 2 passed, 0 skipped; after cards 04-08 |
| AC3 | R4 | `cargo nextest run -p omega --features metal-attn-split-rows --lib -E 'test(~card_11_bf8_row)'` | 2 tests run: 2 passed, 0 skipped; after cards 06,09-11 |
| AC4 | R5, R6 | `cargo nextest run -p omega --features metal-attn-split-rows,metal-attn-variants --lib -E 'test(~card_20_dispatch_matrix)'` | 2 tests run: 2 passed, 0 skipped; legacy identity and seven one-factor flips after cards 12-19 |
| AC5 | R5, R6 | `cargo nextest run -p omega --features metal-attn-split-rows,metal-attn-variants --lib -E 'test(~card_21_cross_axis)'` | 2 tests run: 2 passed, 0 skipped; supported multi-axis manifest and `prefetch` decline at 42,368 required / 32,768 available bytes after card 20 |
| AC6a | R7 | `AB_ATTENTION_VARIANT='kv_storage=f32,mma_precision=legacy,kv_reuse=legacy,tile_height=legacy,query_parallelism=legacy,simd_topology=legacy,prefetch=off' AB_VARIANT_DESCRIBE_ONLY=1 cargo run -p proxima-model-interop --example norm_variant_ab --features std,metal,instrument,metal-attn-split-rows,metal-attn-variants --` | 1 legacy `ab variant` line with seven named values, selected entry/source hash/grid and zero timed cells; after card 22 |
| AC6b | R7 | `AB_ATTENTION_VARIANT='kv_storage=bf16,mma_precision=f16,kv_reuse=shared_k,tile_height=rows_8,query_parallelism=simdgroup_rows,simd_topology=per_head,prefetch=off' AB_VARIANT_DESCRIBE_ONLY=1 cargo run -p proxima-model-interop --example norm_variant_ab --features std,metal,instrument,metal-attn-split-rows,metal-attn-variants --` | 1 non-legacy `ab variant` line with seven selected values, changed source/grid identity and zero timed cells; after card 22 |

## out of scope

- CUDA and WGSL execution paths, model weight quantization, and any BF8 block floating layout.
- Running timing, throughput, or model quality measurements while authoring these cards. The implementation cards capture dispatch and numeric payload evidence; any performance conclusion needs a separate measurement protocol.
- An implicit default change. The current attention path remains selectable for each axis.

## risks

| risk | likelihood | what it costs | what we do about it |
|---|---|---|---|
| “BF8” is mistaken for another float8 standard | high | incompatible bytes | card 00 binds every bit class and edge vector before implementation |
| cache width is mistaken for MMA operand precision | high | incorrect parity claim | separate fields and assertions for cache bytes, Q/K/V MMA operands and f32 accumulators |
| larger tile or prefetch exceeds threadgroup memory | high | Metal compile failure | admission checks use the existing budget and test an over-budget decline |
| the row renderer already has a separate F16 MMA operand toggle over F32 K/V buffers | high | cache and compute precision become conflated | preserve that boundary at `omega/src/msl/cached_attention_row_tiled.rs:37-42`; compare each variant with a same-build baseline and the recorded fixture with source hash |

## context

- The `ai_docs` route for performance-sensitive work is `ai_docs/task-routes.jsonl` task `disciplined-component`; its invariants are in `ai_docs/invariants.jsonl`.
- Existing Granite evidence includes `proxima-tensor/specs/decode-prefill-parity/evidence/attn5/raw/probe/granite.out:1-16` and the per-shape discussion at `proxima-tensor/specs/decode-prefill-parity/SPEC.md:4303-4314`. Those are evidence pointers, not a new verdict.
- The local llama.cpp checkout used here is revision `f1ea206218210afb913ae2f5d2c51faed35915da`; Proxima source was read at `b4e441eb962e96aa31aff2d5ea59ff4024579c45` with unrelated dirty files preserved.
- `proxima-model-interop/src/generate/device_kv.rs:1-27,232-279,365-369` names the existing resident cache ownership and F16 conversion points.
- `omega/src/msl/cached_attention_render.rs:61-103,160-193` currently admits half cache only in decode split, so row tiled BF16/BF8 cache precision requires a separate slice.
