# fp6-e3m2-codec

status: draft
owner: Brian Bruggeman
created: 2026-10-10
revised: 2026-10-11 (BF constants, u32 width, auditor rounds 1-3 closed, requirements merged to one behaviour each)

## problem

On granite 3.1 1b moe (Q8_0 experts, 8.5 bits per weight, 34 bytes per 32 weights) and gemma4 E2B (Q4_0 source, 4.5 bits per weight), a scaled E3M2 block codec (OCP MX element s1e3m2, f16 absmax scale per 32, 26 bytes per 32 weights, 6.5 bits per weight) is added end to end as the next instance of one `FloatLayout` definition that already has two bespoke copies in the tree (bf8 e5m2, bf4 e2m1), and is measured against each model's shipped codec in the same `decode_arms` run on four cells (granite and E2B, each on `prompt1k.txt` and `prompt_short_hippo.txt`) by these bounds: decode ms per token reduced by more than `max(MAD over kept runs, 2% of base)`; positional token agreement with the recorded llama ids no more than `floor(0.02 x N)` positions below the shipped codec's own count in the same run (N = 224 positions for granite, 164 for E2B, both counted from the vendored fixtures); granite `peak_gpu_bytes` lower by at least 250,000,000.

## refutation condition

e3m2 is refuted as a serving default if, on granite, any of these observations holds:

- the `ms_per_token` bound line of `fp6` vs `shipped` has `delta + limit >= 0` on either prompt, so AC18 prints fewer than 4 holding relations. Here `delta` is the fp6 median minus the shipped median and `limit` is `max(MAD kept, 2% of base)`, both printed by `decode_arms` (`print_bounds`, `proxima-model-interop/examples/decode_arms.rs:465-481`, `bound_line` at `:508`);
- the token-agreement example prints `within=false` for either cell (AC16): `shipped_match - fp6_match > floor(0.02 x N)`, which is more than 4 positions of 224 for granite and more than 3 positions of 164 for E2B;
- the granite `peak_gpu_bytes` bound line has `delta` above -250,000,000 on either prompt (AC18 prints fewer than 4 holding relations). The saving derives from the owner's figure of 1.28 GB of Q8_0 expert bytes: `1,280,000,000 x (34 - 26) / 34 = 301,176,470` bytes saved, which is above the bound; AC13 measures the real figure from the converted file.

Any one refutes e3m2 as a serving default. The codec stays in the tree, selectable by config and default off, as a measured negative with its numbers; the refuting AC output is recorded, never edited.

Separately, the float-layout unification is refuted if any of three independent oracles disagrees with the generic encoder or decoder: `f32::to_bits` / `f32::from_bits` for BF32, `half::bf16::from_f32` / `to_f32` for BF16, or the 16 rows of `bf8_vectors.csv` for E5M2 (all three inside AC3).

## requirements

| id | requirement | testable in isolation |
|---|---|---|
| R1 | `FloatLayout { exponent_bits: u8, mantissa_bits: u8, specials: Specials, bias_override: Option<i16> }` in `proxima-primitives/src/float_layout.rs` (sign is always one bit), with derived `bias = 2^(exponent_bits - 1) - 1` (or the override), `width = 1 + exponent_bits + mantissa_bits` (at most 32), `max_normal`, `min_normal`, `min_subnormal`, the enum `Specials { None, SaturateOnOverflow, InfinityOnOverflow }` (`None`: no inf or nan encodings, overflow saturates to the signed max normal; `SaturateOnOverflow`: IEEE inf and nan encodings reserved, finite overflow and inf input clamp to max normal, nan still encodes as nan; `InfinityOnOverflow`: IEEE encodings reserved, overflow rounds to inf), and the constants `E5M2`, `E2M1`, `E3M2`, `E2M3`, `BF32` (E8M23, IEEE binary32 itself), `BF16` (E8M7), `BF8` (the same value as `E5M2`), `BF4` (the same value as `E2M1`), and `E5M2_SATURATING` (E5M2 widths, the GGUF bf8 storage policy that keeps infinities out of stored weights); `specials` per constant: E5M2 and BF8 `InfinityOnOverflow`, E5M2_SATURATING `SaturateOnOverflow`, BF16 and BF32 `InfinityOnOverflow`, E3M2, E2M3, E2M1 and BF4 `None` | yes |
| R2 | Generic `encode(f32, FloatLayout) -> u32` rounds to nearest, ties to even code; clamps overflow to the signed max normal for `Specials::None` and `SaturateOnOverflow`, overflows to infinity for `InfinityOnOverflow`, flushes values below half the minimum subnormal to signed zero; `encode_counting` maps NaN to the positive max normal and counts it for `Specials::None` layouts | yes |
| R3 | Generic `encode` and `decode(u32, FloatLayout) -> f32` agree with three independent oracles that live inside proxima-gguf: `f32::to_bits` / `f32::from_bits` for BF32, `half::bf16` for BF16, and the 16 rows of `bf8_vectors.csv` for E5M2 | yes |
| R4 | `bf8_e5m2::{encode,decode}` and `bf4_e2m1::{encode,decode}` keep their signatures and route through the generic functions (bf8 through `FloatLayout::E5M2_SATURATING`, bf4 through `E2M1`); the existing contract test (256-code E5M2 decode table, E2M1 nibble table, encode asserts) is not modified and stays green | yes |
| R5 | The E3M2 definition is the OCP MX v1.0 text (section "Definition of E3M2" below), vendored as `proxima-gguf/src/quant/fixtures/ocp_mx_v1_0.txt`; generic decode of all 64 codes equals the table derived from it and each golden value encodes to its code | yes |
| R6 | `Codec::Fp6E3M2` in `proxima-primitives/src/codec.rs` with stable tag 32 (30 is `Bf8E5M2`, 31 is `Bf4E2M1`): 32-weight blocks of 26 bytes | yes |
| R7 | Block quantize and dequantize in `proxima-gguf/src/quant/scaled_float.rs`, generic over `FloatLayout`: f16 scale then 24 payload bytes holding 32 six-bit fields, 4 fields per 3 bytes | yes |
| R8 | CPU `dequantize_fp6_e3m2` (`proxima-tensor/src/cpu/epilogue.rs`) and `matmul_fp6_e3m2_f32` (`gemm_dot_quant.rs`) match the golden table times the scale and an f64 dot product | yes |
| R9 | GGUF edge: a proxima-own tensor type id carries e3m2 blocks and round trips through `proxima_gguf`; export to a ggml consumer is rejected (`dtype_to_ggml` returns `None` for the new `DType::Fp6E3M2`, precedent `DType::BFloat8` at `proxima-model-interop/src/dtype.rs:45`; the transform returns `InteropError::UnrepresentableDType`) | yes |
| R10 | omega renders the codec in the row-blocked matvec (`packed_row_blocked_ggml.rs`) with a 4-weights-per-3-bytes unpack, the decode MSL comes from `FloatLayout::E3M2` constants (bias, field masks) | yes |
| R11 | omega admits the codec in the tiled and expert-grouped GEMM through one description: `tiled_decode(Codec::Fp6E3M2)` returns `Some(TiledDecode::new(32, 26, "scaled_e3m2_decode", msl))` (exhaustive over `Codec`, `omega/src/msl/kernel_types_identity.rs:2073`) and the grouped GEMM source renders through it, and none of the three rendered sources (row-blocked, tiled, grouped) carries a model name. In scope; no fallback | yes |
| R12 | Metal output of an e3m2 matvec equals an f64 CPU dot product over the dequantized block values within the golden `NumericPolicy`, on 1024x512 and 1536x6144 rows at 1 and 128 tokens | yes |
| R13 | `convert_codec(model, policy)` in the new `proxima-model-interop/src/convert.rs` converts the tensors a policy names (expert projections `ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`; attention projections `attn_q`, `attn_k`, `attn_v`, `attn_output`; never embeddings, norms, routers `ffn_gate_inp`, or the output head) from F32, F16, Q8_0 and Q4_0, writes the checkpoint through the transform writer, and on the real Q8_0 MoE checkpoint converts as many tensors as the policy matches in the GGUF header and cuts the file by at least 250,000,000 bytes | yes |
| R14 | The `convert_codec` example converts checkpoints offline and prints one summary line per job | yes |
| R15 | A `[codec_policy]` serving setting with `weights = "shipped" \| "fp6_e3m2"` and a tensor class list: default `shipped`, env, TOML and builder agree, an unknown codec name is rejected | yes |
| R16 | A new example reports positional token agreement of fp6 and shipped generations against the recorded llama ids and the bound relation, for both models | yes |
| R17 | The bench table: e3m2 vs shipped through `decode_arms` prints all 11 bound lines for each of four cells (granite and E2B, each on `prompt1k.txt` and `prompt_short_hippo.txt`) | yes |
| R18 | The refutation relations of this spec evaluate on the saved granite bench output: `peak_gpu_bytes` delta at most -250,000,000 and `ms_per_token` `delta + limit < 0`, on both prompts | yes |

## architecture

The owner's framing (2026-10-11): there is no "fp6"; there is a float layout, a tuple of sign, exponent and mantissa widths, and the codecs are its instances. The two existing scalar codecs (`bf8_e5m2.rs:5,46`, `bf4_e2m1.rs:18,39`) each hand-code the same arithmetic; this spec replaces their bodies with the generic functions and adds e3m2 as data. The scaled block layer (absmax f16 scale per 32) is the same for every layout; only the field width and the layout constant change. The owner added (2026-10-11) the brain float constants `BF32`, `BF16`, `BF8`, `BF4`: `BF8` and `BF4` are the same values as `E5M2` and `E2M1`; `BF32` is IEEE binary32 and `BF16` is the half crate's bfloat16. They give the generic code two oracles that live inside proxima-gguf (`f32::to_bits`, and `half = 2.7.1` already a dependency at `proxima-gguf/Cargo.toml:22`). BF32 needs 32 bits, so the generic code is `u32` wide.

```
proxima-primitives/src/float_layout.rs
  pub struct FloatLayout { pub exponent_bits: u8, pub mantissa_bits: u8, pub specials: Specials, pub bias_override: Option<i16> }
  pub enum Specials { None, SaturateOnOverflow, InfinityOnOverflow }
  impl FloatLayout { pub const E5M2; E2M1; E3M2; E2M3; BF32; BF16; BF8 (= E5M2); E5M2_SATURATING; BF4 (= E2M1);
                     pub const fn bias(&self) -> i16; pub const fn width(&self) -> u8;
                     pub fn max_normal(&self) -> f32; pub fn min_normal(&self) -> f32; pub fn min_subnormal(&self) -> f32; }
proxima-gguf/src/quant/float_layout.rs        pub fn encode(value: f32, layout: FloatLayout) -> u32; pub fn decode(code: u32, layout: FloatLayout) -> f32;
                                              pub fn encode_counting(value: f32, layout: FloatLayout, nan_count: &mut u64) -> u32
proxima-gguf/src/quant/bf8_e5m2.rs            encode/decode -> float_layout::{encode,decode}(.., FloatLayout::E5M2_SATURATING) as u8
proxima-gguf/src/quant/bf4_e2m1.rs            the same with E2M1; encode keeps its Result and maps a non-finite input to QuantError::NonFiniteInput
proxima-gguf/src/quant/scaled_float.rs        pub fn quantize_block(values: &[f32; 32], layout: FloatLayout, out: &mut [u8]); pub fn dequantize_block(block: &[u8], layout: FloatLayout, out: &mut [f32; 32])
proxima-gguf/src/quant/fixtures/ocp_mx_v1_0.txt   (new) pdftotext -layout of the OCP MX v1.0 PDF, 592 lines
proxima-gguf/tests/float_layout_oracles.rs    (new) the oracle tests of AC2 to AC5
proxima-primitives/src/codec.rs               Codec::Fp6E3M2 => 32 (layout E3M2, block 32, field width 6)
proxima-tensor/src/cpu/epilogue.rs            dequantize_fp6_e3m2
proxima-tensor/src/cpu/gemm_dot_quant.rs      matmul_fp6_e3m2_f32
proxima-model-interop/src/convert.rs          (new) convert_codec(model, policy)
proxima-model-interop/src/serving_settings.rs [codec_policy]
omega/src/msl/kernel_types_identity.rs        tiled_decode(Codec::Fp6E3M2) -> Some(TiledDecode::new(32, 26, "scaled_e3m2_decode", MSL rendered from FloatLayout::E3M2))
omega/src/msl/packed_row_blocked_ggml.rs      Codec::Fp6E3M2 arm (the Q4_0 arm at :665 is the shape)
proxima-model-interop/examples/convert_codec.rs         (new)
proxima-model-interop/examples/codec_token_agreement.rs (new)
proxima-model-interop/examples/bound_relations.rs (new) parses saved decode_arms bound lines (decode_arms.rs:500,524) and evaluates the refutation relations; Rust only, no awk or shell logic
```

Block byte layout (R7): bytes 0 to 1 are the f16 scale `absmax / 28.0` (28.0 is E3M2's max normal); bytes 2 to 25 are the payload. Weight `j` of a group of 4 occupies bits `6j .. 6j+6` of the little-endian 24-bit word formed by that group's 3 bytes.

`FloatLayout` is the one new type. The pipe question was asked: a layout is not a pipe, it is the data three codecs are parameterized by; the second question (what can a caller do that they could not) is answered by the two deleted copies of the arithmetic and by e3m2, e2m3, BF16 and BF32 arriving as constants. Nothing else is a new type; `quantize_block` takes the layout, the field width is `layout.width()`.

### Definition of E3M2 (OCP MX v1.0, extracted from the vendored text)

Source: Open Compute Project, OCP Microscaling Formats (MX) Specification v1.0, Sep 2023, PDF `ocp-mx-v1.0.pdf` (812323 bytes, 16 pages); text at `.long_ctx_backups/mlx_fp6/ocp/ocp-mx-v1.0.txt` (592 lines), vendored as `proxima-gguf/src/quant/fixtures/ocp_mx_v1_0.txt`. Line numbers below are that text file's; the page is the PDF page number from its footer.

| item | value | text line, page, section |
|---|---|---|
| element in MXFP6 | FP6 (E3M2), block size 32, E8M0 scale | line 316, page 10, Table 1 (section 5.2) |
| value, E > 0 | `(-1)^S x 2^(E - bias) x (1 + 2^-m x M)` | line 344, page 11, section 5.3 |
| value, E = 0 (subnormal) | `(-1)^S x 2^(1 - bias) x (0 + 2^-m x M)` | line 345, page 11, section 5.3 |
| exponent bias | 3 | line 413, page 12, Table 4 (section 5.3.2) |
| infinities, NaN | N/A: no encodings reserved | lines 398-399 and 414-415, page 12, section 5.3.2 and Table 4 |
| zeros | `S 000 00` | line 416, page 12, Table 4 |
| max normal | `S 111 11` = +-2^4 x 1.75 = +-28.0 | line 417, page 12, Table 4 |
| min normal | `S 001 00` = +-2^-2 x 1.0 = +-0.25 | line 418, page 12, Table 4 |
| max subnormal | `S 000 11` = +-2^-2 x 0.75 = +-0.1875 | line 419, page 12, Table 4 |
| min subnormal | `S 000 01` = +-2^-2 x 0.25 = +-0.0625 | line 420, page 12, Table 4 |
| rounding | roundTiesToEven required | lines 398-400, page 12, section 5.3.2 |
| overflow | clamp (saturate) to the max magnitude, sign preserved, required | lines 402-406, page 12, section 5.3.2 |
| underflow | magnitude below the min subnormal after rounding converts to zero | lines 408-410, page 12, section 5.3.2 |
| NaN input | conversion is implementation-defined | lines 409-410, page 12, section 5.3.2 |

The text agrees with `bias = 2^(3 - 1) - 1 = 3` from R1, so `FloatLayout::E3M2` carries `bias_override: None`. E2M3 (same table, line 413) has bias 1, equal to `2^(2 - 1) - 1`. The E5M2 bias of 15 is at line 367 (Table 2, section 5.3.1). The earlier "working definition" paragraph of this spec is struck: it matched the text on every item above. For NaN input the implementation-defined choice is R2.

Oracles, stated because e3m2 has no incumbent (ggml's `ggml_type` enum as read has no FP6; MLX has no fp6 mode): the generic code has three independent oracles inside proxima-gguf: `f32::to_bits` (BF32), `half::bf16` (BF16), and the 16-row bf8 golden CSV plus the 256-code E5M2 table in the contract test (E5M2). e3m2's element oracle is the OCP text (R5). The block and kernel oracles are f64 arithmetic in a test loop that shares nothing with the kernels. Every parity test family carries a degenerate control that must fail. The token oracle is the recorded llama ids under `proxima-model-interop/tests/fixtures/llama-parity/`; no Ollama arm and no llama-server arm is run (the owner's rule), and the bench compares e3m2 against the SHIPPED codec in the same `decode_arms` run.

### decisions

| decision | chosen | why not the alternative |
|---|---|---|
| one `FloatLayout` type, instances as constants | yes | hand-coded scalar codecs that differ by two integers are the compensator pattern; the layout is the thing they compensate for |
| `BF8` and `BF4` as aliases of `E5M2` and `E2M1` | yes | the owner asked for the brain float names; a second value would be a second definition |
| `u32` code width | yes | BF32 is 32 bits; `u8` or `u16` would exclude the f32 oracle |
| one `Specials` enum instead of two bools | yes | the incumbent behavior differs: the shipped `bf8_e5m2::encode` saturates infinity to 0x7b (`bf8_e5m2_bf4_e2m1_encoding_contract.rs`), while `half::bf16::from_f32(inf)` is infinity; the three variants let both oracles hold and make the invalid fourth combination (no reserved encodings yet overflow to inf) unrepresentable |
| E5M2 follows IEEE, the GGUF bf8 codec keeps saturation as a named layout | yes | two in-tree incumbents disagreed: the contract test asserts `encode(inf) == 0x7b` while `bfloat8.rs:59,95` and `bf8_vectors.csv` give 0x7c; the CSV and IEEE ties-to-even agree (0x47700000 is the exact midpoint of 0x7b and the inf code 0x7c, and 0x7c is even), and the difference is one `Specials` value (`E5M2` InfinityOnOverflow vs `E5M2_SATURATING` SaturateOnOverflow), not a second type |
| sign always one bit, bias `2^(e-1)-1` with an override | yes | every published float format uses it (E5M2 15, E2M1 1, E3M2 3, f16 15, bf16 127); the override exists so a deviating format is data, not a fork |
| block-packed with f16 absmax scale per 32 | yes | unscaled 6-bit floats have 3 bits of range; per-block scaling is what Q4_0 and MX do; 32 matches the kernels' lane geometry |
| 4 weights per 3 bytes | yes | dense; one byte per weight is 8 bits |
| proxima-own GGUF type id, rejected at the ggml edge | yes | a fake ggml id would corrupt real consumers (principle 14) |
| grouped GEMM in scope through the one `TiledDecode` description | yes | a converted checkpoint has no shipped-codec expert weights left, so prefill cannot "stay on the shipped codec"; if staging 3-byte groups fails at AC11 the spec is amended with the reason, not reworded |
| offline conversion | yes | load-time quantization hides the codec from the format layer |
| default off | yes | the codec changes outputs; the owner flips the default on the AC16 to AC18 results |

## acceptance criteria

Paths, bound by this assignment block (the `GRANITE` and `E2B` env overrides are `PROXIMA_ARCH_GRANITE_MOE_GGUF` and `PROXIMA_ARCH_GEMMA4_E2B_GGUF`; `PATTERN11` is bound in this block and explained under "command blocks"):

```sh
GRANITE=/Users/brianbruggeman/.ollama/models/blobs/sha256-cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80
E2B=/Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd
OUT=/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/mlx_fp6/out
BENCH=proxima-tensor/specs/decode-prefill-parity/evidence/round2fix/bench
IDS=proxima-model-interop/tests/fixtures/llama-parity
BIN=$OUT/bin/decode_gbps_baseline
PATTERN11='^bound metric=(ms_per_token|prefill_ms|ttft_ms|wall_ms|cpu_ms|peak_rss_bytes|peak_footprint_bytes|peak_gpu_bytes|steady_rss_bytes|steady_footprint_bytes|steady_gpu_bytes) arm=fp6 vs=shipped arm_median='
mkdir -p $OUT/bin
```

`BIN` is produced once by slice 17's setup: `cargo build --release -p proxima-model-interop --example decode_gbps_baseline --features std,metal`, then `rm -f $BIN` and `cp` (the directory exists from the `mkdir -p` above) of the built `target/release/examples/decode_gbps_baseline` to `$BIN` (a binary is never overwritten in place). Every test count is the exact number the slice adds; a slice that adds a different number updates its AC in the same commit. Test names carry layouts and shapes, never model names. A test that needs a checkpoint fails with its path and env override when absent; it never skips. All commands run from the repo root. The oracle tests for AC2 to AC5 live in `proxima-gguf/tests/float_layout_oracles.rs` (new). Corpus for AC3: f32 bit patterns `index x 4093` (wrapping u32) for `index` in `0..1_048_576` plus 12 fixed values (+-0, +-min subnormal, +-`f32::MIN_POSITIVE`, +-`f32::MAX`, +-infinity, 1.0, one NaN), 1,048,588 values; each test asserts that length before comparing.

| id | discharges | command | expected |
|---|---|---|---|
| AC1 | R1 | `cargo nextest run -p proxima-primitives -E 'test(/float_layout_/)'` | `11 passed`: 6 `float_layout_derived_*` (`..._bias_per_layout` 15/1/3/1/127/127 for E5M2/E2M1/E3M2/E2M3/BF16/BF32; `..._width_per_layout` 8/4/6/6/16/32; `..._max_normal_per_layout` 57344/6/28/7.5/3.3895314e38/`f32::MAX`; `..._min_normal_per_layout`; `..._min_subnormal_per_layout` 2^-16/0.5/0.0625/0.125/2^-133/2^-149; `..._bias_override_shifts_range` E3M2 with override 4 has max normal 14.0) and 5 `float_layout_constants_*` (`..._bf8_equals_e5m2`, `..._bf4_equals_e2m1`, `..._bf32_is_e8m23`, `..._bf16_is_e8m7`, `..._specials_and_widths_per_layout` (E5M2 and BF8 InfinityOnOverflow, E5M2_SATURATING SaturateOnOverflow, BF16 and BF32 InfinityOnOverflow, E3M2, E2M3, E2M1, BF4 None)) |
| AC2 | R2 | `cargo nextest run -p proxima-gguf -E 'test(/float_layout_midpoint/) + test(/float_layout_range/) + test(/float_layout_nan/)'` | `10 passed`: 4 `float_layout_midpoint_ties_to_even_{e5m2,e2m1,e3m2,e2m3}` (for every adjacent positive code pair, the exact f32 midpoint of the two decoded values encodes to whichever of the two codes is even); 4 `float_layout_range_*` (`..._saturates_e3m2` 1e9 to 0x1f and -1e9 to 0x3f; `..._saturates_infinity_e5m2_saturating` (on `E5M2_SATURATING`) +inf to 0x7b and -inf to 0xfb; `..._overflows_to_infinity_bf16` +inf and `f32::MAX` to 0x7f80; `..._underflows_to_signed_zero_e3m2` 0.03 to 0x00 and -0.03 to 0x20, where 0.03 is below half of the 0.0625 minimum subnormal); 2 `float_layout_nan_*` (`..._counted_and_clamped_e3m2`: 10 values with 3 NaN give `nan_count` 3 and code 0x1f for each NaN; `..._quiet_code_not_counted_e5m2`: NaN encodes to 0x7e and the counter stays 0) |
| AC3 | R3 | `cargo nextest run -p proxima-gguf -E 'test(/float_layout_bf32/) + test(/float_layout_bf16/) + test(/float_layout_e5m2/)'` | `9 passed`: 3 `float_layout_bf32_*` (`..._encode_is_f32_to_bits`, `..._decode_is_f32_from_bits`, and the control `..._control_bf16_layout_differs_from_to_bits`, which asserts the BF16 layout disagrees with `f32::to_bits` on more than 0 corpus values); 4 `float_layout_bf16_*` (`..._encode_matches_half_from_f32` over the corpus; `..._decode_matches_half_to_f32` over all 65,536 codes; `..._encode_ties_to_even_matches_half` over the 4 f32 values whose low 16 bits are 0x8000 at two exponents; the control `..._control_truncation_differs_from_half`, which asserts round-toward-zero `bits >> 16` disagrees with `half` on more than 0 corpus values); 2 `float_layout_e5m2_*` (`..._matches_bf8_vectors_csv` does `include_str!("../../proxima-tensor/specs/granite-attention-numeric-matrix/bf8_vectors.csv")`, asserts the header `case,f32_bits,bf8_bits` and 16 rows, and checks encode and decode of every row through the generic E5M2 path; `..._control_altered_row_fails` flips one expected bit of one row and asserts the generic path disagrees). NaN inputs compare as `is_nan`, never by bits |
| AC4 | R4 | `cargo nextest run -p proxima-gguf -E 'test(/bf8_e5m2_bf4_e2m1_encoding_contract/) + test(/float_layout_wrapper/)'` | `3 passed`: the 1 existing test `bf8_e5m2_bf4_e2m1_encoding_contract` (`proxima-gguf/tests/bf8_e5m2_bf4_e2m1_encoding_contract.rs:4`, the only test in proxima-gguf whose name contains bf8 or bf4; `bf8_e5m2.rs` and `bf4_e2m1.rs` hold 0 `#[test]`s, read at origin/main; the file is not modified by the slice) plus 2 new: `float_layout_wrapper_bf8_equals_generic_e5m2_saturating` (generic `E5M2_SATURATING`) (all 256 decode codes and 65,536 encode inputs `index x 65537` reinterpreted as f32) and `float_layout_wrapper_bf4_equals_generic_e2m1` (all 16 decode codes and the finite inputs of the same corpus) |
| AC5 | R5 | `cargo nextest run -p proxima-gguf -E 'test(/float_layout_e3m2_golden/)'` | `4 passed`: `float_layout_e3m2_golden_decode_all_64_codes` against the 32 positive values written in the test from Table 4 plus their negatives, with 0x20 decoding to -0.0; `..._encode_each_value` maps each golden value back to its code; `..._control_altered_row_fails` changes one golden row and asserts the check then fails; `..._fixture_carries_table4` asserts the vendored text holds exactly 592 newline bytes (the count `wc -l` prints; the text ends in a form feed, so `str::lines` would give 593) and contains the strings `S 111 112 = ± 24 × 1.75 = ± 28.0` and `S 000 012 = ± 2-2 × 0.25 = ± 0.0625` |
| AC6 | R6 | `cargo nextest run -p proxima-primitives -E 'test(/fp6_e3m2_codec/)'` | `3 passed`: `fp6_e3m2_codec_tag_32_round_trip`, `fp6_e3m2_codec_tag_33_rejected`, `fp6_e3m2_codec_block_geometry_26_bytes_32_elements` |
| AC7 | R7 | `cargo nextest run -p proxima-gguf -E 'test(/scaled_e3m2_block/)'` | `3 passed`: `scaled_e3m2_block_round_trip_error_bound_1024x512` (rows read from the Q8_0 tensors of the granite checkpoint, env `PROXIMA_ARCH_GRANITE_MOE_GGUF`, whose shape is 1024x512, dequantized then quantized; for every element with `q = abs(v) / scale_f16 <= 28.0`, `abs(v - dequantized) <= scale_f16 x h(q)` where `h(q) = 2^(floor(log2 q) - 3)` for `q >= 0.25` (half of the binade spacing `2^(floor(log2 q) - 2)` of 2 mantissa bits) and `h(q) = 0.03125` for the subnormal range `q < 0.25` (half of the 0.0625 subnormal spacing)); `scaled_e3m2_block_saturates_above_28` (an element with `q > 28.0` dequantizes to exactly `28.0 x scale_f16`); `scaled_e3m2_block_control_perturbed_code_exceeds_bound` (one flipped payload bit makes the bound check fail) |
| AC8 | R8 | `cargo nextest run -p proxima-tensor -E 'test(/scaled_e3m2_cpu_/)'` | `3 passed`: `scaled_e3m2_cpu_dequantize_equals_golden_times_scale_1024x512` (blocks built by the test from the AC5 golden codes and an f16 scale; output equals golden x scale exactly in f32); `scaled_e3m2_cpu_matmul_matches_f64_loop_1024x512` (K = 512: `abs(f32 result - f64 result) <= 4e-5 x sum(abs(a_i x b_i))`, above the dot-product forward error bound `K x 2^-24 = 3.05e-5`); `scaled_e3m2_cpu_matmul_control_perturbed_weight_fails` (one changed weight code exceeds that bound) |
| AC9 | R9 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/fp6_e3m2_gguf_/)'` | `3 passed`: `fp6_e3m2_gguf_round_trip_bytes_identical` (transform write then parse returns the same 26-byte blocks under the proxima-own type id); `fp6_e3m2_gguf_export_rejected_dtype_to_ggml_is_none`; `fp6_e3m2_gguf_export_rejected_transform_returns_unrepresentable_dtype` |
| AC10 | R10 | `cargo nextest run -p omega --features metal -E 'test(/scaled_e3m2_render_row_blocked/) + test(/scaled_e3m2_render_decode/)'` | `3 passed`: `scaled_e3m2_render_row_blocked_1024x512` and `..._1536x6144` (each equal to a snapshot file under `omega/tests/fixtures/` (new), and each also asserted independently of the snapshot to contain the group shifts 0, 6, 12, 18, the mask 0x3f, and a 3-byte stride per group of 4 weights); `scaled_e3m2_render_decode_carries_bias_3_and_field_masks` (the source contains bias 3, exponent mask 0x1c and mantissa mask 0x03 computed from `FloatLayout::E3M2`, and re-rendering with `bias_override: Some(4)` changes the bias literal, which is the control) |
| AC11 | R11 | `cargo nextest run -p omega --features metal -E 'test(/scaled_e3m2_render_tiled/) + test(/scaled_e3m2_render_grouped/) + test(/scaled_e3m2_render_model_name_free/)'` | `3 passed`: `scaled_e3m2_render_tiled_decode_elements_32_bytes_26` (asserts elements == 32, bytes == 26 and fn_name == "scaled_e3m2_decode" on the returned `TiledDecode`); `scaled_e3m2_render_grouped_gemm_source_uses_the_tiled_decode` (the grouped GEMM source contains the `scaled_e3m2_decode` function name and, as a substring, the whole MSL string of the `TiledDecode` returned by `tiled_decode(Codec::Fp6E3M2)`); `scaled_e3m2_render_model_name_free` (the rendered row-blocked, tiled and grouped sources, case-folded, contain none of `gemma`, `granite`, `qwen`, `llama`; the test first asserts it scanned 3 sources) |
| AC12 | R12 | `cargo nextest run -p omega --features metal --test scaled_e3m2_parity` | `5 passed`: `scaled_e3m2_parity_1024x512_1_token`, `..._1024x512_128_tokens`, `..._1536x6144_1_token`, `..._1536x6144_128_tokens`, each against the f64 loop within `NumericPolicy`; `scaled_e3m2_parity_control_perturbed_block_fails`. GPU, behind gpu.lock |
| AC13 | R13 | `PROXIMA_ARCH_GRANITE_MOE_GGUF=$GRANITE cargo nextest run --release -p proxima-model-interop --features std -E 'test(/convert_codec_selection/) + test(/convert_codec_source_/) + test(/convert_codec_checkpoint_/)'` | `7 passed`: `convert_codec_selection_expert_and_attention_projections_only` (a header fixture of 7 projection names and 6 excluded names `token_embd`, `output`, `output_norm`, `attn_norm`, `ffn_norm`, `ffn_gate_inp`: exactly the 7 selected); `convert_codec_source_f32_within_block_bound`, `..._f16_...`, `..._q8_0_...`, `..._q4_0_...` (each converted tensor dequantizes within AC7's bound); `convert_codec_checkpoint_reloads_with_every_tensor_present` (the written file parses and holds the source's tensor count); `convert_codec_checkpoint_moe_q8_0_saves_250mb` (reads the tensor names from the GGUF header, counts those matching the policy's classes, converts, asserts converted count equals that count and `source_bytes - output_bytes >= 250_000_000`, prints both). Writes about 3 GB under `$OUT` |
| AC14 | R14 | command block AC14 | `2` |
| AC15 | R15 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/codec_policy/)'` | `3 passed`: `codec_policy_default_is_shipped`, `codec_policy_env_toml_builder_agree`, `codec_policy_unknown_codec_name_rejected` |
| AC16 | R16 | command block AC16 | `6` (granite: 3 cases x 32 + 1 x 128 = 224 positions, `allowed_gap = floor(0.02 x 224) = 4`; E2B: 3 + 32 + 1 + 128 = 164 positions, `allowed_gap = floor(0.02 x 164) = 3`; free-running greedy generation, position `i` matches when the generated id equals `generated_ids[i]`, no teacher forcing; `gap = shipped_match - fp6_match`; the E2B cell is Q4_0-sourced and reported as such; GPU, behind gpu.lock) |
| AC17 | R17 | command block AC17 | `44` (4 cells x the 11 metrics `print_bounds` emits at `decode_arms.rs:465-481`; the `shipped` arm is the same binary on the shipped checkpoint, the `fp6` arm overrides `PROXIMA_DECODE_MODEL_GGUF`; GPU, behind gpu.lock, box load checked before the run) |
| AC18 | R18 | command block AC18 (exactly one line matching `^relations: [0-9] of 4 hold$` is printed after the 4 relation lines) | `relations: 4 of 4 hold` (refutation AC: a lower N is the refutation outcome, recorded not edited; the new example prints one line per relation with its parsed `delta` and `limit`, then the summary line; the four relations are `ms_per_token` `delta + limit < 0` and `peak_gpu_bytes` `delta <= -250000000`, each on the 1k and hippo prompts; `delta` and `limit` are fields 7 and 8 of the `bound_line` and `memory_bound_line` formats at `decode_arms.rs:500,524`) |

### command blocks

Each block is one command; the count it prints is the asserted N, and a count of 0 is red. `PATTERN11` and the other variables are bound in the assignment block above. `PATTERN11` excludes the `n=0` form `decode_arms` prints when an arm has no data (`decode_arms.rs:493,517`) by ending in `arm_median=`.

```sh
# AC14
cargo run --release -p proxima-model-interop --example convert_codec -- --codec fp6_e3m2 --policy default --job $GRANITE=$OUT/granite-fp6.gguf --job $E2B=$OUT/e2b-fp6.gguf | grep -c -E '^converted: [1-9][0-9]* tensors, kept: [0-9]+, bytes: [1-9][0-9]* -> [1-9][0-9]*$'
# AC16
cargo run --release -p proxima-model-interop --example codec_token_agreement -- --fixtures llama_ids.json,long_prompt_llama_ids.json --allowed-gap-rate 0.02 --cell granite=$OUT/granite-fp6.gguf,$GRANITE,$IDS/granite_moe --cell e2b=$OUT/e2b-fp6.gguf,$E2B,$IDS/gemma4_e2b | grep -c -E '^cell=(granite|e2b) codec=(fp6|shipped) match=[1-9][0-9]* of (224|164)$|^cell=(granite|e2b) bound allowed_gap=[34] gap=[0-9]+ within=true$'
# AC17
{ cargo run --release -p proxima-model-interop --example decode_arms -- --prompt-file $BENCH/prompt1k.txt --log $OUT/launches_granite_1k.log --processes 3 --runs 5 --new-tokens 128 --model $GRANITE --arm shipped=$BIN --arm fp6=$BIN --arm-env fp6:PROXIMA_DECODE_MODEL_GGUF=$OUT/granite-fp6.gguf | tee $OUT/bench_granite_1k.out; cargo run --release -p proxima-model-interop --example decode_arms -- --prompt-file $BENCH/prompt_short_hippo.txt --log $OUT/launches_granite_hippo.log --processes 3 --runs 5 --new-tokens 128 --model $GRANITE --arm shipped=$BIN --arm fp6=$BIN --arm-env fp6:PROXIMA_DECODE_MODEL_GGUF=$OUT/granite-fp6.gguf | tee $OUT/bench_granite_hippo.out; cargo run --release -p proxima-model-interop --example decode_arms -- --prompt-file $BENCH/prompt1k.txt --log $OUT/launches_e2b_1k.log --processes 3 --runs 5 --new-tokens 128 --model $E2B --arm shipped=$BIN --arm fp6=$BIN --arm-env fp6:PROXIMA_DECODE_MODEL_GGUF=$OUT/e2b-fp6.gguf | tee $OUT/bench_e2b_1k.out; cargo run --release -p proxima-model-interop --example decode_arms -- --prompt-file $BENCH/prompt_short_hippo.txt --log $OUT/launches_e2b_hippo.log --processes 3 --runs 5 --new-tokens 128 --model $E2B --arm shipped=$BIN --arm fp6=$BIN --arm-env fp6:PROXIMA_DECODE_MODEL_GGUF=$OUT/e2b-fp6.gguf | tee $OUT/bench_e2b_hippo.out; } | grep -c -E "$PATTERN11"
# AC18
cargo run --release -p proxima-model-interop --example bound_relations -- --input $OUT/bench_granite_1k.out --input $OUT/bench_granite_hippo.out --arm fp6 --vs shipped --min-gpu-saving 250000000
```

## out of scope

- activations or KV cache in any float layout (weights only)
- ggml or llama.cpp interoperability for these tensors (rejected at the edge by R17, by design)
- MLX export of the codec (MLX has no fp6 mode; the MLX export is its own spec)
- e2m3, e2m1 or the BF constants as storage codecs in this spec (they arrive as `FloatLayout` constants and as oracles; a storage codec for one is a new tag later)
- any Ollama or llama-server arm: the token oracle is the recorded llama ids; `decode_arms` is run without `--llama-server` and without `--ollama`
- model-specific kernel branches: the codec is generic over `FloatLayout` and a test (AC10) enforces it on the rendered source
- changing the serving default (the owner flips it on the AC16 to AC18 results)

## risks

| risk | likelihood | what it costs | what we do about it |
|---|---|---|---|
| the generic encoder changes bf8's bits | low; AC3 and AC4 each catch it | a landed codec drifts | three independent oracles stay in the tree, none edited |
| the 6-bit unpack costs more ALU than the bandwidth it saves | medium; Q4_0's pair-dot unpack was round two's lever | slower than Q8_0 despite fewer bytes | AC18 measures it (the `ms_per_token` relation); the row-blocked arm follows the Q4_0 pair-dot shape; the first bench gates tiled GEMM work |
| quality loss on E2B, whose source is Q4_0 | high | the E2B cell says nothing about the codec itself | granite (Q8_0 source) is the primary cell; the E2B cells (the E2B cells of AC16 and AC17) are reported as Q4_0-sourced |
| the expert-grouped GEMM cannot stage 3-byte groups | medium | AC11 fails; prefill of a converted checkpoint has no path | the spec is amended with the staging failure's cause and the GEMM slice is redesigned; the shipped codec cannot serve a converted file's experts |
| `delta + limit` on `ms_per_token` is noise-dominated on a short prompt | medium | AC18 flips on box state | box load checked before every timed run; the 3-process x 5-run protocol and the MAD bound are the existing rule, not new |
| `DType::Fp6E3M2` widens a type with blast radius | medium | other `match` sites on `DType` need arms | the slice adds the variant and the compiler lists every site; `DType::BFloat8` (`dtype.rs:45`) is the precedent |
| the OCP PDF text drifts from the vendored fixture | low | R5 loses its primary source | AC5 asserts the fixture's line count (592) and two table strings |

## context

- `proxima-gguf/src/quant/bf8_e5m2.rs:5,46` and `bf4_e2m1.rs:18,39` (the two hand-coded scalar codecs the layout replaces); `proxima-gguf/tests/bf8_e5m2_bf4_e2m1_encoding_contract.rs:4` (the existing contract test and its 256-code E5M2 decode table)
- `proxima-gguf/Cargo.toml:22` (`half = 2.7.1`, the BF16 oracle's crate)
- `proxima-primitives/src/codec.rs:46,86-87,125-126` (bf8 tag 30, bf4 tag 31; 32 is free)
- `proxima-tensor/src/bfloat8.rs:148-187` (the golden-vector test `card_01_bf8_convert_matches_specified_golden_vectors` over `bf8_vectors.csv`, 16 rows, through `Convert<f32, BFloat8>`; AC3 re-runs the same CSV through the generic path)
- `proxima-tensor/src/cpu/epilogue.rs:1367,1416,1468,1835,2014` (per-codec dispatch arms)
- `proxima-model-interop/src/bind.rs:1059,1141,1205,1670`, `capability.rs:312,396` (codec admission and reject sites); `dtype.rs:34` (`dtype_to_ggml`), `transform.rs:108` (the `UnrepresentableDType` site)
- `omega/src/msl/kernel_types_identity.rs:2073` (`tiled_decode`, exhaustive, `TiledDecode::new`); `omega/src/msl/packed_row_blocked_ggml.rs:580-610` (Q8_0 arm) and `:665` (Q4_0 arm)
- `proxima-model-interop/src/transform.rs:1-40` (the writer path the converter uses)
- `proxima-model-interop/examples/decode_arms.rs:137-165` (the flags), `:465-481` and `:508-524` (the bound lines); `proxima-model-interop/examples/model_config_load.rs:10` (the granite env override)
- `proxima-model-interop/tests/fixtures/llama-parity/granite_moe/{llama_ids,long_prompt_llama_ids}.json` (3 cases of 32 generated ids, 1 of 128) and `gemma4_e2b/` (cases of 3, 32, 1 and 128): the token oracle
- OCP Microscaling Formats Specification v1.0: `https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf`, local copy `.long_ctx_backups/mlx_fp6/ocp/ocp-mx-v1.0.pdf`, text `ocp-mx-v1.0.txt`; `.long_ctx_backups/mlx_fp6/facts.md` records that neither MLX nor the read portion of ggml's enum has an fp6 type
- `proxima-tensor/specs/decode-prefill-parity/SPEC.md` rounds two to five (the matvec in-stream GB/s, the bench protocol and bound rule)
