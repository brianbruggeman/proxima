# Metal packed brain-float MoE forward

status: admitted
owner: brian
created: 2026-10-09

## problem

Omega's Metal driver rejects Proxima's BF8 E5M2 and BF4 E2M1 packed expert
weights before execution, so the fixed computed-gather payload that produces
`[-4.5, -6]` on CPU cannot produce the same vector on Metal.

## refutation condition

If the existing Metal packed-operand and serial-reduce path cannot address
BF8 bytes one element at a time and BF4 low/high nibbles as elements 0/1 while
preserving the CPU fixture's selected-expert output, then this path does not
execute this fixed low-precision MoE payload on Metal.

## requirements

| id | requirement | testable in isolation |
|---|---|---|
| R1 | The Metal packed-operand admission accepts the already-declared source-neutral `Codec::Bf8E5M2` and `Codec::Bf4E2M1` identities and leaves `Codec::Iq4Nl` outside the supported set, where the device dtype gate returns `TensorError::NotLowerable`. | yes |
| R2 | MSL decodes BF8 E5M2 bytes and BF4 E2M1 nibbles directly from their packed buffers during the existing FP32 reduction path. Given expert base `b`, input feature `i`, output feature `o`, and `in_dim`, native packed element index is `p = b + o * in_dim + i`; BF8 reads byte `p`, while BF4 reads byte `p / 2` and selects low nibble when `p % 2 == 0`, high nibble otherwise. Rows and experts have no padding; odd-width rows share the boundary byte, so the next row begins in its high nibble. | yes |
| R3 | A computed route selecting expert 1 consumes the exact caller-owned packed span and Metal returns the same two FP32 values as CPU for the fixed fixture. | yes |

## architecture

Extend the existing Metal packed operand admission and generic `operand_read`
decoder for the two scalar codec identities. BF8 maps one byte to one element;
BF4 maps one byte to two elements, first low nibble then high nibble. MSL
decodes each selected scalar as it is consumed by the existing reduction body,
with FP32 activation and accumulation. Keep the generic serial reduction
path for this first kernel; do not broaden the GGML row-block classifier or
claim throughput from this small correctness fixture. The existing computed
gather supplies the route, and the normal `Binding::Indices` path selects
the packed element address. Compile BF kernels with the same existing Metal
language target unless compiler evidence requires a per-kernel target change.

The fixture uses two experts with `out_dim=2`, `in_dim=3`, route `[1]`, and
activation `[2,-1,2]`. The logical stack layout is `[expert,input_feature,output_feature]`;
the packed byte span follows GGUF's native `[expert,output_feature,input_feature]`
row layout, with three contiguous input weights per output row. BF4 has no row
or expert padding: expert 0 bytes are
`[0x12,0x4a,0x93]`, expert 1 bytes are `[0x3c,0x91,0xb4]`, and flattened bytes
are `[0x12,0x4a,0x93,0x3c,0x91,0xb4]`. Expert 1 row 1 starts at element 9,
the high nibble of byte `0x91`. BF8 expert bytes are
`[0x3c,0x38,0xbc,0x40,0x3e,0xb8]` and
`[0xc0,0x3e,0x38,0xb8,0x40,0xbe]`, respectively. Both codecs should produce
`[-4.5,-6]`. Expert 0 differs from expert 1 so an ignored route fails. The
CPU oracle decodes the same packed byte stack with the scalar BF codecs and
transposes each native `[out,in]` expert matrix into the graph's declared
`[input,output]` layout before evaluating the same route and activation. Metal
receives the original packed span and applies the native packed row layout
directly.

### decisions

| decision | chosen | why not the alternative |
|---|---|---|
| First MSL strategy | direct byte/nibble decode in the existing generic reduction | promoting the new codec into a GGML row-block geometry would mix scalar BF semantics with unrelated 256-element blocks |
| output arithmetic | FP32 accumulation | it matches the admitted CPU reference and isolates weight storage precision |
| performance conclusion | none in this phase | a two-output fixture does not measure a useful model-sized kernel |

## acceptance criteria

| id | discharges | command | expected |
|---|---|---|---|
| AC1 | R1-R2 | `cargo test -p omega --lib msl::tests::bf8_e5m2_operand_read_decodes_each_byte -- --exact && cargo test -p omega --lib msl::tests::bf4_e2m1_operand_read_selects_each_nibble -- --exact` | 2 tests pass, 0 fail; source shows BF8 byte addressing at native packed index `p` and BF4 low/high nibble selection at `p`, including odd-row boundary addressing |
| AC2 | R1-R3 | `test "$(uname -s)" = Darwin && cargo test -p omega --test packed_brain_float_moe_metal packed_computed_gather_matches_cpu -- --exact` | 1 test passes, 0 fails on macOS with an available Metal device; device run compiles both kernels, CPU and Metal return `[-4.5,-6]`, mismatch output retains codec, full byte stack, route, CPU vector, and Metal vector; source inspection in the test asserts corrected native packed strides `[0,1,3]`, a raw `device const uchar* in0` binding, direct BF decoder reads, and no expanded FP32 weight buffer |
| AC3 | R1 | `cargo test -p omega --lib metal::prepare_uniforms_pack::unsupported_packed_codec_tests::unsupported_packed_codec_stays_not_lowerable -- --exact` | 1 test passes, 0 fails; an `Iq4Nl` packed block with `DType::BFloat16`, shape `[32]`, and exactly 18 bytes is absent from `codec_from_quantized_block` admission, and `prepare_with_options` returns `MetalError::Tensor(TensorError::NotLowerable { node, .. })` before dispatch |

## out of scope

- GPU autograd or GPU optimizer updates for MoE training
- packed BF row-block/tiled GEMM optimization or throughput claims
- CUDA and WGSL BF4/BF8 support
- changing the E5M2/E2M1 byte contracts, GGUF wire IDs, or GGUF block-scaled formats
- training above the existing tiny CPU pilot or claiming near-1B quality, cost, or feasibility
- publishing or releasing weights

## risks

| risk | likelihood | what it costs | what we do about it |
|---|---|---|---|
| BF4 byte addressing accidentally uses one element per byte or pads each row | medium | odd/even feature reads select incorrect weights | assert flattened low/high selection and exercise the odd-width row boundary in device parity |
| route lowering reads expert 0 or a widened staging copy | medium | a passing shape-only test would not establish selected packed-byte execution | use distinct expert rows, fixed route `[1]`, and CPU/Metal output comparison on the same packed stack |
| generic serial reduction is not suitable for large matrices | high | correctness can be mistaken for practical training performance | keep all performance and scale claims out of this spec; later optimize only with model-shaped measurement |

## context

- `proxima-tensor/specs/packed_brain_float_moe/SPEC.md`
- `proxima-tensor/tests/packed_brain_float_moe_matmul.rs`
- `proxima-gguf/src/quant/bf8_e5m2.rs`
- `proxima-gguf/src/quant/bf4_e2m1.rs`
- `omega/src/msl/signature_tokens_prelude.rs`
- `omega/src/msl/kernel_types_identity.rs`
- `omega/src/msl/emit_and_classify.rs`
- `omega/src/metal/prepare_uniforms_pack.rs`
- `omega/src/metal/arena_encode_dispatch_finish.rs`
- `omega/tests/metal_parity.rs`
- `proxima-tensor/specs/metal_scatter_add/SPEC.md`
