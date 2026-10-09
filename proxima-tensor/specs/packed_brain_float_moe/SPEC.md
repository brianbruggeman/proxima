# Packed brain-float MoE forward

status: admitted
owner: brian
created: 2026-10-09

## problem

The current BF4/BF8 pilot trains a 32-parameter synthetic MoE by expanding each
rounded master tensor into an FP32 Vec before the forward graph, so it does not
exercise packed weight storage or Proxima's quantized expert matmul path.

## refutation condition

If Proxima's existing expert matmul cannot consume BF4 E2M1 or BF8 E5M2 packed
bytes while preserving the declared scalar encoding and FP32 accumulation,
then extending that path is the wrong route and the representation must be
redesigned before adding a GPU training path.

## requirements

| id | requirement | testable in isolation |
|---|---|---|
| R1 | Add BF8 E5M2 and BF4 E2M1 to Proxima's packed expert-weight execution path without routing these scalar encodings through GGUF MXFP4/NVFP4 block semantics. | yes |
| R2 | CPU expert matmul must decode directly from retained packed bytes, accumulate products in FP32, and match an independent FP32 reference over a fixed payload including positive and negative values and both expert rows. | yes |
| R3 | The existing computed expert gather must select the requested packed expert row from caller-owned bytes and pass that row's packed slice to the matmul kernel. | yes |

## architecture

Add BF8 E5M2 as Codec tag 29 and BF4 E2M1 as Codec tag 30 to the existing
source-neutral proxima_primitives::Codec identity. Codec tags serialize this
project's sidecar identity; they are not GGML type numbers. Keep the byte
contracts in proxima-gguf as scalar decode oracles. Do not add either encoding
to proxima_gguf::GgmlType or its wire map. Keep the GGUF adapter's
GgmlType-based constructor unchanged: Codec tag 29/30 is not interpreted as a
GGML wire tag. Change the Codec-to-GgmlType conversion used for GGUF recoding
to return a typed unsupported result for these non-GGUF identities.

Represent a packed table entry as the existing ExpertEntry whose block is
QuantizedBlock::Packed { codec, bytes }, with its existing out_dim/in_dim
shape. For BF8, block_layout is (1 byte, 1 element); for BF4 it is (1 byte,
2 elements), with the first value in the low nibble. Compute these scalar
layouts directly from Codec rather than calling GgmlType::block_layout.
ExpertSource::entry resolves the route to the caller-owned byte span.
QuantizedBlock::matmul_f32_kernel dispatches the matching CPU kernel, and
run_reduce_quantized passes that selected slice directly to it. Under the
existing default-off instrument feature, emit a structured debug record at
the dispatch callsite with the selected expert and exact packed byte slice;
the integration test captures this event while evaluating a Computed gather.
The kernel decodes each operand as it is consumed and accumulates into FP32
outputs.
Update every exhaustive Codec match; non-CPU backends return their existing
typed unsupported-codec error for these two formats in this phase.

This shape abandons the pilot's per-forward FP32 Vec expansion: that path
cannot exercise packed-byte reads or establish storage and bandwidth behavior.
The CPU path becomes the reference oracle for a later GPU and autograd
integration.

### decisions

| decision | chosen | why not the alternative |
|---|---|---|
| Scalar encoding identity | preserve BF8 E5M2 and BF4 E2M1 as the selected formats | a GGUF type name would imply a file-format contract not defined by these scalar codecs |
| First native execution backend | existing CPU quantized expert matmul | it provides the directly inspectable reference path required before a GPU parity claim |
| Matmul output accumulation | FP32 | this isolates packed weight reads while preserving FP32 activation and accumulator semantics |
| Dequantization | decode values as the matmul consumes bytes | a whole-table FP32 expansion erases the packed-storage boundary this phase measures |

## acceptance criteria

| id | discharges | command | expected |
|---|---|---|---|
| AC1 | R1 | cargo test -p proxima-primitives --lib codec_brain_float_tags_roundtrip -- --exact && cargo test -p proxima-gguf --test brain_float_codecs_not_ggml brain_float_codecs_do_not_change_ggml_wire_types -- --exact && cargo test -p proxima-model-interop --lib brain_float_codecs_have_no_ggml_target -- --exact | 3 tests passed, 0 failed; BF8 E5M2 maps to Codec tag 29 and BF4 E2M1 maps to tag 30, both round-trip; GGML wire 29 remains IQ1_M and 30 remains BF16; Codec-to-GgmlType conversion rejects both new source-neutral identities |
| AC2 | R1-R2 | cargo test -p proxima-tensor --test packed_brain_float_moe_matmul bf_packed_moe_matmul_matches_fp32_oracle -- --exact | 1 passed, 0 failed; row-major [2,2,2] weights e0=[1,0.5,-1,2], e1=[-2,1.5,0.5,-0.5] are encoded as BF8 bytes [0x3c,0x38,0xbc,0x40,0xc0,0x3e,0x38,0xb8] and BF4 bytes [0x12,0x4a,0x3c,0x91]; input [2,-1] yields [1.5,-4,-5.5,1.5] by hand multiplication |
| AC3 | R2-R3 | cargo test -p proxima-tensor --features instrument --test packed_brain_float_moe_matmul packed_bf_computed_gather_passes_selected_bytes -- --exact | 1 passed, 0 failed; one Computed-map evaluation with route [1] returns [-5.5,1.5] and its captured dispatch event contains the exact selected slice: BF8 offset 4 length 4 bytes [0xc0,0x3e,0x38,0xb8], BF4 offset 2 length 2 bytes [0x3c,0x91] |
## out of scope

- GPU BF4/BF8 kernels and GPU execution of gathered expert gradients
- training a model above the tiny fixture or claiming near-1B quality, throughput, or cost
- changing BF4/BF8 encodings, adding block scaling, or using GGUF MXFP4/NVFP4 as substitutes
- optimizer state, gradients, activations, or router logits below FP32
- checkpoint publishing or model release

## risks

| risk | likelihood | what it costs | what we do about it |
|---|---|---|---|
| the packed expert API assumes GGUF block geometry | medium | scalar formats may require a parallel and drifting representation | inspect all packed-block consumers and extend the shared seam only if its invariants permit scalar elements |
| a scalar decoder is called per multiply and obscures the hot loop | medium | unnecessary dispatch can erase the benefit of packed reads | dispatch once per format and measure the existing CPU matmul path before optimization |
| the first CPU kernel establishes storage but not scale performance | high | CPU reference could be mistaken for GPU feasibility | retain backend identity in every record and keep GPU support out of this phase's claims |

## context

- proxima-tensor/specs/low_precision_moe_training/SPEC.md
- proxima-tensor/specs/low_precision_moe_training/results/tiny-pilot.json
- proxima-autograd/src/low_precision.rs
- proxima-autograd/src/adjoint.rs
- proxima-tensor/src/cpu/epilogue.rs
- proxima-tensor/src/cpu/run_reduce_scan.rs
- proxima-tensor/src/cpu/quantized_eval.rs
- proxima-primitives/src/codec.rs
- omega/src/msl/packed_row_blocked_ggml.rs
- omega/tests/training_step_parity.rs
- proxima-gguf/src/quant/bf8_e5m2.rs
- proxima-gguf/src/quant/bf4_e2m1.rs
