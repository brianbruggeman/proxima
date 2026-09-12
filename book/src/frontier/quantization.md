# Quantization frontier

The runtime treats a GGUF tensor type as a format identity, not as a model
identity. `proxima-gguf::GgmlType::CURRENT` is the single inventory of the
current ggml wire types, including the newer `MXFP4`, `NVFP4`, `Q1_0`, and
`Q2_0` tags. `block_layout`, `to_wire`, `from_wire`, `name`, and
`is_quantized` are format algebra; backend support is a separate capability
question.

There are four deliberately separate layers:

1. **Wire algebra** — parse, size, validate, and round-trip every registered
   type without assuming a particular model family.
2. **Decode algebra** — map a block to logical values. The CPU owned path is
   the correctness fallback; a packed path is an optimization.
3. **Execution algebra** — `QuantizedBlock`/`PackedCodec` lower the same
   logical matmul into CPU, CUDA, Vulkan/WGSL, or Metal kernels.
4. **Transform algebra** — transpose, split, recode, fuse, and persist bytes
   while preserving the declared type and shape. A transform must either
   prove its layout rule or return a typed error.

This boundary is important for future formats: adding a wire tag is safe and
auditable, but it does not silently claim a decoder or fused kernel. The
capability matrix is where that proof is recorded. The target end state is a
single generic dispatch table from `GgmlType` to decode/pack/fuse/transform
operations, with scalar CPU execution as the oracle and backend kernels
proving parity against it.

## Current implementation boundary

The existing CPU codecs and packed variants cover the K/IQ formats already
landed in `proxima-gguf`/`proxima-tensor`; the binder now routes the available
`Q1_0`, `Q2_0`, `Q4_1`, `Q2_K`, `Q4_0`, `Q5_1`, `IQ2_XS`, `IQ3_XXS`, and
`IQ4_NL`, `Q8_1`, and `Q8_K` decoders through the same owned-or-packed seam. The new scalar
decoders are correctness fallbacks; they do not yet imply a packed GPU kernel.
`MXFP4`, `NVFP4`, and the remaining IQ/TQ formats are registered at the wire
boundary but must not be reported as executable until their decoder and parity
tests land.

Serialized prefill is correctness-gated separately: a plan cache key includes
the requested output set, because a cache-tap-only chunk and a final logits
chunk can have identical tensor dimensions but different live graph roots.
