# proxima-model-interop compatibility matrix

Generated from `src/capability.rs`'s own tables by `examples/generate_compatibility_doc.rs` -- do not hand-edit. Regenerate with `cargo run -p proxima-model-interop --example generate_compatibility_doc --features metal` after any change to `src/capability.rs` or `tests/capability_matrix.rs`'s cell set. `tests/capability_doc_drift.rs` fails the build if this file falls out of sync.

## GGML codec x topology x backend

| codec | topology | backend | status |
| --- | --- | --- | --- |
| F32 | dense | cpu | supported |
| Q8_0 | dense | cpu | supported |
| Q4_K | dense | cpu | supported |
| Q5_K | dense | cpu | supported |
| Q6_K | dense | cpu | supported |
| Q4_0 | dense | cpu | unimplemented -- no encoder or decoder in proxima_gguf::quant (only q3_k/q4_k/q5_k/q6_k/q8_0 exist); bind::gguf_tensor_as_f32 rejects it with UnrepresentableGgmlType before a forward pass can run |
| Q5_0 | dense | cpu | unimplemented -- no encoder or decoder in proxima_gguf::quant (only q3_k/q4_k/q5_k/q6_k/q8_0 exist); bind::gguf_tensor_as_f32 rejects it with UnrepresentableGgmlType before a forward pass can run |
| Q2_K | dense | cpu | unimplemented -- no encoder or decoder in proxima_gguf::quant (only q3_k/q4_k/q5_k/q6_k/q8_0 exist); bind::gguf_tensor_as_f32 rejects it with UnrepresentableGgmlType before a forward pass can run |
| Q3_K | dense | cpu | supported |
| F16 | dense | cpu | unimplemented -- proxima_tensor::cpu::evaluate_quantized_named_with_scratch is f32-only: reject_non_float32 (proxima-tensor/src/cpu.rs) rejects any non-Float32 elementwise node outright |
| F32 | moe | cpu | supported |
| F32 | dense | metal | supported |
| Q8_0 | dense | metal | supported |
| Q4_K | dense | metal | supported |
| Q5_K | dense | metal | supported |
| Q6_K | dense | metal | supported |
| Q3_K | dense | metal | supported |

## Quantized packed-format coverage

| packed codec | cpu kernel |
| --- | --- |
| Q4_K | supported |
| Q5_K | supported |
| Q6_K | supported |
| Q8_0 | supported |
| Q3_K | supported |
| Q4_0 | supported |
| F16 | supported |
| BF16 | supported |
| Q2_K | supported |
| Q5_1 | supported |
| Q5_0 | supported |
| Q4_1 | unsupported |
| Q8_1 | unsupported |
| Q8_K | unsupported |
| IQ1_S | unsupported |
| IQ1_M | unsupported |
| IQ2_XXS | unsupported |
| IQ2_XS | supported |
| IQ2_S | unsupported |
| IQ3_XXS | supported |
| IQ3_S | unsupported |
| IQ4_NL | supported |
| IQ4_XS | unsupported |
| TQ1_0 | unsupported |
| TQ2_0 | unsupported |
| MXFP4 | unsupported |
| NVFP4 | unsupported |
| Q1_0 | unsupported |
| Q2_0 | unsupported |
