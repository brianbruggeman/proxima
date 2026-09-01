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
| Q4_0 | dense | cpu | unimplemented -- no encoder or decoder in proxima_gguf::quant (only q4_k/q5_k/q6_k/q8_0 exist); bind::gguf_tensor_as_f32 rejects it with UnrepresentableGgmlType before a forward pass can run |
| Q5_0 | dense | cpu | unimplemented -- no encoder or decoder in proxima_gguf::quant (only q4_k/q5_k/q6_k/q8_0 exist); bind::gguf_tensor_as_f32 rejects it with UnrepresentableGgmlType before a forward pass can run |
| Q2_K | dense | cpu | unimplemented -- no encoder or decoder in proxima_gguf::quant (only q4_k/q5_k/q6_k/q8_0 exist); bind::gguf_tensor_as_f32 rejects it with UnrepresentableGgmlType before a forward pass can run |
| Q3_K | dense | cpu | unimplemented -- no encoder or decoder in proxima_gguf::quant (only q4_k/q5_k/q6_k/q8_0 exist); bind::gguf_tensor_as_f32 rejects it with UnrepresentableGgmlType before a forward pass can run |
| F16 | dense | cpu | unimplemented -- proxima_tensor::cpu::evaluate_quantized_named_with_scratch is f32-only: reject_non_float32 (proxima-tensor/src/cpu.rs) rejects any non-Float32 elementwise node outright |
| F32 | moe | cpu | supported |
| F32 | dense | metal | supported |
| Q8_0 | dense | metal | supported |
| Q4_K | dense | metal | supported |
| Q5_K | dense | metal | supported |
| Q6_K | dense | metal | supported |

## Quantized packed-format coverage

| packed codec | cpu kernel | metal emitter | wgsl emitter | cuda emitter |
| --- | --- | --- | --- | --- |
| Q4_K | supported | supported | supported | supported |
| Q5_K | supported | supported | supported | supported |
| Q6_K | supported | supported | supported | supported |
| Q8_0 | supported | supported | supported | supported |
| Q4_0 | supported | supported | supported | supported |
| F16 | supported | supported | supported | supported |
| BF16 | supported | supported | supported | supported |
