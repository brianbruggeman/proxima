# proxima-model-interop

Sans-IO GGUF <-> safetensors transform over the one thing both formats agree on: a named tensor is `(name, dtype, shape, bytes)`. Everything beyond that (GGUF typed KV metadata, block-quantized types) is where the formats diverge; the doc comments on `transform::gguf_to_safetensors` and `transform::safetensors_to_gguf` state what each direction preserves and what it loses.

With `std`, `generate::LoadedModel` binds a checkpoint's weights once and runs greedy-decode text generation through `proxima_primitives::pipe::Pipe`. ONNX is out of scope here.

## What it composes

- `proxima-gguf`, `proxima-safetensors`, `proxima-tensor`, `proxima-primitives`, `proxima-core`
- optional: `proxima-omega`, `proxima-prime`, `proxima-autograd`, `proxima-storage`, `proxima-tokenizer`

## Features

`std` (default) plus backend and optimisation switches (`metal`, `cuda`, `vulkan`, `qat`, `moe-*`, ...); see `[features]` in `Cargo.toml`. All are additive.

## License

Licensed under MIT OR Apache-2.0, at your option.

Repository: <https://github.com/brianbruggeman/proxima>
