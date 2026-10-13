# proxima-onnx

Sans-IO ONNX model parser and op lowering. Hand-rolled protobuf message decode over `proxima-protocols`' schema-agnostic wire walker, faithful to `onnx.proto`, plus a `NodeProto` to `proxima_tensor::Op` lowering that composes the existing closed op algebra.

The parser core operates on `&[u8]` and never performs IO. Field numbers and enum discriminants for the nine covered messages are sourced from `onnx.proto3` in the upstream ONNX repository.

## What it composes

- `proxima-protocols` (protobuf wire walker)
- `proxima-tensor` (the `Op` algebra the lowering targets)
- `proxima-primitives` (`Pipe`)

## Tiers and features

`alloc` is the floor (`no_std` + alloc); `std` (default) adds `std::error::Error`, forwards `proxima-protocols/std` and enables the conflaguration-backed `config` module. Diagnostic and bench features are listed under `[features]` in `Cargo.toml`.

## License

Licensed under MIT OR Apache-2.0, at your option.

Repository: <https://github.com/brianbruggeman/proxima>
