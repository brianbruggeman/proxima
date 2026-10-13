# proxima-omega

MSL kernel emission from the `proxima-tensor` `BoundOp` descriptor: the emit half of the GPU backend. The device driver composes on top.

`proxima-tensor`'s `cpu` module interprets a `BoundOp` with nested loops; this crate emits Metal Shading Language source from the same descriptor instead. It stops at source text plus the buffer-index to data mapping a driver needs to dispatch it. Compiling the library, allocating device buffers and dispatching a compute pass are the driver's job.

## What it composes

- `proxima-tensor` (the `BoundOp` descriptor and the CPU evaluator it mirrors)
- `proxima-primitives` (the `Pipe` algebra), `proxima-telemetry`, `proxima-gguf`
- optional doors: `proxima-onnx`, `proxima-mil`

## Tiers and features

- bare (`no_std`, no alloc): error and build-time sizing constants only
- `alloc` (`no_std` + alloc): adds the MSL emitter
- `std` (default): adds `std::error::Error` and the backend-agnostic plan/execute pair
- backend and optimisation features (`cpu`, `metal`, `cuda`, `wgpu-backend`, `onnx-door`, `mil-writer`, ...) are listed under `[features]` in `Cargo.toml`; all are additive.

## License

Licensed under MIT OR Apache-2.0, at your option.

Repository: <https://github.com/brianbruggeman/proxima>
