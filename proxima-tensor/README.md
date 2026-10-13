# proxima-tensor

Tensor vocabulary for proxima, reified as data: a program is a plain `Vec<Op>` built from three op forms (`Input`, `Elementwise`, `Reduce`) over affine index maps, with structural validation, shape inference and bound addressing read off the program.

There is no `Tensor` type here, the same way there is no `Http` type in `proxima-http`. This is the tensor execution and compilation substrate, not a model-family registry or a training framework.

## What it composes

- `proxima-primitives` (the `Pipe` algebra that the op tree reifies)
- `proxima-gguf`, `proxima-clock`, `proxima-telemetry`
- `proxima-prime` (optional, behind the `tensor-bgpool` / `tensor-cohort` features)

## Tiers and features

`alloc` (`no_std` + alloc), `std` (default), `config` (conflaguration-backed settings) and `instrument`. Sizing constants come from `proxima-tensor-runtime.toml` via `build.rs`. Fusion and backend-specific switches are listed under `[features]` in `Cargo.toml`; all are additive.

## License

Licensed under MIT OR Apache-2.0, at your option.

Repository: <https://github.com/brianbruggeman/proxima>
