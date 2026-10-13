# proxima-autograd

Reverse-mode autodiff over a `proxima-tensor` program. There is no `Tensor` value type and no `Optimizer` trait: the graph already is the value, and optimizer updates are elementwise expressions over existing `Op::Input` leaves.

## Pieces

- `adjoint::differentiate`: the adjoint transform, `&[Op] -> Differentiated`
- `activation` (relu, softmax, sigmoid, silu, gelu), `conv::conv2d`, `loss`, `norm`: graph-building functions composing existing `ScalarOp`s
- `optimizer` (adam, sgd, rmsprop, adamw): parameter updates as elementwise expressions
- `train::train_step` / `train::fit`: host-buffer bookkeeping for a training loop (`std` only)

## What it composes

`proxima-tensor`, `proxima-primitives`, `proxima-gguf`, `proxima-telemetry`; `proxima-safetensors` behind the `safetensors` feature.

## Features

`alloc`, `std` (default), `config`, `instrument`, `safetensors`; bench-only features are listed under `[features]` in `Cargo.toml`.

## License

Licensed under MIT OR Apache-2.0, at your option.

Repository: <https://github.com/brianbruggeman/proxima>
