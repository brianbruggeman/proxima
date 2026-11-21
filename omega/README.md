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
- `train` (default-off): adds the training seam over `proxima-autograd` (graph construction plus a state-owning step) configured by one `TrainConfig`; inference builds never compile it
- backend and optimisation features (`cpu`, `metal`, `cuda`, `wgpu-backend`, `onnx-door`, `mil-writer`, ...) are listed under `[features]` in `Cargo.toml`; all are additive.
- `cuda` emits both card-C dialects from the one renderer: CUDA C for NVIDIA and HIP C++ for AMD ROCm, with the dialect and wavefront width selected per emit (`emit_hip_at_wave`: 64-lane CDNA, 32-lane RDNA); `cuda-driver` adds `cuda_driver::rocm`, which resolves `libamdhip64`/`libhiprtc` at run time.

## Training

`train` makes training an omega surface instead of test-only wiring. One `conflaguration`
object carries the engine, driver, numeric policy, optimizer and schedule:

```toml
optimizer = "adamw"

[runtime]
engine = "cpu"

[schedule]
epochs = 2
val_split = 0.05
batch = 2048
ubatch = 512

[adamw]
learning_rate = 1.0e-5
beta1 = 0.9
beta2 = 0.999
epsilon = 1.0e-8
weight_decay = 0.0
```

`TrainConfig::from_file` loads and validates it. `build_optimizer_step` turns a forward
program plus its scalar loss into the forward+backward+optimizer program, and
`TrainingProgram::plan` / `TrainingStep::step` own the parameter and moment buffers across
steps:

```rust,ignore
let config = TrainConfig::from_file("train.toml")?;
let built = build_optimizer_step(&program, loss, &["w1", "b1"], &config)?;
let mut step = built.plan(&named)?;
let loss = step.step(&[
    ("x", QuantizedBlock::Float32(&x)),
    ("y", QuantizedBlock::Float32(&y)),
    ("step", QuantizedBlock::Float32(&step_number)),
])?;
```

State names are `m_<parameter>` and `v_<parameter>`; the bias correction reads the rank-0
`step` input, which the caller binds every step. A gathered gradient comes back as a compact
host-side contribution — applying it stays the caller's job — so `build_optimizer_step`
covers the dense-gradient case and refuses a parameter with no dense gradient instead of
silently skipping it. `benches/llama-finetune.toml` is the benchmark configuration, and its
values are llama.cpp's finetune defaults at tag `b11671`.

## License

Licensed under MIT OR Apache-2.0, at your option.

Repository: <https://github.com/brianbruggeman/proxima>
