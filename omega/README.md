# proxima-omega

Plan and execute tensor programs on a CPU and on a GPU. A program is data — an ordered list of operations over named buffers — and a configuration picks the engine, the driver and the execution policy.

```text
omega::backend::plan_named_with_config   config + program + bound data -> Plan + measurements
omega::backend::execute_plan_named       Plan + bound data             -> Evaluated
```

`Engine` (`omega/src/backend.rs:113`) has two variants, `Cpu` and `Gpu`, because there are two places an op runs. Which driver renders the GPU is a second question: `GpuDriver` (`omega/src/backend.rs:187`) is `Metal`, `Wgpu`, or, with the `cuda-driver` feature, `Cuda` and `Hip`. The engine and driver are `RuntimeConfig` fields (`omega/src/config.rs:25` and `:28`), so moving a program from CPU to Metal is a config edit, not a code change.

Programs come from `proxima-tensor`: either a `Vec<Op>` you build, or a `ProgramSpec` TOML that `proxima_tensor::spec::ProgramSpec::plan_parts` lowers to the same vector. The CPU evaluator is `proxima_tensor::cpu`; the GPU side emits Metal Shading Language or CUDA C / HIP C++ from the same bound descriptor and drives it through the platform API.

Teaching material, in order: [`docs/omega/`](../docs/omega/README.md).

## What it composes

- `proxima-tensor` — the bound-op descriptor, the shape inference, and the CPU evaluator the GPU path mirrors
- `proxima-numeric` — dtypes and the numeric-policy permissions the config carries
- `proxima-primitives` (the `Pipe` algebra), `proxima-telemetry`, `proxima-gguf`
- optional doors: `proxima-onnx`, `proxima-mil`

## Tiers and features

- bare (`no_std`, no alloc): `error` and the build-time `sized` constants only
- `alloc` (`no_std` + alloc): adds the MSL emitter (`msl`)
- `std` (default): adds `std::error::Error`, `config`, and the backend-agnostic `backend` plan/execute pair
- `train` (default-off): adds [`train`], a forward+backward+optimizer step over `proxima-autograd` configured by one `TrainConfig`; inference builds never compile it
- engine and driver features are additive and listed under `[features]` in `Cargo.toml`: `cpu` (`:84`), `metal` (`:188`, in the default set, effective on macOS), `wgpu-backend` (`:382`), `cuda` (`:234`, emits both card-C dialects: CUDA C for NVIDIA, HIP C++ for AMD ROCm, with the dialect and wavefront width chosen per emit via `cuda::emit_hip_at_wave` at `omega/src/cuda.rs:6645`), `cuda-driver` (`:241`, resolves `libamdhip64`/`libhiprtc` at run time and adds `cuda_driver::rocm`)
- `coreml-lane` writes a Core ML package, compiles it with `xcrun coremlcompiler` and caches it by package sha256; it is a package writer, not a selectable engine
- `mlx-gdn` compiles against an MLX installation named by `PROXIMA_MLX_PREFIX`; `mlx` exposes that capability to a native adapter. Neither is a `RuntimeConfig` value.

## What it does not do

- No tensor object and no operators: a program is data, and there is no `a + b` on a value.
- `numerics` (storage, activation and accumulator formats) is recorded in the config and not read by lowering yet (`omega/src/config.rs:102`).
- No autograd here: differentiation lives in `proxima-autograd`; the `train` feature adds a step-wise seam over it, and there is no differentiation of an arbitrary graph on this surface.

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
