# Omega 00: hello, tensor

**Prerequisites:** you know Rust and you have used PyTorch or burn. Nothing about proxima.

**You will:** run a tensor program written in a TOML file, with no Rust at all; run the same plan on the GPU; then run the same three operations from Rust and print the device that served them.

**New concepts (in order):** a program as a file · node, id, dtype, extent, map · `proxima tensor run` · omega, and the two-level backend choice (`engine`, then driver) · the runtime config as environment and as a TOML file · `RuntimeConfigBuilder` · the same program as a `Vec<Op>`.

## 1. Run it

Here is a complete tensor program. It builds `[0, 1, 2, 3]`, builds four copies of `10.0`, adds them, and names the result `sum`. The file is `omega/examples/hello_add.plan.toml`:

```toml
version = 1

[[output]]
name = "sum"
node = "n2"

[[node]]
op = "iota"
id = "n0"
dtype = "float32"
extent = 4

[[node]]
op = "constant"
id = "n1"
dtype = "float32"
shape = [4]
value = 10.0

[[node]]
op = "elementwise"
id = "n2"
dtype = "float32"
body = "add"
inputs = ["n0", "n1"]
maps = ["i->i", "i->i"]
name = "sum"
```

Four things to notice. Each `[[node]]` is one operation with an `id` of your choosing. `n0` is the range `0..4` and `n1` is a constant, so this program needs no input data. `n2` adds them, and its `maps` say how each operand lines up with the output — `i->i` is "same axis, same element". `[[output]]` says the caller reads `n2` under the name `sum`.

Two commands run it. The first starts a daemon, which is the process that evaluates plans; it prints `READY` and the socket it is listening on:

```sh
cargo build -p proxima-cli --bins
./target/debug/proximad serve --unix /tmp/proximad.sock --state-dir /tmp/proximad-state
```

```text
READY /tmp/proximad.sock
```

The second hands the plan to it:

```sh
./target/debug/proxima tensor run --socket /tmp/proximad.sock omega/examples/hello_add.plan.toml
```

```text
sum [4] [10.0, 11.0, 12.0, 13.0]
```

That is the whole first program. No Rust of your own, and no input files. If the TOML is wrong the daemon says which field is wrong (`tools/proxima-cli/src/tensor_run.rs:53`), and the printed line comes from `summarize_values` (`tools/proxima-cli/src/summary.rs:17`), which shows the first eight values and the total count for a big tensor.

The format is a `ProgramSpec` (`proxima-tensor/src/spec/program_spec.rs:26`): five node kinds, namely an `input` leaf, an `elementwise` map, a `reduce` reduction, an `iota` range and a `constant` (`program_spec.rs:130` to `:171`) — shapes written as a number or a symbolic `?0` (`ExtentSpec`, `:87`), one addressing string per operand (`MapSpec`, `:111`), and `[[output]]` entries (`OutputSpec`, `:65`). A node may only reference ids above it, which is why `n2` can name `n0` and `n1` and not the other way round.

## 2. What omega is

Omega is the crate `proxima-omega` (library name `omega`, `omega/Cargo.toml:2` and `:13`). It plans a tensor program for a backend and executes it. There is no tensor object and no operators on values: a program is data, and where it runs is a config value.

The backend choice has two levels, both plain enums in `omega/src/backend.rs`:

- `Engine` (`backend.rs:113`) is where the work happens: `Cpu` or `Gpu`. There are only two, because there are only two places an op runs.
- `GpuDriver` (`backend.rs:187`) is which API renders the GPU: `Metal`, `Wgpu`, and with the `cuda-driver` feature, `Cuda` and `Hip`.

If you configured `cpu`, the driver is ignored (`backend.rs:643`). A `gpu` build with no driver compiled in fails with `BackendError::NoGpuDriver` (`backend.rs:323`): the engine you asked for is the engine that runs. The default features are `["std", "metal", "cpu"]` (`omega/Cargo.toml:21`), so a default build on a Mac has the Metal driver and a default build on Linux has none.

The PyTorch comparison: omega is the layer under `torch.compile` plus the device dispatcher. A `torch.Tensor` is one level above that, carrying its operators with it. In omega you describe the computation once and choose the device separately, which is the point of the whole series.

## 3. Put it on the GPU

On the daemon path the runtime is the environment, read per request (`tools/proxima-cli/src/tensor_run.rs:59`). Two variables move the work to Metal, and the driver has to be compiled into the daemon:

```sh
cargo build -p proxima-cli --bins --features metal
OMEGA_ENGINE=gpu OMEGA_GPU_DRIVER=metal ./target/debug/proximad serve --unix /tmp/proximad-gpu.sock --state-dir /tmp/proximad-state
```

```text
READY /tmp/proximad-gpu.sock
```

```sh
./target/debug/proxima tensor run --socket /tmp/proximad-gpu.sock omega/examples/hello_add.plan.toml
```

```text
sum [4] [10.0, 11.0, 12.0, 13.0]
```

Same plan, same numbers. Leave out `--features metal` and the same two variables get you a refusal that names the missing feature rather than a silent CPU fall back (`tools/proxima-cli/Cargo.toml:28`):

```text
proxima tensor run: upstream error: tensor run failed: HTTP 400 — omega could not plan the program: backend `metal` needs the `metal` cargo feature, which is not compiled in
```

This path does not tell you which device served the request, and it has no runtime-TOML flag yet; both are in [section 6](#6-what-omega-does-not-do-yet). The embedded path in section 5 does tell you, which is the reason to use it.

## 4. The runtime config as a file

When you embed omega, the runtime is a `RuntimeConfig` (`omega/src/config.rs:22`), and it is both a file and a builder. The file is two lines, `omega/examples/omega_hello_tensor.toml:1-2`:

```toml
engine = "cpu"
gpu_driver = "auto"
```

`RuntimeConfig::from_file` (`config.rs:389`) reads TOML, JSON or YAML, overlays it on the defaults, and validates. `deny_unknown_fields` (`config.rs:21`) turns a typo into an error that lists every key it expected, and validation (`config.rs:298`, `:310`) names the field that is wrong:

```text
Error: validation failed:
  - engine: unknown engine 'tpu'; expected cpu|gpu
```

The other 26 fields — worker counts, fusion, cache policy, numeric permissions, candidate selection — have defaults and are [page 01](./01-config.md)'s subject. The two on this page are `engine` (`config.rs:25`) and `gpu_driver` (`config.rs:28`).

The builder produces the same value: `RuntimeConfig::builder()` (`config.rs:397`) returns a `RuntimeConfigBuilder` (`:716`), `with_engine` (`:841`) and `with_gpu_driver` (`:848`) set the fields, and `build()` (`:938`) validates. The repository proves the two forms agree with a test (`omega/src/config_tests.rs:329-340`), which is short enough to read:

```rust
fn runtime_config_builder_matches_serde_config() {
    let source = "engine = 'gpu'\ngpu_driver = 'metal'\n[constraints]\nmax_device_buffer_bytes = 4096\n[numeric]\nreassociation = true\n";
    let from_serde: RuntimeConfig = toml::from_str(source).expect("config TOML deserializes");
    let from_builder = RuntimeConfig::builder()
        .with_engine("gpu")
        .with_gpu_driver("metal")
        .with_constraints(ResourceConstraints::default().with_max_device_buffer_bytes(Some(4096)))
        .with_numeric(NumericPolicy::bit_exact().with_reassociation(true))
        .build()
        .expect("builder config validates");

    assert_eq!(from_builder, from_serde);
}
```

Use the file when whoever deploys the program should choose the backend without a rebuild, and the builder when a test or a caller computes the value. [Page 01](./01-config.md) covers what happens when both, plus the environment, are in play at once.

## 5. Embed it in your program

The same three operations in Rust, printing the device this time. The complete program is `omega/examples/omega_hello_tensor.rs`:

```rust
use anyhow::Result;
use omega::backend::{execute_plan_named, plan_named_with_config};
use omega::config::RuntimeConfig;
use proxima_numeric::DType;
use proxima_tensor::map::{self, IndexMap};
use proxima_tensor::{Extent, NodeId, Op, QuantizedBlock, ScalarOp};

fn main() -> Result<()> {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        concat!(env!("CARGO_MANIFEST_DIR"), "/examples/omega_hello_tensor.toml").into()
    });
    let config = RuntimeConfig::from_file(&path)?;

    let input = |name: &str| Op::Input {
        dtype: DType::FLOAT32,
        shape: vec![Extent::Static(4)],
        name: Some(name.into()),
    };
    let identity = || IndexMap::Affine(map::projection(1, &[0]));
    let program = vec![
        input("a"),
        input("b"),
        Op::Elementwise {
            dtype: DType::FLOAT32,
            body: ScalarOp::Add,
            operands: vec![(NodeId(0), identity()), (NodeId(1), identity())],
            name: None,
        },
    ];
    let left = [1.0_f32, 2.0, 3.0, 4.0];
    let right = [10.0_f32, 20.0, 30.0, 40.0];
    let named = [
        ("a", QuantizedBlock::Float32(&left)),
        ("b", QuantizedBlock::Float32(&right)),
    ];

    let (mut plan, report) = plan_named_with_config(&config, &program, &[], &named, &[NodeId(2)])?;
    let result = execute_plan_named(&mut plan, &named)?;

    println!("backend = {} (config: {path})", report[0].device_identity);
    println!("shape   = {:?}", result.shape());
    println!("a + b   = {:?}", result.root());
    Ok(())
}
```

It builds the `Vec<Op>` by hand on purpose: the three nodes are the ones in section 1, so you can see both forms side by side. The lowering that turns a `ProgramSpec` into that vector is `plan_parts` (`proxima-tensor/src/spec/run.rs:34`), and [page 02](./02-programs-as-data.md) uses it so you never write the vector yourself.

The four steps, in the order the code runs them:

1. **Load the config** — `RuntimeConfig::from_file` (`config.rs:389`).
2. **Describe the program.** Nodes 0 and 1 are `Input`s named `"a"` and `"b"`, four float32 elements each (`Extent::Static(4)`, `op.rs:46`); node 2 adds them. `NodeId` (`op.rs:29`) is a newtype over `u32`, so a node id cannot be confused with any other number.
3. **Bind data by name.** `named` pairs each input name with a `QuantizedBlock` (`proxima-tensor/src/cpu/epilogue.rs:1459`). `Float32` is the plain case; the other variants carry quantized weights, which later pages use. Binding by name rather than position is what lets weights survive a partition that renumbers slices (`Op::Input`'s doc, `op.rs:195-201`).
4. **Plan, then execute.** `plan_named_with_config` (`backend.rs:490`) validates the config, builds a `Plan` (`backend.rs:409`) and reports what it chose (`CandidateMeasurement`, `backend.rs:425`). With the default selection mode there is one candidate and no trial runs (page 01 explains when there could be more). `execute_plan_named` (`backend.rs:1303`) runs it; the same plan takes new data on the next call, which is how a serving loop avoids re-planning per token. The result is an `Evaluated` (`proxima-tensor/src/cpu/arena.rs:20`); `.root()` (`:65`) is the buffer and `.shape()` (`:70`) its shape.

Run it on the CPU config from section 4:

```sh
cargo run -p proxima-omega --example omega_hello_tensor
```

```text
backend = cpu (config: /Users/brianbruggeman/repos/slot-0/proxima-card-PROXIMAOMEGA-63/omega/examples/omega_hello_tensor.toml)
shape   = [4]
a + b   = [11.0, 22.0, 33.0, 44.0]
```

Now put `engine = "gpu"` and `gpu_driver = "metal"` in `omega/examples/configs/gpu.toml` and pass it as the argument. Same binary, no rebuild:

```sh
cargo run -p proxima-omega --example omega_hello_tensor -- omega/examples/configs/gpu.toml
```

```text
backend = Apple M1 Max (config: omega/examples/configs/gpu.toml)
shape   = [4]
a + b   = [11.0, 22.0, 33.0, 44.0]
```

Same program, same numbers, different device, and the `report` line names it. A deliberately bad config shows the field name and exits `1`:

```sh
cargo run -p proxima-omega --example omega_hello_tensor -- omega/examples/configs/bad_engine.toml
```

```text
Error: validation failed:
  - engine: unknown engine 'tpu'; expected cpu|gpu
```

The doctest on `plan_named_with_config` (`backend.rs:468-487`) is the smallest executable statement of the contract: it plans a one-op `Negate` program and asserts `[-1.0, -2.0]`. Doctests run under `cargo test`, not `cargo val`.

## 6. What omega does not do yet

- The daemon path has no runtime-TOML surface. `proxima tensor run` evaluates with `RuntimeConfig::from_env()` per request (`tools/proxima-cli/src/tensor_run.rs:59`). There is no `--runtime <toml>` on `proximad serve`, and the CLI does not report the device that served a plan.
- No tensor object and no operators. There is no `a + b` on a value: you write a `ProgramSpec` or build a `Vec<Op>` and index nodes by `NodeId` (`proxima-tensor/src/op.rs:192`, `:29`). [Page 02](./02-programs-as-data.md) is about the first form.
- Two engines, five drivers, and no more. `Engine` is `Cpu` or `Gpu` (`backend.rs:113`) and the driver is one of `auto|metal|wgpu|cuda|hip|rocm` (`config.rs:310`). There is no `mlx`, `coreml`, `ane` or `npu` value; the `Plan` enum has Cpu, Metal, Wgpu, Cuda and Rocm variants only (`backend.rs:409-421`), and an `Npu`/`Ane` name existed without a lowering and was deleted (`backend.rs:105-111`).
- MLX and Core ML live outside the engine enum. `omega/src/mlx.rs:1-5` says execution "remains behind a native adapter", and the module exists only when `PROXIMA_MLX_PREFIX` points at an installed MLX (`omega/build.rs:25-37`). Core ML is a package writer behind `coreml-lane` (`omega/Cargo.toml:499`).
- Which drivers you have is a compile-time fact. wgpu and CUDA need features that are off by default (`omega/Cargo.toml:382`, `:241`), and asking for `gpu` without one is `NoGpuDriver` (`backend.rs:323`).
- Float32 in, float32 out on this path. Input blocks are `QuantizedBlock` variants (`epilogue.rs:1459`) and `Evaluated` holds `Vec<f32>` (`arena.rs:22`). The `numerics` setting that would choose storage and accumulator formats is recorded in the config and not read by lowering yet (`config.rs:102`).
- No autograd here. Nothing in `backend.rs` or `config.rs` differentiates a program. `proxima-autograd` is a separate crate, used here only as a dev-dependency (`omega/Cargo.toml:710`); [page 03](./03-train.md) is where training happens.

Next: [01: configuration in depth](./01-config.md) — the rest of the config surface, how the four sources layer, and the errors that name your field.
