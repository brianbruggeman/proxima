# Omega 00: hello, tensor

**Prerequisites:** you read Rust (`struct`, `enum`, `Result`, `?`) and have used PyTorch or burn. Nothing about proxima or omega.

**You will:** compute your first tensor on omega in under ten minutes. The first way uses no Rust at all: the computation is a TOML program and one command, and the same plan runs on the CPU or on Metal. The second way embeds those nodes in Rust and drives them from a TOML runtime config, where one line moves the work from the CPU to the GPU.

**New concepts (in order):** omega, the thing that fronts the backends · `Engine` and `GpuDriver`, the two-level backend choice · `RuntimeConfig` as a TOML file · `RuntimeConfigBuilder`, the same value built in code · `ProgramSpec`, a program as TOML data · nodes, ids, maps and outputs · running a plan on `proximad` with no Rust · embedding the same plan in Rust · what omega does not do yet.

Every code block is copied from a real file in this repository and cited as `path:line`. Every `sh` block is followed by a `text` block holding the real output of that command, from a run on an Apple M1 Max. Nothing here is invented output.

## 1. What omega is

Omega is the crate `proxima-omega` (library name `omega`, `omega/Cargo.toml:2` and `:13`). You give it a tensor program, a description of what to compute and over which named input buffers, and a configuration, and it plans that program for a backend and executes it. The backends it can drive are the CPU evaluator (`proxima_tensor::cpu`), Metal (macOS), wgpu (which renders WGSL, behind the `wgpu-backend` feature), CUDA and HIP (behind `cuda-driver`). The planner-and-executor pair lives in `omega/src/backend.rs`; the module doc at `omega/src/lib.rs:1-3` still describes the crate as the MSL emitter, which is where it started.

PyTorch analogy, and where it breaks: omega is closer to the layer under `torch.compile` plus the device dispatcher than to `torch.Tensor`. There is no tensor object with operators. You describe the computation as data, and the backend is chosen by a config value. That is the point of this series: **everything about how and where a program runs is a configuration.**

Two things the crate's own docs do not say, so we say them here. First, `omega/README.md` is out of date: it describes only MSL emission and the driver-less half. Second, mlx and Core ML exist in the source (`omega/src/lib.rs:87` `pub mod mlx`, `:48` `pub mod coreml`) but are not backends you can select by configuration; see [section 8](#8-what-omega-does-not-do-yet).

## 2. Engine and GpuDriver

Omega's backend choice has two levels, both plain enums in `omega/src/backend.rs`:

- `Engine` (`backend.rs:113`) has exactly two variants, `Cpu` and `Gpu`. Its `name()` is the string `"cpu"` or `"gpu"`, and `FromStr` (`backend.rs:261`) parses those two strings and nothing else.
- `GpuDriver` (`backend.rs:187`) says which driver renders the `Gpu` engine: `Metal`, `Wgpu`, and, with `cuda-driver`, `Cuda` and `Hip`.

Why two levels: the doc comment on `Engine` (`backend.rs:105-111`) says a driver is not a peer of the CPU engine. "Where does this run" has two answers, a CPU core or a GPU; "which GPU API" is a separate question you only ask when the first answer is `Gpu`. If `engine` is `cpu`, the driver is ignored (`backend.rs:643`).

Which drivers exist in your build is decided at compile time by cargo features. The default feature set is `["std", "metal", "cpu"]` (`omega/Cargo.toml:21`), so a default build on a Mac has the CPU engine and the Metal driver; a default build on Linux has the CPU engine and no GPU driver (asking for `gpu` there returns `BackendError::NoGpuDriver`, `backend.rs:323`).

## 3. The runtime configuration, as a TOML file

The configuration type is `RuntimeConfig` (`omega/src/config.rs:22`). It derives `Deserialize`, so a TOML file is a first-class way to write one, and `deny_unknown_fields` (`config.rs:21`) means a typo in a key is an error rather than a silently ignored line. Two of its fields are all this page needs:

- `engine: String`, default `"cpu"` (`config.rs:25`)
- `gpu_driver: String`, default `"auto"` (`config.rs:28`)

Everything else (about two dozen more fields: fusion, worker counts, numeric policy, selection) has a default and is [the next page's](./01-config.md) subject. The whole file for this page is two lines, `omega/examples/omega_hello_tensor.toml:1-2`:

```toml
engine = "cpu"
gpu_driver = "auto"
```

`RuntimeConfig::from_file` (`config.rs:389`) reads a TOML, JSON or YAML file, overlays it on the defaults, and validates the result. Validation (`config.rs:298` and `:310`) rejects an `engine` that is not `cpu|gpu` and a `gpu_driver` that is not `auto|metal|wgpu|cuda|hip|rocm`, with a message that names the field.

## 4. The same configuration, as a builder

Proxima's rule is that every configuration is both a file and a fluent builder, and the two must produce an identical value. `RuntimeConfig::builder()` (`config.rs:397`) returns a `RuntimeConfigBuilder` (`config.rs:716`) seeded with the defaults. `with_engine` (`config.rs:841`) and `with_gpu_driver` (`config.rs:848`) set the two fields, and `build()` (`config.rs:938`) validates and returns the `RuntimeConfig`.

The repository checks the equivalence with a test rather than asking you to trust it. This is `omega/src/config_tests.rs:329-340`; the first two keys of the TOML string are the two fields this page uses, and `[constraints]` and `[numeric]` are tables [page 01](./01-config.md) covers:

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

The builder can also read a file in the middle of a chain (`RuntimeConfigBuilder::from_file`, `config.rs:722`): later setters win. How layers combine is the whole of [page 01](./01-config.md); for now, know it exists.

Which should you write? The file when the choice belongs to whoever deploys the program (swap the backend without recompiling), the builder when a test or a caller computes the value. Configuration-first means you reach for the file first.

## 5. The program, as TOML

The second half is the program itself, and it is also data. `ProgramSpec` (`proxima-tensor/src/spec/program_spec.rs:26`) is a tensor program written as TOML. It carries a `version`, and the three tables this page uses are:

- `[[node]]` is one operation. Every node has an `id`, and a node may only reference ids defined above it, which mirrors the program's backwards-reference rule so the two representations cannot disagree (`program_spec.rs:26-33`).
- `[[output]]` names the results you want back (`OutputSpec`, `program_spec.rs:65`). A page-00 program has no `[[bind]]` entries (`BindSpec`, `program_spec.rs:76`), because its leaves are literals rather than caller-supplied buffers.
- `[exec]` is the evaluation policy; this page leaves it at its default.

The node kinds this page needs are the two leaves and the elementwise map (`NodeSpec`, `program_spec.rs:129`): `input` (`:130`), `elementwise` (`:137`), `reduce` (`:146`), `iota` (`:162`) and `constant` (`:171`). Every node carries a `dtype` and, for the leaf kinds, a `shape` whose entries are a bare integer for a static extent or a symbolic name for a data-dependent one (`ExtentSpec`, `program_spec.rs:87`).

Two vectors and their sum, with no inputs and no Rust, `omega/examples/hello_add.plan.toml:1` (shown whole):

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

Read it as four facts:

1. `n0` is `[0, 1, 2, 3]`, from `iota` over four elements.
2. `n1` is four copies of `10.0`, from `constant`.
3. `n2` is the `add` of the two, and the `maps` say how each operand's axes line up with the output's: `i->i` is the identity on one axis. The map grammar is `MapSpec` (`program_spec.rs:111`); a matmul needs a contraction such as `ac->abc`, which is [page 02](./02-programs-as-data.md).
4. `[[output]]` says the caller reads the result of `n2` under the name `sum`.

Nothing in the file says where it runs. That is deliberate: the same plan is the same bytes on every backend. Real programs live in `proxima-tensor/specs/`: 11 files including `conv2d.toml`, `rope.toml`, `attention_block.toml`, `gqa_attention.toml`, `moe_block.toml`, `mistral_layer.toml` and `transformer_block.toml`.

## 6. Run it: no Rust at all

Two binaries make this path work, both in `tools/proxima-cli`: `proxima` (`tools/proxima-cli/Cargo.toml:15`) and `proximad` (`:19`). The daemon evaluates plans; the CLI posts one to it over a Unix socket and prints the outputs.

Start the daemon and note the `READY` line it prints with the socket path:

```sh
cargo build -p proxima-cli --bins
./target/debug/proximad serve --unix /tmp/proximad.sock --state-dir /tmp/proximad-state
```

```text
READY /tmp/proximad.sock
```

Then run the plan. `--socket` also reads `PROXIMAD_SOCKET` (`tools/proxima-cli/src/tensor.rs:36`):

```sh
./target/debug/proxima tensor run --socket /tmp/proximad.sock omega/examples/hello_add.plan.toml
```

```text
sum [4] [10.0, 11.0, 12.0, 13.0]
```

That is the whole first program: a TOML program, a daemon, and one command. The daemon parses the plan into a `ProgramSpec` and rejects it with the failing field named if the TOML is wrong (`tools/proxima-cli/src/tensor_run.rs:53`), so a typo is a message, not a wrong number. The `sum [4] [...]` line comes from `summarize_values` (`tools/proxima-cli/src/summary.rs:17`), which elides past eight values on a large tensor and prints the total instead.

One honest difference from the next section: on this path the runtime config is read from the environment, per request (`tools/proxima-cli/src/tensor_run.rs:59`), so the switch from CPU to Metal here is two environment variables on the daemon plus a build that compiles the driver in, not an edit to `omega_hello_tensor.toml`:

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

Same plan, same numbers. Skip the `--features metal` and the same two variables get you a refusal that names the missing feature, not a silent CPU fallback (`tools/proxima-cli/Cargo.toml:28`):

```text
proxima tensor run: upstream error: tensor run failed: HTTP 400 — omega could not plan the program: backend `metal` needs the `metal` cargo feature, which is not compiled in
```

The daemon has no `--runtime <toml>` flag yet, and this CLI does not print which backend served the request, which is why the next section runs the same nodes in Rust and prints the device; see [section 8](#8-what-omega-does-not-do-yet).

## 7. Embed it in your program

When the program is part of a service, you want the config to be a file *and* the backend identity in your logs. That is what the Rust path gives you. The complete program, `omega/examples/omega_hello_tensor.rs` (under 60 lines):

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

This example hand-builds the `Vec<Op>` that the TOML program describes, on purpose: it is the same three nodes, so you can see the two forms side by side. The lowering that turns a `ProgramSpec` into that vector is `plan_parts` (`proxima-tensor/src/spec/run.rs:34`), and [page 02](./02-programs-as-data.md) uses it so you never write the vector by hand.

Read the Rust in four steps.

1. **Load the config** (`RuntimeConfig::from_file`, `config.rs:389`).
2. **Describe the program.** Nodes 0 and 1 are `Input`s named `"a"` and `"b"`, each a float32 vector of four elements (`Extent::Static(4)`, `op.rs:46`). Node 2 is the `Add` over nodes 0 and 1. `NodeId` (`op.rs:29`) is a newtype over `u32`, so you cannot confuse a node with an arbitrary number.
3. **Bind data by name.** `named` pairs each input name with a `QuantizedBlock` (`proxima-tensor/src/cpu/epilogue.rs:1459`). `Float32` is the plain f32 case; the other variants are integer and packed quantized weights, which later pages meet. Binding by name, not position, is deliberate: it is how a program's weights load by name and survive being split across devices (the doc on `Op::Input`, `op.rs:195-201`).
4. **Plan, then execute.** `plan_named_with_config` (`backend.rs:490`) validates the config again and returns a `Plan` (`backend.rs:409`, a reusable, backend-specific program) plus a report of `CandidateMeasurement`s (`backend.rs:425`). With the default `selection.mode` of `Configured` (`config.rs:581`), there is exactly one candidate, the one you configured, and no trial runs. `execute_plan_named` (`backend.rs:1303`) runs the plan against the named data; you could call it again with new data and reuse the plan, which is how a serving loop avoids re-planning per token. The result is an `Evaluated` (`proxima-tensor/src/cpu/arena.rs:20`); `.root()` (`arena.rs:65`) is the output buffer as `&[f32]` and `.shape()` (`arena.rs:70`) its shape.

A smaller proof that this API does what it says is the doctest on `plan_named_with_config` (`backend.rs:468-487`), which plans a one-op `Negate` program on the default config and asserts `[-1.0, -2.0]`. It is compiled and run by `cargo test`.

The example is auto-discovered by cargo (`omega/Cargo.toml` sets no `autoexamples = false`) and needs no `[[example]]` entry or extra feature: the default features (`omega/Cargo.toml:21`) already include `cpu`, and its dependencies (`anyhow`, `proxima-numeric`, `proxima-tensor`) are already in `omega/Cargo.toml`. On the two-line CPU config from section 3:

```sh
cargo run -p proxima-omega --example omega_hello_tensor
```

```text
backend = cpu (config: /Users/brianbruggeman/repos/slot-0/proxima-card-PROXIMAOMEGA-63/omega/examples/omega_hello_tensor.toml)
shape   = [4]
a + b   = [11.0, 22.0, 33.0, 44.0]
```

Now the claim from the top of the page, made concrete. Write these two lines to `/tmp/gpu.toml` and pass that file instead:

```toml
engine = "gpu"
gpu_driver = "metal"
```

Same binary, no rebuild, no code change:

```sh
cargo run -p proxima-omega --example omega_hello_tensor -- /tmp/gpu.toml
```

```text
backend = Apple M1 Max (config: /tmp/gpu.toml)
shape   = [4]
a + b   = [11.0, 22.0, 33.0, 44.0]
```

Same program, same numbers, a different device, and the `report` line tells you which one served it. A third run against a deliberately bad config shows validation by field name:

```sh
cargo run -p proxima-omega --example omega_hello_tensor -- /tmp/bad.toml
```

```text
Error: validation failed:
  - engine: unknown engine 'tpu'; expected cpu|gpu
```

and the process exits `1`. Two caveats worth knowing. Switching to `gpu` requires that the driver be compiled in; otherwise you get `BackendError::NotCompiled` (`backend.rs:314`) or `NoGpuDriver` and never a silent CPU fallback. And a name you invent is a validation error, not a guess.

## 8. What omega does not do yet

Each item is something a PyTorch user expects on day one, with the line that shows the gap.

- **No tensor object and no operators.** There is no `a + b` on a value. You write a `ProgramSpec`, or build a `Vec<Op>` and index nodes by `NodeId` (`proxima-tensor/src/op.rs:192`, `:29`). [Page 02](./02-programs-as-data.md) is the whole of the first form.
- **The daemon path has no runtime-TOML surface.** `proxima tensor run` evaluates with `RuntimeConfig::from_env()` per request (`tools/proxima-cli/src/tensor_run.rs:59`); there is no `--runtime <toml>` on `proximad serve`, and the CLI does not report the device that served a plan. The file-plus-builder surface in sections 3 and 4 is the embedded path only, today.
- **Selecting a backend is two choices, not six.** `Engine` has only `Cpu` and `Gpu` (`backend.rs:113`); validation accepts only `auto|metal|wgpu|cuda|hip|rocm` for the driver (`config.rs:310`). There is no `mlx`, `coreml`, `ane` or `npu` value. The `Plan` enum has Cpu, Metal, Wgpu, Cuda and Rocm variants only (`backend.rs:409-421`). An `Npu`/`Ane` engine existed as a name only and was deleted (`backend.rs:105-111`).
- **MLX is a capability marker, not a backend.** `omega/src/mlx.rs:1-5` says execution "remains behind a native adapter"; it exists only when `PROXIMA_MLX_PREFIX` points at an installed MLX (`omega/build.rs:25-37`), and `backend.rs` does not reference it.
- **Core ML is a package writer, not a selectable engine.** The `coreml-lane` feature compiles and caches a written package (`omega/Cargo.toml:499`), gated behind `mil-writer` (the same line), and `RuntimeConfig` has no `engine` or `gpu_driver` value that reaches it (`config.rs:298`, `:310`).
- **Which GPU drivers you have is a compile-time fact.** wgpu and CUDA need cargo features (`omega/Cargo.toml:382`, `:241`) that are off by default; a default Linux build has no GPU engine at all (`backend.rs:323`).
- **Float32 in, float32 out on this path.** Input blocks are `QuantizedBlock` variants (`epilogue.rs:1459`) and `Evaluated` stores `Vec<f32>` buffers (`arena.rs:22`). The `numerics` setting that would choose storage, activation and accumulator formats is "recorded here; lowering does not read it yet" (`config.rs:100`).
- **No autograd on this surface.** Nothing in `backend.rs` or `config.rs` differentiates a program. `proxima-autograd` is a separate crate used as a dev-dependency (`omega/Cargo.toml:711`); how training reaches omega is page 03, and nothing on this page claims it is wired.
- **No performance claim.** This page measures nothing. The targets (burn, PyTorch, llama.cpp, whisper, stable diffusion) are the subject of page 07, and a number there will come from a run, not from this page.

Next: [01: configuration in depth](./01-config.md) covers layering (file, environment, builder), `SelectionConfig` (`config.rs:621`), `KernelCacheConfig`, `TensorExecutionConfig`, and validation.
