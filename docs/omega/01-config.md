# Omega 01: configuration in depth

**Prerequisites:** [page 00](./00-hello-tensor.md): `RuntimeConfig` as a file, `RuntimeConfigBuilder` in code, and one run on each.

**You will:** see the whole `RuntimeConfig` surface with its defaults, layer defaults over a file, the environment and explicit setters and know which wins, read a validation error by the field it names, and meet the three nested config types plus the two sibling configs omega uses for kernels and CPU evaluation.

**New concepts (in order):** the field list and its defaults · four sources and their precedence · `from_file` versus `builder().from_file().from_env()` · validation messages that name a field · `SelectionConfig` · `ResourceConstraints` · `NumericPolicy` · `KernelCacheConfig` · `TensorExecutionConfig` · what omega does not do yet.

## 1. The whole surface

`RuntimeConfig` (`omega/src/config.rs:22`) has 28 fields. Page 00 used two; this is all of them, with the line that declares each and the default the `#[setting]` attribute pins:

| field | line | default | what it decides |
|---|---|---|---|
| `engine` | `:25` | `cpu` | backend family: `cpu` or `gpu` |
| `gpu_driver` | `:28` | `auto` | which driver renders `gpu` |
| `hip_wave` | `:31` | `0` | HIP wavefront width; `0` reads it from the device |
| `allow_fusion` | `:34` | `false` | permit bind-time and execution-time algebraic fusion |
| `wavefront_schedule` | `:39` | build-time `sized` value | reorder the program into dependency levels so independent ops share a barrier interval |
| `exact_activations` | `:42` | `true` | require exact activation evaluation where supported |
| `pipeline_cache` | `:48` | `true` | persist compiled Metal pipelines across processes |
| `pipeline_cache_dir` | `:52` | `""` (per-user cache dir) | where those entries live |
| `pipeline_cache_max_bytes` | `:56` | `268435456` | eviction ceiling for that directory |
| `cpu_workers` | `:59` | build-time `sized` value | CPU evaluator workers |
| `independent_ops` | `:62` | build-time `sized` value | independent ops admitted per ready DAG level |
| `parallel_sequences` | `:65` | build-time `sized` value | live generation streams admitted to shared execution |
| `max_in_flight` | `:68` | build-time `sized` value | submissions awaiting backend completion |
| `overlap_transfer_compute` | `:71` | `false` | let the driver overlap transfer with compute |
| `max_device_bytes` | `:74` | `None` | hard device-residency ceiling |
| `decode_kernel` | `:77` | `automatic` | decode projection kernel policy |
| `fusion` | `:80` | strategy default | fusion request; `required` rejects planning when absent |
| `residency` | `:83` | strategy default | cache placement policy |
| `batching` | `:86` | strategy default | row formation policy |
| `speculation` | `:89` | `disabled` | speculative decoding policy |
| `weight_conversion` | `:92` | strategy default | converted-weight cache policy |
| `moe_grouping` | `:95` | `true` | group already-routed expert rows |
| `numeric` | `:98` | `NumericPolicy::bit_exact()` | numeric permissions |
| `numerics` | `:102` | `Numerics::REFERENCE` | storage, activation and accumulator formats |
| `emit` | `:105` | `EmitKnobs::default()` | Metal emit A/B switches |
| `dense_mma_input_layout` | `:108` | derived from operands | dense matrix fragment layout |
| `constraints` | `:111` | no ceilings | hard byte ceilings |
| `selection` | `:114` | `Configured` | cold-plan candidate selection |

Four of those defaults are not literals in this file: `cpu_workers`, `independent_ops`, `parallel_sequences` and `max_in_flight` resolve from `crate::sized` (`config.rs:59-68`), the build-time constants a per-machine build sets. `wavefront_schedule` reads `crate::sized::COMMAND_BUFFER_WAVEFRONT_SCHEDULE` in `Default` (`config.rs:124`). That is deliberate: a build-time floor and a runtime default that were written twice would drift.

## 2. Four sources, and which wins

The precedence is defaults, then a file, then the environment, then explicit setters. `RuntimeConfig::from_file` (`config.rs:389`) is the one-source shortcut: defaults overlaid by a file, validated. The layered form is `RuntimeConfig::builder()` (`config.rs:397`) chained with `from_file` (`config.rs:722`), `from_env` (`config.rs:732`) and any `with_*` setter, whose doc says later setters win.

`omega/examples/config_layers.rs` prints the resolved pair after each combination. Run it with no environment set:

```sh
cargo run -p proxima-omega --example config_layers
```

```text
defaults                           engine=cpu  gpu_driver=auto
file                               engine=cpu  gpu_driver=auto
file + env                         engine=cpu  gpu_driver=auto
file + env + with_engine("cpu")    engine=cpu  gpu_driver=auto
```

The file is `omega/examples/omega_hello_tensor.toml`, whose two lines agree with the defaults, so the first three rows look alike. The environment is where the interesting row appears; every field's env spelling is `OMEGA_` plus the field name (`config.rs:20`), so:

```sh
OMEGA_ENGINE=gpu OMEGA_GPU_DRIVER=metal cargo run -p proxima-omega --example config_layers
```

```text
defaults                           engine=cpu  gpu_driver=auto
file                               engine=cpu  gpu_driver=auto
file + env                         engine=gpu  gpu_driver=metal
file + env + with_engine("cpu")    engine=cpu  gpu_driver=metal
```

Read the last row carefully: the setter overrode `engine` and left `gpu_driver` at the environment's value, because a setter changes one field, not the whole config. Layering is per field, in call order.

## 3. When a value is wrong, the error names the field

Validation runs in `from_file`, in `build()`, and again inside `plan_named_with_config` (`backend.rs:490`), so a config that was assembled without a check still cannot reach a backend. Three real failures, each from the run above.

A value out of range, `omega/examples/omega_hello_tensor.rs` against a file with `cpu_workers = 0`:

```sh
cargo run -p proxima-omega --example omega_hello_tensor -- /tmp/bad_workers.toml
```

```text
Error: validation failed:
  - cpu_workers: must be greater than zero
```

An engine nobody implements, `/tmp/bad.toml` with `engine = "tpu"`:

```text
Error: validation failed:
  - engine: unknown engine 'tpu'; expected cpu|gpu
```

A typo in a key, `/tmp/typo.toml` with `engin = "gpu"` (`deny_unknown_fields`, `config.rs:21`):

```text
Error: toml: TOML parse error at line 2, column 1
  |
2 | engin = "gpu"
  | ^^^^^
unknown field `engin`, expected one of `engine`, `gpu_driver`, `hip_wave`, `allow_fusion`, `wavefront_schedule`, `exact_activations`, `pipeline_cache`, `pipeline_cache_dir`, `pipeline_cache_max_bytes`, `cpu_workers`, `independent_ops`, `parallel_sequences`, `max_in_flight`, `overlap_transfer_compute`, `max_device_bytes`, `decode_kernel`, `fusion`, `residency`, `batching`, `speculation`, `weight_conversion`, `moe_grouping`, `numeric`, `numerics`, `emit`, `dense_mma_input_layout`, `constraints`, `selection`
```

That last message is the surface from section 1, printed by the parser. All three exit `1`.

## 4. The three nested types

`selection` is a `SelectionConfig` (`config.rs:621`) with three fields: `mode` (default `Configured`, `config.rs:624`), `warmup_runs` (`:627`) and `measurement_runs` (`:630`). In `Configured` mode planning takes the backend you configured and does no trial runs; `MinimumMeasuredWallTime` (`config.rs:581`) measures candidates and takes the smallest synchronized mean. The two count fields are overrides; unset means the generated floor (`SelectionConfig::resolved_warmup_runs`, `config.rs:646`).

`constraints` is a `ResourceConstraints` (`config.rs:525`) of three optional byte ceilings: `max_host_tensor_bytes` (`:528`), `max_device_buffer_bytes` (`:531`) and `max_single_device_buffer_bytes` (`:534`). `None` means no ceiling; `Some(0)` is a validation error, because a zero-byte ceiling is a typo rather than a policy.

`numeric` is `proxima_numeric::NumericPolicy` (`proxima-numeric/src/numeric.rs:27`), ten booleans that each permit one class of rewriting: `contraction`, `reassociation`, `nan_assumptions`, `signed_zero`, `approx_functions`, `epilogue_sources`, `activation_quantization`, `bf16_input_rounding`, `flash_narrowing` and `accumulator_narrowing`. All ten are `false` in `NumericPolicy::bit_exact()` (`numeric.rs:126`), which is the default here, so a fresh omega config refuses every rewrite that could change a bit.

## 5. Two sibling configs

`KernelCacheConfig` (`omega/src/kernel_cache_config.rs:19`) is the CUDA PTX cache: one field, `kernel_cache_dir` (`:24`), whose env prefix is `PROXIMA_CUDA` (`:16`) and whose default is `~/.cache/proxima/cuda-ptx` (`:12`). Its module doc says the derive owns the default and there is no second loader (`:1-7`) — that is the house shape for a config that has exactly one tunable.

`TensorExecutionConfig` (`proxima-tensor/src/config.rs:28`) is the CPU evaluator's policy, and it is also the `[exec]` table of a `ProgramSpec` (`program_spec.rs:44`). Its fields are `numeric` (`:33`), `allow_fusion` (`:39`), `exact_activations` (`:45`), `max_host_tensor_bytes` (`:49`), `parallel_threshold` (`:56`), `oversubscribe` (`:63`) and `split_alignment` (`:70`), with the env prefix `TENSOR_EXECUTION` (`:26`). A plan therefore carries its own execution policy: `[exec] numeric.reassociation = true` in a plan TOML is read by the evaluator that runs it.

## 6. What omega does not do yet

- **`numerics` is recorded, not read.** The storage/activation/accumulator formats appear in the config (`config.rs:102`) and lower nothing: the field's own doc says "Recorded here; lowering does not read it yet". A dtype change today is a `ProgramSpec` `dtype` on a node, not a runtime setting.
- **`emit` is Metal-only.** `EmitKnobs` (`config.rs:105`) switches Metal emit decisions and keeps the historical `PROXIMA_*` spellings (`override_prefix`, `config.rs:104`); there is no emit surface for wgpu or CUDA, whose kernels are not emitted from this crate at run time.
- **There is one loader per type, and no global one.** Each config type owns its own `from_file`/`from_env`; nothing reads `proxima.toml` and hands every subsystem its section. Wiring several configs together is the caller's job today.
- **The daemon path still has no runtime file.** `proximad serve` takes flags and reads `OMEGA_*` per request (`tools/proxima-cli/src/tensor_run.rs:59`); there is no `--runtime <toml>`.

Next: [02: programs as data](./02-programs-as-data.md) takes the `ProgramSpec` from page 00 apart: every node kind, binds, outputs, and a real spec from `proxima-tensor/specs/` running through omega.
