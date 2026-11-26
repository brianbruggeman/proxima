# Omega 01: configuration in depth

**Prerequisites:** [page 00](./00-hello-tensor.md), where you ran a plan and used two fields of the runtime config.

**You will:** start with the four settings you will actually touch; watch four sources layer and see which wins per field; read the full surface as five groups of related fields; cause three validation failures that name the field; and meet the nested config types.

**New concepts (in order):** the four settings that matter first · four sources and their precedence · the grouped field list · validation messages · `SelectionConfig` · `ResourceConstraints` · `NumericPolicy` · `KernelCacheConfig` · `TensorExecutionConfig`.

## 1. The four you will touch first

Most runs need nothing beyond page 00's two lines. When a program is slow or memory-hungry, these are the next four:

- `cpu_workers` (`omega/src/config.rs:67`) — how many workers the CPU evaluator uses.
- `allow_fusion` (`:42`) — whether the planner may join adjacent ops. Off by default, so the numbers you see are the ops you wrote.
- `pipeline_cache` (`:56`) — whether compiled Metal pipelines survive a process restart. On by default; it is the difference between a 116 ms first request and 21 ms per token once warm (page 04 measures both).
- `selection.mode` (`omega/src/config.rs:680`) — `Configured` takes the backend you asked for; `MinimumMeasuredWallTime` times the candidates and takes the fastest. The default is `Configured`, which is why nothing you ran on page 00 did any trial runs.

Everything else has a default that is right until you have a reason. Section 2 shows layering with a runnable example; section 3 is the full list, grouped so you can find the one field you need.

## 2. Four sources, and which wins

The precedence is defaults, then a file, then the environment, then explicit setters. `RuntimeConfig::from_file` (`config.rs:412`) is the one-source shortcut: defaults overlaid by a file, validated. The layered form is `RuntimeConfig::builder()` (`config.rs:420`) chained with `from_file` (`config.rs:778`), `from_env` (`config.rs:788`) and any `with_*` setter, whose doc says later setters win.

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

The file is `omega/examples/omega_hello_tensor.toml`, whose two lines agree with the defaults, so the first three rows look alike. The environment changes the picture; every field's env spelling is `OMEGA_` plus the field name (`config.rs:20`), so:

```sh
OMEGA_ENGINE=gpu OMEGA_GPU_DRIVER=metal cargo run -p proxima-omega --example config_layers
```

```text
defaults                           engine=cpu  gpu_driver=auto
file                               engine=cpu  gpu_driver=auto
file + env                         engine=gpu  gpu_driver=metal
file + env + with_engine("cpu")    engine=cpu  gpu_driver=metal
```

Look at the last row: the setter overrode `engine` and left `gpu_driver` where the environment put it. A setter changes one field; layering is per field, in call order.

## 3. The whole surface, grouped

`RuntimeConfig` (`omega/src/config.rs:22`) has 28 fields, all with a default. They fall into five groups; read the group you need.

### Backend and execution shape

| field | line | default | what it decides |
|---|---|---|---|
| `engine` | `:26` | `cpu` | backend family: `cpu` or `gpu` |
| `gpu_driver` | `:29` | `auto` | which driver renders `gpu` |
| `hip_wave` | `:32` | `0` | HIP wavefront width; `0` reads it from the device |
| `cpu_workers` | `:67` | from `sized` | CPU evaluator workers |
| `independent_ops` | `:70` | from `sized` | independent ops admitted per ready DAG level |
| `parallel_sequences` | `:73` | from `sized` | live generation streams admitted to shared execution |
| `max_in_flight` | `:76` | from `sized` | submissions awaiting backend completion |
| `overlap_transfer_compute` | `:79` | `false` | let the driver overlap transfer with compute |
| `wavefront_schedule` | `:47` | from `sized` | group independent ops into dependency levels so the GPU submits them together instead of one dispatch at a time |
| `allow_fusion` | `:42` | `false` | permit bind-time and execution-time algebraic fusion |
| `exact_activations` | `:50` | `true` | require exact activation evaluation where supported |

### Memory and caching

| field | line | default | what it decides |
|---|---|---|---|
| `max_device_bytes` | `:82` | `None` | hard device-residency ceiling |
| `pipeline_cache` | `:56` | `true` | persist compiled Metal pipelines across processes |
| `pipeline_cache_dir` | `:60` | `""` (per-user cache dir) | where those entries live |
| `pipeline_cache_max_bytes` | `:64` | `268435456` | eviction ceiling for that directory |
| `constraints` | `:122` | no ceilings | host and device byte ceilings (`ResourceConstraints`, section 5) |

### Strategy enums

| field | line | default | what it decides |
|---|---|---|---|
| `decode_kernel` | `:85` | `automatic` | decode projection kernel policy |
| `fusion` | `:88` | the enum's `Default` | fusion request; `required` rejects planning when absent |
| `residency` | `:91` | the enum's `Default` | cache placement policy |
| `batching` | `:94` | the enum's `Default` | row formation policy |
| `speculation` | `:97` | `disabled` | speculative decoding policy |
| `weight_conversion` | `:100` | the enum's `Default` | converted-weight cache policy |
| `moe_grouping` | `:103` | `true` | group already-routed expert rows |

The six enum-valued fields carry `#[setting(default)]` (`config.rs:78-93`), so their default is the enum's own `Default` — the member each type documents as the safe one.

### Numerics and emission

| field | line | default | what it decides |
|---|---|---|---|
| `numeric` | `:106` | `NumericPolicy::bit_exact()` | numeric permissions (`NumericPolicy`, section 5) |
| `numerics` | `:110` | `Numerics::REFERENCE` | storage, activation and accumulator formats |
| `emit` | `:113` | `EmitKnobs::default()` | Metal emit A/B switches |
| `dense_mma_input_layout` | `:119` | derived from operands | dense matrix fragment layout |

### Selection

| field | line | default | what it decides |
|---|---|---|---|
| `selection` | `:125` | `Configured` | cold-plan candidate selection (`SelectionConfig`, section 5) |

Five defaults are not literals in this file: `cpu_workers`, `independent_ops`, `parallel_sequences`, `max_in_flight` (`config.rs:76`) and `wavefront_schedule` (`config.rs:47`) come from `crate::sized`, the constants a build sets. A binary built for a small machine therefore does not inherit a desktop-sized default.

## 4. When a value is wrong, the error names the field

Validation runs in `from_file`, in `build()`, and again inside `plan_named_with_config` (`backend.rs:659`), so a config assembled without a check still cannot reach a backend. Three real failures from `omega_hello_tensor`.

A value out of range, against `omega/examples/configs/bad_workers.toml` (`cpu_workers = 0`):

```sh
cargo run -p proxima-omega --example omega_hello_tensor -- omega/examples/configs/bad_workers.toml
```

```text
Error: validation failed:
  - cpu_workers: must be greater than zero
```

An engine nobody implements, `omega/examples/configs/bad_engine.toml` with `engine = "tpu"`:

```text
Error: validation failed:
  - engine: unknown engine 'tpu'; expected cpu|gpu
```

A typo in a key, `omega/examples/configs/typo.toml` with `engin = "gpu"` (`deny_unknown_fields`, `config.rs:22`):

```text
Error: toml: TOML parse error at line 2, column 1
  |
2 | engin = "gpu"
  | ^^^^^
unknown field `engin`, expected one of `engine`, `gpu_driver`, `hip_wave`, `allow_fusion`, `wavefront_schedule`, `exact_activations`, `pipeline_cache`, `pipeline_cache_dir`, `pipeline_cache_max_bytes`, `cpu_workers`, `independent_ops`, `parallel_sequences`, `max_in_flight`, `overlap_transfer_compute`, `max_device_bytes`, `decode_kernel`, `fusion`, `residency`, `batching`, `speculation`, `weight_conversion`, `moe_grouping`, `numeric`, `numerics`, `emit`, `dense_mma_input_layout`, `constraints`, `selection`
```

The real output repeats that block once more under `Caused by:`, because two layers wrap the same parse error. That list is every key the config accepts, which is the grouped table in section 3 read back to you. All three exit `1`.

## 5. The three nested types

`selection` is a `SelectionConfig` (`config.rs:677`) with three fields: `mode` (default `Configured`, `config.rs:637`), `warmup_runs` (`:683`) and `measurement_runs` (`:686`). In `Configured` mode planning takes the backend you configured and does no trial runs; `MinimumMeasuredWallTime` (`config.rs:639`) measures candidates and takes the smallest synchronized mean. The two count fields are overrides; unset means the generated floor (`SelectionConfig::resolved_warmup_runs`, `config.rs:702`).

`constraints` is a `ResourceConstraints` (`config.rs:581`) of three optional byte ceilings: `max_host_tensor_bytes` (`:584`), `max_device_buffer_bytes` (`:587`) and `max_single_device_buffer_bytes` (`:590`). `None` means no ceiling; `Some(0)` is a validation error, because a zero-byte ceiling is a typo rather than a policy.

`numeric` is `proxima_numeric::NumericPolicy` (`proxima-numeric/src/numeric.rs:27`), ten booleans that each permit one class of rewriting: `contraction`, `reassociation`, `nan_assumptions`, `signed_zero`, `approx_functions`, `epilogue_sources`, `activation_quantization`, `input_rounding`, `flash_narrowing` and `accumulator_narrowing`. All ten are `false` in `NumericPolicy::bit_exact()` (`proxima-numeric/src/numeric.rs:150`), which is the default here, so a fresh omega config refuses every rewrite that could change a bit.

## 6. Two sibling configs

`KernelCacheConfig` (`omega/src/kernel_cache_config.rs:20`) is the CUDA PTX cache: one field, `kernel_cache_dir` (`:29`), whose env prefix is `PROXIMA_CUDA` (`:18`) and whose default is `~/.cache/proxima/cuda-ptx` (`:14`). Its module doc says the derive owns the default (`:1-7`): one field, one loader, no second path to the same value.

`TensorExecutionConfig` (`proxima-tensor/src/config.rs:28`) is the CPU evaluator's policy, and it is also the `[exec]` table of a `ProgramSpec` (`program_spec.rs:26`, field `exec` at `:46`). Its fields are `numeric` (`proxima-tensor/src/config.rs:33`), `allow_fusion` (`:39`), `exact_activations` (`:45`), `max_host_tensor_bytes` (`:49`), `parallel_threshold` (`:56`), `oversubscribe` (`:63`) and `split_alignment` (`:70`), with the env prefix `TENSOR_EXECUTION` (`:26`). A plan therefore carries its own execution policy: `[exec] numeric.reassociation = true` in a plan TOML is read by the evaluator that runs it.

## 7. What omega does not do yet

- `numerics` is recorded in the config and ignored by lowering. The storage/activation/accumulator formats appear there (`config.rs:102`) and the field's own doc says "Recorded here; lowering does not read it yet". A dtype change today is a `ProgramSpec` `dtype` on a node.
- `emit` is Metal-only. `EmitKnobs` (`config.rs:113`) switches Metal emit decisions and keeps the historical `PROXIMA_*` spellings (`override_prefix`, `config.rs:112`); there is no emit surface for wgpu or CUDA, whose kernels are not emitted from this crate at run time.
- There is one loader per type, and no global one. Each config type owns its own `from_file`/`from_env`; nothing reads `proxima.toml` and hands every subsystem its section. Wiring several configs together is the caller's job today.
- The daemon path still has no runtime file. `proximad serve` takes flags and reads `OMEGA_*` per request (`tools/proxima-cli/src/tensor_run.rs:59`); there is no `--runtime <toml>`.

Next: [02: programs as data](./02-programs-as-data.md) takes the `ProgramSpec` from page 00 apart: every node kind, binds, outputs, and a real spec from `proxima-tensor/specs/` running through omega.
