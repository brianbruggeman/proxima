# Work packet: extend Omega runtime configuration

## Authoritative revision 2 — implementation contract

This revision replaces the earlier tentative proposal. Implement this contract.

### Exact data and existing types

Extend `omega::config::RuntimeConfig` in place. Keep `engine: String = "cpu"`,
`gpu_driver: String = "auto"`, `allow_fusion: bool = false`, and
`exact_activations: bool = true`. Add these nested fields:

| Field | Type and default | Consumer |
|---|---|---|
| numeric | existing `proxima_tensor::NumericPolicy`, bit_exact | configured binder/emitter; no duplicate numeric-policy type |
| constraints | `ResourceConstraints`, all None | backend and concrete payload allocation owners |
| selection | `SelectionConfig`, configured mode | cold configured planner/measurement selection |

`ResourceConstraints` has exactly three `Option<u64>` fields:
`max_host_tensor_bytes`, `max_device_buffer_bytes`,
`max_single_device_buffer_bytes`. Their byte scopes and refusal rules are in
the execution packet. `Some(0)` fails field validation; None has no additional
operator cap. No max-RSS, plan/kernel-cache, converted-weight-cache, timeout,
CPU-worker or measurement-cache fields are added without a consumer.

`SelectionConfig` contains `mode: SelectionMode` (`Configured` default or
`MinimumMeasuredWallTime`, serde snake_case), `warmup_runs: Option<u16>` and
`measurement_runs: Option<u16>`, both None by default. Getters resolve absent
counts from `sized::RUNTIME_WARMUP_RUNS` and
`sized::RUNTIME_MEASUREMENT_RUNS`, generated from `[runtime] warmup_runs=1`,
`measurement_runs=5` in omega-runtime.toml. Validate explicit or resolved
warmups >=1 and samples >=3 in measured mode. Configured mode performs zero
trials and rejects explicit trial counts instead of silently ignoring them. No cache is
invented: caller reuses the existing Plan.

Use Settings `#[setting(nested)]` for all three nested fields. Add
`#[cfg_attr(feature="config", derive(conflaguration::Settings))]` and
cfg-gated `#[setting(default=false)]` attributes to the six fields of the
existing NumericPolicy. It already has serde under that feature. Enable
proxima-tensor/config for the Omega runtime config using `cargo add`, preserving
its existing dependency flags. Do not edit Cargo.toml by hand.

Keep serde legacy-file support with container/field defaults. Existing field
environment keys remain unchanged. New keys are `OMEGA_NUMERIC_*`,
`OMEGA_CONSTRAINTS_*`, and `OMEGA_SELECTION_*`. Known portable engine/driver
data is parsed independently of compiled hardware; planning reports unavailable
drivers. Unknown engine/driver and enum strings fail validation/parsing.

Implement `Validate` using the existing conflag ValidationMessage aggregation.
Both from_env and from_file validate before returning. Fluent construction
uses `RuntimeConfig::builder()` seeded with Default and consuming
`with_engine`, `with_gpu_driver`, `with_allow_fusion`,
`with_exact_activations`, `with_numeric`, `with_constraints`, `with_selection`,
`from_file`, `from_env`, and `build()->conflaguration::Result<RuntimeConfig>`.
Seed an existing value with `RuntimeConfig::into_builder(self)`. The builder
only owns that config; no second serializable wire representation. Calls apply
in order; a source-loader call replaces the current value, later with_* calls
override it. Nested types expose their own fluent with_* setters on Default.

Also add the tiled packet's compile gate to the existing sizing source:
`[experimental] dense_float_storage=0`, resolve 0/1, emit check-cfg and
`omega_dense_float_storage` only for 1. No manifest feature edit.

### Exactly twelve tests

Own omega/src/config.rs and its config_tests.rs, NumericPolicy derive attributes,
and the existing omega sizing files. Other workers own backend/binder/emitter.

1. `runtime_config_defaults_preserve_legacy_values`: literal cpu/auto/false/true,
   six false permissions, None ceilings/count overrides, Configured.
2. `runtime_config_legacy_toml_deserializes`: four legacy keys with
   gpu/metal/true/false; missing nested fields use defaults.
3. `runtime_config_settings_uses_omega_legacy_prefix`: scoped env overrides to
   gpu/wgpu/true/false, literal expected fields.
4. `runtime_config_nested_env_overrides_nested_defaults`: measured mode,
   warmups2/samples7, max_device_buffer_bytes4096, contraction true;
   assert exactly those fields and other permission bits false.
5. `runtime_config_builder_matches_serde_config`: literal gpu/metal,
   max_device_buffer_bytes4096, reassociation true; both paths equal that
   hand-authored config field by field.
6. `runtime_config_layer_order_is_source_then_explicit_override`: file gpu
   budget8192 then with_engine(cpu), with_constraints(budget4096); expect
   cpu/4096. Reverse call order restores file gpu/8192.
7. `runtime_config_rejects_unknown_engine`: npu, error contains engine/npu.
8. `runtime_config_rejects_unknown_gpu_driver`: vulkan, error contains
   gpu_driver/vulkan; cuda portable data loads even without CUDA runtime.
9. `runtime_config_rejects_zero_resource_ceiling`: each of the three fields
   Some(0) produces its named error; Some(1) is valid config data.
10. `runtime_config_requires_measurement_samples`: measured warmups0/samples2
    produces both errors; warmups1/samples3 admitted. Configured makes no
    claim that those counts ran.
11. `runtime_config_numeric_permissions_round_trip_independently`:
    contraction+approx_functions true, remaining four false through serde,
    Settings and builder. This is the existing NumericPolicy value.
12. `runtime_config_defaults_track_sized_floor`: getters equal generated
    constants; default sizing input resolves literal 1 warmup/5 samples.

Use tempfile for file fixtures and temp_env::with_vars for environment cases.
Command: `cargo nextest run -p omega --lib -E 'test(runtime_config_)'`;
expected exactly 12 tests. Also run the SPEC's CPU/std and alloc checks; the
alloc check excludes runtime I/O deliberately, while compiling the emitter.
