# Execution order and ownership

No code is written until the spec-auditor admits SPEC.md and this file. Root
then dispatches the exact work packet, admitted spec, isolated worktree,
dedicated target directory, dependency revision/patch and test counts together.
Workers do not edit another worker's owned files. Root inspects actual diffs
and payloads before integrating; no commit or publish is authorized.

| Slice | Owner / files | Depends on | AC | Validation command |
|---|---|---|---|---|
| S1 config | Luna; config packet plus numeric Settings derive and existing sizing files | admission | AC1 | `cargo nextest run -p omega --lib -E 'test(runtime_config_)'` |
| S2 binder/CPU policy | Sol; binding packet | admission | AC2 | `cargo nextest run -p proxima-tensor --lib -E 'test(programmable_binding_)'` |
| S3 configured execution/resources/selection | Sol; execution packet, existing backend driver prepare/allocation paths | S1,S2 | AC3 | `cargo nextest run -p omega --lib -E 'test(config_plan_)'` |
| S3b CPU allocation assertions | Sol; execution packet CPU tests | S3 | AC3b | `cargo nextest run -p omega --lib -E 'test(config_cpu_budget_)'` |
| S3c selection assertions | Sol; execution packet selector tests | S3 | AC3c | `cargo nextest run -p omega --lib -E 'test(config_selection_)'` |
| S3d Metal allocation assertions | Sol; execution packet Metal tests | S3 | AC3d | `cargo nextest run -p omega --lib -E 'test(config_metal_)'` |
| S4 tiled storage/policy | Luna; tiled packet, including existing identity and grid consumers | S1 sizing gate | AC4 | `env OMEGA_EXPERIMENTAL_DENSE_FLOAT_STORAGE=1 cargo nextest run -p omega --features metal-tiled-gemm --lib -E 'test(programmable_tiled_)' --no-capture` |
| S4b compile-off assertion | Luna; tiled packet | S4 | AC5 | `env OMEGA_EXPERIMENTAL_DENSE_FLOAT_STORAGE=0 cargo nextest run -p omega --features metal-tiled-gemm --lib -E 'test(programmable_tiled_)' --no-capture` |
| S5 real execution | root source; Luna runs omega/tests/programmable_execution_vectors.rs | S3,S4 | AC6 | `env OMEGA_EXPERIMENTAL_DENSE_FLOAT_STORAGE=1 cargo nextest run -p omega --features metal-tiled-gemm --test programmable_execution_vectors --no-capture` |
| S5b enabled microbench | root source; Luna runs existing performance example | S5 | AC10 | `env OMEGA_EXPERIMENTAL_DENSE_FLOAT_STORAGE=1 OMEGA_TILED_GEMM_MIN_TOKENS=1 cargo run --release --example ouro_loopcd_lab --features proxima-tensor,omega/metal-tiled-gemm -- --performance --microbench --backend metal --samples 5 --omega-config proxima-tensor/specs/omega-programmable/relaxed-metal.toml --output proxima-tensor/specs/omega-programmable/results/micro-enabled` |
| S5c disabled microbench | root source; Luna runs existing performance example | S5 | AC10b | `env OMEGA_EXPERIMENTAL_DENSE_FLOAT_STORAGE=0 OMEGA_TILED_GEMM_MIN_TOKENS=1 cargo run --release --example ouro_loopcd_lab --features proxima-tensor,omega/metal-tiled-gemm -- --performance --microbench --backend metal --samples 5 --omega-config proxima-tensor/specs/omega-programmable/relaxed-metal.toml --output proxima-tensor/specs/omega-programmable/results/micro-disabled` |
| S6 caller/evidence | root integration with delegated runs; evidence packet | S3,S4,S5 | AC9 | Run both commands: `env OMEGA_EXPERIMENTAL_DENSE_FLOAT_STORAGE=1 OMEGA_TILED_GEMM_MIN_TOKENS=1 cargo run --release --example ouro_loopcd_lab --features proxima-tensor,omega/metal-tiled-gemm -- --performance --backend metal --samples 5 --omega-config proxima-tensor/specs/omega-programmable/relaxed-metal.toml --output proxima-tensor/specs/omega-programmable/results/ouro-configured`; `env OMEGA_EXPERIMENTAL_DENSE_FLOAT_STORAGE=0 OMEGA_TILED_GEMM_MIN_TOKENS=1 cargo run --release --example ouro_loopcd_lab --features proxima-tensor,omega/metal-tiled-gemm -- --performance --backend metal --samples 5 --omega-config proxima-tensor/specs/omega-programmable/relaxed-metal.toml --output proxima-tensor/specs/omega-programmable/results/ouro-configured-control` |
| S7 alloc-tier build | Luna; combined tree | S3,S4 | AC7 | `cargo check -p omega --no-default-features --features alloc --message-format=json` |
| S8 CPU/std build | Luna; combined tree | S3,S4 | AC8 | `cargo check -p omega --no-default-features --features std,cpu --message-format=json` |

## Integration protocol

1. Begin S1 and S2 in separate worktrees from the captured dirty starting tree;
   preserve those starting modifications. No branch commit is needed.
2. Integrate S1 before S3 and S4. Never compile a caller against guessed config
   field names. Re-read the admitted packet if an interface changes.
3. Run S3 and S4 in parallel with disjoint ownership. If signatures meet in a
   shared file, root performs that small integration after both worker diffs
   are read. Root owns the final combined worktree.
4. Only one GPU benchmark process runs at a time. CPU builds can proceed
   separately, but capture host load and do not treat loaded-host measurements
   as quiet-host data.
5. For every failed assertion or adverse benchmark row, preserve its actual
   inputs/outputs, identify the path that produced it, amend the implementation
   or roll back the experiment, and re-run its count-bearing AC. No tolerance
   inflation or sample deletion.
6. Independent review reads diff hunks, generated kernel source, resource
   admission/rejection payloads and comparator assertions. Root records what
   ran and observed counts in EVIDENCE.md; user owns the deployment verdict.

## Worker completion format

S6 additionally runs the native `omega_programmable_recount` command in SPEC.md.
Its count is 25 timing records: 16 model, 6 projection, 2 llama and 1 oracle
register. The eight CoV objects per model arm live in `summary.json`; the
recount checks those summaries against the corresponding raw samples. This
verification is part of S6, not a new inference or measurement cohort.

Return changed files and diff, exact commands, actual test counts, raw evidence
paths, policy/resource behavior observed, and any failed assertion with its
payload. Do not report merely "implemented", "green", "faster", or a next-step
list. Keep working within the packet until its concrete assertions are
accounted for, or send the exact unresolved technical observation to root while
root continues the dependent investigation.

## Acceptance on main

Observed on the tree that carries `RuntimeConfig`, `plan_named_with_config` and
`ServingConfig::omega_runtime`. A box is ticked only when the slice's validation
command ran on this tree and its counted result matches the AC. Counts come from
nextest summary lines; the `-E` filters are the commands in the table above.

| Box | Slice | AC | Observed |
|---|---|---|---|
| [x] | S1 config | AC1 | 29 tests run, 29 passed |
| [x] | S2 binder/CPU policy | AC2 | 8 tests run, 8 passed (`proxima-tensor --lib`) |
| [x] | S3 configured execution | AC3 | 12 tests run, 12 passed |
| [x] | S3b CPU allocation | AC3b | 5 tests run, 5 passed |
| [x] | S3c selection | AC3c | 5 tests run, 5 passed |
| [x] | S3d Metal allocation | AC3d | 4 tests run, 4 passed |
| [x] | S4 tiled storage/policy | AC4 | `OMEGA_EXPERIMENTAL_DENSE_FLOAT_STORAGE=1`: 8 tests run, 8 passed |
| [x] | S4b compile-off | AC5 | `OMEGA_EXPERIMENTAL_DENSE_FLOAT_STORAGE=0`: 1 test run, 1 passed (`programmable_tiled_float_storage_feature_off_declines`) |
| [x] | S5 real execution | AC6 | `OMEGA_EXPERIMENTAL_DENSE_FLOAT_STORAGE=1`: 8 tests run, 8 passed; with `=0`: 8 tests run, 8 passed |
| [ ] | S5b/S5c microbenches | AC10, AC10b | not run: `ouro_loopcd_lab` with `--features proxima-tensor` does not compile on this tree (see below) |
| [ ] | S6 caller/evidence | AC9 | not run, same compile failure. The native `omega_programmable_recount` command ran against the source tree's retained sample files: exit 0, 25 timing records (16 model, 6 projection, 2 llama, 1 oracle register) and 187 coordinate comparisons; `recount_float_roundtrip` and `recount_centered_variance` pass (2 tests) |
| [x] | S7 alloc-tier build | AC7 | exit 0; 101 compiler artifacts, 0 error and 0 warning messages |
| [x] | S8 CPU/std build | AC8 | exit 0; 217 compiler artifacts, 0 error and 0 warning messages |

`ouro_loopcd_lab` and `ouro_adaptive_graph_diagnostic` build only as the
no-feature stub on this tree. With `--features proxima-tensor` they fail on
`RecurrentConfig`, `ouro_descriptor_from_hf_config`, `ServingConfig::recurrent`,
`LoadedModel::{load_from_safetensors_with_descriptor, logits_root,
forward_node_values_from_ids}` and `proxima_tensor::spec::{LoopCdMode,
build_forward_with_readouts}`, none of which exist here. Result files over 5 MB
are listed in `EVIDENCE-LOCATIONS.md`.