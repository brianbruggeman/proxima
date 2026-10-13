# Work packet: configure and execute the existing Omega plan

Status: proposed executable packet; no Rust implementation or hardware execution is represented by this document. The root SPEC and auditor admission govern implementation.

## Bound interface and source register

Adapt the existing `Op` program, symbols, named blocks and output IDs into the existing `backend::Plan`, using `RuntimeConfig` once at planning. Execute that same plan repeatedly through `execute_plan_named`; do not add an IR, per-operation device scheduler, runtime registry or global selection cache.

Opened source artifacts:

| Artifact | Observed content and implication |
|---|---|
| `omega/src/backend.rs:397` | `plan_named` has actual CPU, Metal, wgpu and conditional CUDA arms; CPU currently does not consume its numeric argument. |
| `omega/src/backend.rs:835` | CPU plan owns program/symbols/outputs and reusable float buffer pool; regular CPU plan stores `exact_activations=false`. |
| `omega/src/backend.rs:893` | `plan_named_exact` uses exact CPU activation mode and delegates GPU to ordinary planning. |
| `proxima-tensor/src/cpu/quantized_eval.rs:309` | Existing evaluator implementation accepts exact activation and cached-attention flags; its binder call at line 396 hardcodes `NumericPolicy::bit_exact()`. |
| `proxima-tensor/src/bind/gdn_moe_fusion_apply.rs:62` | Existing `fuse_cached_attention=false` only gates that fusion family. It does not mean all algebraic fusion is disabled. |
| `proxima-tensor/src/bind/cached_attention_epilogue_liveness.rs:843` | `bind_plain` itself constructs `BoundOpBuilder`, which still performs chain composition. Renaming this path unfused would misstate its behavior. |
| `proxima-tensor/src/cpu/elementwise_matmul.rs:262` | Elementwise float workspace includes operand values; line 383 sizes a generic per-step float table. |
| `proxima-tensor/src/cpu/width_kernels.rs:697` | `GENERIC_WIDTH_TILE` bounds generic row workspace. Consume this constant in its own crate; do not copy the sizing literal into Omega. |
| `proxima-tensor/src/cpu/run_reduce_scan.rs:1094` | Ordinary reduce allocates operand and step float arrays; line 1839 allocates the width accumulator on its fallback path. |
| `proxima-tensor/src/cpu/run_node.rs:478` | Reduce epilogue additionally allocates coordinate metadata and float operand/step workspace. |
| `omega/src/wgpu_driver.rs:124` | Plan owns input/output capacity maps and pipeline cache. Device caps and numeric policy are already plan fields. |
| `omega/src/wgpu_driver.rs:764` | Storage buffer physical size is rounded to four bytes with a minimum of four. |
| `omega/src/wgpu_driver.rs:837` | Execute validates blocks, then creates/reuses input/output buffers; line 977 creates uniforms/fault storage and line 1098 creates output/fault readback staging. |
| `omega/src/cuda_driver.rs:84` | Persistent arena owns graph buffers and uniforms. It counts allocations but not bytes. |
| `omega/src/cuda_driver.rs:237` | CUDA plans infer/bind/emit once; its cached-attention capability flag is false. |
| `omega/src/cuda_driver.rs:317` | Upload helpers allocate replacement before dropping the previous buffer; transient coexistence must fit the resource check. |
| `omega/src/cuda_driver.rs:563` | Persistent launch allocates output and fault buffers; replacement briefly retains the previous output. |
| `omega/src/cuda_driver.rs:1033` | Complete execution synchronizes the stream before readback. |
| `omega/src/metal/execute_and_hazards.rs:23` | Existing plan constructor passes a cached-attention capability flag into prepare and owns persistent arena/uniform/scratch state. |
| `omega/src/metal/arena_encode_dispatch_finish.rs:124` | Stable arena allocates slots while calculating liveness. The configurable limit must be checked before slot allocation, not after this loop. |
| `omega/src/metal/pipeline_buffers_upload.rs:493` | Output allocator either uses tight bytes or a power-of-two pool bucket. Charge physical bucket capacity. |
| `omega/src/metal/arena_encode_dispatch_finish.rs:269` | Plan uniforms are actual MTLBuffer allocations separate from output arena slots. |
| `omega/src/metal/arena_encode_dispatch_finish.rs:429` | Attention scratch already has a shape/policy sizing function. Reuse it. |
| `omega/src/metal/resident_nocopy_cache.rs:205` | No-copy uploads can expose a page-rounded mapping; charge the reachable buffer span, not merely the logical tensor view. |
| `omega/src/metal/device_buffers_arena_plan.rs:674` | Memory facts report maximum buffer length, recommended working set and allocated bytes. Recommended working set is a fact, not a driver-enforced global reservation. |
| `omega/src/metal/device_buffers_arena_plan.rs:2072` | Refit changes shapes/resolution and can replace uniforms; resource checks also belong before refit mutation. |

The environment exposes no callable `memory_search`/`memory_list` tool. The repository bootstrap/index/routes/invariants were read. The index points to a missing `ai_docs/decision-tensor-weight-conversion-cache.jsonl`; this packet does not infer a converted-weight cache implementation from that pointer.

## Minimal public adaptation

Add to `omega/src/backend.rs`, retaining legacy explicit-argument APIs:

```rust
pub fn plan_named_with_config(
    config: &RuntimeConfig,
    program: &[Op],
    symbols: &[u64],
    named: &[(&str, QuantizedBlock<'_>)],
    outputs: &[NodeId],
) -> Result<(Plan, Vec<CandidateMeasurement>), BackendError>;
```

This is the existing planning seam with policy resolution. `Plan` remains the existing enum. A configured call returns one report row for its selected candidate with no timing samples. A measured call returns one row for every attempted candidate, including typed rejection. Callers already own/reuse `Plan`; no process-global measurement cache is needed.

`CandidateMeasurement` is a caller-owned execution artifact, colocated in `backend.rs`: `engine: Engine`, `driver: Option<GpuDriver>`, `device_identity: String`, `cold_plan_ns: Option<u128>`, `warmup_ns: Vec<u128>`, `execution_ns: Vec<u128>`, `execution_sum_ns: Option<u128>`, `selected: bool`, `rejection: Option<String>`, `host_tensor_reserved_bytes: Option<u64>`, `device_buffer_reserved_bytes: Option<u64>`, `largest_device_buffer_bytes: Option<u64>`, `resolved_op_count: usize`, and `kernel_identities: Vec<String>`. The exact arithmetic mean is `execution_sum_ns / execution_ns.len()` as a rational, not rounded integer division. Absence means unavailable/unmeasured, never zero. `rejection` preserves backend error type in the emitted structured event; its string is display data in the report. Do not add a trait-object report sink.

Add existing `BackendError` variants `UnsupportedConstraint { backend, field, reason }`, `ResourceLimit { backend, field, requested, limit }`, `ResourceOverflow { backend, field }`, `InputContractChanged { node, reason }`, and `NoAdmissibleCandidate { candidates: Vec<CandidateMeasurement> }`. The last error preserves every attempted candidate row even when no plan can be returned. Backend-owned helpers return existing backend errors or propagate these common errors before calling the concrete backend; never translate resource refusal to a CPU fallback.

## Fields that have consumers

Replace the broad resource/cache proposal in the config packet with these ceilings:

| RuntimeConfig field | Exact consumer/scope |
|---|---|
| `constraints.max_host_tensor_bytes: Option<u64>` | Upper bound on owned host tensor payload and floating point operation workspace reachable from this plan/current call. Includes retained CPU scratch capacity and GPU host upload conversion/readback arrays. Excludes borrowed inputs, IR/shape/index metadata, driver/compiler allocations and outputs retained by the caller after return. This is not process RSS. |
| `constraints.max_device_buffer_bytes: Option<u64>` | Conservative upper bound on distinct physical driver buffer capacities reachable from this plan plus current execution temporaries, including input/output arena, uniforms, fault buffers, attention scratch, merge/base tables, readback buffers and replacement coexistence. Charge shared buffers once per physical object within this plan; each plan pays the full reachable capacity, even for global-cache reuse. This is not whole-device VRAM or a cross-plan reservation. |
| `constraints.max_single_device_buffer_bytes: Option<u64>` | Upper bound on every device buffer physical capacity that the plan uses, including reused/no-copy buffers. Hardware per-buffer limits apply independently. CPU rejects this configured field because it has no device buffers. |

All ceilings are optional positive byte counts. `None` preserves the existing unrestricted allocation policy. A configured field without a real bounded path causes `UnsupportedConstraint` at cold planning. Do not accept `max_host_bytes`, per-process RSS caps, cache entry limits, CPU utilization quotas, deadlines or converted-weight byte limits in this change. Driver/compiler RSS and global caches may be reported as measurements; they are outside these ceilings.

Use the revision-2 config packet's exact nested fields: `numeric: NumericPolicy`, `constraints: ResourceConstraints` holding the three optional ceilings above, and `selection: SelectionConfig` holding `mode: SelectionMode` (`Configured` default, `MinimumMeasuredWallTime` alternative), `warmup_runs: Option<u16>` and `measurement_runs: Option<u16>`. Retain the four legacy fields. Optional counts default to `None`; `selection.resolved_warmup_runs()`/`selection.resolved_measurement_runs()` read the generated `sized` floor. `Some` supplies an explicit runtime override. Conflag Settings only supports literal defaults, so do not write an unsupported sized expression inside `#[setting(default=...)]`. The generated `[runtime]` floors seed warmups one and measurements five; measured mode validates resolved warmups at least one and resolved samples at least three. Configured mode rejects explicit trial-count overrides because it never consumes them. Delete measurement-cache capacity and nested cache sections. No data field without a consumer may land.

## Numeric policy, exact activations and all fusion

1. Config parsing accepts portable data (`cpu`, `gpu`, `auto`, `metal`, `wgpu`, `cuda`) independent of the compiled hardware. Planning checks feature and device availability. NPU/ANE/Vulkan are explicit unsupported names.
2. Resolve `numeric` once and pass every permission to each binder and emitter unchanged. Permissions allow rewrites; they do not require a backend to use every permitted rewrite.
3. `exact_activations=true` selects the existing exact CPU quantized path. `false` permits its existing activation-quantized path. GPU kernels consume original activation data already; `false` is permission, not a request to invent GPU activation quantization.
4. `allow_fusion=false` suppresses chain composition, reduce epilogues, cached attention, GDN, MoE/top-k/round grouping and execution-local epilogue fusion. Shape resolution, dead-node pruning and validation remain. `true` permits only backend-supported rewrites admitted by the existing `NumericPolicy`.
5. Consume `WORK-PACKET-BINDING.md`'s additive `bind_unfused(program, shapes, outputs, numeric_policy)`. Configured code chooses that entry when fusion is false, otherwise the existing backend-supported fused entry. Exact CPU seam: `evaluate_quantized_named_with_config(program, symbols, named, outputs, (free_buffers, validated_weight_nodes), expert_sources, &TensorExecutionConfig)`. Extend the existing `TensorExecutionConfig` with numeric policy, fusion, exact-activation and host-payload-bound fields instead of introducing a peer options type or an eleven-argument public API. Its parallel-control fields remain unexposed by Omega. Existing CPU implementations delegate to the shared private body with their historical defaults. This configured entry is config/std-gated; the existing binder and legacy evaluator retain their restricted tiers.
6. Metal supports its existing cached-attention and epilogue forms; CUDA passes cached-attention=false; wgpu passes cached-attention=false and reduce-epilogue=false. Requested fusion is permission, so a backend can use the original graph when that lowering is absent. An unsupported op in the original program remains a typed failure.
7. Metal math mode uses the existing `numeric_policy_as_metal_math_mode` projection. Selection cannot widen numeric flags or change exact activation permission to obtain a faster candidate.

## CPU constrained allocation mechanism

Constrained CPU planning resolves names, verifies shapes/codecs, binds with the requested options and computes a conservative float payload envelope before allocating execution payload. Its first bounded contract admits affine Float32 inputs and ordinary `Elementwise`, ordinary `Reduce` (including scan with affine operands), `Iota`, `Constant`, and affine reduce epilogues. Reject constrained packed/expert/gather/scatter/specialized attention/GDN/MoE/round programs with the first node and missing workspace bound. The unrestricted CPU path retains all existing operation support.

At `proxima-tensor/src/cpu/quantized_eval.rs`, compute checked bytes:

```text
owned_tensor = sum(4 * logical_elements for every reachable computed output)
copied_outputs = sum(4 * elements for requested outputs that are borrowed input nodes)
retained_pool = sum(4 * capacity for every free_buffers Vec)
elementwise_workspace(node) = 4 * (operand_count + step_count * max(1, min(output_elements, GENERIC_WIDTH_TILE)))
reduce_workspace(node) = 4 * (operand_count + step_count + output_width)
scan_workspace(node) = 4 * (operand_count + step_count)
epilogue_workspace(node) = 4 * (epilogue_operand_count + 1 + epilogue_step_count)
reserved = retained_pool + owned_tensor + copied_outputs + max(node_workspace + epilogue_workspace)
```

This deliberately charges all output payloads rather than claiming an optimal liveness bound. Workspace includes float arrays, excludes integer coordinate/layout metadata, and is checked using the same backend-owned shape/width resolution helpers used by execution. No hardware-independent guessed workspace coefficient. For constrained CPU do not create a MatmulSession/cohort or staged packed path; admitted affine Float32 programs use existing sequential dispatch. This is a bounded execution choice, not a thread-utilization contract.

Retained pool capacity is real capacity, not length. Before a repeated call, trim surplus free buffers only as needed to fit the new envelope, then recompute it. Reuse buffers whose capacity suffices. Never increase a buffer after admission without checking its new capacity; use `try_reserve_exact` and charge returned capacity before using it. On allocation error, leave retained buffers valid and return typed failure. Nodes retire back into the same pool. `Evaluated` owns returned outputs; its outstanding payload after return belongs to the caller and is explicitly outside the next call's plan bound.

CPU input shape/codec changes are rejected before buffer-pool mutation. A fresh plan is needed for a different shape or codec. Ordinary data changes with identical shape/codec remain accepted.

## GPU allocation mechanism and exact sites

Add ceilings/reservation fields directly to the existing concrete plan structs. Use backend-owned `checked_buffer_layout` helpers to build a conservative list of physical allocation lengths without allocating driver buffers. The same sizing helpers feed allocation descriptors. Common ceiling comparisons live in `backend.rs`; concrete driver sizing stays with each driver. A reservation list is cold metadata, not a new runtime abstraction.

Before initial plan payload allocation and every execute/refit, calculate `persistent_reachable + current_call_temporaries + replacement_overlap`. Reject checked arithmetic overflow. Validate every single size against the configured single-buffer ceiling and real hardware limit. Compare the sum to the configured total buffer ceiling. Reserve/check before any upload or dispatch. For caps requested on a lowering with unaccounted workspace, fail at planning; do not estimate it with output size alone.

| Backend/file | Functions to change and sizing/accounting requirements |
|---|---|
| `omega/src/wgpu_driver.rs` | `plan_with_policy`/`plan_named_with_policy`: add options-bearing siblings used by common planner; store config ceilings and declared input contract. `execute_plan`: before input uploads, size current capacities plus prospective replacement input/output, all `pack_uniforms` buffers, all gather fault buffers, all output/fault staging. `storage_buffer`: take checked physical size. Check `device.limits().max_buffer_size`, storage binding sizes and existing binding-count limit. `pipeline_for` retains existing per-plan pipeline reuse; its opaque compiler allocation is excluded. |
| `omega/src/cuda_driver.rs` | `CudaDriver::plan`: options-bearing sibling; persist policy/ceilings and input contract in `CudaPlan`. `upload_f32`, `upload_bytes`, `upload_uniform`, `materialize_constant`, `launch_f32_persistent_sized_with_sync`: use common per-arena checked sizing; account both old/new buffer while replacement exists. `CudaF32Arena` gains byte-capacity accounting alongside allocation count. `CudaPlan::execute_named`: full prospective footprint check before the first upload/launch; include per-kernel fault buffers and returned host vectors. Query `memory_info` before additional device allocations as an observed capacity fact; it is not a race-free device reservation. Stream failures release temporary reservations after synchronization/error cleanup. |
| `omega/src/metal/execute_and_hazards.rs` | `plan_with_placed_inputs`: options/ceilings carried into existing `Plan`; prepare with requested fusion. Compute allocation layout before arena/uniform resolution. |
| `omega/src/metal/prepare_uniforms_pack.rs` | `prepare`: use binder options rather than treating the cached-attention flag as all fusion. Preserve existing packed codec classification and shape validation. |
| `omega/src/metal/pipeline_buffers_upload.rs` | `allocate_buffer`, `pool_bucket`: shared checked tight/bucket size calculation used by preflight. `upload_block` and conversion sizing: include uploaded bytes and converted temporary payload. Charge existing pooled capacity, not the requested smaller tensor. |
| `omega/src/metal/arena_encode_dispatch_finish.rs` | `build_buffer_arena`: first compute slot sizes/liveness, compare budgets, then allocate slots from that layout. `build_plan_uniforms`, `attention_scratch_buffer`, `shared_buffer_from`, `encode_op`: preflight uniforms, attention scratch, faults, descriptors/temporary tables. Reuse `cached_attention_scratch_len`/`shared_scratch_elements`; do not duplicate attention formulas. `refit_uniform_buffers`/`apply_refit`: compare prospective old+new coexistence before replacing buffers. Readback contributes owned host float vectors. |
| `omega/src/metal/device_buffers_arena_plan.rs` | Existing `Plan` gains ceiling/reservation/layout state. `ensure_merged_group_resolved`, `upload_base_table`, `ensure_round_group_resolved`: account merged output/base table capacities before allocation. `refit_symbols`: resource validation before committing patches/shapes. |
| `omega/src/metal/resident_nocopy_cache.rs` | `create_no_copy_buffer`, `upload_block_copy`, `upload_block_as_half`, `allocate_fault_buffer`, `upload_uniforms`: use shared layout lengths. Cache hit buffers are charged by actual `length()` and deduplicated by physical buffer pointer within this plan. Registered whole-mapping buffers are charged by whole buffer length. |

Metal's existing maximum-buffer hard limit is enforced. Recommended working set and `currentAllocatedSize` are reported separately; do not use them as proof of whole-device occupancy enforcement. Global buffer pools/caches remain governed by their existing policy; bytes they retain but this plan cannot reach are outside the documented per-plan bound. A caller requiring a process/global bound gets an unsupported-field error because that field does not exist in this config.

Repeated execution revalidates input names/shapes/codecs before mutation. Same-shape new data is accepted. Same logical element count with different packed codec is rejected before choosing old kernel/buffer. Resource failure before dispatch preserves usable plan state. If a driver failure occurs after submit, finish/synchronize the submitted work before releasing its buffers. Drop of an unselected candidate releases its plan-owned buffers; shared cache objects retain their existing cache ownership. Selection tests must account for this distinction.

## Selection and tuning at the cold boundary

Configured mode resolves one engine/driver and plans it, with no trial execution. `gpu/auto` follows the existing target preference and checks actual availability; an explicit unavailable driver errors. GPU failure does not imply a CPU fallback.

Measured mode evaluates a finite ordered candidate set: configured target first, then CPU, Metal, wgpu, CUDA, deduplicated by engine/driver. Only compiled arms are attempted; discovery, planning, resource and unsupported-graph failures produce rejection rows. Each candidate uses identical program/symbols/input bytes/outputs/numeric flags/fusion/exact activation permissions. Each candidate drops before the next except the best current plan, whose reachable reservation must not be charged as free memory. Device probes report identity for attribution.

Time cold planning with `Instant` around the real planner. Keep it separate from warmup and steady execution. For every admitted candidate, run configured warmups, then each configured sample through the complete existing `execute_plan_named` call. The interval ends only after output readback/error checks: Metal command buffer completion, wgpu mapping/poll completion, CUDA stream synchronization, CPU return. Release each result before the next sample; do not keep samples' tensor payloads alive. Retain every integer nanosecond observation; select by arithmetic mean synchronized execution wall time, then stable candidate order for ties. Compare exact checked sums divided by counts without rounding; configured sample counts are equal, so direct checked sum comparison is sufficient. Keep sample arrays in original order. This establishes best observed admissible candidate on those inputs, not a global minimum or performance guarantee.

No implicit tuner widens precision. Existing Metal math mode is projected from policy. Existing dispatch type and command-buffer chunks are not newly exposed fields in this minimal packet; their current callers continue to use their existing setters. Do not build a Cartesian sweep of unexposed env knobs. Initial measured candidates use fixed resolved controls. Renderer knobs/feature gates and tiled geometry continue to belong to their existing emitters; adding measured kernel forms requires a shape-applicable lowering, a separate identity, same semantic policy and raw samples before admission. The separate tiled packet's buildcfg `omega_dense_float_storage` derives from `[experimental].dense_float_storage=0` through existing `SizingSource`, with `OMEGA_EXPERIMENTAL_DENSE_FLOAT_STORAGE=1` as its explicit experiment override; it does not add a selectable hardware backend or a parallel runtime.

Environment overrides read by existing renderers must be captured in report identity, and must not change between candidate/sample runs. Prefer explicit config at planning for the admitted controls. No environment mutation inside execution.

## Hand-authored acceptance fixtures

All examples below use actual `Op`/`append`/`IndexMap` constructors, `NodeId` and named `QuantizedBlock::Float32` slices. Expected vectors are literal, not produced by another backend under test.

Fixture A: names `input=[1,-2,3,-4]`, `bias=[10,20,30,40]`, rank-one extent four, `sum=input+bias`, `negated=-sum`, outputs `[sum,negated]`. Expected outputs `[11,18,33,36]` and `[-11,-18,-33,-36]`, each shape `[4]`. Repeated call changes input to `[2,-1,4,-3]`; expected `[12,19,34,37]` and `[-12,-19,-34,-37]`.

Fixture B: `left=[[1,2,3],[4,5,6]]`, `right=[[7,8],[9,10],[11,12]]`, affine multiply/reduce contraction. Expected output shape `[2,2]`, values `[58,64,139,154]`. This is a correctness fixture; llama.cpp timings are not its numeric oracle.

Fixture C: reservation accounting receives physical buffer rows `input=16`, `bias=16`, `sum=16`, `negated=16`, uniform buffers `24` and `16`, readback buffers `16` and `16`, plus four explicit upload staging buffers `16`, `16`, `24`, `16`. Expected conservative simultaneous total `208`, largest `24`, physical objects `12`. These literal uniform sizes follow wgpu `pack_elementwise_uniforms` for rank-one binary and unary bodies. Configured uploads use caller-owned mapped staging and encoder copies because `Queue::write_buffer` hides staging ownership in the driver. Staging survives through synchronized completion. A cap `208` admits; `207` refuses before input upload. A single cap `24` admits; `23` refuses its binary uniform. GPU driver objects/compiler bytes are not part of this sum. Host readback reservation is `48` bytes: both result vectors contribute `32`, and the current mapped byte copy contributes at most `16` alongside them.

Wgpu host workspace additionally reserves the maximum transient uniform serialization vector, taking the maximum of that phase and the readback phase because serialization ends before readback. `pack_uniforms_byte_len` counts the exact existing packer's fields with checked arithmetic and allocates no serialization vector during admission. Fixture C remains host `48` because its largest uniform serialization is `24`. CUDA's precompiled `CudaPlan::uniforms` vectors are retained plan compilation metadata and explicitly excluded from the host tensor/workspace ceiling; their uploaded device buffers are included. CUDA host tensor payload is returned output vectors or constant materialization workspace, whichever phase is larger. Compiler, driver and IR metadata allocations are outside both backends' host tensor ceilings; neither field claims total RSS.

Fixture D: generic resource ledger rows `[16,32,8]`, same second object referenced twice, replacement `64` created while old `32` remains. Expected existing distinct capacity `56`, temporary coexistence `120`, post-replacement capacity `88`; release temporary fault row `8` produces `80`. Test duplicate references do not double-charge, but replacement overlap does. These are accounting fixtures, not measured hardware occupancy.

CPU Fixture A with fusion disabled has two reachable owned output vectors, total `32` bytes. Conservative scalar workspace via the formula above is binary `4*(2+1*4)=24`, unary `4*(1+1*4)=20`; reserved without retained pool `56`. The actual unary/binary fast paths allocate less; this is the declared conservative admission envelope. Cap `56` admits, `55` refuses before allocation. A retained free vector of capacity eight contributes `32` bytes; trim it if it would prevent otherwise feasible admission. Do not confuse capacity with its current length.

## Named tests and exact count gates

Each named item is one non-parameterized test; expected counts below do not include unrelated tests. Unit tests use plain sync `#[test]`; parameterized/async extensions use `#[proxima::test]`. No sleeps. Fake candidate timing takes literal sample arrays and drives a pure selector helper; hardware timing never uses a fake clock.

Common planning group, **12 tests**, prefix `config_plan_`:

1. `config_plan_cpu_executes_named_two_outputs` — Fixture A, both exact vectors/shapes/output IDs.
2. `config_plan_cpu_reuses_same_shape_with_fresh_data` — Fixture A repeat vectors.
3. `config_plan_cpu_exact_flag_reaches_evaluator` — assert the stored flag and policy passed to existing evaluator; use exact/ordinary packed activation fixture from root CPU packet for execution difference.
4. `config_plan_numeric_bits_reach_cpu_binder` — contraction and approx-functions true, other four false, captured binder options equal those literals.
5. `config_plan_fusion_false_keeps_chain_nodes` — Fixture A has two elementwise emitted nodes with separate outputs and no fused body; positive fusion fixture from root binder packet proves opt-in actually changes emitted structure.
6. `config_plan_explicit_unavailable_driver_does_not_fallback` — fake unavailable wgpu discovery yields its typed failure and zero CPU attempts.
7. `config_plan_npu_ane_vulkan_names_are_rejected` — three invalid names, each explicit unsupported error.
8. `config_plan_changed_input_codec_fails_before_mutation` — planned Float32 receives packed block for same node; zero uploads/dispatches/pool changes.
9. `config_plan_resource_sum_checks_overflow` — rows `[u64::MAX,1]`, expected overflow, zero allocation requests.
10. `config_plan_resource_replacement_counts_overlap` — Fixture D totals `56/120/88/80`.
11. `config_plan_device_single_cap_checks_reused_capacity` — logical view `16`, physical reused capacity `32`, cap `24`: refuse; no upload.
12. `config_plan_output_negative_control_detects_wrong_vector` — first pass the literal oracle `[11,18,33,36]` through the same actual-output comparator and assert zero mismatches among four coordinates. Then compare with deliberate `[11,18,33,37]`; exactly one mismatch reports output sum index three, expected 36/observed 37. The test passes by asserting rejection, not by accepting wrong data.

CPU bounded group, **5 tests**, prefix `config_cpu_budget_`:

1. `config_cpu_budget_admits_literal_envelope` — Fixture A cap `56`, literal vectors and reservation `56`.
2. `config_cpu_budget_rejects_one_byte_below_envelope` — cap `55`, requested `56`, zero execution payload allocations.
3. `config_cpu_budget_trims_retained_capacity_before_repeat` — retained vector capacity eight/length one; cap `56`, pool trim then repeat vectors; reservation stays bounded.
4. `config_cpu_budget_rejects_unbounded_specialized_workspace` — constrained cached-attention/packed specialized node names its missing workspace bound before allocation.
5. `config_cpu_budget_rejects_device_only_constraint` — CPU max-single-device ceiling names unsupported field.

Pure selection group, **5 tests**, prefix `config_selection_`:

1. `config_selection_configured_executes_no_trials` — one candidate row, zero warmup/sample runs.
2. `config_selection_stable_order_deduplicates_configured` — configured wgpu, all arms available: order `[wgpu,cpu,metal,cuda]`; no duplicates.
3. `config_selection_mean_uses_raw_sample_order` — CPU samples `[9,2,5]`, Metal `[4,4,4]`: sums `16/12`, counts `3/3`, exact means `16/3` and `4`, Metal selected, report arrays retain literal original order.
4. `config_selection_tie_uses_stable_order` — configured CPU `[3,5,4]`, Metal `[4,3,5]`: both sum twelve/count three, CPU selected.
5. `config_selection_rejections_are_retained_and_empty_set_errors` — CPU resource refusal, Metal absent, wgpu unsupported-op, CUDA absent: four rejection records and `NoAdmissibleCandidate`.

Hardware driver group, **4 tests per available driver**, prefixes `config_metal_`, `config_wgpu_`, `config_cuda_`:

1. `...named_program_and_repeat` — Fixture A initial/repeat vectors and shapes; actual device identity included. The same device-output comparator first accepts the literal oracle `[11,18,33,36]` with zero mismatches and detects exactly one mismatch for `[11,18,33,37]`.
2. `...affine_matmul_named_program` — Fixture B literal output vector.
3. `...cap_failure_precedes_upload_and_dispatch` — exact backend-owned resource row table from first test; set total below its required reservation, assert zero upload/submit calls. For wgpu use literal Fixture C `207/208`; for CUDA/Metal record packed-uniform literal lengths after opening their packer assertions, do not assume wgpu ABI.
4. `...same_shape_execution_resource_census` — execute repeated Fixture A; show persistent physical buffer IDs/capacities and all transient rows; sum distinct resources and compare to admission reservation; deliberate duplicate reference must not inflate total.

Hardware tests are gated by the actual feature/target and report absent hardware as an availability result, not a skip masquerading as execution. CPU-only tests simulate unavailable arms. CUDA compilation does not claim CUDA runtime hardware. NPU has zero hardware tests because it has no driver.

```text
cargo nextest run -p proxima-omega --lib -E 'test(config_plan_)'
expected: 12 passed, 0 failed, 0 skipped
cargo nextest run -p proxima-omega --lib -E 'test(config_cpu_budget_)'
expected: 5 passed, 0 failed, 0 skipped
cargo nextest run -p proxima-omega --lib -E 'test(config_selection_)'
expected: 5 passed, 0 failed, 0 skipped
cargo nextest run -p proxima-omega --lib --features metal -E 'test(config_metal_)'
expected on macOS Metal host: 4 passed, 0 failed, 0 skipped
cargo nextest run -p proxima-omega --lib --features wgpu-backend -E 'test(config_wgpu_)'
expected on available wgpu host: 4 passed, 0 failed, 0 skipped
cargo nextest run -p proxima-omega --lib --features cuda-driver -E 'test(config_cuda_)'
expected on available CUDA host: 4 passed, 0 failed, 0 skipped
```

Run native build checks after tests: `cargo check -p proxima-omega --all-targets`, `cargo check -p proxima-omega --no-default-features --features std,cpu`, `cargo check -p proxima-omega --no-default-features --features alloc,cuda`, and `cargo clippy -p proxima-omega --lib -- -D warnings`. These are compile/lint supplements, not count-bearing behavior acceptance criteria. Root will assign build/test execution to a bounded worker. Do not write or run shell/Python/Perl gate scripts. Actual example/doctest commands and their nonzero counts must be admitted after inspecting existing assertions; do not invent a count for an unopened target.

## Dependency order and file ownership

1. Root CPU/binder slice: options-bearing binder and CPU policy/workspace preflight in existing `proxima-tensor/src/bind/{builder_compose_window,cached_attention_epilogue_liveness,gdn_moe_fusion_apply}.rs`, `cpu/{quantized_eval,elementwise_matmul,run_reduce_scan,run_node,mod}.rs`, relevant exports/tests. No backend implementation starts before these exact contracts are admitted.
2. Config slice: `omega/src/config.rs`, generated defaults via `omega/omega-runtime.toml`/`build.rs`/`sized.rs`; Settings/serde/fluent consistency. Remove proposal fields without consumers.
3. Execution worker: `omega/src/backend.rs`, `wgpu_driver.rs`, `cuda_driver.rs`, listed Metal plan/prepare/allocation files; private resource sizing helpers and named tests. Preserve legacy constructor defaults while the new config path binds all resolved options.
4. Cold selection worker (same execution ownership): finite candidate ordering, real complete-call timing and caller-owned report. No selection until resource policy actually reaches every selected backend.
5. Root validation/benchmark worker: raw candidate rows and numeric boundary vectors; update TASKS and evidence ledger. Tiled MSL geometry belongs to its separate worker; execution changes may only consume an admitted existing lowering/candidate identity.

Each dependency finishes its named assertion count before dependent integration. No commits without owner authorization. No Rust edits before root spec admission.

## Performance evidence and future NPU extension

llama.cpp is the performance oracle. Existing Proxima is the before-change baseline. Root's oracle packet owns pinned source/build/model/token/prompt identity and native commands. For matched workloads capture every cold duration, warmup/sample, synchronized wall execution, bytes/count resource rows, CPU/GPU counters where available, process RSS as measurement, and tensor output vectors separately. Preserve losses and errors alongside gains. Candidate means are selectors; they cannot by themselves support a causal speed claim. Read traces/lowering/source identities for each changed candidate. Synthetic fixtures above are semantic/resource controls and are not llama.cpp performance comparisons.

NPU/ANE extension requires a real engine/driver/Plan arm, availability probe/device identity, backend-owned supported-op/dtype/codec/fusion/numeric capability checks, shape inference/binding into executable lowering, checked buffer/workspace accounting with cleanup/refit/repeat semantics, synchronized complete-call timing/readback, literal numerical fixtures and matched oracle workload. None of those contracts is supplied by the current narrow MLX GDN bridge. Keep NPU/ANE names rejected until those implementations exist; do not add a selectable placeholder, pretend MLX dispatch selects ANE, or synthesize a hardware support claim.
