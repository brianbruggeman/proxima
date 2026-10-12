# packet: Omega scheduling, capabilities, residency and strategy seams

**Route:** GPT-6 Sol (preselected). The task changes public planning and
execution contracts across graph, CPU/Metal/CUDA/WGPU owners, with numerical,
allocation, dependency and backend-selection consequences. Luna may run bounded
builds and collect artifacts after the interface is fixed.

**Owned paths:** `omega/src/backend.rs`, new files only below `omega/src/backend/`,
existing Metal/WGPU/CUDA driver files as actually needed,
`proxima-tensor/src/bind/` only where existing binder consumers need the policy,
`proxima-tensor/src/config.rs`, `proxima-tensor/src/cpu/quantized_eval.rs`,
`proxima-tensor/src/cpu/gemm_q8k.rs`, and `proxima-tensor/src/cpu/elementwise_matmul.rs`
for explicit per-evaluation worker propagation, and `omega/tests/` for
cross-driver contract assertions. Do not edit
`omega/src/config.rs` (packet CONFIG) or model serving/decode state (packet
SERVING). Root owns source integration if a shared signature must change.

**Existing APIs:** `RuntimeConfig`, configured planner entry points,
`Plan` concrete variants and `backend::set_dispatch_type` /
`set_command_buffer_chunks`; existing bind `Program`/`Op`, `NumericPolicy`,
allocation owners and dependency/hazard maps. Backend candidate measurement
selects a plan; it is not distributed execution. CPU worker count currently
reads `PROXIMA_MATMUL_WORKERS` in `proxima-tensor/src/cpu/gemm_q8k.rs`.

**Required integration:** consume resolved per-plan worker and in-flight caps;
preserve DAG ordering, alias/hazard/lifetime edges, cancellation and numeric
policy; report actual worker/queue/device/kernel choices and rejected
candidates. Add real concurrency only at owners with explicit ownership,
completion and resource limits. Preserve zero-allocation behavior where
existing hot paths provide it; count any new setup/hot-path allocation. Extend
current tiling/fusion/placement/cache mechanisms, not a second plan/IR/executor.
Only expose a strategy when it changes existing execution or returns a typed
capability error.

**Full-field evidence:** supply actual consumer/rejection events for the 26
distinct paths in `fixtures.json#/policy_consumption`, with serving supplying
its owned paths. Each `execution_result` has non-null `field_path`,
`requested_value`, `effective_value_or_error`, `owner`, `status`, and `source`.
Use `consumed` only with the actual effective value and a retained event/result
source; use `typed_unsupported` or `rejected_by_validation` only with a named
error variant and field from the actual rejection. A static `owner_mapping`
row is inventory, never execution evidence. E2E assembles the shared-owner
artifact at `results/executor/policy-consumption.jsonl` and implements the AC2
native recount command. Require exactly 26 unique paths and zero missing,
duplicate, extra, owner_mapping or unaccounted rows.

**Worked DAG:** literal edges `(0,1),(0,2),(1,3),(2,3)`. Expected execution
levels are `[(0),(1,2),(3)]`; serial and overlapped schedules both return the
independent oracle vector `[11,18,33,36]`. Force `3` before `2`; the hazard
checker returns exactly one violation. Under capacity `64`, allocations
`24+24+16` fit and an additional `1` is refused before allocator count rises.

**Invariants:** semantic numeric options never derive from a performance
strategy; cache key includes effective format/geometry/backend; device
unavailability never falls through to CPU; cancellation cannot reclaim
submitted buffers early; transfer/compute overlap claims require overlapping
driver timestamps and joined command identities; distinguish host commit-to-
wait ticks from device busy time; strategy defaults remain unchanged unless an
admitted measured result explicitly changes them.

**Dependency order:** unified config first; then backend plan/capability
consumer; then serving can submit compatible rows. Do not invent independent
multi-device execution without a driver owner. Compile backend feature
contracts before timing.

**Validation:** AC2, AC5, AC6, AC7, AC12 and AC13 in `SPEC.md`; add explicit
native build checks per available driver. Tests must report selected count,
excluded count and actual hardware execution count separately. Only one GPU
timing process may run at a time.

**Negative controls:** missing dependency edge, corrupted oracle coordinate,
one-byte resource overflow, stale cache generation, forbidden reassociation,
unavailable required driver and device request that attempts CPU fallback.
The AC2 row checker also accepts the independent 26-row literal schema control
and rejects exactly once when its first row's status becomes `owner_mapping`.

**Artifacts:** `results/executor/` for allocation rows, dependency events,
candidate rejection/effective selection, source/dispatch digests and all
positive/negative literal comparisons. Preserve slower/no-overlap arms.

**Isolation:** after admission, own worktree is
`/private/tmp/proxima-unified-executor` on branch `codex/unified-executor` and
build output `/private/tmp/proxima-target-unified-executor`. Executor work is
serialized with serving's GPU timing run; only one Metal timing process runs at
a time. Return reviewed patches and command output for root integration.
