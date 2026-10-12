# round 2 synthesis: unified programmable execution

Status: proposed plan AB2; no implementation or execution result is asserted.
All paths below are relative to this directory unless a crate path is named.

Provenance: incumbent preserved at `PLAN-AB-R1.md:20-85` (AB); blind
`PLAN-C.md:9-39` (C); `PLAN-R2-CRITIQUE.md:8-14` (K), all opened directly.
K identifies missing physical high-water capture cadence
and policy-owner instrumentation order. The acceptance contract remains
`SPEC.md:200-217`, including R16 at `SPEC.md:34`. Numeric values below are
required counts or proposed capture settings, never observed measurements.

1. **Freeze source and establish observation before changing owners.** [AB1,
   C1; K ordering/observability] Preserve the existing resume snapshot and
   checkpoint9 raw failure under `results/worktree-snapshots/` and
   `results/checkpoints/e2e-checkpoint-9/`. Bind source diff, executable digest,
   model digests, input IDs, precision, context, output budget, arrivals and
   instrumentation settings in each new checkpoint manifest. Open the vmmap,
   resource, plan-source and run-status records named by
   `WORK-PACKET-RESIDENCY.md:20`, retaining unmatched ranges.

   Instrument current allocation/reservation owners and actual policy consumers
   before their behavior changes. Config owns requested/resolved policy;
   backend/driver owners emit allocation, dispatch and rejection results;
   serving owns admission, sequence and scheduler consumption. The runner joins
   their events; it cannot invent effective values. Preserve the initial
   consumer evidence, including absent or unsupported consumers, then recount
   the final changed owners in step 3. Keep all 26 AC2 field paths distinct.

   Specify the capture contract in `results/checkpoints/<id>/capture.json`:
   emit monotonically sequenced reservation/acquire/retire/alias/fence events
   at every ownership transition; sample system process physical footprint
   before and after startup, load, compile, dispatch, completion, fence
   submission/completion, cancellation and teardown transitions. Also schedule
   periodic capture every 10 ms from process launch until teardown completes,
   with a declared maximum lateness of 10 ms per scheduled sample. Record
   scheduled and actual timestamps, sampling duration, availability and source
   API, plus the native process physical-footprint lifetime high-water value.
   If that value is unavailable or resets, AC18 fails; periodic samples cannot
   substitute for it. Never collapse RSS,
   physical footprint, mapped residency or Metal allocated bytes. The checker
   rejects capped evidence on a missing transition sample, missed periodic
   deadline, unavailable required measurement, dropped event, sequence gap,
   counter reset or cap excursion. A sampled maximum is only an observed
   maximum; retain it alongside the native lifetime peak and record unobserved
   intervals explicitly.
   Admission reservations must conservatively cover those intervals, including
   runtime baseline and opaque driver allocations; unresolved accounting cannot
   support completed cap evidence. Capture buffers and compiler/source capture
   themselves belong in the memory budget.

2. **Integrate R16/AC18 memory ownership as one boundary.** [AB2, C1;
   K risk/coupling] Follow `WORK-PACKET-RESIDENCY.md:49-80`: reservation and
   lifecycle events first; finite mapped weight ranges and fence leases second;
   dense binding, embedding/output projection and bounded graph/Plan arenas
   third; exact KV and selected MoE expert ranges fourth; cache/compiler/source
   capture and old/new generation overlap closure last. One reservation owner
   accounts unique physical backing and aliases without double charging;
   unknown ranges remain charged and unresolved. Admission precedes allocation.
   Retirement follows the last completion fence, never discard advice alone.
   Preserve all graph-cut dependencies, full logits, reduction order, KV
   generations and post-router expert IDs. Do not retain every segmented Plan.

   Owner seams are mapping residency and GPU registration; `bind.rs` and
   `generate/model_weights.rs`; existing graph partition/Plan and Metal arena
   owners; `generate/residency_caches.rs` and `generate/decode.rs`; expert
   sidecar/slab owners; existing backend fences. Their concrete starting files
   are enumerated at `WORK-PACKET-RESIDENCY.md:9-16`. The shared reservation
   contract precedes all dependent edits; these owner changes are coupled and
   must not be represented as independent PRs. Focused owner diffs remain
   independently attributable within the boundary.

   Run exact-fit and one-byte-over controls before allocator/dispatch counters
   change, the literal ledger oracle and its mutation, plus the packet's lease,
   alias, graph-cut, numeric, KV, route and cancellation controls. Run the real
   Qwen dense/Granite MoE/llama checkpoint below before another integration.
   At this boundary, run AC18 lifecycle capture for Qwen dense and Granite MoE
   on the supported serial arm and retain separate AC13 llama attribution.
   After scheduling is integrated in step 5, run AC18 across every admitted
   model, workload and arm. Refusal or cap termination is incomplete execution:
   continue repairing this boundary without relaxing the ceiling or workload.

3. **Complete Omega configuration and actual policy consumption.** [AB3, C2;
   K ordering] Extend existing RuntimeConfig/build defaults, CPU, graph/backend,
   placement and driver owners; keep numeric permissions, hard constraints and
   strategies independent. Validate AC1 as well as AC2/AC5. The 26-row consumer
   ledger points to actual effective results or typed rejections from those
   owners; declarations and static owner mappings are excluded. Recount the
   ledger after memory changes and after later consumer changes. Run the real
   model checkpoint. Config/backend/serving share field identity, not ownership.

4. **Make serving resumable before scheduling.** [AB4, C3] Extend existing
   decode, PrefixState/cache and BackendRuntime owners, preserving sampler/RNG,
   token history, position, KV identity/generation and pending-buffer lifetimes
   through cancellation. Run AC3 literals/mutations and the real checkpoint.

5. **Integrate bounded scheduling with its feature disabled by default.**
   [AB5, C4] Reuse existing bounded Pipe/queue primitives; pack only compatible
   rows with explicit stream/position/KV/generation/sampler mapping. Preserve
   prefill service bounds, admission/backpressure and cancellation. Run enabled
   AC4 and disabled AC16 independently, AC17 paired scheduler microbenchmarks,
   and the real checkpoint. Source consumers introduced here refresh AC2 rows.

6. **Integrate strategies separately, with one checkpoint per strategy.**
   [AB6, C5] Order: decode kernels/fusion (AC6), exact cache/plan reuse with
   placement/overlap (AC7), permitted weight conversion (AC9), post-router MoE
   grouping (AC10), then target-verified speculation (AC8). Each uses existing
   owners and preserves numeric policy and route IDs. Retain complete output
   changes in both directions, source/dispatch joins and raw timing samples.
   Each strategy reruns affected memory, policy and state controls before its
   real model checkpoint; no later strategy can hide a failed earlier gate.

7. **Recount the full paired matrix and close remaining ACs.** [AB7-8, C6-7]
   Run AC11-13 for Ouro, Qwen, Granite and Gemma with both workload classes and
   serial/unified arms. Each runtime retains 96 cohorts and 192 request rows,
   including the prescribed warmups and five measured samples per cell. Recount
   exact model/input/precision/context/budget/arrival identity, output/timing
   lengths, routes and every resource/residency/dispatch join. Repeat AC18 for
   this final source condition. Run AC14/15 and update task and ai_docs records
   from opened artifacts; any source correction requires its controls and a new
   checkpoint. Preserve every unexplained observation as an explicit residual.

The real integration checkpoint used in steps 2-4 is exact Qwen dense and
Granite MoE on single and mixed-arrival workloads, through the currently
supported shared serial execution arm, paired with llama.cpp on identical
admitted inputs and settings. Do not count the existing rejected
`parallel_sequences=3` request as a unified model run. The
capture/instrumentation-only checkpoint runs both models on that currently
supported serial path and preserves cap-exceeding failure if present; after
the memory boundary, it must complete under cap.
Do not count the existing rejected `parallel_sequences=3` request as a unified
model run. From step 5, after the bounded scheduler is integrated and enabled,
each checkpoint exercises both serial and unified arms. Retain five raw measured
samples plus the admitted warmup; IDs/text/EOS, timing, CPU, memory, routes,
plan/kernel source hashes, cache, KV, allocation, dispatch and fence records.
Enforce the 500,000,000-byte Proxima physical-footprint ceiling from startup
through teardown using step 1's capture contract. Partial generation, refusal,
stopped processes, missing comparator rows, unavailable high-water, or rejected
telemetry leave that checkpoint incomplete. Final AC18 covers all four models;
this smaller checkpoint never substitutes for it.

Acceptance mapping (commands and literal payloads remain those in SPEC):

| Step | ACs and required observations |
|---|---|
| 1-2 | AC18/R16: Qwen and Granite, single and mixed-arrival workloads on the supported serial arm; lifecycle events and native high-water; exact-fit admitted and one-byte-over rejected before allocation; literal ledger unchanged and mutation rejected once |
| 5 and 7 | AC18/R16: every admitted model/workload/arm after scheduling integration; allocation before creation; zero unjoined allocations, unowned ranges and cap excursions; exact-fit admitted and one-byte-over rejected; literal ledger unchanged and mutation rejected once |
| 3 | AC1: 16 config tests; AC2: 18 tests and 26 actual consumer rows with no missing/extra/duplicate/owner-mapping rows; AC5: 12 capability tests with actual driver execution or typed rejection |
| 4 | AC3: 20 tests, literal packed/RNG state and seven identity mutations |
| 5 | AC4: 16 tests and exact fairness trace; AC16: four feature-off rejections and zero dispatch; AC17: five raw runs per arm at 2/8/32 streams, allocation/op and selection timing |
| 6 | AC6/AC7: 12 tests each; AC8/AC9/AC10: four tests each; each independent literal oracle and its specified corruption controls |
| 7 | AC11/AC12/AC13: complete four-model matrix and joins, paired comparator controls, raw samples and N-1 CoV; AC14: one alloc-tier Omega library artifact and zero diagnostics; AC15: zero lint/compiler diagnostics |

Resolve conflicts explicitly: both AB and C place memory before policy; retain
that integration order because owner allocations share the same live budget.
Move observation of policy consumers ahead of that boundary
to address K; complete their behavioral adaptation afterward. Retain AB's
strategy order because cache identity and converted storage precede MoE and
speculative consumers. Take C's explicit AC18/full-matrix closure and restore
AC1 explicitly. Cadence and failed-sample rejection are additions grounded in
K, not a new execution subsystem.

Rollback: save each boundary's exact starting diff and source condition, then
restore only responsible hunks if payload, lifetime or cap evidence refutes a
change. Rerun controls and the restored condition under a new artifact ID.
Keep the known full-file/full-Plan path available only as an explicitly
uncapped diagnostic condition; it cannot be a capped fallback. Do not raise
the limit or reduce admitted precision, context, concurrency, routes or token
budget. No commit, push or publication is included.

Artifact layout: immutable `results/checkpoints/<id>/{manifest,capture}.json`,
`proxima/` and `llama/` raw payloads, command output, exact patch/source hashes
and rejection records; memory rows under `results/memory-checkpoint-<id>/`;
policy evidence at `results/executor/policy-consumption.jsonl` with immutable
per-checkpoint copies; recounts under `results/recount/<id>/` and
`results/recount/policy-consumption/`; final matrix at `results/checkpoint-<id>/`
and `results/llama-paired-<id>/`. Preserve earlier failed rows without rewriting.
