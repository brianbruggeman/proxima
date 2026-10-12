# Resume: unified programmable execution and serving

Work in `/Users/brianbruggeman/repos/slot-0/proxima`. Read the applicable AGENTS.md
instructions. I authorize implementation, instrumentation, tests, benchmarks,
local integration and subagents. Generate a detailed executable plan, admit its
spec, then execute it. Do not stop after an inventory, plan, config struct or
isolated kernel. I want the shared interface and its execution integrations
buttoned up, with actual end-to-end evidence.

## Objective

Adapt the existing Proxima/Omega programmable interface so a typed
conflaguration/Settings input and equivalent fluent API describe resources,
workload policy, numerical permissions and performance strategy. Carry them
through admission, scheduling, lowering and actual execution. Parallelization
is a first-class part of this contract.

Reuse existing graph IR, model descriptors, bound weights, plans, kernels,
cache owners and Pipe primitives. Ouro is a workload that exposes shared
pipeline gaps. No separate Ouro runtime, duplicate transformer implementation,
second IR, or parallel configuration system. Preserve ordinary dense and MoE
models. Our relevant workloads include Ouro, Gemma, Qwen and Granite 1B/400M
MoE; inspect the local model artifacts for their exact identities.

The objective is the lowest measured wall time under supplied resource and
semantic constraints. Measure utilization to explain execution; do not promise
global optimality or full occupancy. llama.cpp is our PERFORMANCE oracle.
Numerical/semantic checks have their own independent fixtures and references.

## Startup: recover current work, do not restart it

1. Read the repository model routing policy.
   Use Luna for bounded source work, evidence extraction, tests and builds.
   Assign foundational scheduling/ownership/backend contract work to a model
   whose documented capability fits it, with the reason recorded before work.
2. Search persistent memory if available. If unavailable, record that and use
   `ai_docs/AGENT.md`, `index.jsonl`, `task-routes.jsonl`, matching invariants,
   `ouro-mainline.jsonl` and `omega-programmable.jsonl`.
3. Read `proxima-tensor/specs/ouro-mainline/{SPEC,TASKS,SCHEDULING-INVENTORY,EVIDENCE,DISCIPLINE}.md`.
   The scheduling inventory's final sections contain the latest source-traced
   parallelism/configuration gaps and performance proposals.
4. Read `proxima-tensor/specs/omega-programmable/{SPEC,TASKS}.md` and its work
   packets. These describe an earlier bounded implementation and contain old
   temporary-worktree paths. Reconcile against the current source/evidence;
   extend the contract for this request rather than rerunning old work blindly
   or treating old exclusions as cancellation of the current objective.
5. Inspect git status/diffs. The tree contains substantial uncommitted work.
   Preserve it. Do not reset, clean, overwrite unrelated changes, commit, push,
   publish or send messages outside this session without authorization.

Apply $guiding-principles, $disciplined-component, $conflag and $spec-first.
Use algorithm-development for known algorithms, instrument/bench-metrics for
performance, and proxima-pipe for scheduling composition. Follow their actual
instructions; do not turn them into a ceremonial checklist.

## Current implementation and evidence

- Shared Ouro graph/registry/serving integration, immutable HF weight ownership,
  configured Omega planning and BF16 tiled projection work exist in the tree.
- The latest repair preserves ordered generic broadcast reductions when numeric
  policy denies TreeReduce, reusing the original fold and cooperative output
  writes. Do not undo it to recover performance by violating numeric policy.
- The repair's exact commands, failures, source brackets, native outputs and
  independent artifact audit are in `ouro-mainline/results/BROADCAST-FOLD-COMMANDS.md`
  and `results/broadcast-fold-final-audit.md`. Fresh replay/C2/strict-live/cached
  cohorts were executed. Mainline TASKS rows are checked against those artifacts.
- Strict-policy live timings increased in the retained comparison. Relaxed
  cached timings are a separate condition. Preserve both; no causal speedup
  claim follows from comparing those checkpoints.
- The matched C7 llama comparison and actual dispatched MSL are in EVIDENCE.md,
  CALLER-C7-AUDIT.md and results/C7-PROFILE-AUDIT.jsonl. The tiled change's
  observed reduction was in long prefill, not decode. Short and repeated-ball
  fixtures are synthetic length controls, not representative conversation or
  answer-quality evidence.
- These records cover local CPU/Metal. They do not establish CUDA/NPU hardware
  performance or multi-stream scheduling. Read raw artifacts before quoting.

## Required integration work

Build an explicit capability matrix from source consumers, then implement the
shared contract and close its gaps. Every exposed control must affect execution
or return a typed capability error; accepted settings must not silently do nothing.

1. **Unify policy resolution.** Extend existing config/owners. Resolve file,
   environment and fluent inputs once with documented precedence. Separate hard
   resource limits, numerical permissions, service objectives and strategy
   overrides. Match cache identity to effective policy/layout/backend. Preserve
   the no_std/alloc floor and generated build-time defaults for restricted tiers.
2. **Parallel execution.** Integrate CPU worker budgets, kernel parallelism,
   independent DAG work, bounded in-flight submissions, transfer/compute overlap
   and supported device placement. CPU currently reads PROXIMA_MATMUL_WORKERS
   outside RuntimeConfig; dispatch_type and command_buffer_chunks are currently
   Metal-specific. Existing backend candidate selection selects one plan; it
   does not implement distributed execution. Preserve dependency and hazard
   edges, buffer lifetimes, cancellation and numeric permissions.
3. **Resumable serving and batching.** Adapt the existing decode loop into
   bounded resumable work using existing substrate. Separate ready decode work
   from bursty prefill chunks, combine compatible decode rows, and retain exact
   per-stream KV, positions, sampler/RNG, output order and cancellation ownership.
   Include bounded admission/backpressure, fairness and prefill service. Existing
   parallel_sequences and phase interleaving requests are rejected; changing
   those guards alone is not implementation.
4. **Cache/residency-aware selection.** Use exact active-stream ownership,
   validity and placement. Reuse existing prefix trie/Bloom machinery where
   screening saves work. Bloom positives never authorize reuse. Verify model,
   configuration, preceding context, positions, logical layer/recurrent visit
   and allocation generation. Preserve service for cold/evicted streams. Current
   Bloom candidate lookup scans and allocates; do not call it a hot-path solution.
5. **Performance strategies as shared integrations.** Cover decode-shaped
   kernels, attention/epilogue fusion, plan/encoding reuse, intermediate/device
   residency, token-selection boundaries, overlap, existing speculative verify,
   weight-conversion caching/precision and expert-grouped MoE execution. Identify
   what already works before extending it. Instrument and test each experiment
   with a toggle; retain negative results. Do not change expert routing, skip
   required Ouro passes, equate Q8 with floating FP8, or relax precision silently.
6. **Portable capability contract.** Preserve CPU/Metal/CUDA/WGPU driver seams
   and the requested NPU extension boundary. Required unsupported features must
   be named and rejected. An absent hardware driver must not masquerade as CPU
   fallback or simulated hardware evidence. Implement and exercise available
   backends; distinguish compiled contract tests from actual device execution.
7. **Observable effective plans.** Record requested and effective policies,
   selected backend/kernel/schedule, cache/residency decisions, resource use and
   rejected alternatives. Automatic choices require evidence; explicit overrides
   make experiments reproducible. Merely reporting an enabled flag is insufficient.

## Execution discipline

- NEVER use Python, Perl, Bash or other scripts. Use native tools/commands,
  apply_patch and repository Rust tooling. Do not run existing shell wrappers;
  use the retained native nextest config when needed.
- Write an admitted spec with count-bearing runnable acceptance criteria before
  changing public contracts. Include literal worked examples for packed streams,
  positions/KV isolation, dependency hazards, resource pressure and cancellation.
- Dispatch detailed packets with exact owned files, existing APIs, invariants,
  input/output examples, dependency order, commands, expected counts, negative
  controls and artifact destinations. Luna must not have to invent the contract.
  Parallelize independent agent work; serialize shared-file edits and GPU timing.
- Builds/tests run through subagents. Debug for development; release for timing.
  No commits are authorized. New experiments use default-off compile-time gates
  where required and named incumbent baselines. Record every failed attempt,
  change, rollback and its evidence in the discipline log.
- At EVERY implementation checkpoint run real-model end-to-end generation.
  Preserve full output IDs/text/EOS, per-token timing and errors. Add representative
  longer conversations and mixed arrivals to the synthetic fixtures; retain the
  exact corpus. Exercise dense and MoE behavior, not only Ouro or Qwen.
- Compare controlled before/after and matched llama conditions: model/weights,
  precision, prompt IDs, context, output budget, concurrency and instrumentation.
  Record time to first token, inter-token distributions, total response time,
  throughput, CPU, RSS, error rate and sample CoV%. Retain all raw samples.
- Join timings to actual dispatches, emitted kernel source, copies/allocations,
  cache decisions and output changes in both directions. GPU_EXEC_TICKS is host
  commit-through-wait time, not GPU busy time. Do not turn aggregate deltas into
  causal claims or add together overlapping counters.
- Root reads actual diffs and payloads; subagent summaries are pointers. Include
  independent review and controls that deliberately fail. Fix integration defects,
  update JSONL memory, specs, teaching docs, work packets and discipline records.

## Completion

Account for every requirement in the admitted spec with actual implementation,
counted execution artifacts and explicitly bounded backend capabilities. No inert
knobs, disconnected alternate pipeline, unexecuted checkpoint or undocumented
regression. Report what changed, what ran, observed payloads, gains and losses,
CoV and hardware limits. Do not claim every candidate optimization improves
performance. The user owns the verdict; the evidence must support it in one read.

Start by reading the durable context, preparing concrete worker packets and
executing the admitted plan. Do not ask me to repeat decisions preserved here.
