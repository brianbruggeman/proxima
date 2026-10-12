# execution plan: unified programmable Proxima/Omega

Status: admitted implementation plan, source integration not complete.

## Routing and evidence

GPT-6 Astra was selected for foundational plan synthesis because this work
crosses model loading, graph/Plan ownership, KV lifetime, backend allocations,
and public policy contracts. Opus was unavailable in the model catalog. Luna
handled the bounded source/evidence extraction. Persistent-memory tools were
unavailable; local `ai_docs` is the source of durable repository context.

Plan A and Plan B were authored independently; Plan B was blind to A. A direct
source critique and a three-judge Borda panel evaluated A/B/AB. All three
judges ranked AB first: Borda AB=6, A=3, B=0. This is one round, so the plan
tournament has not converged; a second round is in progress.

## Sequence

1. **Freeze evidence and make the checkpoint runner observable.** Keep the dirty
   worktree snapshot and checkpoint9 failure. Reconcile its raw `vmmap`, plan,
   resource and run-status payloads; inventory exact model/input digests and
   command settings. Retain per-allocation owner, address/range, backing
   identity, alias, bytes, lifetime, generation and completion-fence rows.
   Unjoined malloc/graphics regions remain unresolved. Process physical
   footprint, RSS, mapped resident pages and Metal allocated bytes remain
   distinct fields. Prepare immutable output paths and controls before changing
   a production owner.

2. **Repair the memory path as one integration boundary.** First add lifecycle
   reservations/events at current owners. Replace whole-file residency proof
   and registration with finite source ranges whose host mapping and GPU view
   remain leased through the final command fence. Bound dense BF16 binding,
   embedding lookup, output projection and retained Plan arena slots while
   preserving reduction order and full logits. Reuse existing graph partition
   and Plan owners; carry every cut dependency. Budget KV, selected MoE expert
   ranges, conversion/cache state and transition overlap in the same owner
   budget. Make over-budget allocation fail before allocation. Preserve the
   prior full-path implementation behind an explicit diagnostic condition;
   never let that condition masquerade as a capped run.

   **Boundary gate:** Qwen dense and Granite MoE each complete the admitted
   single and mixed-arrival workloads under the 500,000,000-byte process
   physical-footprint ceiling, with paired llama.cpp runs and retained payloads.
   Include full IDs/text/EOS, token timings, actual routes, allocations,
   dispatches, cache/placement events and errors. A pre-allocation refusal or a
   cap termination is an incomplete checkpoint; continue this same boundary.

3. **Finish Omega policy consumption.** Extend existing RuntimeConfig, CPU,
   graph/backend, placement and driver owners. Keep numeric permissions,
   hard limits and strategies independent. Finish the actual 26-field consumer
   ledger/recount and typed unavailable-driver outcomes. Run AC2/AC5 controls,
   then the dense/MoE/llama checkpoint gate.

4. **Add resumable stream state.** Extend the existing decode/PrefixState/cache
   owners and BackendRuntime. Preserve per-stream token history, sampler/RNG,
   positions, KV identity/generation, cancellation and submitted-buffer lifetime.
   Run AC3 literals and mutations, then the model checkpoint gate.

5. **Add bounded scheduling behind `unified-scheduling`.** Reuse fitting Pipe
   and bounded-queue primitives. Pack only compatible rows with their stream,
   position, KV slot/generation and sampler state. Enforce admission/backpressure,
   cancellation and the admitted prefill service bound. The feature-off path
   returns its typed error. Run AC4/AC16/AC17 controls and the model checkpoint
   gate; microbenchmarks do not substitute for model runs.

6. **Integrate one performance strategy per source checkpoint.** Use existing
   owners for (a) decode kernels/fusion (AC6), (b) exact cache/plan reuse,
   placement and overlap (AC7), (c) permitted weight conversion (AC9), (d)
   post-routing MoE row grouping (AC10), and (e) target-verified speculation
   (AC8). Preserve route IDs and numeric permissions. Retain both-direction
   output changes, raw timing samples, CoV and dispatch/source joins. After each
   strategy change, run the full dense/MoE/llama checkpoint gate before the next
   source integration.

7. **Run the complete paired matrix and recount.** Reconcile AC11–AC13 against
   all four local model identities, both workload classes, both arms, 96 cohort
   runs and 192 stream-request records per runtime. Compare identical inputs,
   precision, context, token budget, concurrency, arrivals and instrumentation.
   Retain all regressions, dropped/unjoined rows, raw samples and N-1 CoV.

8. **Close tier, lint, docs and memory records.** Run AC14/AC15 after shared
   interfaces stabilize; then rerun affected model checkpoints for any source
   correction. Update task rows and `ai_docs` from opened artifacts. Retain every
   unknown allocation or unexplained behavior as a residual.

## Rollback and design constraints

Every boundary keeps its exact pre-change source condition and immutable raw
artifacts. Roll back only the responsible hunks when payloads, lifecycle,
resource limits or paired evidence refute them; rerun the restored condition
before proceeding. No source integration is complete until its dense, MoE and
llama checkpoint exists. Never raise the memory ceiling or silently reduce
context, precision, concurrency, routes, or token count to force a pass.

Abandoned approaches: (1) rerunning the known whole-checkpoint/whole-plan path
as an initial capped baseline, because checkpoint9 captured multi-gigabyte
physical footprint and the full output-head product alone exceeded the cap;
(2) changing only the scheduler/cache while whole-file prefault and whole-plan
retention remain; (3) treating equal-sized malloc and graphics VM regions as
identified owners without pointer joins; (4) storing every segmented Plan at
once, which moves rather than bounds retained arena ownership. These choices
are recorded as source/evidence constraints, not claims that a replacement is
already implemented.

## Plan tournament trail

| Candidate | Judge 1 | Judge 2 | Judge 3 | Borda |
|---|---:|---:|---:|---:|
| AB | 1st (2) | 1st (2) | 1st (2) | 6 |
| A | 2nd (1) | 2nd (1) | 2nd (1) | 3 |
| B | 3rd (0) | 3rd (0) | 3rd (0) | 0 |

Judge 1 axis scores (risk/order/rollback/missing/coupling/observability/scope):
AB 3/4/3/5/3/5/3, A 3/3/3/2/3/4/4, B 5/1/2/2/2/3/4. Judge 2: AB
4/5/4/5/4/5/5, A 3/4/3/2/3/2/3, B 1/1/2/2/2/1/2. Judge 3: AB
4/5/4/4/4/5/5, A 3/4/3/2/3/3/3, B 1/1/2/2/2/2/3. The critique scored A
9/35 and identified gaps; its raw-C9 limitation was corrected by root review
of the files in `results/checkpoints/e2e-checkpoint-9/`.
