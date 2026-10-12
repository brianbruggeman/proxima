# independent plan C

Author: `/root/round2_plan_b` (blind to Plan A and Plan AB)
Model route: GPT-6 Luna, bounded independent implementation planning.
No source edits or execution were performed for this plan.

## Ordered implementation

1. Freeze the dirty resume point and checkpoint9. Instrument baseline,
   reservations and allocation acquire/retire at current owners, then implement
   S16/AC18 across the reservation owner, mmap windows, binder, Omega plan
   arenas, KV, experts and release fences. Keep RSS, physical footprint, mapped
   residency and Metal allocation separate. Require all four models × two
   workloads × two arms × five measured samples, complete address/owner/backing/
   alias/generation/fence joins, zero unmatched/unowned ranges/cap events,
   physical footprint ≤500,000,000 bytes, exact-fit admission and one-byte-over
   rejection before allocation/dispatch. Retain `results/memory-checkpoint-1/`.
2. Finish Omega config/policy at the existing RuntimeConfig, planner, backend,
   CPU and Metal owners. Preserve independent limits/numeric permissions/
   objectives/strategies. Close AC2/AC5 with 18 planning tests, 12 capability
   tests and 26 actual consumer rows plus controls.
3. Make generation resumable in existing decode/PrefixState/cache/BackendRuntime
   owners. Close AC3 with 20 tests and literal packed-stream, pause/resume,
   cancellation and seven identity-mutation records.
4. Add bounded mixed-arrival scheduling behind default-off
   `unified-scheduling`. Close AC4/AC16 with 16 enabled tests and 4 typed
   feature-off errors; AC17 with 5 raw serial/scheduler runs at 2, 8 and 32
   streams, including allocation/op and CoV.
5. Integrate strategy owners one at a time: AC6 decode/fusion (12 tests), AC7
   residency/reuse/overlap (12), AC8 target-verified speculation (4), AC9
   conversion identity (4), AC10 post-route MoE grouping (4). Preserve typed
   unsupported errors and literal mutation controls.
6. At each integrated source checkpoint, run real dense and MoE models plus
   matched llama.cpp. Keep full trajectories, route/KV/dispatch/resource,
   errors and five raw samples. A refusal or cap stop is incomplete. Final
   AC11–AC13 covers 4 models × 2 workloads × 2 arms, 96 cohort runs and 192
   stream-request rows per runtime with matched conditions and N-1 CoV.
7. Close AC14 alloc-tier, AC15 clippy, full recount and cap evidence; update
   JSONL memory/evidence, discipline, tasks and residuals. No commit.

## Existing seams from the independent author

- Policy/defaults and planners: `omega/src/config.rs:15-80,187-225`,
  `omega/src/backend.rs:452-475,1393-1430`.
- CPU workers: `proxima-tensor/src/cpu/gemm_q8k.rs:126-175`.
- Serving and continuation: `proxima-model-interop/src/serving.rs`,
  `generate/decode.rs:2132-2152,3684-3729,6084-6104`.
- Memory: `generate/pregather.rs:2638-2669`,
  `mapping_residency.rs:296-365`, `generate/model_weights.rs:25-53`,
  `generate/residency_caches.rs:904-908`.
- Plan allocation/completion: `omega/src/metal/arena_encode_dispatch_finish.rs:165-241`,
  `device_buffers_arena_plan.rs:2191-2204`,
  `placements_execute_named.rs:1594-1608`.

The author opened the admitted spec, source, and raw memory packet. The author
identified AC18 as missing from the prior admitted AC1–AC17 set; root added R16,
AC18 and the literal ledger fixture for independent re-audit. Address attribution
remains unresolved until actual pointer/fence rows join the retained VM regions.
