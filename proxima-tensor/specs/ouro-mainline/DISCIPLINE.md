# Shared Ouro integration discipline

This adapts existing owners. It adds no executor, attention implementation,
sampler or kernel. The experimental BF16/F16 dense kernel extension remains
behind its existing default-off build setting. An absent recurrent descriptor
continues to select the ordinary graph; three frozen incumbent graphs retain
every Op and root for that comparison.

## Constraints that changed the implementation

- Reuse: abandoned keeping the lab transformer behind a wrapper. The lab now
  calls LoadedModel; recurrence and optional Q/K normalization are shared data.
- Ownership: rejected aliasing KV state because physical weights are shared.
  Logical visits receive distinct cache owners; fixtures retain their rows.
- Cold/hot separation: descriptors, recurrent identity and runtime policy are
  selected before execution. Diagnostic readouts expose existing graph nodes.
- Configuration: rejected forwarding only numeric permission while dropping
  resource constraints. Ordinary plan misses call Omega's configured planner;
  paths without the required ownership contract reject the configuration.
- Tier boundary: the plain recurrence mode lives in the tensor root. The alloc
  build covers that core; the graph specification remains behind its existing
  std/config boundary. An alloc artifact is not evidence that the graph builder
  compiled without std.
- Evidence: rejected using the full-prefix diagnostic as proof of timed kernel
  dispatch. Request brackets, actual bound extents and source hashes are required.

## Execution record

| Checkpoint | Change or condition | Native evidence | Disposition |
|---|---|---|---|
| Shared graph | Optional Q/K norm, physical/logical recurrence indices, completed-pass readout roots | `results/shared-graph/validation`, frozen `results/baseline` | Literal and deliberately wrong-state controls retained; no latency claim from these fixtures. |
| Default serving | GGUF through LoadedModel prefill, KV append, sampling | `results/proxima-cached-r2`, `results/llama-cached-r2` | Full requests and resource samples retained. Different microbatch widths prevent attributing their gap to one kernel. |
| Configured C6 | Same runtime policy with dense-float gate off/on; production microbatch32 | `results/cached-recount` | Both complete cohorts retained. Wall/TTFT values and CoV are in EVIDENCE.md; the measured batches are below the tile floor. |
| HF descriptor repair | Canonical-only legacy fastpath; explicit descriptor shape admission | `results/interop` | Ordinary/custom descriptor and wrong-shape assertions retained. Real HF vector comparison is part of the same implementation checkpoint. |
| Diagnostic root repair | Capture actual guided raw hidden state and adaptive strength | Shared graph run r5; caller Model source | Existing graph nodes exposed and remapped; no added model Ops. Derived probability metadata remains separately labeled. |
| C7 first capture attempts | Matched batch256/context320 but packed-only selector | Archived partial `proxima-matched-batch-off*` attempts and COMMANDS.md | Rejected as dispatch evidence: packed classifier excludes BF16. Initial missing-feature explanation retracted; instrumentation was transitive. Replacement uses all nodes at step zero in both arms. |
| C7 dispatch capture | Matched batch256/context320, both arms capture all nodes at step zero | `results/C7-PROFILE-AUDIT.jsonl`, `CALLER-C7-AUDIT.md`, `results/proxima-matched-batch-{off,on}-r3` | Node116 BF16 projection inside request brackets maps to scalar reduction off and simdgroup matrix multiplication on. EVIDENCE.md retains short/long request and decode durations, including cells without the long-prefill reduction. |
| C7 independent recount repair | Exact float parsing and centered Welford sample variance | `results/cached-checkpoint/ouro-mainline-recount-tests-r5.txt`, `results/cached-matched-batch-recount-r5` | Fixed comparison bound retained. Literal source durations round-trip, known variance is reproduced, changed standard deviation is rejected, and the old unequal-batch cohort is rejected. |
| Multiple-stream scheduling inventory | Existing phase and parallel-sequence settings reject interleaving | `proxima-model-interop/src/serving.rs:1355`, `proxima-model-interop/src/generate/decode.rs:3139` | Single-request prefill chunking is not cross-stream scheduling. The prefix resume API clones host KV and creates a runtime per call; no concurrent-serving measurement follows from current cached cohorts. |
| Shared HF weight owner | Existing bound payloads have one Arc-owned destructor across forward variants; graph and KV state remain distinct | `results/weight-owner/ouro-owner-tests-r5.txt`, `results/mainline-methods-live-r3`, `results/owner-checkpoint-cached-on-r1` | Rejected cache clearing, per-arm reloads and last-owner count polling. The live runner retains all four arms through eight warmups and forty samples; the cached run retains twelve requests and their complete tokens. The earlier resident-name rebound remains recorded. |
| Independent C2 method comparison | Frozen Metal candidate against native F32 arithmetic, two prefixes and four methods | `results/weight-owner/c2-vector-ledger.jsonl`, `results/weight-owner/final-artifact-audit.md` | All non-exact coordinates retained in both directions. Selection equality and the admitted numerical bound do not establish bit equality, BF16 activation reproduction or general answer quality. |
| Final lab helper checkpoint | Named tap alias, existing Model context reuse, helper argument grouping, iterator expressions | `results/weight-owner/lab-*-clippy.diff`, `results/owner-checkpoint-cached-on-r2`, `results/weight-owner/owner-r2-profile-pairs.json` | Strict lab compiler findings repaired without source lint suppression. Caller4/recount2 repeated, then twelve fresh cached requests. All profile pairs and complete trajectories retained; EVIDENCE.md gives the new durations and CoV without attributing differences to these rewrites. |

## Ordered broadcast fold checkpoint

The generic broadcast reduction now reuses the existing serial fold when
TreeReduce is denied, then publishes to the existing cooperative write tail.
Disabling all fusion and routing through the unsupported ordinary serial
renderer were rejected. `results/BROADCAST-FOLD-COMMANDS.md` and
`results/broadcast-fold-final-audit.md` index fresh replay/C2/live/cached payloads.
Strict live means increased in every retained cell; complete signed coordinate
changes and CoV are retained. Relaxed cached timings are a separate condition,
with no causal speedup attribution. The missing capture-parent attempt and
initial compiler findings are preserved with their repairs.

No zero-allocation, process-memory-cap, global-optimality or full-device-utilization
claim follows from these records. The measured client path includes production
planning, execution, callbacks and token decoding; setup and standalone vector
diagnostics are identified separately. No commits or publication are part of
this checkpoint.
