# Discipline log — MoE batched indexed expert product (`metal-moe-mul-mat-id`)

Component: collapse the ~1211 routed-expert packed-row matvecs/token into one
batched indexed matmul over the k selected experts (llama.cpp `mul_mat_id`
shape). Targets the fat bucket of qwen35moe GPU decode. Model-generic: must hold
for gemma4 26b-a4b too.

Base: local main `5ec9c8698`. Worktree: `proxima-wt-moe-fusion`.
Workflow: `wf_fecbb835-f14` (Baseline→Design→Build→Correctness→Bench).
Landing: main-loop coherent-commit onto LOCAL main after diff review (not pushed).

## Incumbent (gate 6/13 — named, not "industry standard")
- **llama.cpp** via Ollama `qwen3.6:35b-a3b`, same GGUF blob. Its `ggml_mul_mat_id`
  batched indexed expert matmul IS the design point this component copies.
- Measured (loaded box, wf_fecbb835-f14 Baseline): Ollama 31.53 ms/tok; proxima
  76.76 ms/tok; 2.43x. Earlier quieter-box read: Ollama 17.4, proxima 54.6. The
  absolute numbers are load-sensitive; the ratio and the per-kind op counts
  (packed-row 1211, cooperative 551, elementwise 503) are the stable signal.

## Home-turf 80% arm (gate 13)
Per-token routed-expert product — fires every token, every MoE layer.

## Invariants engaged (guiding-principles — what each ruled out)
- I1 batched indexed matmul, k as grid-z, k-independent products + one combine.
- I2 algebra-preserving: Metal kernel + bind admission only; NO spec/builder
  change. Ruled out: any Op-graph rewrite.
- I3 default-off compile-time flag; main default byte-unchanged (firewall).
- I4 no new proxima-library type unless the pipe question fails (answered by
  writing the expression). Ruled out: a `MergedExpertProduct` library type — the
  kernel is an omega/msl edge concern.
- I5 (prior-failure trap, HARD): the naive horizontal merge was CLOSED as a net
  LOSS (74.9 vs 57.7) because the grouped form added per-round slicing +
  cooperative reduces. Ruled out: grouping the existing per-round product shape.
- I6 correctness: proxima's own default per-expert path is the oracle; flag-on
  must be bit-reproducible (P14).

## Rows (every tweak, incl. rollbacks; a measurement is an input, not a verdict)
| row | change | flag_off ms | flag_on ms | llama ms | packed-row ops off/on | CoV | read |
|-----|--------|-------------|------------|----------|-----------------------|-----|------|
| 0 | baseline (wf Baseline, loaded box) | 76.76 | — | 31.53 | 1211 / — | — | 2.43x; op counts stable |
| 1 | Build: admission + `round_count` field + GridSpec.depth wiring; kernel NOT implemented | — | — | — | — | — | compiles both feature sets (exit 0); renderer DECLINES round_count folds (elementwise_reduce_core.rs:211-217, typed EmitError, NOT silent-wrong); flag-OFF verified INTACT (Paris, feature absent — a324ffea; the wf Correctness "flag-off broken" was a mis-built binary) |
| 2 | kernel slice (z-addressed MSL + arena placement) | — | — | — | — | — | STOPPED honestly at 40-min ceiling, zero code, rather than a false-positive hack (a3e5a39e). Findings: GridSpec.depth already wired; real gap = MSL z-addressing in the packed-row body + arena eager-contiguous placement. Blocker: `gid` is scalar `[[thread_position_in_grid]]`; Metal rejects scalar+vector mix → must widen gid→uint3 in the 1 selected render body (precedent: horizontal-merge splice emit_and_classify.rs:454-466). Body confirmed packed-row-blocked |
| 3 | compliance review of scaffold (ab0590177) — CRITICAL | — | — | — | — | — | The `round_count: Option<u32>` on the EXISTING `Reduce` variant is a silent-corruption footgun: every non-Metal consumer (cpu interpreter ~8 sites, cuda, wgsl, wgpu) destructures `Reduce { .. }` and silently runs round 0 only if the flag is enabled there — wrong numbers, no error (violates "no silent failures"; the exact "information destroyed at a match boundary" my own critique rules name). Flag lives in proxima-tensor (not scoped to omega/metal) → Cargo feature unification can turn it on for a CPU/cuda build. Fix: encode as a NEW `BoundOpKind` variant (in-pattern with MoeTopK/GatedDeltaNet) → exhaustive-match forces every backend to handle or explicitly decline. Scaffold NOT landed until re-encoded. Also: 5 inline `std::env::var_os` paths in gdn_moe_fusion_apply.rs violate rust.md (4 pre-existing + the new one) → move to top-of-file `use std::env`. Clean: no new pub type, default-off confirmed, confined to omega+proxima-tensor, round_count:None at all 8 sites |

| 4 | re-scaffold field→variant (a5c3e2aa) + landing gate (aa300d64) | — | — | — | — | — | DONE: `RoundBatchedReduce` variant (types_layout_boundop.rs:425-441, `round_count: u32`), ~20 backend match sites forced to explicit typed declines (cpu NotLowerable, msl/cuda/wgsl EpilogueNotSupported/Unsupported), no silent catch-all in a correctness path. Flag-off cfg-gated (gdn_moe_fusion_apply.rs:129-141, all `#[cfg(feature="metal-moe-mul-mat-id")]`) → flag-off CANNOT produce the variant → default byte-intact (verified by SOURCE READ + re-scaffold flag-off→Paris). Gate AC1/AC2/AC7 PASS. AC3 "critical fail" = mis-copied flag-ON binary (only flag-on emits "no CPU interpreter yet"; source proves flag-off can't). AC8: 1 file outside = a FORCED exhaustive-match arm in proxima-model-interop/tests (expected — the type-safe variant touches every consumer); AC8 refined to permit consumer-crate test arms. Also fixed latent `reduce_round_count` `_=>None` wildcard + 5 inline std::env paths |

## ~~Kernel plan — reuse the splice~~ RETRACTED (was wrong)

~~Reuse the horizontal-merge base_table splice for RoundBatchedReduce.~~ WRONG,
disproven by a1f04dc3 + my own read of `gdn_moe_fusion_apply.rs:1367-1423`. The
splice needs real per-member (weight/activation/output) byte offsets from k
SEPARATE BoundOps — but `apply_moe_round_group_fusion` DELETES rounds 1..k
(`drop.extend(rest)`, `continue` at 1382/1386-1387) and carries only round 0's
operands/out_layout unchanged into the `RoundBatchedReduce` (1404-1417). Their
per-round route/output addresses are GONE post-bind, so a base_table has nothing
correct to hold — filling it with round 0's address k times reads round 0 k
times (the wrong-kernel case the type doc at 1355-1365 + 406-408 forbids, and the
ROW-571 false positive G-collapse exists to catch).

## Kernel plan (slice 2) — CORRECTED: the crux is bind/arena, not Metal

The blocker is NOT a Metal-renderer gap. Both backends decline
(`cpu/run_node.rs:205-208` NotLowerable; msl EpilogueNotSupported) BECAUSE the
per-round data does not exist after fusion — the CPU decline (no codegen concern,
only needs real addresses) is the proof the gap is upstream of both backends.
The admission's own doc (gdn_moe_fusion_apply.rs:1355-1365) names the residual:
the k route/output positions must be pre-placed CONTIGUOUS so a z-addressed read
reaches every round through ONE MORE STRIDE. So slice 2 is, in order:
1. BIND/ARENA (the crux): in `apply_moe_round_group_fusion`, for each group prove
   the k rounds' `route` index buffers and output destinations have a uniform
   per-round element stride (arena places them contiguously); carry that stride
   into round 0's operand/out `Layout`s by widening `extents` with one round
   axis; REJECT the group back to plain per-round `Reduce`s when not contiguous
   (cheapest correct fix, option a). Only `route`/output vary per round — `stack`
   (weight) and `x` (activation) are the SAME NodeId across rounds
   (moe_round_reduce_operand 1261-1291), so there is no per-round weight base.
2. METAL (mechanical, downstream): the z-read is `push_gather_fetch`'s EXISTING
   per-axis `coord[dim]*stride[dim]` loop walking the new round axis — NOT the
   SliceBase splice. GridSpec::depth = round_count is already wired.
The landed scaffold (85ac70da5) is correct-but-incomplete: safe (declines
everywhere, default-off), but the admission drops the addresses — step 1 above is
the real work.

## Notes
Design (wf Design phase) avoids both prior traps: `round_count: Option<u32>` on
BoundOp (None everywhere = byte-identical default, like `GridSpec.depth`), z rides
the existing rank-generic `push_gather_fetch` stride loop, pre-places k outputs
contiguously (the ROW 571 fix). One private struct `MoeRoundGroup` (bind-internal,
precedented by `merge_candidates`), zero new public types. The Build implemented
the admission + field + an explicit renderer decline, NOT the kernel body — so
gate (5) dispatch-count-drop is unproven and gates (3)/(4) parity are pending the
kernel. Ollama reclaim required before every GPU run.
