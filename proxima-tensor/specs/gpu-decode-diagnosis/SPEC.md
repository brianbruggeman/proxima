# batched-expert-completion (problem-map-first)

status: draft
owner: brian
created: 2026-09-19

## problem

The batched-expert op-collapse (metal-moe-mul-mat-id) is the right lever and will
work; it is failing to LAND only because it has been executed reactively — one
agent per crash — burning ~30 agents to discover defects that a single read of
the full path would have enumerated at once. This spec makes the remaining work
converge instead of thrash: read the entire batched-expert execution path once,
list EVERY remaining defect by file:line, then fix them in one coherent pass.

## refutation condition

If, after the up-front read (R1), the coherent fix pass (R2) still hits a NEW
crash/defect that the read did not predict, the read was incomplete — that is the
process failure this spec exists to kill, and it means the map (R1) must be
widened before any further code, not that another reactive patch is dispatched.

## requirements

| id | requirement | testable in isolation |
|---|---|---|
| R1 | a COMPLETE problem map of the RoundBatchedReduce execution path at REAL scale (256 experts, k=8, prefill seq=15 AND decode seq=1), produced by READING — every defect and every unhandled edge, each at file:line, with the mechanism, NOT by running until it crashes | yes |
| R2 | one coherent fix pass closes EVERY item in R1's map (no reactive one-at-a-time dispatch); flag-on GPU then runs the real checkpoint end-to-end without fault | yes |
| R3 | correctness: flag-on generated token stream == flag-off, bit-exact (the reorder + collapse change no values) | yes |
| R4 | the measurement (the "chase number", done ONCE at the end, not per-patch): op_count 1211→≤200 AND flag-on mean TTNT vs flag-off — a real win only if op_count↓ AND ttnt not worse (ROW-543 protocol) | yes |

## architecture

The known head of R1's map (from this session's reactive crashes — the map must
CONTINUE past these by reading, to find what the NEXT pass would otherwise hit):
- gather-index-out-of-range at prefill: `node %256 gather fetched index 0, out of
  range for extent 256`, fault decoded at `omega/src/metal/arena_encode_dispatch_finish.rs:~1140`.
  Suspected: `ensure_round_group_resolved`'s contiguous route/output placement
  (`device_buffers_arena_plan.rs`) or the splice's `round_table[z].route_base`
  offset is computed for decode (seq=1) but wrong for prefill (seq>1) — the route
  index buffer is per-position and the placement/z-stride may not carry the
  sequence axis.
- the map must ALSO cover, by reading (not crashing): the memory-budget of the
  device-direct contiguous output alloc at real 256-expert shape (never measured);
  caller-placed-output (unsupported today, a documented follow-up in the renderer);
  the cache-key coarseness residual (`packed_row_block_shape_token` default for
  RoundBatchedReduce); and any other RoundBatchedReduce arm that still defaults/
  declines across omega msl+metal at seq>1.

### decisions

| decision | chosen | why not the alternative |
|---|---|---|
| read the whole path before any fix | one read-only map (R1) | the session's waste was N agents each finding one defect by crashing; a single trace finds them together |
| one coherent fix pass | R2 closes the whole map at once | reactive per-defect patching is the exact failure mode being corrected |
| measure ONCE at the end | R4 after the path runs clean | "patch + chase number" per step is the token waste; the number is only meaningful once the path executes |

## acceptance criteria

| id | discharges | command | expected |
|---|---|---|---|
| AC1 | R1 | the map document lists each defect: file:line + mechanism + the seq (prefill/decode) it bites + the fix | ≥1 item (the gather bug) with mechanism proven by reading; explicit "path fully traced, no further unhandled RoundBatchedReduce edge found" statement |
| AC2 | R2 | after the single fix pass: `reclaim; ./target/release/examples/gguf_generate "$BLOB" "$Q" 16 gpu 2>&1 \| grep -ciE 'metal run failed\|out of range\|fell back'` | `0` (no fault, no CPU fallback) |
| AC3 | R3 | flag-on France 16-tok GPU vs flag-off, `diff` of generated_text | both Paris; `0` differing lines |
| AC4 | R4 | `PROXIMA_METAL_OP_PROFILE_STEP=1` flag-on 32-tok → packed-row op_count; then 3× off + 3× on mean TTNT | op_count ≤200; VERDICT stated: win (op_count↓ AND ttnt≤off) / op-count-win-ttnt-regression / measured-slower — with the 6 numbers |

## out of scope

- Re-litigating whether batching is the right lever (owner settled: it works).
- Prefill/TTFT as a separate optimization.
- The other two buckets (cooperative, elementwise) — after this lands and is measured.
- Pushing to origin.

## risks

| risk | likelihood | what it costs | what we do about it |
|---|---|---|---|
| the read misses a defect (R2 hits a new crash) | medium | back to reactive | the refutation condition: widen the map, do not dispatch a reactive patch |
| the memory-budget alloc spikes at 256 experts | medium | OOM / residency fault | the map (R1) computes the number BEFORE building, not after a crash |
| op_count↓ but TTNT worse (ROW-543 redux) | medium | a hollow win | R4 measures TTNT, not just op_count; a regression is reported as such, not landed |

## context

- Machinery on main: `3e03ec4ba` (RoundBatchedReduce cpu+metal, default-off, inert). Route hoist held at worktree base `41da75882` (makes admission fire → exposes the gather bug).
- The gather crash + the "landed but never proven at real scale" pattern: `moe-mul-mat-id/discipline.md`.
- Prior reactive trail (the waste this spec corrects): a3536954 (route hoist, blocked on gather), a1c74ed8 (renderer, synthetic-only), a704796d (route-hoist feasibility).
