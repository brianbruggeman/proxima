# batched-expert-completion -- slices

The anti-reactive sequence: map ALL problems by reading, fix them in ONE pass,
measure ONCE. No coding until the map (slice 1) is complete.

| # | slice | discharges | validation command | expected | done | note |
|---|---|---|---|---|---|---|
| 1 | COMPLETE problem map of the RoundBatchedReduce path at real scale, by READING only | AC1 | the map doc | 5 findings rooted; crash narrowed to 2 candidates | [x] | DONE (a08a2950). Reframe: prefill=15 sequential n=1 passes, NOT one seq=15 dispatch; match_moe_topk is n=1-gated so RoundBatchedReduce is INERT under batched prefill. FINDINGS: #1 silent output-placement drop (arena_encode_dispatch_finish.rs:816-826 — caller placement discarded, writes to group.output_buffer; SAME class as GatedDeltaNet state_out, never guarded; likely crash cause + cheapest); #2 route_member_bytes=buffer.length() correct only by coincidence (device_buffers_arena_plan.rs:1205); #3 MoeTopK extra-output hardcoded 1-elem, no size assert (arena_encode_dispatch_finish.rs:996); #4 packed_row_block_shape_token returns 'S' for RoundBatchedReduce instead of delegating to round_zero_reduce_bound like grid_threads/tiled_gemm do (emit_and_classify.rs:737 — one-liner, cache collision); #5 multi-GB alloc UNREACHABLE under n=1 admission (tens of KiB). CRASH (#6) narrowed to: (a)=#1 placement drop, or (b) BufferArena plan-stable slot reuse across the 15 sequential steps — settle by fixing #1 then, if it persists, a metal-plan-stable-buffers on/off differential |
| 2 | ONE coherent fix pass closing EVERY map item at once (no per-defect reactive dispatch) | AC2, AC3 | `reclaim; ./target/release/examples/gguf_generate "$BLOB" "$Q" 16 gpu 2>&1 \| grep -ciE 'metal run failed\|out of range\|fell back'`; and flag-on vs flag-off `diff` | `0` faults/fallback; both Paris, `0` diff | [ ] | only after slice 1 says "fully traced"; if a NEW crash appears, the map was incomplete — widen it, do not patch reactively |
| 3 | measure ONCE (the "chase number", at the end) | AC4 | `PROXIMA_METAL_OP_PROFILE_STEP=1` flag-on 32-tok packed-row op_count; 3× off + 3× on mean TTNT | op_count ≤200; VERDICT: win / op-count-win-ttnt-regression / measured-slower, with 6 numbers | [ ] | ROW-543 protocol: op_count↓ alone is not a win |

## resume

Last landed: batched-expert machinery on main 3e03ec4ba (inert); route hoist at worktree base 41da75882 (admission fires → gather bug live).
Map v2 (a08a2950 deeper dig) RE-DIAGNOSED the crash, ruling out the v1 leads in code: #1 placement-drop is UNREACHABLE (expert_count=256 → single_range=None, load_model.rs:1260); buffer-race RULED OUT (DispatchType::Serial only, all nodes re-inserted/step). REAL lead: the fault report clamps a NEGATIVE fetched index to display "0" (push_gather_fault_check, signature_tokens_prelude.rs:~586); render_moe_topk's tournament `masked == max_value` (~1116) emits a -1 sentinel when no thread matches exact-float-equality → consumed as a negative expert gather index → crash. Latent (not the crash): #1, #2, #3, #4, #7 (PlanUniforms doc/code mismatch). NOTE: both slice-2 fix agents died on a transient SSL API error, applied nothing (worktree clean at 41da75882).
Next action: CONFIRM the -1 hypothesis by instrumentation IN FLIGHT (adafe615) — read the real fault value + raw route buffer + whether a round has no tournament winner — then fix render_moe_topk ONLY if confirmed, else report the real mechanism. Then the latent findings (#2/#3/#4/#7) as hardening, then measure once.
Open question: is the -1 pre-existing in render_moe_topk (batched gather's fault-check merely exposes it) or introduced by the route reorder? The confirm agent answers via "why does flag-off survive".

## struck

- (superseded) the reactive per-defect loop: a3536954/a1c74ed8/a704796d/ab2fb870 — each found one problem by crashing. Replaced by slice 1's single read.
