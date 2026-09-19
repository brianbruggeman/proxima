# batched-expert-completion -- slices

The anti-reactive sequence: map ALL problems by reading, fix them in ONE pass,
measure ONCE. No coding until the map (slice 1) is complete.

| # | slice | discharges | validation command | expected | done | note |
|---|---|---|---|---|---|---|
| 1 | COMPLETE problem map of the RoundBatchedReduce path at real scale (256 experts, prefill seq=15 + decode seq=1), by READING only | AC1 | the map doc: each defect file:line + mechanism + seq + fix, ending "path fully traced" | ≥1 rooted (the gather bug) + the output-alloc byte number + the fully-traced statement | [~] | IN FLIGHT (a08a2950) — read-only, no crash-to-discover |
| 2 | ONE coherent fix pass closing EVERY map item at once (no per-defect reactive dispatch) | AC2, AC3 | `reclaim; ./target/release/examples/gguf_generate "$BLOB" "$Q" 16 gpu 2>&1 \| grep -ciE 'metal run failed\|out of range\|fell back'`; and flag-on vs flag-off `diff` | `0` faults/fallback; both Paris, `0` diff | [ ] | only after slice 1 says "fully traced"; if a NEW crash appears, the map was incomplete — widen it, do not patch reactively |
| 3 | measure ONCE (the "chase number", at the end) | AC4 | `PROXIMA_METAL_OP_PROFILE_STEP=1` flag-on 32-tok packed-row op_count; 3× off + 3× on mean TTNT | op_count ≤200; VERDICT: win / op-count-win-ttnt-regression / measured-slower, with 6 numbers | [ ] | ROW-543 protocol: op_count↓ alone is not a win |

## resume

Last landed: batched-expert machinery on main 3e03ec4ba (inert); route hoist at worktree base 41da75882 (admission fires → gather bug live).
Next action: slice 1 problem-map IN FLIGHT (a08a2950). When it returns "fully traced", dispatch ONE fix pass for the whole map (slice 2), then measure once (slice 3).
Open question: does the map come back "fully traced" or "could not trace X"? If the latter, widen the read before any code — that is the process being corrected.

## struck

- (superseded) the reactive per-defect loop: a3536954/a1c74ed8/a704796d/ab2fb870 — each found one problem by crashing. Replaced by slice 1's single read.
