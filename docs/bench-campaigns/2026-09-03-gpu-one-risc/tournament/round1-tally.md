# plan-rigor round 1 — Borda tally

Candidates: A (Plan A, round-0 incumbent), B (blind Plan B), S (synthesis_AB).
Label maps (randomized per judge): judge1 P1=A P2=B P3=S; judge2 P1=S P2=A P3=B; judge3 P1=B P2=S P3=A.

| judge | ranking (labels) | ranking (candidates) | S | B | A |
|---|---|---|---|---|---|
| 1 | [P3, P2, P1] | [S, B, A] | 2 | 1 | 0 |
| 2 | [P1, P3, P2] | [S, B, A] | 2 | 1 | 0 |
| 3 | [P2, P1, P3] | [S, B, A] | 2 | 1 | 0 |
| **Borda** | | | **6** | **3** | **0** |

Winner: S (synthesis_AB), unanimous, 3 first-place votes. Becomes the round-2 incumbent.

Per-axis low scores on the winner (to carry into round 2):
- judge1: per-dispatch `record_route` is a mutex BTreeMap insert ~1196x/token inside the orchestration slice, uncosted; 8.1 (`-fa 1`) and 8.2 (roofline) queued AFTER 4.3's board prediction; 5.5's parallel `&[Option<u64>]` slice loses block↔declaration binding (positional desync across two crates); 1.1 / 7.2 worker cards have reading lists where commands should be; split-K build card missing after 10.1.
- judge2: 5.4 is the one card where wrong-but-green is possible (bounds semantics for every Reduce; autograd adjoint path).
- judge3: 0.5's `git apply --check` at 2b95210 fails for the 3 worktrees based on other HEADs (gpuker@bfc150d, q4k@a2175c2, lat@14f1304) — check against each worktree's own HEAD; 0.5's `git diff` drops UNTRACKED files (proxima-wt-all has 6); two Q4_K features coexist under `--all-features` in omega-gate [2/6],[3/6] — needs a precedence rule or the gate runs with explicit feature sets on that branch; 4.1's plan_hits expectation must be a FORMULA (hits = forward_calls − distinct (new_count, bucket) shapes; at PROXIMA_MAX_TOKENS=8 with prefill (31,0) and decode (1,256): 8 − 2 = 6 hits), not "tokens−1"; G6's measurer queue must be "any card whose reprove runs the decode harness, a probe, or a bench takes the lock", not a fixed list; 0.3's replicate runs on a tree carrying 0.1+0.2 (instrument/test only) — state it; line cites off by one: ScatterNotSupported msl.rs:933 / wgsl.rs:364 / cuda.rs:241, IndexMap::scatter map.rs:175, u.out_base msl.rs:2734.
