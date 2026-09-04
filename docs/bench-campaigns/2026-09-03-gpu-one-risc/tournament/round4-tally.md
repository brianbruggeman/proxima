# plan-rigor round 4 (cap round) — Borda tally

Candidates: S3 (synthesis_3, round-3 winner / incumbent), B4 (blind), S4 (synthesis_4).
Label maps: j10 P1=S4 P2=S3 P3=B4; j11 P1=B4 P2=S4 P3=S3; j12 P1=S3 P2=B4 P3=S4.

| judge | ranking (labels) | ranking (candidates) | S4 | B4 | S3 |
|---|---|---|---|---|---|
| j10 | [P1, P3, P2] | [S4, B4, S3] | 2 | 1 | 0 |
| j11 | [P2, P1, P3] | [S4, B4, S3] | 2 | 1 | 0 |
| j12 | [P3, P2, P1] | [S4, B4, S3] | 2 | 1 | 0 |
| **Borda** | | | **6** | **3** | **0** |

Unanimous: S4 wins round 4 with three first-place votes. The incumbent (S3) lost, so the
"same candidate wins two consecutive rounds" test is not met at the round-4 cap. Per the
plan-rigor cap rule the tournament emits S4 as the final plan with the flag **NO CONVERGENCE**.
Trajectory across four rounds: every synthesis beat its incumbent unanimously (6-3-0 four times),
and every critique found first-execution breaks in the incumbent that the synthesis closed; the
residual set shrank from execution-blocking (rounds 1-3) to scheduling, provenance and one
under-quoted API (round 4).

Per-axis on S4 (j10 / j11 / j12): risk 9/8/9, ordering 9/9/9, rollback 8/8/9, missing 8/8/8,
coupling 9/8/9, observability 9/9/9, scope 8/7/8.

Residuals the judges named on S4 (applied to the assembled §5 where they are text; recorded here
where they are structural):
- j10, j11: `$WT`/`$TD`/`$LOCK` shell variables in commands — the expanded cards in plan.md §5
  carry absolute paths at every command (the round-3 j9 fix), so this residual is already closed
  in the deliverable.
- j10: exit 75 has no retry/queue policy — G3 now states one (re-issue after a 300 s pause, up to
  six times, waits logged, seventh is a STOP).
- j10: D7 over-generalises — the ORACLE test at `proxima-model-interop/src/bind.rs:2764` is
  `#[ignore]` with no `metal` cfg; G4 now states each test's cfg.
- j10: 9.1's REJECT half should drive `adjoint.rs:806`'s data-dependent guard and
  `error.rs:73-88` — added as expect cases (fix6 J1).
- j12: the MILLI rung is a 5-token cell (`bind.rs:3103`) and its env vars are inert — every milli
  row is labelled "milli budget 5, bench budget 8" (fix6 J4).
- j12: the `UNIFORM_BUFFERS` LRU bound needs its own N, test and rollback — it is commit 1 of 6.5
  with a capacity+1 test and is kept on rollback (fix6 J2).
- j12: nothing gates G0's bare-`bind.rs` rule — 11.1 greps the plan for bare cites (fix6 J3).
- j11: §X.11 claimed B4's 4.40e9 ceiling sits below R13's prefill peak — false (4.40e9 > 4.305e9);
  corrected in 5.4 §X.11.
- j11: 5.2 asserts an `AlignedBuffer` → `&[f32]` accessor it never quotes — opens/expect added
  (fix6 J5).
- j10, j11, j12: the board `[46.8, 57.3]` is a seven-term composition carrying two cross-tree
  anchors (−29% Q4_K, −20% wide reduce) and δ_b's pre-registered endpoints; stated as the weakest
  number in the document and not changed.
- j10, j11: 33 of 44 cards serialise on one mutex; no parallel lane; stated, not fixed.
- j11: brief item 3 (one emitter core) is the least-closed of the eight — one kind lands, after
  the board, behind a measured gate.
