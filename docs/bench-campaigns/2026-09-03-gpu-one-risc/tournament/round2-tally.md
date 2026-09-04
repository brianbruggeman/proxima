# plan-rigor round 2 — Borda tally (judge 3 pending at write time; outcome already decided)

Candidates: S1 (synthesis_AB, round-1 winner / incumbent), B2 (blind), S2 (synthesis_2).
Label maps: j4 P1=S1 P2=B2 P3=S2; j5 P1=S2 P2=S1 P3=B2; j6 P1=B2 P2=S2 P3=S1.

| judge | ranking (candidates) | S2 | B2 | S1 |
|---|---|---|---|---|
| j4 | [S2, B2, S1] | 2 | 1 | 0 |
| j5 | [S2, B2, S1] | 2 | 1 | 0 |
| j6 | [S2, B2, S1] | 2 | 1 | 0 |
| **Borda** | | **6** | **3** | **0** |

Unanimous: S2 wins round 2 with three first-place votes. The incumbent S1 lost, so convergence
("same candidate wins two consecutive rounds") is NOT reached; S2 becomes the round-3 incumbent.

Concrete breaks j6 found in S2 (verified against main; MUST be fixed in round 3):
- 0.6 (hands): `$wt` sits inside the single-quoted `sh -c` body, so the inner shell expands it
  empty and all ten worktrees' untracked files collide into `$R/-untracked/`; and the
  `git apply --check` loop runs `git -C .../proxima` (main at 4be2f3a) while the expect says "against
  that worktree's own recorded HEAD" — every check reports red and the kill misfires. Use
  `git -C /path/to/proxima-wt-$wt apply --check` (or `git --work-tree`/a temp checkout at the
  recorded HEAD) and pass `$wt` as a positional to `sh -c 'script' _ "$wt"`.
- 0.3 expects literal `plan_hits=0 plan_misses=8` against its own G4 rule (no literal token
  counts) — write it as `plan_hits == 0 && plan_misses == F`.
- 4.1 N2 binds the distinct-shape count to prefill `(31, 256)`; 31 is unverified — use the 0.12
  symbols dump's observed value, never a literal.
- 5.2/5.4 injectivity by leaf NAME convention (`"*.write_row"`) — needs a structural proof at bind
  time (indices = Iota(coeff 1) + Constant, or a declared strictly-monotonic leaf checked once), with
  a scheduled fallback.
- 8.2 onnxruntime `./build.sh --parallel` is under the lock but unbounded (cap `--parallel N`, record
  peak RSS, or run it on a separate host).

Residual defects the judges named on S2 (to carry into round 3):
- j4: 5.2 proves scatter injectivity by a leaf NAME convention (`"*.write_row"`), not structurally,
  and the affine fallback is declared closed by over-reading map.rs:118-124 (which rejects a
  Reduce-WIDE field, not an out_map offset) — Phase 5 is single-path with no scheduled fallback.
- j4: the 21-card spine passes through 4.1, pre-registered as a timing LOSS into [63,72] whose
  payoff only arrives at 5.3/6.2; the board is worse than R13 for most of the critical path.
- j4: 7.1's "≤5 unclassifiable functions" threshold has no evidence behind it.
- j4: brief item 1 stays known-false (0.14 RED) until 7.6 behind the whole emitter reorganisation.
- j4: 49 worktree lines written with a literal ellipsis `…/proxima-wt-riscNN`; a duplicated/
  truncated `### 7.1` heading (an artifact of the two-block output).
- j5: 4.1 is the one deliberate regression upstream of five cards (honest, but the highest-risk card).
- j5: 8.2's onnxruntime source-build branch is under the lock but otherwise unbounded.
- both: volume — 45 cards / 49 worktrees / 21-card critical path.
