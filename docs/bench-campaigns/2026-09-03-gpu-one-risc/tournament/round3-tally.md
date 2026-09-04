# plan-rigor round 3 — Borda tally

Candidates: S2 (synthesis_2, round-2 winner / incumbent), B3 (blind), S3 (synthesis_3).
Label maps: j7 P1=S2 P2=B3 P3=S3; j8 P1=S3 P2=S2 P3=B3; j9 P1=B3 P2=S3 P3=S2.

| judge | ranking (labels) | ranking (candidates) | S3 | B3 | S2 |
|---|---|---|---|---|---|
| j7 | [P3, P2, P1] | [S3, B3, S2] | 2 | 1 | 0 |
| j8 | [P1, P3, P2] | [S3, B3, S2] | 2 | 1 | 0 |
| j9 | [P2, P1, P3] | [S3, B3, S2] | 2 | 1 | 0 |
| **Borda** | | | **6** | **3** | **0** |

Unanimous: S3 wins round 3 with three first-place votes. The incumbent S2 lost, so convergence
("same candidate wins two consecutive rounds") is NOT reached after round 3. Round 4 is the cap
round: S3 is the incumbent; if S3 wins round 4 the tournament converges; if synthesis_4 wins,
the cap rule emits the final winner with a "no convergence" flag.

Per-axis (j7 / j8 / j9 on S3): risk 8/9/8, ordering 9/9/8, rollback 9/9/8, missing 8/8/7,
coupling 8/9/8, observability 9/9/8, scope 8/8/6.

Residual defects the judges named on S3 (to carry into round 4 and into the assembled §5):
- j7, j8: card 9.1 opens `map.rs:238` (`IndexMap::as_gather_from_output`) but its blast, expect
  and kill never scope the autograd adjoint path that rides the same convention.
- j7: card 9.2 asserts `UNIFORM_BUFFER_REUSES` unchanged with no expect line and no ON/OFF arm,
  on the one field a per-token uniform patch is most likely to move.
- j7, j8: card 8.3's `>= 600` line-count continuation gate is an evidence-free threshold of the
  same shape S3 retired from S2 (`<= 5` unclassifiable functions).
- j8: on-device argmax is neither a card nor an entry in the abandoned list.
- j9: `$WT` / `$TD` / `$LOCK` are shell variables; shell state does not persist between a hands
  model's tool calls, so every command must carry absolute paths (or the env block must be
  re-run with every command).
- j9: card 0.7's capture loop uses `read -r -d ''`, which is bash-specific; the host shell is
  zsh and the card says "run verbatim".
- j9: card 6.1 relies on the KV arena being zero-initialised (`exp(-inf) = 0` exactly on rows
  `[cached_len, bucket)`) without a citation or a test that the arena is zero-filled.
- j9: 5.1 (`SpanSlot`), 8.1 (`trait Dialect`), 9.2 (`Plan.dynamic_bases`) add surface with a
  principle citation but without both binary questions written in-line (only 3.1's `Route` has
  them).
- j9: 10.4 holds the single mutex through an onnxruntime C++ build; `--wait 5400` makes other
  cards exit 75 rather than queue.
- j9 (all three plans): `profiles_one_real_decode_step_by_per_op_gpu_time` hardcodes
  `max_tokens = 5` (`bind.rs:3105`) and sets `PROXIMA_METAL_OP_PROFILE_STEP=3` itself
  (`:3122`), so `PROXIMA_MAX_TOKENS=8` does not control the MILLI rung; S3's F-from-
  `generated.0.len()` contract absorbs it, but G4 must say so.
- j7: 33 of 44 cards are measuring cards serialised on one mutex; the schedule length is the
  measuring-card count, not the 30-card path; stated, not fixed.
- j7: the board band `[45.5, 53.8]` carries `δ_b` as a free symbol in five of eight ladder rows
  until 6.3 measures it.
