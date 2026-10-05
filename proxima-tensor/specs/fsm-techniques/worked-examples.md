# fsm techniques: worked examples (derived by hand before code)

## block seal trace

A block seal marks a prefix of a row cache as immutable once a whole block of rows has fallen behind a small horizon. Per-block summaries can then be folded at the seal point, and a rewind may never cut into sealed rows.

Inputs: block size b = 4 rows, seal horizon H = 1 row.

Rule: after every operation, `sealed_end = max(sealed_end, floor(max(0, len - H) / b) * b)`. A rewind to length t is refused with a typed rewind-into-sealed error, leaving the state unchanged, exactly when `t < sealed_end`. An append adds one row; a rewind to length t keeps t rows.

Sequence: a1 a2 a3 a4 a5, r1 (to 4), a6 a7 a8 a9 a10, r2 (to 8), r3 (to 7).

| op | len - H | floor(.. / b) * b | len/sealed_end | note |
|---|---|---|---|---|
| a1 | 0 | 0 | 1/0 | |
| a2 | 1 | 0 | 2/0 | |
| a3 | 2 | 0 | 3/0 | |
| a4 | 3 | 0 | 4/0 | block 0 is full but 3 < 4: inside the horizon, not sealed |
| a5 | 4 | 4 | 5/4 | block 0 sealed |
| r1 to 4 | | | 4/4 | 4 >= 4, allowed |
| a6 | 4 | 4 | 5/4 | |
| a7 | 5 | 4 | 6/4 | |
| a8 | 6 | 4 | 7/4 | |
| a9 | 7 | 4 | 8/4 | 7 < 8: block 1 still inside the horizon |
| a10 | 8 | 8 | 9/8 | block 1 sealed |
| r2 to 8 | | | 8/8 | 8 >= 8, allowed |
| r3 to 7 | | | 8/8 | 7 < 8, refused with the rewind-into-sealed error carrying keep 7 and sealed end 8; state unchanged |

RESULT block seal trace: trace=[a1:1/0,a2:2/0,a3:3/0,a4:4/0,a5:5/4,r1->4:ok 4/4,a6:5/4,a7:6/4,a8:7/4,a9:8/4,a10:9/8,r2->8:ok 8/8,r3->7:RewindIntoSealed 8/8] final_len=8 sealed_end=8 sealed_blocks=2

## eviction victim

The eviction decision is a pure function over entries in ascending stamp order, tried against an ordered rule list; the first rule that finds a candidate decides. Today's code (`PromptCache::eviction_victim`) is the list `[branch_first, lowest_stamp]`: an entry with `branch_base` set before anything a request produced, otherwise the lowest stamp. The stamp is reissued on every store, so the lowest stamp is the least recently stored entry.

Rules:

- `branch_first`: the first entry in ascending stamp order whose branch flag is set.
- `lowest_stamp`: the first entry.
- `fewest_uses`: the lowest use count, ties to the lower stamp.
- `earliest_inserted`: the lowest insertion counter.

Inputs, entries in ascending stamp order as `(stamp, branch, uses, inserted)`:

| entry | stamp | branch | uses | inserted |
|---|---|---|---|---|
| A | 3 | no | 4 | 1 |
| B | 5 | yes | 0 | 4 |
| C | 7 | yes | 1 | 2 |
| D | 9 | no | 2 | 3 |

Derivation, each as the walk over A to D:

- today's rule `[branch_first, lowest_stamp]`: A has no branch flag, B has one, so `branch_first` finds B at stamp 5; the victim is 5.
- `[lowest_stamp]`: the first entry is A, stamp 3.
- `[fewest_uses]`: use counts 4, 0, 1, 2; the lowest is B, stamp 5.
- `[earliest_inserted]`: insertion counters 1, 4, 2, 3; the lowest is A, stamp 3.
- today's rule over the two entries A and D (no branch flag set): `branch_first` finds nothing, `lowest_stamp` gives A, stamp 3.
- `[fewest_uses]` over X (2, no, 1, 1) and Y (4, no, 1, 2): equal use counts, the tie goes to the lower stamp, stamp 2.
- any rule list over an empty entry set: no victim.

`uses` and `inserted` are fields the cache entry does not carry today; this example shows the decision function over them, not that they exist.

RESULT eviction victim: today=[branch_first,lowest_stamp] -> 5; lowest_stamp -> 3; fewest_uses -> 5; earliest_inserted -> 3; no_branch today -> 3; fewest_uses_tie -> 2; empty -> none

## per-layer recompute selection

A blend-shaped stage loads cached rows for 3 chunks of 4 rows each (12 rows, indices 0 to 11; chunk c holds rows 4c to 4c+3), then recomputes only the rows whose loaded values deviate most. Selection is chained layer to layer: each later layer chooses only among the rows the previous layer selected.

Ratios per layer are `[0.5, 0.25, 0.125]`, the first being the check layer. The count per layer is `ceil(ratio * 12)`: `ceil(6.0) = 6`, `ceil(3.0) = 3`, `ceil(1.5) = 2`, so `[6, 3, 2]`.

The deviation `d` of a row is the sum over its values of `|loaded - recomputed|`.

Check layer, all 12 rows, `d` by row index 0 to 11: `[0.05, 0.90, 0.10, 0.40, 0.02, 0.30, 0.75, 0.08, 0.60, 0.15, 0.04, 0.50]`.

- ranked by `d`: row 1 (0.90), row 6 (0.75), row 8 (0.60), row 11 (0.50), row 3 (0.40), row 5 (0.30), then row 9 (0.15) and below;
- the top 6, sorted by row index: `[1,3,5,6,8,11]`;
- per chunk: chunk 0 holds rows 1 and 3, chunk 1 holds rows 5 and 6, chunk 2 holds rows 8 and 11, so 2, 2, 2.

Layer 2 sees only the 6 selected rows, with deviations: row 1 0.50, row 3 0.80, row 5 0.20, row 6 0.70, row 8 0.10, row 11 0.60.

- ranked: row 3 (0.80), row 6 (0.70), row 11 (0.60), then rows 1, 5, 8;
- the top 3, sorted: `[3,6,11]`.

Layer 3 sees only rows 3, 6 and 11, with deviations 0.30, 0.90, 0.50.

- ranked: row 6 (0.90), row 11 (0.50), row 3 (0.30);
- the top 2, sorted: `[6,11]`.

Each layer's set is a subset of the one before it, so the chain narrows 12 to 6 to 3 to 2 rows and a row dropped at one layer never returns.

RESULT per-layer recompute selection: check=[1,3,5,6,8,11] layer2=[3,6,11] layer3=[6,11] counts=[6,3,2] per_chunk_check=[2,2,2]

## block scoring

A read stage scores sealed blocks against a pooled query and attends to the top-n non-local blocks plus the local blocks. Head dimension 2, 4 sealed blocks with per-dimension key min and max:

- B0: min (0,0), max (1,1);
- B1: min (-2,-1), max (0,3);
- B2: min (1,-3), max (2,-1);
- B3: min (-1,-1), max (1,1).

Pooled query q = (2,-1). The score of block i is the sum over dims j of `max(q_j * kmax_ij, q_j * kmin_ij)`:

- B0: max(2,0) + max(-1,0) = 2 + 0 = 2;
- B1: max(0,-4) + max(-3,1) = 0 + 1 = 1;
- B2: max(4,2) + max(1,3) = 4 + 3 = 7;
- B3: max(2,-2) + max(-1,1) = 2 + 1 = 3.

Scores by block: `[2, 1, 7, 3]`.

Selection: the most recent sealed block is local (`local_blocks = 1`, B3) and never competes for a top slot. The non-local count is `nonlocal = M - local = 3`, and `n = min(nonlocal, max(n_min, ceil(keep_ratio * nonlocal)))` with `n_min = 1`, taken over the scores of B0 to B2, `[2, 1, 7]`:

- keep_ratio 0.25: `ceil(0.75) = 1`, n = 1, top = {B2}, attended = {B2} plus local {B3} = {2,3};
- keep_ratio 0.75: `ceil(2.25) = 3`, n = min(3, 3) = 3, top = {B2, B0, B1}, attended = {0,1,2,3}.

An earlier form counted n over all four blocks and let the local block compete, which listed `{0,2,3}` at 0.75; the stated rule counts non-local blocks only, so the attended count is a function of lengths and not of the scores.

RESULT block scoring: scores=[2,1,7,3] keep0.25:n=1,attended={2,3} keep0.75:n=3,attended={0,1,2,3}

## block read rows

A block read attends to the top-n sealed non-local blocks plus the local blocks, and the unsealed tail. The top-n are chosen among the sealed non-local blocks and local blocks are always added, so the number of rows read is a pure function of lengths and never of the scores.

Only the full-attention layers apply the block read; a windowed layer reads `min(len, window)` rows either way. gemma4 E2B has 35 layers (`gemma4.block_count = 35`) with a sliding window of 512 (`gemma4.attention.sliding_window = 512`). Layers 4, 9, 14, 19, 24, 29 and 34 report `is_swa = 0`, so `layers = 7`. Each of those 7, including any that shares another layer's cache, evaluates its own attention over the rows. Counting 35 layers (every layer of the model) would be wrong for this checkpoint: 28 of its layers are windowed.

Per layer, per decode step s from 0 to T-1:

- `len_s = P + s + 1` (rows visible, including the new row);
- `sealed_end_s = floor(max(0, len_s - H) / b) * b` and `M_s = sealed_end_s / b`;
- `L' = min(L, M_s)` and `n_s = min(M_s - L', max(n_min, ceil(keep_ratio * (M_s - L'))))`;
- `rows_s = (n_s + L') * b + (len_s - sealed_end_s)`;
- `kv_rows_read = layers * sum_s rows_s`.

Instance: block size b = 64, hot tail H = 64, n_min = 16, local blocks L = 1, layers = 7, prompt P = 4096, decode steps T = 2.

- len = 4097 and 4098; sealed_end = 4032 for both (floor(4033/64) = 63 and floor(4034/64) = 63); M = 63, L' = 1, non-local blocks 62;
- keep_ratio 0.1: n = min(62, max(16, ceil(6.2) = 7)) = 16; rows = 17*64 + 65 = 1153 and 17*64 + 66 = 1154; per layer 2307; total 7 * 2307 = 16149;
- keep_ratio 0.9: n = min(62, max(16, ceil(55.8) = 56)) = 56; rows = 57*64 + 65 = 3713 and 57*64 + 66 = 3714; per layer 7427; total 7 * 7427 = 51989.

RESULT block read rows: P=4096 T=2 layers=7 b=64 H=64 n_min=16 L=1 keep=0.1 -> 16149; keep=0.9 -> 51989

## sampled read budget

A sampled read's output has relative error at most epsilon with probability at least 1 - delta. The number of sampled rows n is sized from a base-rate sample of per-row weights, so the budget is a function of the observed spread and never of the scores of the rows being read.

Rule (two-sided normal quantile): `n = ceil((z * sd / (epsilon * mu))^2)` with `z = z_(1 - delta/2)`, and `mu` and `sd` the sample mean and sample standard deviation (n-1 denominator) over the base-rate sample.

Instance: epsilon = 0.05, delta = 0.05, so z = z_0.975 = 1.959964. Base-rate sample of 8 per-row weights `[0.9, 1.1, 1.0, 1.2, 0.8, 1.0, 1.1, 0.9]`.

- mu = 8.0 / 8 = 1.0;
- squared deviations 0.01, 0.01, 0, 0.04, 0.04, 0, 0.01, 0.01, sum 0.12;
- variance = 0.12 / 7 = 0.0171429; sd = 0.130931;
- z * sd / (epsilon * mu) = 1.959964 * 0.130931 / 0.05 = 5.13239; squared = 26.3414; n = ceil(26.3414) = 27.

RESULT sampled read budget: mu=1.0 sd=0.1309 z=1.959964 n=27

## row tolerance

Each sublayer ends in a reduction accumulated in f32 as a tree. A tree reduction of n terms has relative error at most `ceil(log2 n) * u`, `u = 2^-24`. Rows are compared K and V at every layer, so `tau = 2 * L * ceil(log2 n) * u` with L = block_count and n the widest reduction in the model: `n = max(embedding_length, head_count * key_length, max over layers of feed_forward_length, expert_feed_forward_length)`. The attention output projection reduces over `head_count * key_length`. The comparison is `max_i |a_i - b_i| <= tau * max_i |b_i|` per row.

Inputs (headers under `proxima-model-interop/tests/fixtures/llama-parity/<model>/gguf_kv.txt`):

- gemma4_e2b: L 35; embedding 1536; attention 8 x 512 = 4096; feed forward 12288 (layers 15 to 34; layers 0 to 14 are 6144, as in `proxima-model-interop/src/gemma4/bind.rs::e2b_shaped`); n = 12288, ceil(log2) = 14; tau = 2*35*14*2^-24 = 5.841e-05.
- gemma4_26b: L 30; embedding 2816; attention 16 x 512 = 8192; feed forward 2112; expert 704; n = 8192, ceil(log2) = 13; tau = 2*30*13*2^-24 = 4.649e-05.
- granite_moe: L 24; embedding 1024; attention 16 x 64 = 1024 (no `key_length` key, so head width is `rope.dimension_count` 64); expert feed forward 512; n = 1024, ceil(log2) = 10; tau = 2*24*10*2^-24 = 2.861e-05.

Assumption: a kernel that reduces sequentially instead of as a tree would exceed these bounds; if a measured gap exceeds tau the test fails and tau is not widened. The earlier gemma4 E2B value (5.424e-05) used n = 6144, missing the 20 shared-cache layers' 12288, and omitted the attention width.

RESULT row tolerance: gemma4_e2b=5.841e-05 gemma4_26b=4.649e-05 granite_moe=2.861e-05

## readout tolerance

The readout of a settle decision is a log probability: `logprob = logit - logsumexp(logits)`.

Model: the logits carry relative error tau (the row tolerance above) and an assumed bound `|logit| <= 40`. The error of one logit is at most `40 * tau`. A log probability moves by the chosen logit's error plus the logsumexp's error, so `tol_logprob = 2 * tau * 40 = 80 * tau`.

For the top-1 minus top-2 probability margin, `|dp| <= p * |dlogp|` and `p1 + p2 <= 1`, so `tol_margin = tol_logprob`.

The bound on the logit:
- gemma4_e2b and gemma4_26b: both headers declare `gemma4.final_logit_softcapping = 30.0`, so `|logit| <= 30 < 40` holds by construction.
- granite_moe: the header declares no soft cap. Its logits are divided by `granitemoe.logit_scale = 6.0` after the head, and `|logit| <= 40` on the scaled value is an assumption, unmeasured.

Results:
- gemma4_e2b: 80 * 5.841e-05 = 4.673e-03
- gemma4_26b: 80 * 4.649e-05 = 3.719e-03
- granite_moe: 80 * 2.861e-05 = 2.289e-03 (rests on the unmeasured bound)

RESULT readout tolerance: tol_logprob gemma4_e2b=4.673e-03 gemma4_26b=3.719e-03 granite_moe=2.289e-03 tol_margin=tol_logprob

## cartridge concatenation rotation

A stored block of key rows is moved to a new position by rotating each (even, odd) pair of every row by the angle of the position delta. For a pair with angle `a` per position and delta `d`: `even' = even * cos(a*d) - odd * sin(a*d)` and `odd' = odd * cos(a*d) + even * sin(a*d)`. A negative delta rotates the other way. Value rows are not rotated.

Inputs: head dim 4 (2 pairs), per-pair angle per position `(pi/2, pi/4)`. Block A holds 2 rows. Block B (2 rows, base position 0) is concatenated after A, so each B row moves by delta = 2 positions: angles `(pi, pi/2)`, `cos = [-1, 0]`, `sin = [0, 1]`. These are exact in the hand arithmetic; in f32 `sin(pi)` is not exactly 0, so a test of this value uses a tolerance of 1e-6.

B's key rows (layout `[row][pair]`): row 0 even `[1,2]` odd `[0,3]`; row 1 even `[0,4]` odd `[2,-1]`.

- row 0 pair 0: even' = 1*(-1) - 0*0 = -1; odd' = 0*(-1) + 1*0 = 0
- row 0 pair 1: even' = 2*0 - 3*1 = -3; odd' = 3*0 + 2*1 = 2
- row 1 pair 0: even' = 0*(-1) - 2*0 = 0; odd' = 2*(-1) + 0*0 = -2
- row 1 pair 1: even' = 4*0 - (-1)*1 = 1; odd' = (-1)*0 + 4*1 = 4

RESULT cartridge concatenation rotation: k_b_rotated even=[[-1,-3],[0,1]] odd=[[0,2],[-2,4]] v_unchanged

## threshold judge

A threshold judge is one pure function of one readout. The readout is the chosen token's logprob. The rule: settle iff `readout >= h`, otherwise escalate. The comparison is inclusive, so a readout equal to `h` settles. Here `h = -0.5`.

- lp = -0.2: -0.2 >= -0.5, settle
- lp = -0.9: -0.9 < -0.5, escalate
- lp = -0.5: -0.5 >= -0.5, settle (boundary, inclusive)

RESULT threshold judge: lp=-0.2 -> settle; lp=-0.9 -> escalate; lp=-0.5 -> settle

## classifier judge

A classifier judge is one more pure function of class probabilities. The judge model emits `[p_accept, p_escalate]`; the probabilities are readouts the settle hook consumes, not values it produces. `accept_class = 0` is a configured value. The rule: settle iff `argmax = accept_class`, otherwise escalate.

- [0.7, 0.3]: argmax is index 0 (0.7 > 0.3), equals `accept_class`, settle
- [0.4, 0.6]: argmax is index 1 (0.6 > 0.4), differs from `accept_class`, escalate

RESULT classifier judge: [0.7,0.3] -> settle; [0.4,0.6] -> escalate

## conformal judge

A conformal judge is a pure function of vote counts over sixteen sampled answers. The bound `max_disagree` is the calibrated q-hat in units of samples, an integer, because dividing by the sample count is not exact: one disagreeing draw of 16 is 62.5 per thousand. Here `samples = 16` and `max_disagree = 13` (a q-hat of 0.8125 over 16 samples is 13 disagreements, given as a calibrated value; fitting it is out of scope). An answer is in the prediction set iff `samples - votes <= max_disagree`. The judge settles iff the set has exactly one answer, and the settled answer is that one; otherwise it escalates.

- votes {A:9, B:4, C:3}: disagreements 16-9 = 7, 16-4 = 12, 16-3 = 13, all <= 13, set {A,B,C}, size 3, escalate
- votes {A:12, B:2, C:2}: disagreements 16-12 = 4, 16-2 = 14, 16-2 = 14, only A is <= 13, set {A}, size 1, settle A

RESULT conformal judge: samples=16 max_disagree=13 votes{A:9,B:4,C:3} -> escalate (|C|=3); votes{A:12,B:2,C:2} -> settle A (|C|=1)

## isotonic judge

An isotonic judge is a pure function of the per-token margins of the answer. The margin of a token is `p_top1 - p_top2`; `u = 1 - mean(margins)`. The fitted table `g` is supplied (a calibration output; producing it is a fitter and out of scope), keyed by the serving configuration and read at the exact grid point `u`: grid `u = 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0` with `g = 0, 0, 0.5, 0.5, 0.5, 0.5, 1, 1`. The judge settles iff `g(u) <= theta`, otherwise it escalates. Here `theta = 0.5`. All means and u values below are exact in binary.

- margins [0.875, 0.625, 0.75]: sum 2.25, mean 0.75, u = 0.25, g(0.25) = 0, 0 <= 0.5, settle
- margins [0.5, 0.5, 0.5]: mean 0.5, u = 0.5, g(0.5) = 0.5, 0.5 <= 0.5, settle (boundary, inclusive)
- margins [0.25, 0.0, 0.125]: sum 0.375, mean 0.125, u = 0.875, g(0.875) = 1, 1 > 0.5, escalate

RESULT isotonic judge: u=0.25 g=0 -> settle; u=0.5 g=0.5 -> settle; u=0.875 g=1 -> escalate

## always judge

The always judge is the default: an absent cascade is one tier with judge `always` and one call, and the last tier of any cascade always settles. It is a pure function of nothing: it settles for every request regardless of readouts. In vote terms `samples = 1`, votes `[1]`, disagreement `1 - 1 = 0`, which is inside every bound (`0 <= max_disagree` for any non-negative `max_disagree`).

- logprob -0.2 (a confident token): settle
- logprob -3.0 (an unconfident token): settle
- readout missing: settle, because the judge reads no readout

RESULT always judge: any readout -> settle (3 of 3 requests)

## top-fraction selection

The selection rule: `count = max(min_rows, ceil(fraction * M))`; `rank(i)` is the number of scores strictly greater than `score[i]` plus the number of equal scores at a lower index, so ranks are distinct and exactly `count` rows are selected; a row is selected when `rank < count` or its index is in the kept rows. Inputs: fraction 0.25, min_rows 2, keep_rows {5}, scores (12 values, indices 0 to 11, all distinct) `[0.31, 0.92, 0.15, 0.77, 0.64, 0.08, 0.55, 0.99, 0.23, 0.71, 0.40, 0.86]`.

- M = 12 (fraction times M is an integer): count = max(2, ceil(0.25 * 12) = 3) = 3; ranks: 0.99 (idx 7) rank 0, 0.92 (idx 1) rank 1, 0.86 (idx 11) rank 2; selected {1,7,11} union keep {5} = {1,5,7,11}
- M = 11 (the first 11 scores; fraction times M is not an integer): count = max(2, ceil(2.75) = 3) = 3; top three: 0.99 (7), 0.92 (1), 0.77 (3); union {5} = {1,3,5,7}

The two sets differ, so the count formula is distinguishable at the two sizes.

Equal scores: M = 6, fraction 0.5, min_rows 1, no kept rows, scores `[0.5, 0.9, 0.5, 0.5, 0.1, 0.9]`: count = max(1, ceil(3.0) = 3) = 3. With the tie rule: idx 1 (0.9) greater 0, equal-lower 0, rank 0; idx 5 (0.9) rank 0 + 1 = 1; idx 0 (0.5) greater 2, rank 2; idx 2 (0.5) rank 2 + 1 = 3; idx 3 (0.5) rank 2 + 2 = 4; idx 4 (0.1) greater 5, rank 5; selected (rank < 3) = {0,1,5}, 3 rows. With strictly-greater rank alone, idx 0, 2 and 3 all have rank 2, so {0,1,2,3,5} would be selected, 5 rows.

RESULT top-fraction selection: fraction=0.25 min_rows=2 keep={5} M=12 -> {1,5,7,11}; M=11 -> {1,3,5,7}; ties M=6 fraction=0.5 min_rows=1 scores=[0.5,0.9,0.5,0.5,0.1,0.9] -> {0,1,5} (3 selected; strictly-greater rank alone selects 5)

## block read selection

The selection rule: the top blocks are chosen among the sealed blocks that are not local blocks; `nonlocal` is the count of those blocks and `n = min(nonlocal, max(min_blocks, ceil(keep_ratio * nonlocal)))`. The read set is the union of the selected blocks, the local blocks and the unsealed tail. Inputs: block size b = 16, 4 sealed blocks (rows 0 to 63) plus an unsealed tail of 5 rows (rows 64 to 68, 69 rows total), keep_ratio 0.5, min_blocks 1, local_blocks 1, block scores `[6, 9, 4, 1]` (block 3, the most recent sealed block, has the lowest score).

- local = block 3 (always, despite score 1), so the non-local blocks are 0, 1, 2 with scores 6, 9, 4 and nonlocal = 3
- n = min(3, max(1, ceil(0.5 * 3) = 2)) = 2
- top 2 of the non-local scores: block 1 (9), block 0 (6); block 2 (4) is not selected
- the tail rows 64 to 68 are all attended
- attended blocks {0,1,3} = 3 * 16 = 48 rows, plus 5 tail rows = 53 rows of 69; the 16 rows of block 2 are not attended

Counting n over all four blocks (`ceil(0.5 * 4) = 2`) also gives 2 here, so this selection does not distinguish the two forms; the non-local form is the stated rule.

RESULT block read selection: b=16 sealed=4 tail=5 n=2 top={0,1} local={3} attended_blocks={0,1,3} attended_rows=53

## action speculation

The entry type is an action, not a token. Environment `E`: `values: BTreeMap<u8, i64>` (a missing key reads 0) and `version: u64`; `apply(A(key, delta, version))` does `values[key] += delta; version += 1`. True policy: `true_action(E) = A(key = E.version % 3, delta = (E.values[key] rem 5) + 1, version = E.version)`. Guessing drafter: `guessed_action(E) = true_action(E)` except `delta + 1` when `E.version % 4 == 3`; the draft runs on a clone and applies its own guesses. Run: 12 steps, rows per draft = min(4, remaining); a round whose index `r` satisfies `r % 3 == 2` drafts from the environment as it was at the start of the previous round (a stale snapshot). Verify: observed row i = `true_action` on `E_i`, where `E_0` is the real environment and `E_(i+1) = E_i.apply(draft[i])`; row cache i = `E_i.apply(observed[i])`.

Sequential run (each delta is `values[key] rem 5 + 1`):

- S0 A(0,1,0): key 0 holds 0, 0 rem 5 = 0, delta 1; values {0:1}
- S1 A(1,1,1): key 1 holds 0, delta 1; {0:1,1:1}
- S2 A(2,1,2): key 2 holds 0, delta 1; {0:1,1:1,2:1}
- S3 A(0,2,3): key 0 holds 1, 1 rem 5 = 1, delta 2; {0:3,1:1,2:1}
- S4 A(1,2,4): key 1 holds 1, delta 2; {0:3,1:3,2:1}
- S5 A(2,2,5): key 2 holds 1, delta 2; {0:3,1:3,2:3}
- S6 A(0,4,6): key 0 holds 3, 3 rem 5 = 3, delta 4; {0:7,1:3,2:3}
- S7 A(1,4,7): key 1 holds 3, delta 4; {0:7,1:7,2:3}
- S8 A(2,4,8): key 2 holds 3, delta 4; {0:7,1:7,2:7}
- S9 A(0,3,9): key 0 holds 7, 7 rem 5 = 2, delta 3; {0:10,1:7,2:7}
- S10 A(1,3,10): key 1 holds 7, delta 3; {0:10,1:10,2:7}
- S11 A(2,3,11): key 2 holds 7, delta 3; {0:10,1:10,2:10}

Final environment `{0:10, 1:10, 2:10}`, version 12.

Each policy is also stated as the three values of one accept rule: similarity floor, minimum run, row cap. The accepted count of a round is the length of the leading run of rows whose similarity to the observed row is at least the floor, cut at the row cap, and zero when that run is shorter than the minimum run.

Equality policy (a row is accepted iff action and version are equal; rule values: floor 1.0 under exact equality, minimum run 0, row cap 8, which never binds at 4 rows):

- r0: the four drafted rows are S0, S1, S2 and A(0,3,3) (version 3 satisfies `3 % 4 == 3`, so the delta 2 is guessed as 3); observed S0..S2 are equal, observed[3] = A(0,2,3) differs; accepted 3, Rollback; commits S0..S2 and S3 (4 steps)
- r1: from E4, drafted rows S4, S5, S6 and A(1,5,7) (version 7, `7 % 4 == 3`, delta 4 guessed as 5); accepted 3, Rollback; commits S4..S7 (8 steps)
- r2 (stale, `2 % 3 == 2`): drafts from E4 while the real environment is E8; the first drafted row S4 differs from the first observed row S8; accepted 0, Rollback; commits S8 (9 steps)
- r3: from E9, rows = min(4, 3) = 3; drafted rows S9, S10 and A(2,4,11) (version 11, `11 % 4 == 3`, delta 3 guessed as 4); accepted 2, Rollback; commits S9, S10, S11 (12 steps)
- totals: rounds 4, rejected rounds 4, accepted rows 3 + 3 + 0 + 2 = 8

Anchor+macro policy (rule values: floor 1.0, minimum run 3, no row cap; the aligned prefix must reach 3, else accepted 0):

- r0: aligned 3, accepted 3; r1: aligned 3, accepted 3; r2: stale, accepted 0
- r3: aligned 2 < 3, so accepted 0; commits S9 (10 steps)
- r4 (from E10, rows 2): the first drafted row S10 is aligned, the second A(2,4,11) is not; aligned 1 < 3, accepted 0; commits S10 (11 steps)
- r5 (stale, `5 % 3 == 2`, from the start of r4 = E10, rows 1): the only drafted row S10 differs from observed S11; accepted 0; commits S11 (12 steps)
- totals: rounds 6, rejected rounds 6, accepted rows 3 + 3 = 6

Verifier-exact policy (a row is accepted iff the whole entry is equal; rule values: floor 1.0 under a similarity that is the fraction of the three fields key, delta, version that agree, minimum run 0, no row cap): the rows that differ are the same rows as under equality, so the rounds are the same as equality; totals rounds 4, rejected 4, accepted rows 8.

RESULT action speculation: sequential_final={0:10,1:10,2:10} v12; equality rounds=4 rejected=4 accepted_rows=8; anchor_macro(min_skip=3) rounds=6 rejected=6 accepted_rows=6; verifier_exact rounds=4 rejected=4 accepted_rows=8
