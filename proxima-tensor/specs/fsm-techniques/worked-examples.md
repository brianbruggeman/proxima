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
