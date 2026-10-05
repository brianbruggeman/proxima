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
