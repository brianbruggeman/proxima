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
