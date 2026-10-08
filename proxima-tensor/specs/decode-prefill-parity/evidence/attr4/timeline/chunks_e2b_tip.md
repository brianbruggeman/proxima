steady decode steps used: 21 (steps >= 2, excluding [23]); medians, CoV% in brackets

| chunk | ops | dispatches (capture step) | host encode us | encode us/op | gpu busy us | gpu us/dispatch | commit to gpu start us | gpu idle before us | steps where the chunk was committed after the previous chunk's gpu end |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 24 | 16 | 62 [22.4] | 2.60 | 259 [2.1] | 16.22 | 149 | 221 | 0 of 21 |
| 2 | 39 | 28 | 99 [22.2] | 2.54 | 336 [1.6] | 12.01 | 325 | 23 | 0 of 21 |
| 3 | 54 | 44 | 150 [23.3] | 2.77 | 453 [0.3] | 10.30 | 537 | 22 | 0 of 21 |
| 4 | 73 | 61 | 206 [22.2] | 2.83 | 690 [0.4] | 11.30 | 815 | 26 | 0 of 21 |
| 5 | 98 | 81 | 265 [22.5] | 2.70 | 916 [3.8] | 11.31 | 1265 | 25 | 0 of 21 |
| 6 | 133 | 103 | 337 [22.6] | 2.54 | 1318 [3.2] | 12.80 | 1882 | 31 | 0 of 21 |
| 7 | 179 | 139 | 465 [21.7] | 2.59 | 2254 [1.6] | 16.22 | 2747 | 32 | 0 of 21 |
| 8 | 242 | 181 | 574 [20.7] | 2.37 | 3986 [1.4] | 22.02 | 4462 | 39 | 0 of 21 |

| per step | median ms | CoV % | min | max |
|---|---|---|---|---|
| step wall ms | 11.121 | 7.37 | 10.659 | 14.864 |
| evaluate ms | 11.077 | 7.38 | 10.614 | 14.804 |
| sum of host encode windows ms | 2.099 | 21.72 | 2.051 | 3.452 |
| sum of chunk gpu busy ms | 10.195 | 1.05 | 9.846 | 10.348 |
| lead idle (entry to first gpu start) ms | 0.221 | 37.25 | 0.188 | 0.614 |
| inter-chunk gpu idle ms | 0.195 | 4.82 | 0.177 | 0.213 |
|   of which chunk committed after previous gpu end (waiting on encode) ms | -0.000 | NaN | -0.000 | -0.000 |
| last gpu end ms | 10.613 | 1.25 | 10.228 | 10.966 |
| evaluate minus last gpu end ms (tail on the host) | 0.410 | 125.25 | 0.368 | 3.837 |
| wall minus evaluate ms (sampling, token feedback) | 0.049 | 17.06 | 0.042 | 0.069 |
