steady decode steps used: 21 (steps >= 2, excluding [23]); medians, CoV% in brackets

| chunk | ops | dispatches (capture step) | host encode us | encode us/op | gpu busy us | gpu us/dispatch | commit to gpu start us | gpu idle before us | steps where the chunk was committed after the previous chunk's gpu end |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 24 | 21 | 68 [6.0] | 2.82 | 269 [1.4] | 12.82 | 135 | 201 | 0 of 21 |
| 2 | 51 | 55 | 169 [4.0] | 3.32 | 719 [0.7] | 13.08 | 257 | 26 | 0 of 21 |
| 3 | 51 | 54 | 175 [3.3] | 3.43 | 791 [2.1] | 14.66 | 824 | 26 | 0 of 21 |
| 4 | 51 | 55 | 171 [3.2] | 3.36 | 765 [2.5] | 13.91 | 1473 | 30 | 0 of 21 |
| 5 | 52 | 55 | 175 [2.9] | 3.36 | 799 [2.3] | 14.53 | 2085 | 31 | 0 of 21 |
| 6 | 51 | 54 | 169 [4.0] | 3.32 | 740 [2.6] | 13.71 | 2737 | 30 | 0 of 21 |
| 7 | 51 | 55 | 179 [3.2] | 3.51 | 770 [2.6] | 14.00 | 3327 | 36 | 0 of 21 |
| 8 | 52 | 49 | 150 [3.9] | 2.89 | 782 [2.0] | 15.97 | 3973 | 33 | 0 of 21 |

| per step | median ms | CoV % | min | max |
|---|---|---|---|---|
| step wall ms | 6.323 | 1.23 | 6.123 | 6.514 |
| evaluate ms | 6.269 | 1.24 | 6.062 | 6.460 |
| sum of host encode windows ms | 1.254 | 3.10 | 1.231 | 1.372 |
| sum of chunk gpu busy ms | 5.613 | 1.15 | 5.361 | 5.673 |
| lead idle (entry to first gpu start) ms | 0.201 | 7.62 | 0.192 | 0.252 |
| inter-chunk gpu idle ms | 0.210 | 4.58 | 0.185 | 0.231 |
|   of which chunk committed after previous gpu end (waiting on encode) ms | -0.000 | NaN | -0.000 | -0.000 |
| last gpu end ms | 6.022 | 1.23 | 5.760 | 6.133 |
| evaluate minus last gpu end ms (tail on the host) | 0.246 | 16.32 | 0.238 | 0.435 |
| wall minus evaluate ms (sampling, token feedback) | 0.057 | 8.21 | 0.054 | 0.072 |
