steady decode steps used: 21 (steps >= 2, excluding [23]); medians, CoV% in brackets

| chunk | ops | dispatches (capture step) | host encode us | encode us/op | gpu busy us | gpu us/dispatch | commit to gpu start us | gpu idle before us | steps where the chunk was committed after the previous chunk's gpu end |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 24 | 21 | 68 [8.8] | 2.82 | 262 [0.9] | 12.48 | 136 | 204 | 0 of 21 |
| 2 | 17 | 18 | 60 [8.9] | 3.53 | 222 [1.0] | 12.35 | 353 | 21 | 0 of 21 |
| 3 | 24 | 26 | 88 [4.2] | 3.66 | 391 [0.6] | 15.03 | 503 | 22 | 0 of 21 |
| 4 | 32 | 34 | 111 [8.4] | 3.47 | 460 [2.2] | 13.53 | 803 | 21 | 0 of 21 |
| 5 | 43 | 46 | 149 [4.0] | 3.47 | 673 [3.0] | 14.64 | 1137 | 25 | 0 of 21 |
| 6 | 58 | 62 | 193 [2.7] | 3.33 | 872 [3.0] | 14.06 | 1639 | 27 | 0 of 21 |
| 7 | 78 | 83 | 260 [2.9] | 3.33 | 1208 [2.8] | 14.56 | 2280 | 26 | 0 of 21 |
| 8 | 107 | 108 | 318 [4.6] | 2.97 | 1594 [2.2] | 14.76 | 3180 | 32 | 0 of 21 |

| per step | median ms | CoV % | min | max |
|---|---|---|---|---|
| step wall ms | 6.330 | 1.35 | 6.064 | 6.480 |
| evaluate ms | 6.280 | 1.34 | 6.010 | 6.432 |
| sum of host encode windows ms | 1.243 | 3.66 | 1.219 | 1.401 |
| sum of chunk gpu busy ms | 5.662 | 1.38 | 5.360 | 5.716 |
| lead idle (entry to first gpu start) ms | 0.204 | 9.56 | 0.189 | 0.260 |
| inter-chunk gpu idle ms | 0.173 | 5.87 | 0.154 | 0.198 |
|   of which chunk committed after previous gpu end (waiting on encode) ms | -0.000 | NaN | -0.000 | -0.000 |
| last gpu end ms | 6.042 | 1.35 | 5.723 | 6.099 |
| evaluate minus last gpu end ms (tail on the host) | 0.241 | 13.13 | 0.228 | 0.367 |
| wall minus evaluate ms (sampling, token feedback) | 0.052 | 16.24 | 0.047 | 0.077 |
