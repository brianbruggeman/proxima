steady decode steps used: 21 (steps >= 2, excluding [23]); medians, CoV% in brackets

| chunk | ops | dispatches (capture step) | host encode us | encode us/op | gpu busy us | gpu us/dispatch | commit to gpu start us | gpu idle before us | steps where the chunk was committed after the previous chunk's gpu end |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 24 | 16 | 67 [19.5] | 2.80 | 260 [2.0] | 16.26 | 141 | 221 | 0 of 21 |
| 2 | 116 | 91 | 307 [20.9] | 2.64 | 1028 [0.3] | 11.29 | 130 | 25 | 4 of 21 |
| 3 | 117 | 96 | 313 [21.6] | 2.67 | 997 [2.3] | 10.38 | 887 | 29 | 0 of 21 |
| 4 | 117 | 93 | 304 [20.7] | 2.60 | 1117 [3.8] | 12.01 | 1592 | 27 | 0 of 21 |
| 5 | 117 | 91 | 301 [21.0] | 2.57 | 1377 [2.4] | 15.13 | 2457 | 31 | 0 of 21 |
| 6 | 117 | 90 | 297 [21.2] | 2.53 | 1562 [2.5] | 17.35 | 3526 | 33 | 0 of 21 |
| 7 | 117 | 90 | 297 [21.2] | 2.54 | 1425 [2.1] | 15.83 | 4857 | 35 | 0 of 21 |
| 8 | 117 | 86 | 270 [20.4] | 2.31 | 2429 [1.1] | 28.24 | 6035 | 32 | 0 of 21 |

| per step | median ms | CoV % | min | max |
|---|---|---|---|---|
| step wall ms | 11.106 | 7.85 | 10.747 | 15.128 |
| evaluate ms | 11.058 | 7.85 | 10.694 | 15.058 |
| sum of host encode windows ms | 2.163 | 20.49 | 2.045 | 3.440 |
| sum of chunk gpu busy ms | 10.180 | 1.06 | 9.811 | 10.312 |
| lead idle (entry to first gpu start) ms | 0.221 | 38.64 | 0.185 | 0.621 |
| inter-chunk gpu idle ms | 0.218 | 25.25 | 0.189 | 0.382 |
|   of which chunk committed after previous gpu end (waiting on encode) ms | -0.000 | 211.97 | -0.000 | 0.204 |
| last gpu end ms | 10.639 | 1.10 | 10.276 | 10.888 |
| evaluate minus last gpu end ms (tail on the host) | 0.418 | 135.11 | 0.336 | 4.170 |
| wall minus evaluate ms (sampling, token feedback) | 0.048 | 16.53 | 0.042 | 0.070 |
