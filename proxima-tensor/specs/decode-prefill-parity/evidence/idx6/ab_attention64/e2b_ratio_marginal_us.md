value `marginal_us`; per-process ratio arm/base (interleaved in one process), median [CoV% of the ratios] (min-max); base and arm absolute us per process
| group | extents | dispatches | arm | ratio median [CoV%] (min-max) | delta us (median ratio x base median) | base us per process | arm us per process | bits vs base (process 1) |
|---|---|---|---|---|---|---|---|---|
| 3e45a325 | [1, 1, 8, 256] | 28 | long64 | 1.355 [5.65] (1.335-1.480) | +5.468 | 15.317 / 15.392 / 38.911 | 20.758 / 20.550 / 57.592 | 0/2048 max_abs=0e0 |
| 61c90d1b | [1, 1, 8, 512] | 7 | long64 | 1.330 [1.37] (1.314-1.350) | +10.004 | 28.844 / 30.311 / 75.639 | 38.364 / 39.814 / 102.100 | 0/4096 max_abs=0e0 |
| 90c72f24 | [1, 1, 8, 512] | 7 | long64 | 1.143 [2.46] (1.096-1.146) | +0.986 | 6.172 / 6.881 / 17.983 | 6.767 / 7.867 / 20.603 | 0/4096 max_abs=0e0 |
| df362c2d | [1, 1, 8, 256] | 28 | long64 | 1.061 [6.87] (1.016-1.161) | +0.316 | 5.192 / 4.572 / 12.619 | 5.275 / 4.850 / 14.647 | 0/2048 max_abs=0e0 |
