value `single_us`; per-process ratio arm/base (interleaved in one process), median [CoV% of the ratios] (min-max); base and arm absolute us per process
| group | extents | dispatches | arm | ratio median [CoV%] (min-max) | delta us (median ratio x base median) | base us per process | arm us per process | bits vs base (process 1) |
|---|---|---|---|---|---|---|---|---|
| 3e45a325 | [1, 1, 8, 256] | 28 | long64 | 1.311 [15.98] (1.293-1.698) | +15.404 | 49.583 / 30.250 / 49.542 | 65.000 / 51.375 / 64.042 | 0/2048 max_abs=0e0 |
| 61c90d1b | [1, 1, 8, 512] | 7 | long64 | 1.333 [0.47] (1.333-1.344) | +25.306 | 76.000 / 76.250 / 75.833 | 102.125 / 101.625 / 101.083 | 0/4096 max_abs=0e0 |
| 90c72f24 | [1, 1, 8, 512] | 7 | long64 | 1.097 [0.58] (1.093-1.106) | +2.250 | 23.250 / 22.875 / 23.125 | 25.417 / 25.292 / 25.375 | 0/4096 max_abs=0e0 |
| df362c2d | [1, 1, 8, 256] | 28 | long64 | 1.077 [0.27] (1.076-1.082) | +1.375 | 18.000 / 17.875 / 17.917 | 19.375 / 19.333 / 19.292 | 0/2048 max_abs=0e0 |
