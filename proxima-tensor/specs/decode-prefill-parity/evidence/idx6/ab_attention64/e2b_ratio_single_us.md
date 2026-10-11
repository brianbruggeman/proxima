value `single_us`; per-process ratio arm/base (interleaved in one process), median [CoV% of the ratios] (min-max); base and arm absolute us per process
| group | extents | dispatches | arm | ratio median [CoV%] (min-max) | delta us (median ratio x base median) | base us per process | arm us per process | bits vs base (process 1) |
|---|---|---|---|---|---|---|---|---|
| 3e45a325 | [1, 1, 8, 256] | 28 | long64 | 1.264 [2.99] (1.262-1.330) | +4.658 | 17.625 / 17.500 / 47.750 | 22.250 / 22.125 / 63.500 | 0/2048 max_abs=0e0 |
| 61c90d1b | [1, 1, 8, 512] | 7 | long64 | 1.317 [0.78] (1.310-1.330) | +10.345 | 32.458 / 32.625 / 77.042 | 42.750 / 42.750 / 102.500 | 0/4096 max_abs=0e0 |
| 90c72f24 | [1, 1, 8, 512] | 7 | long64 | 1.067 [2.24] (1.041-1.089) | +0.602 | 8.750 / 9.042 / 23.375 | 9.333 / 9.417 / 25.458 | 0/4096 max_abs=0e0 |
| df362c2d | [1, 1, 8, 256] | 28 | long64 | 1.017 [3.19] (1.012-1.071) | +0.125 | 7.250 / 7.125 / 18.125 | 7.375 / 7.208 / 19.417 | 0/2048 max_abs=0e0 |
