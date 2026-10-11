value `single_us`; per-process ratio arm/base (interleaved in one process), median [CoV% of the ratios] (min-max); base and arm absolute us per process
| group | extents | dispatches | arm | ratio median [CoV%] (min-max) | delta us (median ratio x base median) | base us per process | arm us per process | bits vs base (process 1) |
|---|---|---|---|---|---|---|---|---|
| 38d6abd7 | [1, 8, 2, 64] | 24 | long64 | 0.981 [1.24] (0.969-0.993) | -0.125 | 6.667 / 6.375 / 6.708 | 6.542 / 6.333 / 6.500 | 0/1024 max_abs=0e0 |
| 90a8b14d | [1, 8, 2, 64] | 24 | long64 | 1.889 [1.78] (1.879-1.942) | +22.370 | 25.125 / 25.208 / 25.167 | 47.458 / 48.958 / 47.292 | 0/1024 max_abs=0e0 |
