value `marginal_us`; per-process ratio arm/base (interleaved in one process), median [CoV% of the ratios] (min-max); base and arm absolute us per process
| group | extents | dispatches | arm | ratio median [CoV%] (min-max) | delta us (median ratio x base median) | base us per process | arm us per process | bits vs base (process 1) |
|---|---|---|---|---|---|---|---|---|
| 3e45a325 | [1, 1, 8, 256] | 28 | long64 | 1.453 [0.91] (1.438-1.464) | +18.011 | 39.894 / 32.042 / 39.786 | 58.408 / 46.072 / 57.797 | 0/2048 max_abs=0e0 |
| 61c90d1b | [1, 1, 8, 512] | 7 | long64 | 1.329 [1.88] (1.288-1.331) | +24.817 | 75.661 / 75.342 / 69.208 | 100.583 / 100.317 / 89.117 | 0/4096 max_abs=0e0 |
| 90c72f24 | [1, 1, 8, 512] | 7 | long64 | 1.158 [1.08] (1.146-1.171) | +2.816 | 17.961 / 17.508 / 17.817 | 20.589 / 20.506 / 20.633 | 0/4096 max_abs=0e0 |
| df362c2d | [1, 1, 8, 256] | 28 | long64 | 1.159 [0.81] (1.145-1.163) | +2.000 | 12.636 / 12.583 / 12.439 | 14.467 / 14.583 / 14.461 | 0/2048 max_abs=0e0 |
