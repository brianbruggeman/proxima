3 files; value `single_us` from `group` lines; median [CoV%] (min-max) over runs
| group | extents | arm | n | single_us median [CoV%] (min-max) | bit compare vs base (first file) | cpu ms per replay | cpu % of wall | wall ms per replay | static tg bytes, bound buffer MB, rss peak MB, footprint MB, Metal MB, load |
|---|---|---|---|---|---|---|---|---|---|
| 28e53828 | [971, 1, 8, 256] | base | 3 | 931.958 [0.19] (930.500-934.042) | differing=0/1988608 max_ulp=0 | 0.059 | 4.8 | 1.222 | tg_static=8576 bound_buffer_mb=385.3 rss_peak_mb=3433.0 footprint_mb=224.7 gpu_alloc_mb=3374.6 load=4.07 |
| 28e53828 | [971, 1, 8, 256] | both_cvt | 3 | 913.000 [0.04] (912.458-913.250) | differing=1988238/1988608 max_ulp=1909662226 | 0.055 | 5.0 | 1.109 | tg_static=8576 bound_buffer_mb=385.3 rss_peak_mb=3433.0 footprint_mb=224.7 gpu_alloc_mb=3374.6 load=4.07 |
| 28e53828 | [971, 1, 8, 256] | emit_half | 3 | 912.167 [0.23] (911.500-915.417) | differing=1988238/1988608 max_ulp=1909662226 | 0.056 | 5.0 | 1.109 | tg_static=8576 bound_buffer_mb=385.3 rss_peak_mb=3433.0 footprint_mb=224.7 gpu_alloc_mb=3374.6 load=4.07 |
| 28e53828 | [971, 1, 8, 256] | qk_cvt | 3 | 914.375 [0.40] (913.208-920.125) | differing=1985774/1988608 max_ulp=1911280476 | 0.056 | 5.0 | 1.113 | tg_static=8576 bound_buffer_mb=385.3 rss_peak_mb=3433.0 footprint_mb=224.7 gpu_alloc_mb=3374.6 load=4.07 |
