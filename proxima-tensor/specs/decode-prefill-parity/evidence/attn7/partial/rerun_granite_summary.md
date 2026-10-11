3 files; value `single_us` from `group` lines; median [CoV%] (min-max) over runs
| group | extents | arm | n | single_us median [CoV%] (min-max) | bit compare vs base (first file) | cpu ms per replay | cpu % of wall | wall ms per replay | static tg bytes, bound buffer MB, rss peak MB, footprint MB, Metal MB, load |
|---|---|---|---|---|---|---|---|---|---|
| 2ff6fe0d | [1, 8, 2, 64] | base | 3 | 49.125 [0.64] (48.750-49.375) | differing=0/33792 max_ulp=0 max_abs=0e0 | 0.064 | 21.7 | 0.294 | tg_static=1056 bound_buffer_mb=5.2 rss_peak_mb=1641.8 footprint_mb=150.3 gpu_alloc_mb=1469.1 load=8.58 |
| 2ff6fe0d | [1, 8, 2, 64] | ctrl | 3 | 49.125 [2.16] (47.375-49.250) | differing=0/33792 max_ulp=0 max_abs=0e0 | 0.060 | 20.4 | 0.297 | tg_static=1056 bound_buffer_mb=5.2 rss_peak_mb=1641.8 footprint_mb=150.3 gpu_alloc_mb=1469.1 load=8.58 |
| 2ff6fe0d | [1, 8, 2, 64] | div32 | 3 | 32.292 [0.65] (32.083-32.500) | differing=0/33792 max_ulp=0 max_abs=0e0 | 0.060 | 23.8 | 0.253 | tg_static=1056 bound_buffer_mb=5.2 rss_peak_mb=1641.8 footprint_mb=150.3 gpu_alloc_mb=1469.1 load=8.58 |
| 2ff6fe0d | [1, 8, 2, 64] | i32_all | 3 | 25.125 [0.57] (25.125-25.375) | differing=0/33792 max_ulp=0 max_abs=0e0 | 0.060 | 24.5 | 0.245 | tg_static=1056 bound_buffer_mb=5.2 rss_peak_mb=1641.8 footprint_mb=150.3 gpu_alloc_mb=1469.1 load=8.58 |
