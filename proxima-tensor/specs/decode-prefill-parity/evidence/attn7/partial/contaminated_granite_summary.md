3 files; value `single_us` from `group` lines; median [CoV%] (min-max) over runs
| group | extents | arm | n | single_us median [CoV%] (min-max) | bit compare vs base (first file) | cpu ms per replay | cpu % of wall | wall ms per replay | static tg bytes, bound buffer MB, rss peak MB, footprint MB, Metal MB, load |
|---|---|---|---|---|---|---|---|---|---|
| 2ff6fe0d | [1, 8, 2, 64] | base | 3 | 61.833 [12.58] (49.292-61.875) | differing=0/33792 max_ulp=0 max_abs=0e0 | 0.077 | 26.2 | 0.297 | tg_static=1056 bound_buffer_mb=5.2 rss_peak_mb=1657.4 footprint_mb=149.8 gpu_alloc_mb=1469.1 load=7.64 |
| 2ff6fe0d | [1, 8, 2, 64] | ctrl | 3 | 62.250 [14.37] (49.250-65.125) | differing=0/33792 max_ulp=0 max_abs=0e0 | 0.071 | 23.4 | 0.301 | tg_static=1056 bound_buffer_mb=5.2 rss_peak_mb=1657.4 footprint_mb=149.8 gpu_alloc_mb=1469.1 load=7.64 |
| 2ff6fe0d | [1, 8, 2, 64] | div32 | 3 | 42.042 [14.43] (32.375-42.125) | differing=0/33792 max_ulp=0 max_abs=0e0 | 0.071 | 26.0 | 0.275 | tg_static=1056 bound_buffer_mb=5.2 rss_peak_mb=1657.4 footprint_mb=149.8 gpu_alloc_mb=1469.1 load=7.64 |
| 2ff6fe0d | [1, 8, 2, 64] | i32_all | 3 | 32.375 [13.88] (25.250-32.583) | differing=0/33792 max_ulp=0 max_abs=0e0 | 0.068 | 26.1 | 0.254 | tg_static=1056 bound_buffer_mb=5.2 rss_peak_mb=1657.4 footprint_mb=149.8 gpu_alloc_mb=1469.1 load=7.64 |
