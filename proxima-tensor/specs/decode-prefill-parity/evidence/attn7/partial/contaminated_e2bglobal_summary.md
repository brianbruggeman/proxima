3 files; value `single_us` from `group` lines; median [CoV%] (min-max) over runs
| group | extents | arm | n | single_us median [CoV%] (min-max) | bit compare vs base (first file) | cpu ms per replay | cpu % of wall | wall ms per replay | static tg bytes, bound buffer MB, rss peak MB, footprint MB, Metal MB, load |
|---|---|---|---|---|---|---|---|---|---|
| 7134997e | [1, 1, 8, 512] | base | 3 | 42.292 [0.10] (42.250-42.333) | differing=0/131584 max_ulp=0 max_abs=0e0 | 0.067 | 20.2 | 0.332 | tg_static=8224 bound_buffer_mb=7.1 rss_peak_mb=3451.4 footprint_mb=116.3 gpu_alloc_mb=3246.7 load=6.68 |
| 7134997e | [1, 1, 8, 512] | ctrl | 3 | 42.625 [0.66] (42.500-43.042) | differing=0/131584 max_ulp=0 max_abs=0e0 | 0.062 | 19.8 | 0.320 | tg_static=8224 bound_buffer_mb=7.1 rss_peak_mb=3451.4 footprint_mb=116.3 gpu_alloc_mb=3246.7 load=6.68 |
| 7134997e | [1, 1, 8, 512] | div32 | 3 | 35.083 [0.38] (35.042-35.292) | differing=0/131584 max_ulp=0 max_abs=0e0 | 0.062 | 24.4 | 0.253 | tg_static=8224 bound_buffer_mb=7.1 rss_peak_mb=3451.4 footprint_mb=116.3 gpu_alloc_mb=3246.7 load=6.68 |
| 7134997e | [1, 1, 8, 512] | i32_all | 3 | 32.667 [0.38] (32.542-32.792) | differing=0/131584 max_ulp=0 max_abs=0e0 | 0.062 | 25.9 | 0.241 | tg_static=8224 bound_buffer_mb=7.1 rss_peak_mb=3451.4 footprint_mb=116.3 gpu_alloc_mb=3246.7 load=6.68 |
