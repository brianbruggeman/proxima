3 files; value `single_us` from `group` lines; median [CoV%] (min-max) over runs
| group | extents | arm | n | single_us median [CoV%] (min-max) | bit compare vs base (first file) | cpu ms per replay | cpu % of wall | wall ms per replay | static tg bytes, bound buffer MB, rss peak MB, footprint MB, Metal MB, load |
|---|---|---|---|---|---|---|---|---|---|
| e3603c69 | [1000, 8, 2, 64] | base | 3 | 1138.125 [0.10] (1136.667-1138.958) | differing=0/1024000 max_ulp=0 | 0.057 | 4.1 | 1.395 | tg_static=9600 bound_buffer_mb=131.8 rss_peak_mb=1618.5 footprint_mb=183.2 gpu_alloc_mb=1515.3 load=3.22 |
| e3603c69 | [1000, 8, 2, 64] | mma_f32 | 3 | 2777.250 [1.15] (2775.125-2831.875) | differing=1024000/1024000 max_ulp=2523666303 | 0.083 | 2.8 | 2.991 | tg_static=0 bound_buffer_mb=131.8 rss_peak_mb=1618.5 footprint_mb=183.2 gpu_alloc_mb=1515.3 load=3.22 |
| e3603c69 | [1000, 8, 2, 64] | mma_h_f32 | 3 | 2592.958 [0.14] (2592.792-2599.000) | differing=1024000/1024000 max_ulp=2523666303 | 0.074 | 2.7 | 2.788 | tg_static=0 bound_buffer_mb=131.8 rss_peak_mb=1618.5 footprint_mb=183.2 gpu_alloc_mb=1515.3 load=3.22 |
| e3603c69 | [1000, 8, 2, 64] | mma_h_h | 3 | 2594.375 [2.85] (2536.500-2684.000) | differing=1024000/1024000 max_ulp=3227282183 | 0.075 | 2.5 | 2.963 | tg_static=0 bound_buffer_mb=131.8 rss_peak_mb=1618.5 footprint_mb=183.2 gpu_alloc_mb=1515.3 load=3.22 |
