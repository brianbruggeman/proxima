3 files; value `single_us` from `group` lines; median [CoV%] (min-max) over runs
| group | extents | arm | n | single_us median [CoV%] (min-max) | bit compare vs base (first file) | cpu ms per replay | cpu % of wall | wall ms per replay | static tg bytes, bound buffer MB, rss peak MB, footprint MB, Metal MB, load |
|---|---|---|---|---|---|---|---|---|---|
| a335ade5 | [1, 1, 8, 256] | base | 3 | 22.000 [0.33] (22.000-22.125) | differing=0/66048 max_ulp=0 max_abs=0e0 | 0.067 | 24.8 | 0.271 | tg_static=4128 bound_buffer_mb=3.6 rss_peak_mb=3446.2 footprint_mb=116.0 gpu_alloc_mb=3246.7 load=6.68 |
| a335ade5 | [1, 1, 8, 256] | ctrl | 3 | 21.958 [0.19] (21.917-22.000) | differing=0/66048 max_ulp=0 max_abs=0e0 | 0.060 | 22.8 | 0.264 | tg_static=4128 bound_buffer_mb=3.6 rss_peak_mb=3446.2 footprint_mb=116.0 gpu_alloc_mb=3246.7 load=6.68 |
| a335ade5 | [1, 1, 8, 256] | div32 | 3 | 18.083 [1.13] (18.042-18.417) | differing=0/66048 max_ulp=0 max_abs=0e0 | 0.063 | 26.7 | 0.227 | tg_static=4128 bound_buffer_mb=3.6 rss_peak_mb=3446.2 footprint_mb=116.0 gpu_alloc_mb=3246.7 load=6.68 |
| a335ade5 | [1, 1, 8, 256] | i32_all | 3 | 17.417 [1.27] (17.083-17.500) | differing=0/66048 max_ulp=0 max_abs=0e0 | 0.060 | 26.3 | 0.228 | tg_static=4128 bound_buffer_mb=3.6 rss_peak_mb=3446.2 footprint_mb=116.0 gpu_alloc_mb=3246.7 load=6.68 |
