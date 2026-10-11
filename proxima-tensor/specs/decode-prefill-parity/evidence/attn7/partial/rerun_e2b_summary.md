3 files; value `single_us` from `group` lines; median [CoV%] (min-max) over runs
| group | extents | arm | n | single_us median [CoV%] (min-max) | bit compare vs base (first file) | cpu ms per replay | cpu % of wall | wall ms per replay | static tg bytes, bound buffer MB, rss peak MB, footprint MB, Metal MB, load |
|---|---|---|---|---|---|---|---|---|---|
| a335ade5 | [1, 1, 8, 256] | base | 3 | 25.625 [63.02] (22.125-64.667) | differing=0/66048 max_ulp=0 max_abs=0e0 | 0.062 | 22.0 | 0.280 | tg_static=4128 bound_buffer_mb=3.6 rss_peak_mb=3457.6 footprint_mb=108.7 gpu_alloc_mb=3246.7 load=8.05 |
| a335ade5 | [1, 1, 8, 256] | ctrl | 3 | 22.250 [67.93] (21.833-64.708) | differing=0/66048 max_ulp=0 max_abs=0e0 | 0.056 | 19.3 | 0.291 | tg_static=4128 bound_buffer_mb=3.6 rss_peak_mb=3457.6 footprint_mb=108.7 gpu_alloc_mb=3246.7 load=8.05 |
| a335ade5 | [1, 1, 8, 256] | div32 | 3 | 18.583 [66.65] (18.125-52.792) | differing=0/66048 max_ulp=0 max_abs=0e0 | 0.067 | 27.2 | 0.241 | tg_static=4128 bound_buffer_mb=3.6 rss_peak_mb=3457.6 footprint_mb=108.7 gpu_alloc_mb=3246.7 load=8.05 |
| a335ade5 | [1, 1, 8, 256] | i32_all | 3 | 17.917 [65.52] (17.333-49.792) | differing=0/66048 max_ulp=0 max_abs=0e0 | 0.054 | 22.2 | 0.240 | tg_static=4128 bound_buffer_mb=3.6 rss_peak_mb=3457.6 footprint_mb=108.6 gpu_alloc_mb=3246.7 load=8.05 |
