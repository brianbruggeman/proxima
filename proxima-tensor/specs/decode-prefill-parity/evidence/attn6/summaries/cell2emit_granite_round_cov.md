3 files; value `round_cov_pct` from `group` lines; median [CoV%] (min-max) over runs
| group | extents | arm | n | round_cov_pct median [CoV%] (min-max) | bit compare vs base (first file) | cpu ms per replay | cpu % of wall | wall ms per replay | static tg bytes, bound buffer MB, rss peak MB, footprint MB, Metal MB, load |
|---|---|---|---|---|---|---|---|---|---|
| e3603c69 | [1000, 8, 2, 64] | base | 3 | 19.200 [5.57] (18.340-20.481) | differing=0/1024000 max_ulp=0 | 0.058 | 4.2 | 1.397 | tg_static=9600 bound_buffer_mb=131.8 rss_peak_mb=1618.4 footprint_mb=179.7 gpu_alloc_mb=1515.3 load=4.30 |
| e3603c69 | [1000, 8, 2, 64] | both_cvt | 3 | 1.759 [11.07] (1.720-2.094) | differing=1023906/1024000 max_ulp=1998121882 | 0.056 | 4.2 | 1.330 | tg_static=9600 bound_buffer_mb=131.8 rss_peak_mb=1618.4 footprint_mb=179.7 gpu_alloc_mb=1515.3 load=4.30 |
| e3603c69 | [1000, 8, 2, 64] | emit_half | 3 | 1.836 [15.56] (1.563-2.137) | differing=1023906/1024000 max_ulp=1998121882 | 0.056 | 4.2 | 1.326 | tg_static=9600 bound_buffer_mb=131.8 rss_peak_mb=1618.4 footprint_mb=179.7 gpu_alloc_mb=1515.3 load=4.30 |
| e3603c69 | [1000, 8, 2, 64] | qk_cvt | 3 | 1.458 [25.03] (1.368-2.124) | differing=1016688/1024000 max_ulp=1997885750 | 0.056 | 4.2 | 1.325 | tg_static=9600 bound_buffer_mb=131.8 rss_peak_mb=1618.4 footprint_mb=179.7 gpu_alloc_mb=1515.3 load=4.30 |
