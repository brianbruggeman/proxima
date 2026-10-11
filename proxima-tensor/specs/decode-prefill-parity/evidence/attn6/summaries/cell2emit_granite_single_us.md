3 files; value `single_us` from `group` lines; median [CoV%] (min-max) over runs
| group | extents | arm | n | single_us median [CoV%] (min-max) | bit compare vs base (first file) | cpu ms per replay | cpu % of wall | wall ms per replay | static tg bytes, bound buffer MB, rss peak MB, footprint MB, Metal MB, load |
|---|---|---|---|---|---|---|---|---|---|
| e3603c69 | [1000, 8, 2, 64] | base | 3 | 1132.000 [0.45] (1129.083-1138.958) | differing=0/1024000 max_ulp=0 | 0.058 | 4.2 | 1.397 | tg_static=9600 bound_buffer_mb=131.8 rss_peak_mb=1618.4 footprint_mb=179.7 gpu_alloc_mb=1515.3 load=4.30 |
| e3603c69 | [1000, 8, 2, 64] | both_cvt | 3 | 1111.500 [0.30] (1106.333-1112.417) | differing=1023906/1024000 max_ulp=1998121882 | 0.056 | 4.2 | 1.330 | tg_static=9600 bound_buffer_mb=131.8 rss_peak_mb=1618.4 footprint_mb=179.7 gpu_alloc_mb=1515.3 load=4.30 |
| e3603c69 | [1000, 8, 2, 64] | emit_half | 3 | 1108.792 [0.16] (1107.625-1111.125) | differing=1023906/1024000 max_ulp=1998121882 | 0.056 | 4.2 | 1.326 | tg_static=9600 bound_buffer_mb=131.8 rss_peak_mb=1618.4 footprint_mb=179.7 gpu_alloc_mb=1515.3 load=4.30 |
| e3603c69 | [1000, 8, 2, 64] | qk_cvt | 3 | 1104.500 [0.75] (1093.750-1110.125) | differing=1016688/1024000 max_ulp=1997885750 | 0.056 | 4.2 | 1.325 | tg_static=9600 bound_buffer_mb=131.8 rss_peak_mb=1618.4 footprint_mb=179.7 gpu_alloc_mb=1515.3 load=4.30 |
