3 files; value `single_us` from `group` lines; median [CoV%] (min-max) over runs
| group | extents | arm | n | single_us median [CoV%] (min-max) | bit compare vs base (first file) | cpu ms per replay | cpu % of wall | wall ms per replay | static tg bytes, bound buffer MB, rss peak MB, footprint MB, Metal MB, load |
|---|---|---|---|---|---|---|---|---|---|
| e3603c69 | [1000, 8, 2, 64] | base | 3 | 1131.583 [0.60] (1125.292-1138.792) | differing=0/1024000 max_ulp=0 | 0.057 | 4.2 | 1.378 | tg_static=9600 bound_buffer_mb=131.8 rss_peak_mb=1622.4 footprint_mb=184.3 gpu_alloc_mb=1515.3 load=3.55 |
| e3603c69 | [1000, 8, 2, 64] | both | 3 | 1036.792 [0.39] (1035.125-1042.875) | differing=0/1024000 max_ulp=0 | 0.057 | 4.5 | 1.266 | tg_static=9600 bound_buffer_mb=131.8 rss_peak_mb=1622.4 footprint_mb=184.3 gpu_alloc_mb=1515.3 load=3.55 |
| e3603c69 | [1000, 8, 2, 64] | kpb64 | 3 | 1123.292 [1.18] (1107.750-1134.042) | differing=0/1024000 max_ulp=0 | 0.056 | 4.3 | 1.302 | tg_static=9600 bound_buffer_mb=131.8 rss_peak_mb=1622.4 footprint_mb=184.3 gpu_alloc_mb=1515.3 load=3.55 |
| e3603c69 | [1000, 8, 2, 64] | msg4 | 3 | 1111.333 [0.29] (1111.250-1116.875) | differing=533198/1024000 max_ulp=80089 | 0.059 | 4.4 | 1.340 | tg_static=13696 bound_buffer_mb=131.8 rss_peak_mb=1622.4 footprint_mb=184.3 gpu_alloc_mb=1515.3 load=3.55 |
