3 files; value `round_cov_pct` from `group` lines; median [CoV%] (min-max) over runs
| group | extents | arm | n | round_cov_pct median [CoV%] (min-max) | bit compare vs base (first file) | cpu ms per replay | cpu % of wall | wall ms per replay | static tg bytes, bound buffer MB, rss peak MB, footprint MB, Metal MB, load |
|---|---|---|---|---|---|---|---|---|---|
| e3603c69 | [1000, 8, 2, 64] | base | 3 | 1.702 [3.51] (1.699-1.806) | differing=0/1024000 max_ulp=0 | 0.057 | 4.2 | 1.378 | tg_static=9600 bound_buffer_mb=131.8 rss_peak_mb=1622.4 footprint_mb=184.3 gpu_alloc_mb=1515.3 load=3.55 |
| e3603c69 | [1000, 8, 2, 64] | both | 3 | 1.670 [30.75] (1.375-2.467) | differing=0/1024000 max_ulp=0 | 0.057 | 4.5 | 1.266 | tg_static=9600 bound_buffer_mb=131.8 rss_peak_mb=1622.4 footprint_mb=184.3 gpu_alloc_mb=1515.3 load=3.55 |
| e3603c69 | [1000, 8, 2, 64] | kpb64 | 3 | 1.919 [12.28] (1.835-2.300) | differing=0/1024000 max_ulp=0 | 0.056 | 4.3 | 1.302 | tg_static=9600 bound_buffer_mb=131.8 rss_peak_mb=1622.4 footprint_mb=184.3 gpu_alloc_mb=1515.3 load=3.55 |
| e3603c69 | [1000, 8, 2, 64] | msg4 | 3 | 2.666 [10.05] (2.558-3.085) | differing=533198/1024000 max_ulp=80089 | 0.059 | 4.4 | 1.340 | tg_static=13696 bound_buffer_mb=131.8 rss_peak_mb=1622.4 footprint_mb=184.3 gpu_alloc_mb=1515.3 load=3.55 |
