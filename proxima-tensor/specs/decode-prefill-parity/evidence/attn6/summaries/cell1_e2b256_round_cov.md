3 files; value `round_cov_pct` from `group` lines; median [CoV%] (min-max) over runs
| group | extents | arm | n | round_cov_pct median [CoV%] (min-max) | bit compare vs base (first file) | cpu ms per replay | cpu % of wall | wall ms per replay | static tg bytes, bound buffer MB, rss peak MB, footprint MB, Metal MB, load |
|---|---|---|---|---|---|---|---|---|---|
| 28e53828 | [971, 1, 8, 256] | base | 3 | 1.087 [19.10] (0.775-1.118) | differing=0/1988608 max_ulp=0 | 0.059 | 4.5 | 1.330 | tg_static=8576 bound_buffer_mb=385.3 rss_peak_mb=3425.4 footprint_mb=226.3 gpu_alloc_mb=3374.6 load=3.49 |
| 28e53828 | [971, 1, 8, 256] | both | 3 | 0.425 [11.76] (0.367-0.465) | differing=1045269/1988608 max_ulp=996093 | 0.054 | 4.7 | 1.133 | tg_static=4480 bound_buffer_mb=385.3 rss_peak_mb=3425.4 footprint_mb=226.3 gpu_alloc_mb=3374.6 load=3.49 |
| 28e53828 | [971, 1, 8, 256] | kpb64 | 3 | 0.382 [9.34] (0.337-0.406) | differing=1045269/1988608 max_ulp=996093 | 0.052 | 4.6 | 1.133 | tg_static=4480 bound_buffer_mb=385.3 rss_peak_mb=3425.4 footprint_mb=226.3 gpu_alloc_mb=3374.6 load=3.49 |
| 28e53828 | [971, 1, 8, 256] | msg4 | 3 | 0.723 [11.40] (0.620-0.779) | differing=0/1988608 max_ulp=0 | 0.055 | 4.9 | 1.133 | tg_static=8576 bound_buffer_mb=385.3 rss_peak_mb=3425.4 footprint_mb=226.3 gpu_alloc_mb=3374.6 load=3.49 |
