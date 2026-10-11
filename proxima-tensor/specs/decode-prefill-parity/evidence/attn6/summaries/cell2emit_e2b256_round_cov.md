3 files; value `round_cov_pct` from `group` lines; median [CoV%] (min-max) over runs
| group | extents | arm | n | round_cov_pct median [CoV%] (min-max) | bit compare vs base (first file) | cpu ms per replay | cpu % of wall | wall ms per replay | static tg bytes, bound buffer MB, rss peak MB, footprint MB, Metal MB, load |
|---|---|---|---|---|---|---|---|---|---|
| 28e53828 | [971, 1, 8, 256] | base | 3 | 32.643 [39.35] (14.537-32.748) | differing=0/1988608 max_ulp=0 | 0.059 | 4.8 | 1.222 | tg_static=8576 bound_buffer_mb=385.3 rss_peak_mb=3433.0 footprint_mb=224.7 gpu_alloc_mb=3374.6 load=4.07 |
| 28e53828 | [971, 1, 8, 256] | both_cvt | 3 | 0.593 [8.23] (0.552-0.650) | differing=1988238/1988608 max_ulp=1909662226 | 0.055 | 5.0 | 1.109 | tg_static=8576 bound_buffer_mb=385.3 rss_peak_mb=3433.0 footprint_mb=224.7 gpu_alloc_mb=3374.6 load=4.07 |
| 28e53828 | [971, 1, 8, 256] | emit_half | 3 | 0.646 [15.46] (0.595-0.797) | differing=1988238/1988608 max_ulp=1909662226 | 0.056 | 5.0 | 1.109 | tg_static=8576 bound_buffer_mb=385.3 rss_peak_mb=3433.0 footprint_mb=224.7 gpu_alloc_mb=3374.6 load=4.07 |
| 28e53828 | [971, 1, 8, 256] | qk_cvt | 3 | 0.608 [11.39] (0.498-0.614) | differing=1985774/1988608 max_ulp=1911280476 | 0.056 | 5.0 | 1.113 | tg_static=8576 bound_buffer_mb=385.3 rss_peak_mb=3433.0 footprint_mb=224.7 gpu_alloc_mb=3374.6 load=4.07 |
