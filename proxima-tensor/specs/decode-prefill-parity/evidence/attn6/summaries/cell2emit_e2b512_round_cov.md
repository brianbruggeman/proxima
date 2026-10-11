3 files; value `round_cov_pct` from `group` lines; median [CoV%] (min-max) over runs
| group | extents | arm | n | round_cov_pct median [CoV%] (min-max) | bit compare vs base (first file) | cpu ms per replay | cpu % of wall | wall ms per replay | static tg bytes, bound buffer MB, rss peak MB, footprint MB, Metal MB, load |
|---|---|---|---|---|---|---|---|---|---|
| 96585e0b | [971, 1, 8, 512] | base | 3 | 0.604 [158.24] (0.585-19.455) | differing=0/3977216 max_ulp=0 | 0.075 | 2.6 | 2.898 | tg_static=8576 bound_buffer_mb=355.3 rss_peak_mb=3423.1 footprint_mb=226.6 gpu_alloc_mb=3374.6 load=3.64 |
| 96585e0b | [971, 1, 8, 512] | both_cvt | 3 | 0.298 [15.00] (0.233-0.312) | differing=3975986/3977216 max_ulp=1896514884 | 0.076 | 2.5 | 2.988 | tg_static=8576 bound_buffer_mb=355.3 rss_peak_mb=3423.1 footprint_mb=226.6 gpu_alloc_mb=3374.6 load=3.64 |
| 96585e0b | [971, 1, 8, 512] | emit_half | 3 | 0.335 [15.32] (0.262-0.354) | differing=3975986/3977216 max_ulp=1896514884 | 0.068 | 2.3 | 3.000 | tg_static=8576 bound_buffer_mb=355.3 rss_peak_mb=3423.1 footprint_mb=226.6 gpu_alloc_mb=3374.6 load=3.64 |
| 96585e0b | [971, 1, 8, 512] | qk_cvt | 3 | 0.308 [15.57] (0.236-0.318) | differing=3971229/3977216 max_ulp=1889379524 | 0.074 | 2.4 | 3.001 | tg_static=8576 bound_buffer_mb=355.3 rss_peak_mb=3423.1 footprint_mb=226.6 gpu_alloc_mb=3374.6 load=3.64 |
