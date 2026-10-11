3 files; value `round_cov_pct` from `group` lines; median [CoV%] (min-max) over runs
| group | extents | arm | n | round_cov_pct median [CoV%] (min-max) | bit compare vs base (first file) | cpu ms per replay | cpu % of wall | wall ms per replay | static tg bytes, bound buffer MB, rss peak MB, footprint MB, Metal MB, load |
|---|---|---|---|---|---|---|---|---|---|
| 96585e0b | [971, 1, 8, 512] | base | 3 | 0.501 [14.95] (0.471-0.621) | differing=0/3977216 max_ulp=0 | 0.072 | 2.5 | 2.912 | tg_static=8576 bound_buffer_mb=355.3 rss_peak_mb=3425.2 footprint_mb=230.6 gpu_alloc_mb=3374.6 load=4.00 |
| 96585e0b | [971, 1, 8, 512] | both | 3 | 0.490 [28.45] (0.422-0.717) | differing=2195339/3977216 max_ulp=597454 | 0.068 | 1.8 | 3.719 | tg_static=4480 bound_buffer_mb=355.3 rss_peak_mb=3425.2 footprint_mb=230.6 gpu_alloc_mb=3374.6 load=4.00 |
| 96585e0b | [971, 1, 8, 512] | kpb64 | 3 | 0.751 [18.25] (0.676-0.956) | differing=2195339/3977216 max_ulp=597454 | 0.078 | 2.1 | 3.726 | tg_static=4480 bound_buffer_mb=355.3 rss_peak_mb=3425.2 footprint_mb=230.6 gpu_alloc_mb=3374.6 load=4.00 |
| 96585e0b | [971, 1, 8, 512] | msg4 | 3 | 0.298 [3.06] (0.287-0.305) | differing=0/3977216 max_ulp=0 | 0.071 | 2.5 | 2.869 | tg_static=8576 bound_buffer_mb=355.3 rss_peak_mb=3425.2 footprint_mb=230.6 gpu_alloc_mb=3374.6 load=4.00 |
