3 files; value `single_us` from `group` lines; median [CoV%] (min-max) over runs
| group | extents | arm | n | single_us median [CoV%] (min-max) | bit compare vs base (first file) | cpu ms per replay | cpu % of wall | wall ms per replay | static tg bytes, bound buffer MB, rss peak MB, footprint MB, Metal MB, load |
|---|---|---|---|---|---|---|---|---|---|
| 96585e0b | [971, 1, 8, 512] | base | 3 | 2622.375 [0.04] (2620.875-2622.625) | differing=0/3977216 max_ulp=0 | 0.075 | 2.6 | 2.898 | tg_static=8576 bound_buffer_mb=355.3 rss_peak_mb=3423.1 footprint_mb=226.6 gpu_alloc_mb=3374.6 load=3.64 |
| 96585e0b | [971, 1, 8, 512] | both_cvt | 3 | 2764.833 [0.07] (2763.000-2767.042) | differing=3975986/3977216 max_ulp=1896514884 | 0.076 | 2.5 | 2.988 | tg_static=8576 bound_buffer_mb=355.3 rss_peak_mb=3423.1 footprint_mb=226.6 gpu_alloc_mb=3374.6 load=3.64 |
| 96585e0b | [971, 1, 8, 512] | emit_half | 3 | 2774.875 [0.11] (2774.625-2780.000) | differing=3975986/3977216 max_ulp=1896514884 | 0.068 | 2.3 | 3.000 | tg_static=8576 bound_buffer_mb=355.3 rss_peak_mb=3423.1 footprint_mb=226.6 gpu_alloc_mb=3374.6 load=3.64 |
| 96585e0b | [971, 1, 8, 512] | qk_cvt | 3 | 2769.000 [0.06] (2767.208-2770.250) | differing=3971229/3977216 max_ulp=1889379524 | 0.074 | 2.4 | 3.001 | tg_static=8576 bound_buffer_mb=355.3 rss_peak_mb=3423.1 footprint_mb=226.6 gpu_alloc_mb=3374.6 load=3.64 |
