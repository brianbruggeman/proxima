3 files; value `single_us` from `group` lines; median [CoV%] (min-max) over runs
| group | extents | arm | n | single_us median [CoV%] (min-max) | bit compare vs base (first file) | cpu ms per replay | cpu % of wall | wall ms per replay | static tg bytes, bound buffer MB, rss peak MB, footprint MB, Metal MB, load |
|---|---|---|---|---|---|---|---|---|---|
| e3603c69 | [1000, 8, 2, 64] | base | 3 | 1137.125 [0.41] (1130.292-1139.125) | differing=0/1024000 max_ulp=0 | 0.058 | 4.2 | 1.391 | tg_static=9600 bound_buffer_mb=131.8 rss_peak_mb=1616.5 footprint_mb=191.5 gpu_alloc_mb=1515.3 load=3.59 |
| e3603c69 | [1000, 8, 2, 64] | c32_f32 | 3 | 1693.500 [0.04] (1692.417-1693.625) | differing=1024000/1024000 max_ulp=2523666303 | 0.059 | 3.1 | 1.921 | tg_static=0 bound_buffer_mb=131.8 rss_peak_mb=1616.5 footprint_mb=191.5 gpu_alloc_mb=1515.3 load=3.59 |
| e3603c69 | [1000, 8, 2, 64] | c32_hf | 3 | 1694.250 [0.04] (1693.083-1694.333) | differing=1024000/1024000 max_ulp=2523666303 | 0.067 | 3.5 | 1.925 | tg_static=0 bound_buffer_mb=131.8 rss_peak_mb=1616.5 footprint_mb=191.5 gpu_alloc_mb=1515.3 load=3.59 |
| e3603c69 | [1000, 8, 2, 64] | c32_hh | 3 | 1786.167 [0.05] (1784.750-1786.250) | differing=1024000/1024000 max_ulp=3223982185 | 0.060 | 3.0 | 2.009 | tg_static=0 bound_buffer_mb=131.8 rss_peak_mb=1616.5 footprint_mb=191.5 gpu_alloc_mb=1515.3 load=3.59 |
