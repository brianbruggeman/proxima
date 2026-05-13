**par.rs now covers rayon's primary data-parallel surface. Supported shapes: par_reduce, par_each, par_map_collect (sync + async), par_filter, par_sort / par_sort_by / par_sort_unstable, par_chunks_mut, scope, install, par_bridge. Not supported: Rayon's work-stealing deque (proxima uses a shared injector without per-worker deques), rayon::join (use futures::join! instead), rayon's ParallelIterator adaptor chain (filter_map, flat_map, partition, etc. beyond the primitives above).**

Platform: M1. Run via `scripts/bench-vs-rayon.sh`.

Unit: mean wall time from criterion `estimates.json` (lower is better).
CoV: std_dev / mean as reported by criterion.

## bg_pool_tiny_jobs

| arm | M1 mean | M1 CoV | Linux mean | Linux CoV | unit | notes |
|-----|---------|--------|------------|-----------|------|-------|
| proxima_typed | pending | pending | pending | pending | ns | |
| proxima_dyn | pending | pending | pending | pending | ns | |
| rayon | pending | pending | pending | pending | ns | rayon dyn dispatch |
| rayon_via_bgpool | pending | pending | pending | pending | ns | rayon pool inside bgpool wrapper |
| proxima_rayon_backed | pending | pending | pending | pending | ns | proxima typed API on rayon pool |
| tokio_spawn_blocking | pending | pending | pending | pending | ns | control |

## bg_pool_cpu_imbalanced

| arm | M1 mean | M1 CoV | Linux mean | Linux CoV | unit | notes |
|-----|---------|--------|------------|-----------|------|-------|
| proxima_typed | pending | pending | pending | pending | ns | |
| proxima_dyn | pending | pending | pending | pending | ns | |
| rayon | pending | pending | pending | pending | ns | rayon dyn dispatch |
| rayon_via_bgpool | pending | pending | pending | pending | ns | rayon pool inside bgpool wrapper |
| proxima_rayon_backed | pending | pending | pending | pending | ns | proxima typed API on rayon pool |
| tokio_spawn_blocking | pending | pending | pending | pending | ns | control |

## bg_pool_fork_join

| arm | M1 mean | M1 CoV | Linux mean | Linux CoV | unit | notes |
|-----|---------|--------|------------|-----------|------|-------|
| proxima_typed | pending | pending | pending | pending | ns | |
| proxima_dyn | pending | pending | pending | pending | ns | |
| proxima_par_reduce_512 | pending | pending | pending | pending | ns | threshold=512 |
| proxima_par_reduce_2048 | pending | pending | pending | pending | ns | threshold=2048 |
| proxima_par_reduce_8192 | pending | pending | pending | pending | ns | threshold=8192 |
| proxima_par_iter_map_sum | pending | pending | pending | pending | ns | |
| proxima_par_iter_map_collect | pending | pending | pending | pending | ns | |
| proxima_par_iter_map_async_sum | pending | pending | pending | pending | ns | async map variant |
| proxima_par_stream_then_16 | pending | pending | pending | pending | ns | buffered concurrency, cap=16 |
| proxima_par_stream_then_64 | pending | pending | pending | pending | ns | buffered concurrency, cap=64 |
| proxima_par_stream_then_ordered_64 | pending | pending | pending | pending | ns | ordered reorder buf, cap=64 |
| proxima_par_filter | pending | pending | pending | pending | ns | recursive split + filter + concat |
| rayon_par_iter_filter | pending | pending | pending | pending | ns | rayon par_iter().filter().collect() |
| proxima_par_sort_by | pending | pending | pending | pending | ns | parallel merge sort |
| rayon_par_sort_by | pending | pending | pending | pending | ns | rayon par_sort_by |
| proxima_par_chunks_mut | pending | pending | pending | pending | ns | parallel chunk mutation |
| rayon_par_chunks_mut | pending | pending | pending | pending | ns | rayon par_chunks_mut |
| proxima_par_bridge | pending | pending | pending | pending | ns | mutex-iterator fan-out |
| rayon_par_bridge | pending | pending | pending | pending | ns | rayon par_bridge |
| rayon | pending | pending | pending | pending | ns | rayon dyn dispatch |
| rayon_par_iter | pending | pending | pending | pending | ns | rayon recursive split + work-steal |
| rayon_via_bgpool | pending | pending | pending | pending | ns | rayon pool inside bgpool wrapper |
| proxima_rayon_backed | pending | pending | pending | pending | ns | proxima typed API on rayon pool |
| tokio_spawn_blocking | pending | pending | pending | pending | ns | control |
