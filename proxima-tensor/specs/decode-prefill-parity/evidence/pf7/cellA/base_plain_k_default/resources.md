
### resources per process (telemetry `run=done` records and `/usr/bin/time -l` of the same process)

| process | gen 1 ttft ms | gens 2-5 ttft ms median (min-max) | gens 2-5 wall ms median | gens 2-5 cpu % median | steady rss MB | steady footprint MB | gpu peak MB | time -l max rss MB | time -l peak footprint MB | text hashes (distinct) |
|---|---|---|---|---|---|---|---|---|---|---|
| /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/pf7/raw/cellA/base_plain_k_default/p1/stderr.log | 346 | 216 (215-217) | 370.4 | 18.4 | 1748.8 | 221.9 | 1625.0 | 1854.4 | 325.4 | aff93fd561a93eff |
| /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/pf7/raw/cellA/base_plain_k_default/p2/stderr.log | 340 | 216 (216-217) | 368.9 | 17.9 | 1775.9 | 219.7 | 1625.1 | 1881.6 | 325.0 | aff93fd561a93eff |
| /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/pf7/raw/cellA/base_plain_k_default/p3/stderr.log | 342 | 217 (216-217) | 372.5 | 18.7 | 1756.6 | 217.6 | 1625.0 | 1859.5 | 322.3 | aff93fd561a93eff |

warm ttft (gens 2-5) pooled over processes: median 216.0 ms, CoV of the per-process medians 0.27%
