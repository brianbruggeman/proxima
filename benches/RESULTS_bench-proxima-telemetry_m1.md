# proxima-telemetry bench results — m1

Per-component micro-bench coverage map for proxima-telemetry primitives.
Each component is benched in isolation behind its own feature flag.
No comparison axis — this answers "is each primitive fast enough on its own?"
Linux column is populated by CI; M1 column is populated by local developer runs.
Run `scripts/bench-proxima-telemetry.sh` to refresh.

---

## per-component

| component | metric | M1 | Linux | CoV | bench file |
|-----------|--------|-----|-------|-----|------------|
| c1-ring | mean | pending | pending | n/a | bench_c1_ring |
| c2-id | mean | pending | pending | n/a | bench_c2_id |
| c3-level | mean | pending | pending | n/a | bench_c3_level |
| c4-attr | mean | pending | pending | n/a | bench_c4_attr |
| c5-trace | mean | pending | pending | n/a | bench_c5_trace |
| c6-metric-basic | mean | PENDING | PENDING | n/a | bench_c6_counter |
| c7-metric-histogram | mean | PENDING | PENDING | n/a | bench_c7_histogram |
| c8-log | mean | PENDING | PENDING | n/a | bench_c8_log |
| c9-recorder | mean | PENDING | PENDING | n/a | bench_c9_recorder |
| c10-out-otlp-http | mean | PENDING | PENDING | n/a | bench_c10_otlp_http |
| c12-out-native | mean | PENDING | PENDING | n/a | bench_c12_native |

---
