telemetry /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/combine4/census/census_granite_prefill/decode_telemetry.log: 2 step records, 383 dispatch rows

| seq | step | new tokens | wall ms | evaluate ms | prepare ms | pre_encode ms | pipeline compile ms (misses) | encode ms | gpu_exec ms | chunks | lead idle ms | inter-chunk idle ms | chunk busy sum ms | last gpu end ms | residual after last gpu end ms (evaluate - prepare - pre_encode - last gpu end) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 1000 | 353.933 | 343.531 | 38.550 | 36.598 | 25.568 (33) | 2.646 | 262.004 | 1 | 61.786 | -0.000 | 205.793 | 267.579 | 0.804 |
| 1 | 1 | 1 | 58.123 | 57.972 | 38.339 | 10.074 | 5.025 (10) | 0.438 | 5.178 | 8 | 2.877 | 0.193 | 5.780 | 8.850 | 0.710 |

no decode steps >= 2 in this log (N=0)

detail seq 0 (step 0): wall 353.933 ms, evaluate 343.531 ms, chunk busy sum 205.793 ms, last gpu end 267.579 ms

| chunk | ops | dispatches (census index) | encode start-end ms | gpu start-end ms | gpu busy ms | idle before ms | dispatch before the idle | dispatch after the idle |
|---|---|---|---|---|---|---|---|---|
| 1 | 0-382 | 383 (0-382) | 0.000-5.771 | 61.786-267.579 | 205.793 | 61.786 | none (host: prepare, plan, encode) | #0 constant omega_constant_r0_v41400000 [] |

largest host gaps in this step, ranked:

1. 61.786 ms before chunk 1: after none (host: prepare, plan, encode); before #0 constant omega_constant_r0_v41400000 []
2. 0.804 ms residual: evaluate minus prepare, pre_encode and the last chunk's gpu end (readback, kv append, host)

host phases, ms from the earliest phase start (step_phase has chunk 0):

| chunk | phase | start ms | end ms | duration ms |
|---|---|---|---|---|
| 0 | prepare | 0.000 | 38.550 | 38.550 |
| 0 | pre_encode | 39.011 | 75.608 | 36.597 |
| 1 | encode | 75.608 | 81.379 | 5.771 |
| 1 | commit | 81.384 | 81.416 | 0.032 |
| 1 | scheduled | 136.993 | 136.993 | 0.000 |
| 1 | gpu | 137.394 | 343.187 | 205.793 |

commit-to-GPU-start split per chunk (commit end to scheduled callback, scheduled callback to GPU start):

| chunk | commit end to scheduled ms | scheduled to gpu start ms |
|---|---|---|
| 1 | 55.576 | 0.401 |
