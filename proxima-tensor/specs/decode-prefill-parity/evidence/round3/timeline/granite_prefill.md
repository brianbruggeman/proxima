telemetry census/census_granite_prefill/decode_telemetry.log: 2 step records, 383 dispatch rows

| seq | step | new tokens | wall ms | evaluate ms | prepare ms | pre_encode ms | pipeline compile ms (misses) | encode ms | gpu_exec ms | chunks | lead idle ms | inter-chunk idle ms | chunk busy sum ms | last gpu end ms | residual after last gpu end ms (evaluate - prepare - pre_encode - last gpu end) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 1000 | 389.990 | 380.257 | 38.743 | 42.259 | 32.573 (33) | 3.242 | 292.188 | 1 | 56.509 | -0.000 | 241.839 | 298.348 | 0.907 |
| 1 | 1 | 1 | 80.567 | 80.390 | 56.410 | 14.131 | 8.022 (10) | 0.860 | 4.483 | 8 | 3.195 | 0.183 | 5.586 | 8.964 | 0.885 |

no decode steps >= 2 in this log (N=0)

detail seq 0 (step 0): wall 389.990 ms, evaluate 380.257 ms, chunk busy sum 241.839 ms, last gpu end 298.348 ms

| chunk | ops | dispatches (census index) | encode start-end ms | gpu start-end ms | gpu busy ms | idle before ms | dispatch before the idle | dispatch after the idle |
|---|---|---|---|---|---|---|---|---|
| 1 | 0-382 | 383 (0-382) | 0.000-6.342 | 56.509-298.348 | 241.839 | 56.509 | none (host: prepare, plan, encode) | #0 constant omega_constant_r0_v41400000 [] |

largest host gaps in this step, ranked:

1. 56.509 ms before chunk 1: after none (host: prepare, plan, encode); before #0 constant omega_constant_r0_v41400000 []
2. 0.907 ms residual: evaluate minus prepare, pre_encode and the last chunk's gpu end (readback, kv append, host)

host phases, ms from the earliest phase start (step_phase has chunk 0):

| chunk | phase | start ms | end ms | duration ms |
|---|---|---|---|---|
| 0 | prepare | 0.000 | 38.743 | 38.743 |
| 0 | pre_encode | 39.271 | 81.530 | 42.259 |
| 1 | encode | 81.530 | 87.871 | 6.342 |
| 1 | commit | 87.873 | 87.906 | 0.033 |
| 1 | scheduled | 137.796 | 137.796 | 0.000 |
| 1 | gpu | 138.038 | 379.877 | 241.839 |

commit-to-GPU-start split per chunk (commit end to scheduled callback, scheduled callback to GPU start):

| chunk | commit end to scheduled ms | scheduled to gpu start ms |
|---|---|---|
| 1 | 49.890 | 0.242 |
