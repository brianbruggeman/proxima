telemetry census/census_granite_prefill_segments/decode_telemetry.log: 2 step records, 383 dispatch rows

| seq | step | new tokens | wall ms | evaluate ms | prepare ms | pre_encode ms | pipeline compile ms (misses) | encode ms | gpu_exec ms | chunks | lead idle ms | inter-chunk idle ms | chunk busy sum ms | last gpu end ms | residual after last gpu end ms (evaluate - prepare - pre_encode - last gpu end) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 1000 | 387.692 | 377.100 | 43.387 | 34.634 | 28.872 (33) | 2.760 | 292.053 | 1 | 56.712 | -0.000 | 241.453 | 298.165 | 0.914 |
| 1 | 1 | 1 | 61.582 | 61.507 | 41.838 | 10.129 | 5.016 (10) | 0.799 | 4.745 | 8 | 2.932 | 0.202 | 5.687 | 8.821 | 0.718 |

no decode steps >= 2 in this log (N=0)

detail seq 0 (step 0): wall 387.692 ms, evaluate 377.100 ms, chunk busy sum 241.453 ms, last gpu end 298.165 ms

| chunk | ops | dispatches (census index) | encode start-end ms | gpu start-end ms | gpu busy ms | idle before ms | dispatch before the idle | dispatch after the idle |
|---|---|---|---|---|---|---|---|---|
| 1 | 0-382 | 383 (0-382) | 0.000-6.282 | 56.712-298.165 | 241.453 | 56.712 | none (host: prepare, plan, encode) | #0 constant omega_constant_r0_v41400000 [] |

largest host gaps in this step, ranked:

1. 56.712 ms before chunk 1: after none (host: prepare, plan, encode); before #0 constant omega_constant_r0_v41400000 []
2. 0.914 ms residual: evaluate minus prepare, pre_encode and the last chunk's gpu end (readback, kv append, host)

host phases, ms from the earliest phase start (step_phase has chunk 0):

| chunk | phase | start ms | end ms | duration ms |
|---|---|---|---|---|
| 0 | prepare | 0.000 | 43.387 | 43.387 |
| 0 | pre_encode | 43.968 | 78.601 | 34.633 |
| 1 | encode | 78.601 | 84.884 | 6.282 |
| 1 | commit | 84.886 | 84.915 | 0.029 |
| 1 | scheduled | 135.272 | 135.272 | 0.000 |
| 1 | gpu | 135.313 | 376.766 | 241.453 |

commit-to-GPU-start split per chunk (commit end to scheduled callback, scheduled callback to GPU start):

| chunk | commit end to scheduled ms | scheduled to gpu start ms |
|---|---|---|
| 1 | 50.357 | 0.041 |
