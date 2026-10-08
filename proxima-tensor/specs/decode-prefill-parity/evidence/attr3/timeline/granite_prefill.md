telemetry census_granite_prefill/decode_telemetry.log: 2 step records, 383 dispatch rows

| seq | step | new tokens | wall ms | evaluate ms | prepare ms | pre_encode ms | pipeline compile ms (misses) | encode ms | gpu_exec ms | chunks | lead idle ms | inter-chunk idle ms | chunk busy sum ms | last gpu end ms | residual after last gpu end ms (evaluate - prepare - pre_encode - last gpu end) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 1000 | 836.776 | 826.572 | 39.155 | 460.449 | 455.085 (33) | 2.498 | 320.673 | 1 | 65.763 | -0.000 | 260.333 | 326.096 | 0.872 |
| 1 | 1 | 1 | 247.831 | 247.752 | 40.058 | 190.270 | 173.575 (10) | 0.808 | 12.106 | 8 | 3.621 | 0.261 | 12.816 | 16.698 | 0.727 |

no decode steps >= 2 in this log (N=0)

detail seq 0 (step 0): wall 836.776 ms, evaluate 826.572 ms, chunk busy sum 260.333 ms, last gpu end 326.096 ms

| chunk | ops | dispatches (census index) | encode start-end ms | gpu start-end ms | gpu busy ms | idle before ms | dispatch before the idle | dispatch after the idle |
|---|---|---|---|---|---|---|---|---|
| 1 | 0-382 | 383 (0-382) | 0.000-5.637 | 65.763-326.096 | 260.333 | 65.763 | none (host: prepare, plan, encode) | #0 constant omega_constant_r0_v41400000 [] |

largest host gaps in this step, ranked:

1. 65.763 ms before chunk 1: after none (host: prepare, plan, encode); before #0 constant omega_constant_r0_v41400000 []
2. 0.872 ms residual: evaluate minus prepare, pre_encode and the last chunk's gpu end (readback, kv append, host)
