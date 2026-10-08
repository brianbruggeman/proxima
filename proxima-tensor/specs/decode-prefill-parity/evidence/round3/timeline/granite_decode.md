telemetry census/census_granite_decode/decode_telemetry.log: 24 step records, 398 dispatch rows

| seq | step | new tokens | wall ms | evaluate ms | prepare ms | pre_encode ms | pipeline compile ms (misses) | encode ms | gpu_exec ms | chunks | lead idle ms | inter-chunk idle ms | chunk busy sum ms | last gpu end ms | residual after last gpu end ms (evaluate - prepare - pre_encode - last gpu end) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 1000 | 374.895 | 365.096 | 38.231 | 33.619 | 24.436 (33) | 1.172 | 288.497 | 1 | 54.750 | -0.000 | 237.797 | 292.547 | 0.698 |
| 1 | 1 | 1 | 59.466 | 59.389 | 39.577 | 9.996 | 4.927 (10) | 0.854 | 4.989 | 8 | 3.213 | 0.177 | 5.699 | 9.090 | 0.726 |
| 2 | 2 | 1 | 6.404 | 6.332 | 0.000 | 0.072 | 0.000 (0) | 0.746 | 4.840 | 8 | 0.230 | 0.203 | 5.570 | 6.004 | 0.256 |
| 3 | 3 | 1 | 6.411 | 6.353 | 0.000 | 0.073 | 0.000 (0) | 0.748 | 4.899 | 8 | 0.234 | 0.194 | 5.655 | 6.083 | 0.197 |
| 4 | 4 | 1 | 6.441 | 6.390 | 0.000 | 0.063 | 0.000 (0) | 0.769 | 4.973 | 8 | 0.241 | 0.180 | 5.701 | 6.123 | 0.204 |
| 5 | 5 | 1 | 6.377 | 6.324 | 0.000 | 0.067 | 0.000 (0) | 0.742 | 4.903 | 8 | 0.206 | 0.173 | 5.669 | 6.048 | 0.209 |
| 6 | 6 | 1 | 6.373 | 6.323 | 0.000 | 0.068 | 0.000 (0) | 0.728 | 4.907 | 8 | 0.206 | 0.184 | 5.661 | 6.051 | 0.204 |
| 7 | 7 | 1 | 6.347 | 6.299 | 0.000 | 0.065 | 0.000 (0) | 0.734 | 4.908 | 8 | 0.203 | 0.183 | 5.659 | 6.045 | 0.189 |
| 8 | 8 | 1 | 6.492 | 6.442 | 0.000 | 0.075 | 0.000 (0) | 0.756 | 5.005 | 8 | 0.223 | 0.183 | 5.739 | 6.146 | 0.221 |
| 9 | 9 | 1 | 6.416 | 6.369 | 0.000 | 0.062 | 0.000 (0) | 0.715 | 5.015 | 8 | 0.197 | 0.185 | 5.719 | 6.101 | 0.205 |
| 10 | 10 | 1 | 6.127 | 6.077 | 0.000 | 0.063 | 0.000 (0) | 0.765 | 4.616 | 8 | 0.195 | 0.150 | 5.440 | 5.784 | 0.230 |
| 11 | 11 | 1 | 6.504 | 6.404 | 0.000 | 0.095 | 0.000 (0) | 0.745 | 4.880 | 8 | 0.244 | 0.194 | 5.643 | 6.081 | 0.227 |
| 12 | 12 | 1 | 6.219 | 6.164 | 0.000 | 0.070 | 0.000 (0) | 0.738 | 4.747 | 8 | 0.191 | 0.185 | 5.527 | 5.902 | 0.192 |
| 13 | 13 | 1 | 6.597 | 6.535 | 0.000 | 0.085 | 0.000 (0) | 0.811 | 4.947 | 8 | 0.435 | 0.174 | 5.633 | 6.243 | 0.207 |
| 14 | 14 | 1 | 6.353 | 6.290 | 0.000 | 0.081 | 0.000 (0) | 0.788 | 4.769 | 8 | 0.214 | 0.155 | 5.618 | 5.987 | 0.221 |
| 15 | 15 | 1 | 6.374 | 6.300 | 0.000 | 0.084 | 0.000 (0) | 0.843 | 4.668 | 8 | 0.210 | 0.180 | 5.631 | 6.021 | 0.195 |
| 16 | 16 | 1 | 6.403 | 6.341 | 0.000 | 0.079 | 0.000 (0) | 0.844 | 4.728 | 8 | 0.216 | 0.179 | 5.656 | 6.050 | 0.211 |
| 17 | 17 | 1 | 6.290 | 6.230 | 0.000 | 0.080 | 0.000 (0) | 0.871 | 4.572 | 8 | 0.161 | 0.162 | 5.596 | 5.919 | 0.231 |
| 18 | 18 | 1 | 6.408 | 6.343 | 0.000 | 0.084 | 0.000 (0) | 0.912 | 4.612 | 8 | 0.207 | 0.164 | 5.705 | 6.076 | 0.184 |
| 19 | 19 | 1 | 6.285 | 6.220 | 0.000 | 0.078 | 0.000 (0) | 0.911 | 4.500 | 8 | 0.196 | 0.163 | 5.588 | 5.946 | 0.195 |
| 20 | 20 | 1 | 6.297 | 6.232 | 0.000 | 0.082 | 0.000 (0) | 0.917 | 4.493 | 8 | 0.206 | 0.166 | 5.601 | 5.974 | 0.177 |
| 21 | 21 | 1 | 6.193 | 6.130 | 0.000 | 0.077 | 0.000 (0) | 0.914 | 4.404 | 8 | 0.200 | 0.155 | 5.479 | 5.834 | 0.219 |
| 22 | 22 | 1 | 6.225 | 6.137 | 0.000 | 0.099 | 0.000 (0) | 0.931 | 4.324 | 8 | 0.241 | 0.164 | 5.403 | 5.808 | 0.231 |
| 23 | 23 | 1 | 6.440 | 6.361 | 0.000 | 0.096 | 0.000 (0) | 3.560 | 1.844 | 8 | 0.408 | 0.187 | 5.431 | 6.025 | 0.240 |

median over 22 decode steps >= 2: wall 6.377 ms, evaluate 6.324, chunk busy sum 5.633, lead idle 0.210, inter-chunk idle 0.180, last gpu end 6.045, residual after last gpu end 0.209

detail seq 23 (step 23): wall 6.440 ms, evaluate 6.361 ms, chunk busy sum 5.431 ms, last gpu end 6.025 ms

| chunk | ops | dispatches (census index) | encode start-end ms | gpu start-end ms | gpu busy ms | idle before ms | dispatch before the idle | dispatch after the idle |
|---|---|---|---|---|---|---|---|---|
| 1 | 0-23 | 21 (0-20) | 0.000-0.262 | 0.408-0.673 | 0.265 | 0.408 | none (host: prepare, plan, encode) | #0 elementwise fused_identity_o0__multiply_s0_o1_g10 omega_elementwise_r2_n2_fused_identity_o0__multiply_s0_o1_g10 [1, 1024] |
| 2 | 24-40 | 18 (21-38) | 0.269-0.485 | 0.694-0.919 | 0.225 | 0.022 | #20 matvec Q8_0 omega_reduce_r4_o3_n2_multiply_add_zero [1, 8, 64, 1024] | #21 UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__add_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__subtract_s0_s1 omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__add [1, 8, 32] |
| 3 | 41-64 | 26 (39-64) | 0.490-0.795 | 0.940-1.334 | 0.394 | 0.020 | #38 UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__subtract_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__add_s0_s1 omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__sub [1, 16, 32] | #39 cached attention partial omega_cached_attention_q1_h8_g2_d64_s3c800000_ln9223372036854775808_up [1, 8, 2, 64] |
| 4 | 65-96 | 34 (65-98) | 0.799-1.185 | 1.357-1.806 | 0.449 | 0.023 | #64 matvec f32 omega_reduce_r3_o2_n2_multiply_add_zero_epi10_fused_identity_o10__add_ [1, 8, 1024] | #65 RMSNorm sumsq + fused epilogue omega_reduce_r2_o1_n2_multiply_add_zero_epi4_fused_identity_o4__multip [1, 1024] |
| 5 | 97-139 | 46 (99-144) | 1.189-1.715 | 1.833-2.466 | 0.634 | 0.026 | #98 matvec Q8_0 omega_reduce_r4_o3_n2_multiply_add_zero [1, 8, 64, 1024] | #99 matvec Q8_0 omega_reduce_r4_o3_n2_multiply_add_zero [1, 16, 64, 1024] |
| 6 | 140-197 | 62 (145-206) | 1.720-2.405 | 2.511-3.335 | 0.824 | 0.045 | #144 matvec f32 omega_reduce_r3_o2_n2_multiply_add_zero_epi10_fused_identity_o10__add_ [1, 8, 1024] | #145 RMSNorm sumsq + fused epilogue omega_reduce_r2_o1_n2_multiply_add_zero_epi4_fused_identity_o4__multip [1, 1024] |
| 7 | 198-275 | 83 (207-289) | 2.410-3.251 | 3.356-4.489 | 1.133 | 0.020 | #206 matvec Q8_0 omega_reduce_r4_o3_n2_multiply_add_zero_epi3_fused_identity_o3__negate [1, 8, 1024, 512] | #207 matvec Q8_0 omega_reduce_r4_o3_n2_multiply_add_zero_g10 [1, 8, 512, 1024] |
| 8 | 276-382 | 108 (290-397) | 3.256-4.329 | 4.519-6.025 | 1.506 | 0.030 | #289 RMSNorm sumsq + fused epilogue omega_reduce_r2_o1_n2_multiply_add_zero_epi4_fused_identity_o4__multip [1, 1024] | #290 matvec Q8_0 omega_reduce_r4_o3_n2_multiply_add_zero [1, 8, 64, 1024] |

largest host gaps in this step, ranked:

1. 0.408 ms before chunk 1: after none (host: prepare, plan, encode); before #0 elementwise fused_identity_o0__multiply_s0_o1_g10 omega_elementwise_r2_n2_fused_identity_o0__multiply_s0_o1_g10 [1, 1024]
2. 0.240 ms residual: evaluate minus prepare, pre_encode and the last chunk's gpu end (readback, kv append, host)
3. 0.045 ms before chunk 6: after #144 matvec f32 omega_reduce_r3_o2_n2_multiply_add_zero_epi10_fused_identity_o10__add_ [1, 8, 1024]; before #145 RMSNorm sumsq + fused epilogue omega_reduce_r2_o1_n2_multiply_add_zero_epi4_fused_identity_o4__multip [1, 1024]
4. 0.030 ms before chunk 8: after #289 RMSNorm sumsq + fused epilogue omega_reduce_r2_o1_n2_multiply_add_zero_epi4_fused_identity_o4__multip [1, 1024]; before #290 matvec Q8_0 omega_reduce_r4_o3_n2_multiply_add_zero [1, 8, 64, 1024]
5. 0.026 ms before chunk 5: after #98 matvec Q8_0 omega_reduce_r4_o3_n2_multiply_add_zero [1, 8, 64, 1024]; before #99 matvec Q8_0 omega_reduce_r4_o3_n2_multiply_add_zero [1, 16, 64, 1024]
6. 0.023 ms before chunk 4: after #64 matvec f32 omega_reduce_r3_o2_n2_multiply_add_zero_epi10_fused_identity_o10__add_ [1, 8, 1024]; before #65 RMSNorm sumsq + fused epilogue omega_reduce_r2_o1_n2_multiply_add_zero_epi4_fused_identity_o4__multip [1, 1024]

host phases, ms from the earliest phase start (step_phase has chunk 0):

| chunk | phase | start ms | end ms | duration ms |
|---|---|---|---|---|
| 0 | pre_encode | 0.000 | 0.096 | 0.096 |
| 1 | encode | 0.096 | 0.358 | 0.262 |
| 1 | commit | 0.358 | 0.364 | 0.006 |
| 2 | encode | 0.365 | 0.581 | 0.216 |
| 1 | scheduled | 0.436 | 0.436 | 0.000 |
| 1 | gpu | 0.504 | 0.769 | 0.265 |
| 2 | commit | 0.581 | 0.585 | 0.004 |
| 3 | encode | 0.586 | 0.891 | 0.305 |
| 2 | scheduled | 0.635 | 0.635 | 0.000 |
| 2 | gpu | 0.790 | 1.015 | 0.225 |
| 3 | commit | 0.891 | 0.894 | 0.003 |
| 4 | encode | 0.895 | 1.280 | 0.385 |
| 3 | scheduled | 0.952 | 0.952 | 0.000 |
| 3 | gpu | 1.036 | 1.430 | 0.394 |
| 4 | commit | 1.281 | 1.284 | 0.004 |
| 5 | encode | 1.285 | 1.811 | 0.526 |
| 4 | scheduled | 1.342 | 1.342 | 0.000 |
| 4 | gpu | 1.453 | 1.902 | 0.449 |
| 5 | commit | 1.811 | 1.815 | 0.004 |
| 6 | encode | 1.816 | 2.501 | 0.685 |
| 5 | scheduled | 1.876 | 1.876 | 0.000 |
| 5 | gpu | 1.929 | 2.562 | 0.634 |
| 6 | commit | 2.501 | 2.505 | 0.004 |
| 7 | encode | 2.506 | 3.347 | 0.842 |
| 6 | scheduled | 2.563 | 2.563 | 0.000 |
| 6 | gpu | 2.607 | 3.431 | 0.824 |
| 7 | commit | 3.348 | 3.351 | 0.003 |
| 8 | encode | 3.352 | 4.425 | 1.073 |
| 7 | scheduled | 3.408 | 3.408 | 0.000 |
| 7 | gpu | 3.451 | 4.584 | 1.133 |
| 8 | commit | 4.425 | 4.429 | 0.004 |
| 8 | scheduled | 4.485 | 4.485 | 0.000 |
| 8 | gpu | 4.615 | 6.121 | 1.506 |

commit-to-GPU-start split per chunk (commit end to scheduled callback, scheduled callback to GPU start):

| chunk | commit end to scheduled ms | scheduled to gpu start ms |
|---|---|---|
| 1 | 0.072 | 0.068 |
| 2 | 0.050 | 0.155 |
| 3 | 0.058 | 0.083 |
| 4 | 0.058 | 0.111 |
| 5 | 0.061 | 0.053 |
| 6 | 0.058 | 0.044 |
| 7 | 0.058 | 0.043 |
| 8 | 0.056 | 0.130 |
