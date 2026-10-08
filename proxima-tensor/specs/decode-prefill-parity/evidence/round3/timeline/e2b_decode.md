telemetry census/census_e2b_decode/decode_telemetry.log: 24 step records, 653 dispatch rows

| seq | step | new tokens | wall ms | evaluate ms | prepare ms | pre_encode ms | pipeline compile ms (misses) | encode ms | gpu_exec ms | chunks | lead idle ms | inter-chunk idle ms | chunk busy sum ms | last gpu end ms | residual after last gpu end ms (evaluate - prepare - pre_encode - last gpu end) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 971 | 768.761 | 760.404 | 20.735 | 52.716 | 42.475 (48) | 1.338 | 680.045 | 1 | 102.208 | -0.000 | 583.824 | 686.032 | 0.921 |
| 1 | 1 | 1 | 59.296 | 59.170 | 20.928 | 22.053 | 11.585 (13) | 1.448 | 8.327 | 8 | 4.712 | 0.211 | 10.297 | 15.220 | 0.969 |
| 2 | 2 | 1 | 11.094 | 11.041 | 0.000 | 0.137 | 0.000 (0) | 1.325 | 8.424 | 8 | 0.275 | 0.202 | 10.101 | 10.578 | 0.326 |
| 3 | 3 | 1 | 11.148 | 11.097 | 0.000 | 0.121 | 0.000 (0) | 1.286 | 8.624 | 8 | 0.253 | 0.192 | 10.237 | 10.682 | 0.294 |
| 4 | 4 | 1 | 10.942 | 10.897 | 0.000 | 0.109 | 0.000 (0) | 1.229 | 8.566 | 8 | 0.224 | 0.202 | 10.071 | 10.497 | 0.291 |
| 5 | 5 | 1 | 11.119 | 11.067 | 0.000 | 0.127 | 0.000 (0) | 1.373 | 8.442 | 8 | 0.227 | 0.211 | 10.175 | 10.612 | 0.328 |
| 6 | 6 | 1 | 11.042 | 10.993 | 0.000 | 0.115 | 0.000 (0) | 1.347 | 8.427 | 8 | 0.193 | 0.193 | 10.209 | 10.595 | 0.283 |
| 7 | 7 | 1 | 10.977 | 10.918 | 0.000 | 0.133 | 0.000 (0) | 1.571 | 7.961 | 8 | 0.215 | 0.190 | 10.049 | 10.454 | 0.331 |
| 8 | 8 | 1 | 11.024 | 10.970 | 0.000 | 0.130 | 0.000 (0) | 1.586 | 7.914 | 8 | 0.212 | 0.206 | 10.093 | 10.511 | 0.329 |
| 9 | 9 | 1 | 11.068 | 11.015 | 0.000 | 0.131 | 0.000 (0) | 1.592 | 7.965 | 8 | 0.225 | 0.191 | 10.116 | 10.532 | 0.352 |
| 10 | 10 | 1 | 10.987 | 10.935 | 0.000 | 0.131 | 0.000 (0) | 1.589 | 7.915 | 8 | 0.251 | 0.188 | 10.031 | 10.471 | 0.333 |
| 11 | 11 | 1 | 10.993 | 10.941 | 0.000 | 0.128 | 0.000 (0) | 1.574 | 7.973 | 8 | 0.212 | 0.197 | 10.088 | 10.497 | 0.316 |
| 12 | 12 | 1 | 10.930 | 10.878 | 0.000 | 0.131 | 0.000 (0) | 1.584 | 7.883 | 8 | 0.217 | 0.193 | 9.995 | 10.405 | 0.343 |
| 13 | 13 | 1 | 10.611 | 10.552 | 0.000 | 0.149 | 0.000 (0) | 1.581 | 7.492 | 8 | 0.243 | 0.183 | 9.648 | 10.074 | 0.328 |
| 14 | 14 | 1 | 10.998 | 10.945 | 0.000 | 0.132 | 0.000 (0) | 1.491 | 8.077 | 8 | 0.218 | 0.183 | 10.082 | 10.484 | 0.329 |
| 15 | 15 | 1 | 11.076 | 11.025 | 0.000 | 0.128 | 0.000 (0) | 1.469 | 8.220 | 8 | 0.219 | 0.194 | 10.163 | 10.576 | 0.320 |
| 16 | 16 | 1 | 11.122 | 11.075 | 0.000 | 0.129 | 0.000 (0) | 1.477 | 8.229 | 8 | 0.212 | 0.207 | 10.208 | 10.626 | 0.320 |
| 17 | 17 | 1 | 10.724 | 10.674 | 0.000 | 0.128 | 0.000 (0) | 1.515 | 7.751 | 8 | 0.213 | 0.186 | 9.812 | 10.211 | 0.335 |
| 18 | 18 | 1 | 11.007 | 10.953 | 0.000 | 0.136 | 0.000 (0) | 1.477 | 8.117 | 8 | 0.222 | 0.189 | 10.077 | 10.488 | 0.329 |
| 19 | 19 | 1 | 11.012 | 10.960 | 0.000 | 0.133 | 0.000 (0) | 1.540 | 8.032 | 8 | 0.221 | 0.188 | 10.101 | 10.509 | 0.317 |
| 20 | 20 | 1 | 10.950 | 10.899 | 0.000 | 0.123 | 0.000 (0) | 1.557 | 7.965 | 8 | 0.200 | 0.188 | 10.057 | 10.445 | 0.331 |
| 21 | 21 | 1 | 10.898 | 10.838 | 0.000 | 0.159 | 0.000 (0) | 1.619 | 7.720 | 8 | 0.313 | 0.190 | 9.813 | 10.316 | 0.363 |
| 22 | 22 | 1 | 13.713 | 13.651 | 0.000 | 0.133 | 0.937 (1) | 1.542 | 7.895 | 8 | 0.241 | 0.171 | 9.931 | 10.343 | 3.175 |
| 23 | 23 | 1 | 11.479 | 11.421 | 0.000 | 0.143 | 0.000 (0) | 5.402 | 4.529 | 8 | 0.365 | 0.743 | 9.844 | 10.952 | 0.327 |

median over 22 decode steps >= 2: wall 11.012 ms, evaluate 10.960, chunk busy sum 10.082, lead idle 0.222, inter-chunk idle 0.192, last gpu end 10.497, residual after last gpu end 0.329

detail seq 23 (step 23): wall 11.479 ms, evaluate 11.421 ms, chunk busy sum 9.844 ms, last gpu end 10.952 ms

| chunk | ops | dispatches (census index) | encode start-end ms | gpu start-end ms | gpu busy ms | idle before ms | dispatch before the idle | dispatch after the idle |
|---|---|---|---|---|---|---|---|---|
| 1 | 0-23 | 16 (0-15) | 0.000-0.197 | 0.365-0.626 | 0.262 | 0.365 | none (host: prepare, plan, encode) | #0 elementwise fused_identity_o0__multiply_s0_o1_g10 omega_elementwise_r2_n2_fused_identity_o0__multiply_s0_o1_g10 [1, 1536] |
| 2 | 24-62 | 28 (16-43) | 0.208-0.492 | 0.648-0.982 | 0.334 | 0.022 | #15 cached attention merge omega_cached_attention_q1_h1_g8_d256_s3f800000_ln511_up0_x4_e1_k1_ds_m [1, 1, 8, 256] | #16 matvec Q4_0 omega_reduce_r5_o2_n2_multiply_add_zero [1, 1, 8, 256, 1536] |
| 3 | 63-116 | 44 (44-87) | 0.496-0.922 | 1.026-1.473 | 0.448 | 0.044 | #43 RMSNorm sumsq + fused epilogue omega_reduce_r2_o1_n2_multiply_add_zero_epi5_fused_identity_o5__multip [1, 1536] | #44 matvec Q4_0 omega_reduce_r3_o2_n2_multiply_add_zero_epi5_fused_identity_o5__identi [1, 1536, 256] |
| 4 | 117-189 | 61 (88-148) | 0.926-1.512 | 1.673-2.359 | 0.686 | 0.200 | #87 matvec Q4_0 omega_reduce_r3_o2_n2_multiply_add_zero [1, 256, 1536] | #88 RMSNorm sumsq + fused epilogue omega_reduce_r2_o1_n2_multiply_add_zero_epi6_fused_identity_o6__multip [1, 1536] |
| 5 | 190-287 | 81 (149-229) | 1.516-2.320 | 2.413-3.272 | 0.859 | 0.054 | #148 RMSNorm sumsq + fused epilogue omega_reduce_r2_o1_n2_multiply_add_zero_epi5_fused_identity_o5__multip [1, 1536] | #149 matvec Q4_0 omega_reduce_r3_o2_n2_multiply_add_zero_epi5_fused_identity_o5__identi [1, 1536, 256] |
| 6 | 288-420 | 103 (230-332) | 2.326-3.328 | 3.461-4.676 | 1.215 | 0.189 | #229 matvec Q4_0 omega_reduce_r3_o2_n2_multiply_add_zero [1, 1536, 6144] | #230 matvec Q4_0 omega_reduce_r3_o2_n2_multiply_add_zero_epi9_fused_identity_o9__multip [1, 1536, 6144] |
| 7 | 421-599 | 139 (333-471) | 3.335-4.713 | 4.883-7.151 | 2.269 | 0.206 | #332 RMSNorm sumsq + fused epilogue omega_reduce_r2_o1_n2_multiply_add_zero_epi5_fused_identity_o5__multip [1, 1536] | #333 matvec Q4_0 omega_reduce_r3_o2_n2_multiply_add_zero_epi5_fused_identity_o5__identi [1, 1536, 256] |
| 8 | 600-841 | 181 (472-652) | 4.717-6.566 | 7.180-10.952 | 3.772 | 0.029 | #471 RMSNorm sumsq + fused epilogue omega_reduce_r2_o1_n2_multiply_add_zero_epi5_fused_identity_o5__multip [1, 1536] | #472 RMSNorm sumsq + fused epilogue omega_reduce_r2_o1_n2_multiply_add_zero_epi4_fused_identity_o4__multip [1, 1536] |

largest host gaps in this step, ranked:

1. 0.365 ms before chunk 1: after none (host: prepare, plan, encode); before #0 elementwise fused_identity_o0__multiply_s0_o1_g10 omega_elementwise_r2_n2_fused_identity_o0__multiply_s0_o1_g10 [1, 1536]
2. 0.327 ms residual: evaluate minus prepare, pre_encode and the last chunk's gpu end (readback, kv append, host)
3. 0.206 ms before chunk 7: after #332 RMSNorm sumsq + fused epilogue omega_reduce_r2_o1_n2_multiply_add_zero_epi5_fused_identity_o5__multip [1, 1536]; before #333 matvec Q4_0 omega_reduce_r3_o2_n2_multiply_add_zero_epi5_fused_identity_o5__identi [1, 1536, 256]
4. 0.200 ms before chunk 4: after #87 matvec Q4_0 omega_reduce_r3_o2_n2_multiply_add_zero [1, 256, 1536]; before #88 RMSNorm sumsq + fused epilogue omega_reduce_r2_o1_n2_multiply_add_zero_epi6_fused_identity_o6__multip [1, 1536]
5. 0.189 ms before chunk 6: after #229 matvec Q4_0 omega_reduce_r3_o2_n2_multiply_add_zero [1, 1536, 6144]; before #230 matvec Q4_0 omega_reduce_r3_o2_n2_multiply_add_zero_epi9_fused_identity_o9__multip [1, 1536, 6144]
6. 0.054 ms before chunk 5: after #148 RMSNorm sumsq + fused epilogue omega_reduce_r2_o1_n2_multiply_add_zero_epi5_fused_identity_o5__multip [1, 1536]; before #149 matvec Q4_0 omega_reduce_r3_o2_n2_multiply_add_zero_epi5_fused_identity_o5__identi [1, 1536, 256]

host phases, ms from the earliest phase start (step_phase has chunk 0):

| chunk | phase | start ms | end ms | duration ms |
|---|---|---|---|---|
| 0 | pre_encode | 0.000 | 0.142 | 0.142 |
| 1 | encode | 0.142 | 0.340 | 0.198 |
| 1 | commit | 0.340 | 0.349 | 0.009 |
| 2 | encode | 0.350 | 0.634 | 0.284 |
| 1 | scheduled | 0.434 | 0.434 | 0.000 |
| 1 | gpu | 0.507 | 0.768 | 0.262 |
| 2 | commit | 0.634 | 0.637 | 0.003 |
| 3 | encode | 0.638 | 1.064 | 0.426 |
| 2 | scheduled | 0.683 | 0.683 | 0.000 |
| 2 | gpu | 0.790 | 1.124 | 0.334 |
| 3 | commit | 1.064 | 1.068 | 0.003 |
| 4 | encode | 1.069 | 1.654 | 0.585 |
| 3 | scheduled | 1.124 | 1.124 | 0.000 |
| 3 | gpu | 1.168 | 1.615 | 0.448 |
| 4 | commit | 1.654 | 1.657 | 0.004 |
| 5 | encode | 1.658 | 2.463 | 0.804 |
| 4 | scheduled | 1.748 | 1.748 | 0.000 |
| 4 | gpu | 1.815 | 2.501 | 0.686 |
| 5 | commit | 2.463 | 2.467 | 0.004 |
| 6 | encode | 2.468 | 3.470 | 1.002 |
| 5 | scheduled | 2.514 | 2.514 | 0.000 |
| 5 | gpu | 2.556 | 3.414 | 0.859 |
| 6 | commit | 3.470 | 3.476 | 0.006 |
| 7 | encode | 3.477 | 4.855 | 1.378 |
| 6 | scheduled | 3.541 | 3.541 | 0.000 |
| 6 | gpu | 3.603 | 4.818 | 1.215 |
| 7 | commit | 4.855 | 4.858 | 0.003 |
| 8 | encode | 4.859 | 6.708 | 1.849 |
| 7 | scheduled | 4.957 | 4.957 | 0.000 |
| 7 | gpu | 5.025 | 7.293 | 2.269 |
| 8 | commit | 6.709 | 6.716 | 0.007 |
| 8 | scheduled | 6.789 | 6.789 | 0.000 |
| 8 | gpu | 7.322 | 11.094 | 3.772 |

commit-to-GPU-start split per chunk (commit end to scheduled callback, scheduled callback to GPU start):

| chunk | commit end to scheduled ms | scheduled to gpu start ms |
|---|---|---|
| 1 | 0.085 | 0.073 |
| 2 | 0.045 | 0.107 |
| 3 | 0.057 | 0.043 |
| 4 | 0.091 | 0.067 |
| 5 | 0.047 | 0.042 |
| 6 | 0.066 | 0.062 |
| 7 | 0.099 | 0.067 |
| 8 | 0.072 | 0.534 |
