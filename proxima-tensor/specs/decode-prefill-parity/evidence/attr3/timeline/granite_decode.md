telemetry census_granite_decode/decode_telemetry.log: 24 step records, 398 dispatch rows

| seq | step | new tokens | wall ms | evaluate ms | prepare ms | pre_encode ms | pipeline compile ms (misses) | encode ms | gpu_exec ms | chunks | lead idle ms | inter-chunk idle ms | chunk busy sum ms | last gpu end ms | residual after last gpu end ms (evaluate - prepare - pre_encode - last gpu end) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 1000 | 399.854 | 389.197 | 38.419 | 29.437 | 24.558 (33) | 0.787 | 316.953 | 1 | 62.043 | -0.000 | 258.514 | 320.557 | 0.785 |
| 1 | 1 | 1 | 83.048 | 82.880 | 61.991 | 10.800 | 5.425 (10) | 2.718 | 3.092 | 8 | 3.427 | 0.233 | 5.554 | 9.214 | 0.875 |
| 2 | 2 | 1 | 6.515 | 6.440 | 0.000 | 0.083 | 0.000 (0) | 2.753 | 2.852 | 8 | 0.363 | 0.382 | 5.416 | 6.161 | 0.195 |
| 3 | 3 | 1 | 6.446 | 6.375 | 0.000 | 0.083 | 0.000 (0) | 2.653 | 2.884 | 8 | 0.341 | 0.320 | 5.422 | 6.083 | 0.209 |
| 4 | 4 | 1 | 6.470 | 6.402 | 0.000 | 0.081 | 0.000 (0) | 2.752 | 2.826 | 8 | 0.330 | 0.378 | 5.424 | 6.132 | 0.189 |
| 5 | 5 | 1 | 6.468 | 6.403 | 0.000 | 0.076 | 0.000 (0) | 2.674 | 2.925 | 8 | 0.328 | 0.355 | 5.451 | 6.134 | 0.193 |
| 6 | 6 | 1 | 6.458 | 6.386 | 0.000 | 0.081 | 0.000 (0) | 2.713 | 2.856 | 8 | 0.333 | 0.359 | 5.423 | 6.115 | 0.190 |
| 7 | 7 | 1 | 6.464 | 6.393 | 0.000 | 0.076 | 0.000 (0) | 2.656 | 2.928 | 8 | 0.299 | 0.360 | 5.437 | 6.095 | 0.221 |
| 8 | 8 | 1 | 6.374 | 6.303 | 0.000 | 0.071 | 0.000 (0) | 2.666 | 2.848 | 8 | 0.306 | 0.346 | 5.386 | 6.038 | 0.194 |
| 9 | 9 | 1 | 6.507 | 6.439 | 0.000 | 0.072 | 0.000 (0) | 2.719 | 2.926 | 8 | 0.361 | 0.359 | 5.416 | 6.135 | 0.231 |
| 10 | 10 | 1 | 6.553 | 6.467 | 0.000 | 0.159 | 0.000 (0) | 2.824 | 2.681 | 8 | 0.353 | 0.390 | 5.370 | 6.113 | 0.195 |
| 11 | 11 | 1 | 6.430 | 6.350 | 0.000 | 0.091 | 0.000 (0) | 2.692 | 2.814 | 8 | 0.329 | 0.368 | 5.351 | 6.048 | 0.211 |
| 12 | 12 | 1 | 6.359 | 6.287 | 0.000 | 0.081 | 0.000 (0) | 2.670 | 2.832 | 8 | 0.301 | 0.376 | 5.331 | 6.008 | 0.199 |
| 13 | 13 | 1 | 6.386 | 6.314 | 0.000 | 0.073 | 0.000 (0) | 2.794 | 2.725 | 8 | 0.318 | 0.367 | 5.360 | 6.045 | 0.197 |
| 14 | 14 | 1 | 6.436 | 6.365 | 0.000 | 0.076 | 0.000 (0) | 2.705 | 2.870 | 8 | 0.320 | 0.414 | 5.347 | 6.081 | 0.208 |
| 15 | 15 | 1 | 6.429 | 6.356 | 0.000 | 0.086 | 0.000 (0) | 2.856 | 2.695 | 8 | 0.353 | 0.380 | 5.349 | 6.083 | 0.187 |
| 16 | 16 | 1 | 6.399 | 6.327 | 0.000 | 0.077 | 0.000 (0) | 2.773 | 2.768 | 8 | 0.330 | 0.390 | 5.343 | 6.062 | 0.188 |
| 17 | 17 | 1 | 6.410 | 6.345 | 0.000 | 0.078 | 0.000 (0) | 2.796 | 2.730 | 8 | 0.315 | 0.372 | 5.376 | 6.063 | 0.204 |
| 18 | 18 | 1 | 6.473 | 6.398 | 0.000 | 0.085 | 0.000 (0) | 2.906 | 2.593 | 8 | 0.346 | 0.353 | 5.384 | 6.082 | 0.231 |
| 19 | 19 | 1 | 6.543 | 6.460 | 0.000 | 0.115 | 0.000 (0) | 2.862 | 2.663 | 8 | 0.381 | 0.382 | 5.355 | 6.119 | 0.227 |
| 20 | 20 | 1 | 6.533 | 6.448 | 0.000 | 0.105 | 0.000 (0) | 2.864 | 2.653 | 8 | 0.401 | 0.354 | 5.353 | 6.108 | 0.235 |
| 21 | 21 | 1 | 6.615 | 6.531 | 0.000 | 0.104 | 0.000 (0) | 3.566 | 2.007 | 8 | 0.373 | 0.424 | 5.375 | 6.172 | 0.256 |
| 22 | 22 | 1 | 6.625 | 6.535 | 0.000 | 0.122 | 0.000 (0) | 3.011 | 2.546 | 8 | 0.418 | 0.370 | 5.370 | 6.157 | 0.256 |
| 23 | 23 | 1 | 6.614 | 6.525 | 0.000 | 0.112 | 0.000 (0) | 2.903 | 2.635 | 8 | 0.409 | 0.389 | 5.375 | 6.174 | 0.240 |

median over 22 decode steps >= 2: wall 6.468 ms, evaluate 6.398, chunk busy sum 5.375, lead idle 0.341, inter-chunk idle 0.372, last gpu end 6.108, residual after last gpu end 0.208

detail seq 23 (step 23): wall 6.614 ms, evaluate 6.525 ms, chunk busy sum 5.375 ms, last gpu end 6.174 ms

| chunk | ops | dispatches (census index) | encode start-end ms | gpu start-end ms | gpu busy ms | idle before ms | dispatch before the idle | dispatch after the idle |
|---|---|---|---|---|---|---|---|---|
| 1 | 0-23 | 21 (0-20) | 0.000-0.241 | 0.409-0.682 | 0.273 | 0.409 | none (host: prepare, plan, encode) | #0 elementwise fused_identity_o0__multiply_s0_o1_g10 omega_elementwise_r2_n2_fused_identity_o0__multiply_s0_o1_g10 [1, 1024] |
| 2 | 24-74 | 55 (21-75) | 0.255-0.743 | 0.894-1.618 | 0.724 | 0.212 | #20 matvec Q8_0 omega_reduce_r4_o3_n2_multiply_add_zero [1, 8, 64, 1024] | #21 UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__add_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__subtract_s0_s1 omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__add [1, 8, 32] |
| 3 | 75-125 | 54 (76-129) | 0.749-1.232 | 1.644-2.391 | 0.747 | 0.026 | #75 matvec f32 omega_reduce_r3_o2_n2_multiply_add_zero [1, 1024, 32] | #76 UNCLASSIFIED omega_moe_topk_e32_k8_stacked omega_moe_topk_e32_k8_stacked [1] |
| 4 | 126-176 | 55 (130-184) | 1.236-1.740 | 2.419-3.139 | 0.719 | 0.028 | #129 RMSNorm sumsq + fused epilogue omega_reduce_r2_o1_n2_multiply_add_zero_epi4_fused_identity_o4__multip [1, 1024] | #130 matvec Q8_0 omega_reduce_r4_o3_n2_multiply_add_zero [1, 8, 64, 1024] |
| 5 | 177-228 | 55 (185-239) | 1.745-2.239 | 3.169-3.924 | 0.755 | 0.030 | #184 cached attention merge omega_cached_attention_q1_h8_g2_d64_s3c800000_ln9223372036854775808_up [1, 8, 2, 64] | #185 matvec Q8_0 omega_reduce_r5_o2_n2_multiply_add_zero_epi2_fused_identity_o2__multip [1, 8, 2, 64, 1024] |
| 6 | 229-279 | 54 (240-293) | 2.243-2.724 | 3.953-4.639 | 0.685 | 0.030 | #239 matvec Q8_0 omega_reduce_r4_o3_n2_multiply_add_zero_g10 [1, 8, 512, 1024] | #240 matvec f32 omega_reduce_r3_o2_n2_multiply_add_zero_epi10_fused_identity_o10__add_ [1, 8, 1024] |
| 7 | 280-330 | 55 (294-348) | 2.728-3.231 | 4.666-5.387 | 0.721 | 0.027 | #293 UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__add_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__subtract_s0_s1 omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__add [1, 8, 32] | #294 UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__subtract_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__add_s0_s1 omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__sub [1, 16, 32] |
| 8 | 331-382 | 49 (349-397) | 3.238-3.668 | 5.423-6.174 | 0.751 | 0.035 | #348 UNCLASSIFIED omega_moe_topk_e32_k8_stacked omega_moe_topk_e32_k8_stacked [1] | #349 matvec Q8_0 omega_reduce_r4_o3_n2_multiply_add_zero_g10 [1, 8, 1024, 512] |

largest host gaps in this step, ranked:

1. 0.409 ms before chunk 1: after none (host: prepare, plan, encode); before #0 elementwise fused_identity_o0__multiply_s0_o1_g10 omega_elementwise_r2_n2_fused_identity_o0__multiply_s0_o1_g10 [1, 1024]
2. 0.240 ms residual: evaluate minus prepare, pre_encode and the last chunk's gpu end (readback, kv append, host)
3. 0.212 ms before chunk 2: after #20 matvec Q8_0 omega_reduce_r4_o3_n2_multiply_add_zero [1, 8, 64, 1024]; before #21 UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__add_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__subtract_s0_s1 omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__add [1, 8, 32]
4. 0.035 ms before chunk 8: after #348 UNCLASSIFIED omega_moe_topk_e32_k8_stacked omega_moe_topk_e32_k8_stacked [1]; before #349 matvec Q8_0 omega_reduce_r4_o3_n2_multiply_add_zero_g10 [1, 8, 1024, 512]
5. 0.030 ms before chunk 5: after #184 cached attention merge omega_cached_attention_q1_h8_g2_d64_s3c800000_ln9223372036854775808_up [1, 8, 2, 64]; before #185 matvec Q8_0 omega_reduce_r5_o2_n2_multiply_add_zero_epi2_fused_identity_o2__multip [1, 8, 2, 64, 1024]
6. 0.030 ms before chunk 6: after #239 matvec Q8_0 omega_reduce_r4_o3_n2_multiply_add_zero_g10 [1, 8, 512, 1024]; before #240 matvec f32 omega_reduce_r3_o2_n2_multiply_add_zero_epi10_fused_identity_o10__add_ [1, 8, 1024]
