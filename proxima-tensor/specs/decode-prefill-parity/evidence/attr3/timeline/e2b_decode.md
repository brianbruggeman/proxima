telemetry census_e2b_decode/decode_telemetry.log: 24 step records, 653 dispatch rows

| seq | step | new tokens | wall ms | evaluate ms | prepare ms | pre_encode ms | pipeline compile ms (misses) | encode ms | gpu_exec ms | chunks | lead idle ms | inter-chunk idle ms | chunk busy sum ms | last gpu end ms | residual after last gpu end ms (evaluate - prepare - pre_encode - last gpu end) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 971 | 1227.414 | 1218.903 | 20.316 | 503.256 | 493.168 (48) | 1.492 | 688.239 | 1 | 109.073 | -0.000 | 585.255 | 694.328 | 1.002 |
| 1 | 1 | 1 | 267.285 | 267.231 | 21.554 | 217.274 | 184.013 (13) | 5.034 | 16.436 | 8 | 5.270 | 0.266 | 22.011 | 27.547 | 0.856 |
| 2 | 2 | 1 | 13.311 | 13.257 | 0.000 | 0.166 | 0.000 (0) | 4.650 | 7.198 | 8 | 0.366 | 0.608 | 11.303 | 12.278 | 0.814 |
| 3 | 3 | 1 | 11.421 | 11.374 | 0.000 | 0.289 | 0.000 (0) | 4.323 | 5.616 | 8 | 0.286 | 0.635 | 9.896 | 10.817 | 0.268 |
| 4 | 4 | 1 | 11.192 | 11.141 | 0.000 | 0.131 | 0.000 (0) | 4.630 | 5.106 | 8 | 0.294 | 0.724 | 9.718 | 10.736 | 0.274 |
| 5 | 5 | 1 | 11.340 | 11.288 | 0.000 | 0.120 | 0.000 (0) | 4.311 | 5.718 | 8 | 0.274 | 0.656 | 9.972 | 10.902 | 0.266 |
| 6 | 6 | 1 | 11.403 | 11.350 | 0.000 | 0.128 | 0.000 (0) | 4.715 | 5.328 | 8 | 0.303 | 0.729 | 9.921 | 10.953 | 0.269 |
| 7 | 7 | 1 | 11.489 | 11.441 | 0.000 | 0.147 | 0.000 (0) | 4.712 | 5.438 | 8 | 0.298 | 0.752 | 9.938 | 10.988 | 0.307 |
| 8 | 8 | 1 | 11.438 | 11.390 | 0.000 | 0.115 | 0.000 (0) | 4.633 | 5.471 | 8 | 0.258 | 0.723 | 9.998 | 10.979 | 0.296 |
| 9 | 9 | 1 | 11.050 | 11.001 | 0.000 | 0.111 | 0.000 (0) | 4.669 | 5.034 | 8 | 0.258 | 0.713 | 9.608 | 10.579 | 0.311 |
| 10 | 10 | 1 | 11.113 | 11.055 | 0.000 | 0.138 | 0.000 (0) | 4.668 | 5.013 | 8 | 0.337 | 0.708 | 9.582 | 10.626 | 0.291 |
| 11 | 11 | 1 | 11.143 | 11.085 | 0.000 | 0.132 | 0.000 (0) | 4.785 | 4.954 | 8 | 0.320 | 0.734 | 9.597 | 10.651 | 0.302 |
| 12 | 12 | 1 | 11.347 | 11.289 | 0.000 | 0.131 | 0.000 (0) | 4.814 | 5.100 | 8 | 0.322 | 0.714 | 9.807 | 10.843 | 0.315 |
| 13 | 13 | 1 | 11.772 | 11.715 | 0.000 | 0.127 | 0.000 (0) | 5.342 | 5.030 | 8 | 0.293 | 1.128 | 9.880 | 11.301 | 0.287 |
| 14 | 14 | 1 | 11.565 | 11.510 | 0.000 | 0.137 | 0.000 (0) | 4.671 | 5.503 | 8 | 0.307 | 0.721 | 10.020 | 11.048 | 0.325 |
| 15 | 15 | 1 | 11.360 | 11.313 | 0.000 | 0.119 | 0.000 (0) | 4.656 | 5.407 | 8 | 0.276 | 0.707 | 9.927 | 10.909 | 0.285 |
| 16 | 16 | 1 | 11.294 | 11.246 | 0.000 | 0.109 | 0.000 (0) | 4.770 | 5.183 | 8 | 0.258 | 0.713 | 9.883 | 10.853 | 0.284 |
| 17 | 17 | 1 | 11.399 | 11.352 | 0.000 | 0.108 | 0.000 (0) | 4.668 | 5.448 | 8 | 0.258 | 0.695 | 10.008 | 10.962 | 0.283 |
| 18 | 18 | 1 | 11.323 | 11.275 | 0.000 | 0.111 | 0.000 (0) | 4.679 | 5.348 | 8 | 0.256 | 0.701 | 9.944 | 10.901 | 0.264 |
| 19 | 19 | 1 | 11.281 | 11.234 | 0.000 | 0.104 | 0.000 (0) | 4.648 | 5.353 | 8 | 0.281 | 0.688 | 9.889 | 10.858 | 0.272 |
| 20 | 20 | 1 | 11.311 | 11.264 | 0.000 | 0.109 | 0.000 (0) | 4.759 | 5.225 | 8 | 0.256 | 0.739 | 9.894 | 10.890 | 0.265 |
| 21 | 21 | 1 | 11.336 | 11.290 | 0.000 | 0.106 | 0.000 (0) | 4.697 | 5.340 | 8 | 0.263 | 0.702 | 9.947 | 10.912 | 0.272 |
| 22 | 22 | 1 | 13.700 | 13.645 | 0.000 | 0.111 | 0.853 (1) | 4.764 | 5.173 | 8 | 0.593 | 0.411 | 9.797 | 10.800 | 2.733 |
| 23 | 23 | 1 | 11.161 | 11.105 | 0.000 | 0.121 | 0.000 (0) | 4.765 | 5.036 | 8 | 0.283 | 0.735 | 9.682 | 10.700 | 0.284 |

median over 22 decode steps >= 2: wall 11.347 ms, evaluate 11.290, chunk busy sum 9.896, lead idle 0.286, inter-chunk idle 0.713, last gpu end 10.901, residual after last gpu end 0.285

detail seq 23 (step 23): wall 11.161 ms, evaluate 11.105 ms, chunk busy sum 9.682 ms, last gpu end 10.700 ms

| chunk | ops | dispatches (census index) | encode start-end ms | gpu start-end ms | gpu busy ms | idle before ms | dispatch before the idle | dispatch after the idle |
|---|---|---|---|---|---|---|---|---|
| 1 | 0-23 | 16 (0-15) | 0.000-0.150 | 0.283-0.545 | 0.262 | 0.283 | none (host: prepare, plan, encode) | #0 elementwise fused_identity_o0__multiply_s0_o1_g10 omega_elementwise_r2_n2_fused_identity_o0__multiply_s0_o1_g10 [1, 1536] |
| 2 | 24-139 | 91 (16-106) | 0.154-0.977 | 1.097-2.123 | 1.026 | 0.552 | #15 cached attention merge omega_cached_attention_q1_h1_g8_d256_s3f800000_ln511_up0_x4_e1_k1_ds_m [1, 1, 8, 256] | #16 matvec Q4_0 omega_reduce_r5_o2_n2_multiply_add_zero [1, 1, 8, 256, 1536] |
| 3 | 140-256 | 96 (107-202) | 0.980-1.801 | 2.150-3.109 | 0.958 | 0.028 | #106 RMSNorm sumsq + fused epilogue omega_reduce_r2_o1_n2_multiply_add_zero_epi5_fused_identity_o5__multip [1, 1536] | #107 matvec Q4_0 omega_reduce_r3_o2_n2_multiply_add_zero_epi5_fused_identity_o5__identi [1, 1536, 256] |
| 4 | 257-373 | 93 (203-295) | 1.804-2.629 | 3.138-4.167 | 1.029 | 0.029 | #202 UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__subtract_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__add_s0_s1 omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__sub [1, 8, 256] | #203 cached attention partial omega_cached_attention_q1_h1_g8_d512_s3f800000_ln9223372036854775808_u [1, 1, 8, 512] |
| 5 | 374-490 | 91 (296-386) | 2.632-3.456 | 4.196-5.458 | 1.263 | 0.028 | #295 RMSNorm sumsq + fused epilogue omega_reduce_r2_o1_n2_multiply_add_zero_epi5_fused_identity_o5__multip [1, 1536] | #296 matvec Q4_0 omega_reduce_r3_o2_n2_multiply_add_zero_epi5_fused_identity_o5__identi [1, 1536, 256] |
| 6 | 491-607 | 90 (387-476) | 3.459-4.276 | 5.489-6.968 | 1.479 | 0.031 | #386 RMSNorm sumsq + fused epilogue omega_reduce_r3_o2_n2_multiply_add_zero_epi4_fused_identity_o4__multip [1, 8, 512] | #387 UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__subtract_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__add_s0_s1 omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__sub [1, 8, 256] |
| 7 | 608-724 | 90 (477-566) | 4.280-5.048 | 7.002-8.327 | 1.326 | 0.033 | #476 RMSNorm sumsq + fused epilogue omega_reduce_r2_o1_n2_multiply_add_zero_epi5_fused_identity_o5__multip [1, 1536] | #477 matvec Q4_0 omega_reduce_r3_o2_n2_multiply_add_zero_epi5_fused_identity_o5__identi [1, 1536, 256] |
| 8 | 725-841 | 86 (567-652) | 5.050-5.780 | 8.361-10.700 | 2.338 | 0.034 | #566 matvec Q4_0 omega_reduce_r5_o2_n2_multiply_add_zero [1, 1, 8, 256, 1536] | #567 RMSNorm sumsq + fused epilogue omega_reduce_r2_o1_n2_multiply_add_zero_epi5_fused_identity_o5__multip [1, 1536] |

largest host gaps in this step, ranked:

1. 0.552 ms before chunk 2: after #15 cached attention merge omega_cached_attention_q1_h1_g8_d256_s3f800000_ln511_up0_x4_e1_k1_ds_m [1, 1, 8, 256]; before #16 matvec Q4_0 omega_reduce_r5_o2_n2_multiply_add_zero [1, 1, 8, 256, 1536]
2. 0.284 ms residual: evaluate minus prepare, pre_encode and the last chunk's gpu end (readback, kv append, host)
3. 0.283 ms before chunk 1: after none (host: prepare, plan, encode); before #0 elementwise fused_identity_o0__multiply_s0_o1_g10 omega_elementwise_r2_n2_fused_identity_o0__multiply_s0_o1_g10 [1, 1536]
4. 0.034 ms before chunk 8: after #566 matvec Q4_0 omega_reduce_r5_o2_n2_multiply_add_zero [1, 1, 8, 256, 1536]; before #567 RMSNorm sumsq + fused epilogue omega_reduce_r2_o1_n2_multiply_add_zero_epi5_fused_identity_o5__multip [1, 1536]
5. 0.033 ms before chunk 7: after #476 RMSNorm sumsq + fused epilogue omega_reduce_r2_o1_n2_multiply_add_zero_epi5_fused_identity_o5__multip [1, 1536]; before #477 matvec Q4_0 omega_reduce_r3_o2_n2_multiply_add_zero_epi5_fused_identity_o5__identi [1, 1536, 256]
6. 0.031 ms before chunk 6: after #386 RMSNorm sumsq + fused epilogue omega_reduce_r3_o2_n2_multiply_add_zero_epi4_fused_identity_o4__multip [1, 8, 512]; before #387 UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__subtract_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__add_s0_s1 omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__sub [1, 8, 256]
