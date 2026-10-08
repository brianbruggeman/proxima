run file granite_prefill_r1.out: 20 omit lines
run file granite_prefill_r2.out: 20 omit lines

groups=20 runs_per_group_max=2 sum_of_median_cost_ms=242.531 median_base_ms=253.311

| rank | entry | sha8 | threads | extents | dispatches | runs | median cost ms | cost range ms | us per dispatch | median base ms |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | omega_reduce_r4_o3_n2_multiply_add_zero_epi3_fused_identity_o3__negate_o2__exponential_s1__add_s | b4b75617 | 2048 | [1000, 8, 1024, 512] | 24 | 2 | 58.7613 | 58.5567 to 58.9659 | 2448.39 | 253.278 |
| 2 | omega_reduce_r4_o3_n2_multiply_add_zero_g10 | 6cd35d13 | 4096 | [1000, 8, 512, 1024] | 24 | 2 | 58.0358 | 57.8197 to 58.2518 | 2418.16 | 253.267 |
| 3 | omega_reduce_r4_o3_n2_multiply_add_zero_g10 | 6cd35d13 | 2048 | [1000, 8, 1024, 512] | 24 | 2 | 54.1366 | 54.1300 to 54.1432 | 2255.69 | 253.196 |
| 4 | omega_cached_attention_h8_g2_d64_s3c800000_ln9223372036854775808_up0_r8_n2_b64_rt | 1388fb73 | 64000 | [1000, 8, 2, 64] | 24 | 2 | 27.2304 | 27.2216 to 27.2391 | 1134.60 | 253.015 |
| 5 | omega_reduce_r3_o2_n2_multiply_add_zero_epi10_fused_identity_o10__add_o8_o9__add_s1_o7__add_s2_o | 2c18a81c | 1024000 | [1000, 8, 1024] | 24 | 2 | 13.5080 | 13.3127 to 13.7032 | 562.83 | 253.247 |
| 6 | omega_reduce_r3_o2_n2_multiply_add_zero | 7cc38fcb | 4096 | [1000, 1024, 32] | 24 | 2 | 12.2804 | 12.2356 to 12.3251 | 511.68 | 253.265 |
| 7 | omega_reduce_r5_o2_n2_multiply_add_zero_epi2_fused_identity_o2__multiply_s0_o1__add_s1_o0 | 944aea16 | 65536 | [1000, 8, 2, 64, 1024] | 24 | 2 | 8.7299 | 8.5864 to 8.8734 | 363.75 | 253.532 |
| 8 | omega_reduce_r4_o3_n2_multiply_add_zero | 549f5659 | 32768 | [1000, 8, 64, 1024] | 48 | 2 | 7.9687 | 7.3387 to 8.5987 | 166.01 | 253.493 |
| 9 | omega_reduce_r4_o3_n2_multiply_add_zero | 549f5659 | 65536 | [1000, 16, 64, 1024] | 24 | 2 | 7.0636 | 6.7893 to 7.3379 | 294.32 | 253.254 |
| 10 | omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__subtract_s0_s1_fused_multiply | 17b21f30 | 512000 | [1000, 16, 32] | 24 | 2 | 0.9364 | 0.7871 to 1.0856 | 39.01 | 253.378 |
| 11 | omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__add_s0_s1_fused_multiply_o0_o | 817c00aa | 256000 | [1000, 8, 32] | 24 | 2 | 0.5621 | 0.4959 to 0.6283 | 23.42 | 253.600 |
| 12 | omega_constant_r0_v3e6147ae | d552d3d4 | 1 | [] | 1 | 2 | 0.4906 | 0.3283 to 0.6528 | 490.55 | 253.539 |
| 13 | omega_constant_r0_v3a800000 | 6ff32071 | 1 | [] | 1 | 2 | 0.2598 | -0.0359 to 0.5555 | 259.80 | 253.569 |
| 14 | omega_elementwise_r2_n1_identity_g1 | 6ea595c3 | 1024 | [1, 1024] | 1 | 2 | 0.2566 | -0.0055 to 0.5186 | 256.55 | 253.479 |
| 15 | omega_elementwise_r2_n2_fused_identity_o0__multiply_s0_o1_g10 | f27d6389 | 1024000 | [1000, 1024] | 1 | 2 | 0.1633 | 0.1188 to 0.2077 | 163.25 | 252.945 |
| 16 | omega_reduce_r3_o2_n2_multiply_add_zero_epi1_fused_identity_o1__multiply_s0_o0 | c9fdfb3e | 393248 | [1, 1024, 49155] | 1 | 2 | 0.1553 | -0.0504 to 0.3610 | 155.30 | 253.301 |
| 17 | omega_constant_r0_v3f800000 | ff543d11 | 1 | [] | 1 | 2 | 0.0992 | 0.0293 to 0.1690 | 99.15 | 253.569 |
| 18 | omega_constant_r0_v3e2aaaab | 6ff5fb48 | 1 | [] | 1 | 2 | 0.0869 | -0.2813 to 0.4552 | 86.95 | 253.309 |
| 19 | omega_constant_r0_v41400000 | 2277fa6e | 1 | [] | 1 | 2 | -0.3293 | -0.7594 to 0.1007 | -329.35 | 252.923 |
| 20 | omega_reduce_r2_o1_n2_multiply_add_zero_epi4_fused_identity_o4__multiply_s0_o3__add_s1_o2__squar | 23d555d5 | 256000 | [1000, 1024] | 49 | 2 | -7.8644 | -8.2414 to -7.4875 | -160.50 | 253.363 |
