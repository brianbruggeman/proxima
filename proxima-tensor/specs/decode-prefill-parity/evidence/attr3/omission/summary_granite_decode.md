run file granite_decode_r1.out: 26 omit lines
run file granite_decode_r2.out: 26 omit lines
run file granite_decode_r3.out: 26 omit lines

groups=26 runs_per_group_max=3 sum_of_median_cost_ms=5.571 median_base_ms=5.576

| rank | entry | sha8 | threads | extents | dispatches | runs | median cost ms | cost range ms | us per dispatch | median base ms |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | omega_cached_attention_q1_h8_g2_d64_s3c800000_ln9223372036854775808_up0_x4_e1_k1_ds | 9ee4b248 | 65536 | [1, 8, 2, 64] | 24 | 3 | 1.2114 | 1.2058 to 1.2275 | 50.48 | 5.498 |
| 2 | omega_reduce_r4_o3_n2_multiply_add_zero_g10 | 2f64e24e | 65536 | [1, 8, 512, 1024] | 24 | 3 | 0.7652 | 0.7643 to 0.7859 | 31.88 | 5.713 |
| 3 | omega_reduce_r4_o3_n2_multiply_add_zero_epi3_fused_identity_o3__negate_o2__exponential_s1__add_s | 68576a1f | 32768 | [1, 8, 1024, 512] | 24 | 3 | 0.7273 | 0.7187 to 1.0150 | 30.30 | 5.601 |
| 4 | omega_reduce_r4_o3_n2_multiply_add_zero_g10 | 2f64e24e | 32768 | [1, 8, 1024, 512] | 24 | 3 | 0.5520 | 0.5183 to 0.6797 | 23.00 | 5.509 |
| 5 | omega_reduce_r2_o1_n2_multiply_add_zero_epi4_fused_identity_o4__multiply_s0_o3__add_s1_o2__squar | 23d555d5 | 256 | [1, 1024] | 49 | 3 | 0.4889 | 0.4658 to 0.5003 | 9.98 | 5.515 |
| 6 | omega_reduce_r4_o3_n2_multiply_add_zero | 00a58208 | 4096 | [1, 8, 64, 1024] | 48 | 3 | 0.3650 | 0.2837 to 0.3771 | 7.60 | 5.727 |
| 7 | omega_reduce_r5_o2_n2_multiply_add_zero_epi2_fused_identity_o2__multiply_s0_o1__add_s1_o0 | 4ffe0e12 | 8192 | [1, 8, 2, 64, 1024] | 24 | 3 | 0.2610 | 0.0765 to 0.3202 | 10.88 | 5.577 |
| 8 | omega_reduce_r3_o2_n2_multiply_add_zero_epi10_fused_identity_o10__add_o8_o9__add_s1_o7__add_s2_o | d491ed9e | 32768 | [1, 8, 1024] | 24 | 3 | 0.2536 | 0.1977 to 0.2597 | 10.57 | 5.629 |
| 9 | omega_reduce_r4_o3_n2_multiply_add_zero | 00a58208 | 8192 | [1, 16, 64, 1024] | 24 | 3 | 0.2361 | 0.1868 to 0.2565 | 9.84 | 5.755 |
| 10 | omega_reduce_r3_o2_n2_multiply_add_zero | 16be5817 | 8192 | [1, 1024, 32] | 24 | 3 | 0.1980 | 0.0989 to 0.2243 | 8.25 | 5.558 |
| 11 | omega_cached_attention_q1_h8_g2_d64_s3c800000_ln9223372036854775808_up0_x4_e1_k1_ds_merge | b74c01aa | 16384 | [1, 8, 2, 64] | 24 | 3 | 0.1622 | 0.1041 to 0.3069 | 6.76 | 5.712 |
| 12 | omega_reduce_r3_o2_n2_multiply_add_zero_epi1_fused_identity_o1__multiply_s0_o0 | c9fdfb3e | 393248 | [1, 1024, 49155] | 1 | 3 | 0.1374 | 0.1164 to 0.1707 | 137.40 | 5.448 |
| 13 | omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__subtract_s0_s1_fused_multiply | 5c56afee | 512 | [1, 16, 32] | 24 | 3 | 0.1073 | 0.0538 to 0.1821 | 4.47 | 5.600 |
| 14 | omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__add_s0_s1_fused_multiply_o0_o | 776034fd | 256 | [1, 8, 32] | 24 | 3 | 0.0987 | 0.0685 to 0.2129 | 4.11 | 5.454 |
| 15 | omega_reduce_r2_o1_n1_identity_add_zero | 5186d131 | 32 | [1, 113] | 1 | 3 | 0.0163 | -0.0357 to 0.0173 | 16.30 | 5.753 |
| 16 | omega_elementwise_r1_n2_multiply | c5fcdbd8 | 113 | [113] | 1 | 3 | 0.0100 | -0.0369 to 0.0109 | 10.00 | 5.760 |
| 17 | omega_reduce_r3_o2_n4_fused_greater_o0_o1__greater_o2_o3__multiply_s0_s1_add_zero | b7329559 | 14464 | [1, 113, 435] | 1 | 3 | 0.0097 | 0.0067 to 0.0210 | 9.70 | 5.731 |
| 18 | omega_elementwise_r3_n1_identity | efb05eff | 49155 | [1, 113, 435] | 1 | 3 | 0.0070 | -0.0005 to 0.0407 | 7.00 | 5.682 |
| 19 | omega_reduce_r3_o2_n5_fused_equal_o0_o1__add_o2_o3__select_s0_s1_o4_minimum_positive_infinity | 8f117d35 | 14464 | [1, 113, 435] | 1 | 3 | 0.0040 | -0.0023 to 0.0447 | 4.00 | 5.572 |
| 20 | omega_elementwise_r2_n1_identity_g1 | 6ea595c3 | 1024 | [1, 1024] | 1 | 3 | 0.0027 | -0.0410 to 0.1493 | 2.70 | 5.391 |
| 21 | omega_reduce_r3_o2_n1_identity_maximum_negative_infinity | 565c16a0 | 14464 | [1, 113, 435] | 1 | 3 | 0.0020 | -0.1580 to 0.0063 | 2.00 | 5.715 |
| 22 | omega_reduce_r2_o1_n1_identity_maximum_negative_infinity | 15126115 | 32 | [1, 113] | 1 | 3 | -0.0020 | -0.1867 to 0.0164 | -2.00 | 5.561 |
| 23 | omega_iota_r1 | 2ed77224 | 435 | [435] | 1 | 3 | -0.0045 | -0.0300 to 0.0169 | -4.50 | 5.572 |
| 24 | omega_elementwise_r2_n2_fused_identity_o0__multiply_s0_o1_g10 | 7d27f080 | 1024 | [1, 1024] | 1 | 3 | -0.0073 | -0.0450 to 0.0072 | -7.30 | 5.547 |
| 25 | omega_reduce_r2_o1_n1_identity_minimum_positive_infinity | 5f55fef1 | 32 | [1, 113] | 1 | 3 | -0.0108 | -0.1115 to -0.0061 | -10.80 | 5.473 |
| 26 | omega_iota_r1 | 2ed77224 | 113 | [113] | 1 | 3 | -0.0199 | -0.0840 to 0.0045 | -19.90 | 5.545 |
