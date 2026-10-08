run file e2b_decode_r1.out: 16 omit lines
run file e2b_decode_r2.out: 45 omit lines
run file e2b_decode_r3.out: 45 omit lines
run file e2b_decode_r4.out: 45 omit lines

groups=45 runs_per_group_max=4 sum_of_median_cost_ms=10.766 median_base_ms=10.539

| rank | entry | sha8 | threads | extents | dispatches | runs | median cost ms | cost range ms | us per dispatch | median base ms |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | omega_reduce_r3_o2_n2_multiply_add_zero_epi2_fused_identity_o2__divide_s0_o1__tanh_s1__multiply_ | c161db61 | 4194304 | [1, 1536, 262144] | 1 | 3 | 1.1017 | 1.0862 to 1.1165 | 1101.70 | 10.553 |
| 2 | omega_reduce_r3_o2_n2_multiply_add_zero | b7305889 | 12288 | [1, 12288, 1536] | 20 | 3 | 0.9440 | 0.8869 to 0.9448 | 47.20 | 10.557 |
| 3 | omega_reduce_r3_o2_n2_multiply_add_zero_epi9_fused_identity_o9__multiply_o5_o6__multiply_s1_o4__ | 1d7873ae | 98304 | [1, 1536, 12288] | 20 | 4 | 0.8809 | 0.8203 to 0.9163 | 44.04 | 10.479 |
| 4 | omega_reduce_r2_o1_n2_multiply_add_zero_epi5_fused_identity_o5__multiply_s0_o4__add_s1_o3__squar | f41c3c0e | 256 | [1, 1536] | 70 | 3 | 0.8336 | 0.7917 to 0.8540 | 11.91 | 10.555 |
| 5 | omega_reduce_r3_o2_n2_multiply_add_zero | 301c3740 | 98304 | [1, 1536, 12288] | 20 | 4 | 0.8079 | 0.7429 to 0.8355 | 40.40 | 10.443 |
| 6 | omega_cached_attention_q1_h1_g8_d256_s3f800000_ln511_up0_x4_e1_k1_ds | b2651dd9 | 17408 | [1, 1, 8, 256] | 28 | 3 | 0.7211 | 0.6850 to 0.7653 | 25.75 | 10.544 |
| 7 | omega_reduce_r2_o1_n2_multiply_add_zero_epi4_fused_identity_o4__multiply_s0_o3__add_s1_o2__squar | 23d555d5 | 256 | [1, 1536] | 71 | 4 | 0.6863 | 0.6233 to 0.7610 | 9.67 | 10.461 |
| 8 | omega_reduce_r2_o1_n2_multiply_add_zero_epi6_fused_identity_o6__multiply_s0_o4__add_s1_o3__squar | d8ad59bf | 256 | [1, 1536] | 35 | 3 | 0.3913 | 0.3863 to 0.4050 | 11.18 | 10.559 |
| 9 | omega_reduce_r4_o3_n2_multiply_add_zero | fa756124 | 16384 | [1, 8, 256, 1536] | 28 | 3 | 0.3899 | 0.3569 to 0.4130 | 13.93 | 10.549 |
| 10 | omega_reduce_r3_o2_n2_multiply_add_zero | b7305889 | 12288 | [1, 6144, 1536] | 15 | 3 | 0.3891 | 0.3845 to 0.3986 | 25.94 | 10.552 |
| 11 | omega_cached_attention_q1_h1_g8_d512_s3f800000_ln9223372036854775808_up0_x4_e1_k1_ds | a99a3e2e | 32768 | [1, 1, 8, 512] | 7 | 3 | 0.3870 | 0.3753 to 0.3974 | 55.29 | 10.566 |
| 12 | omega_reduce_r3_o2_n2_multiply_add_zero_epi9_fused_identity_o9__multiply_o5_o6__multiply_s1_o4__ | 1d7873ae | 49152 | [1, 1536, 6144] | 15 | 4 | 0.3793 | 0.3673 to 0.4535 | 25.29 | 10.537 |
| 13 | omega_reduce_r5_o2_n2_multiply_add_zero | 70da2b90 | 12288 | [1, 1, 8, 256, 1536] | 28 | 3 | 0.3718 | 0.3299 to 0.3743 | 13.28 | 10.540 |
| 14 | omega_reduce_r3_o2_n2_multiply_add_zero_epi5_fused_identity_o5__identity_o0__multiply_s0_s0__mul | fb097ee4 | 2048 | [1, 1536, 256] | 35 | 3 | 0.3628 | 0.3616 to 0.3841 | 10.37 | 10.573 |
| 15 | omega_reduce_r3_o2_n2_multiply_add_zero | 301c3740 | 49152 | [1, 1536, 6144] | 15 | 4 | 0.3567 | 0.3419 to 0.4245 | 23.78 | 10.501 |
| 16 | omega_cached_attention_q1_h1_g8_d256_s3f800000_ln511_up0_x4_e1_k1_ds_merge | fb384d9b | 8192 | [1, 1, 8, 256] | 28 | 3 | 0.2239 | 0.1886 to 0.2247 | 8.00 | 10.547 |
| 17 | omega_reduce_r3_o2_n2_multiply_add_zero | 301c3740 | 12288 | [1, 256, 1536] | 35 | 4 | 0.2104 | 0.1430 to 0.3116 | 6.01 | 10.546 |
| 18 | omega_reduce_r4_o3_n2_multiply_add_zero | fa4a9da9 | 2048 | [1, 1, 256, 1536] | 24 | 3 | 0.1904 | 0.1724 to 0.1959 | 7.93 | 10.584 |
| 19 | omega_reduce_r4_o3_n2_multiply_add_zero | fa756124 | 32768 | [1, 8, 512, 1536] | 7 | 3 | 0.1515 | 0.1232 to 0.1520 | 21.64 | 10.576 |
| 20 | omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__subtract_s0_s1_fused_multiply | 5c56afee | 1024 | [1, 8, 128] | 28 | 4 | 0.1418 | 0.1320 to 0.1649 | 5.06 | 10.526 |
| 21 | omega_reduce_r5_o2_n2_multiply_add_zero | 70da2b90 | 12288 | [1, 1, 8, 512, 1536] | 7 | 3 | 0.1284 | 0.1155 to 0.1301 | 18.34 | 10.399 |
| 22 | omega_reduce_r3_o2_n2_multiply_add_zero_epi1_fused_identity_o1__multiply_s0_o0 | 588991df | 2293760 | [1, 1536, 8960] | 1 | 4 | 0.1181 | 0.1146 to 0.1627 | 118.10 | 10.546 |
| 23 | omega_reduce_r3_o2_n2_multiply_add_zero_epi4_fused_identity_o4__multiply_s0_o3__add_s1_o2__squar | a638c75f | 64 | [1, 1, 256] | 12 | 3 | 0.0856 | 0.0752 to 0.0931 | 7.13 | 10.568 |
| 24 | omega_cached_attention_q1_h1_g8_d512_s3f800000_ln9223372036854775808_up0_x4_e1_k1_ds_merge | 5c2e7fcd | 8192 | [1, 1, 8, 512] | 7 | 4 | 0.0802 | 0.0557 to 0.1035 | 11.46 | 10.543 |
| 25 | omega_reduce_r3_o2_n2_multiply_add_zero_epi3_fused_identity_o3__multiply_s0_o2__add_s1_o1__squar | 9b87f663 | 64 | [1, 1, 256] | 12 | 3 | 0.0744 | 0.0732 to 0.0899 | 6.20 | 10.552 |
| 26 | omega_reduce_r2_o1_n1_identity_add_zero | 055f1105 | 128 | [1, 512] | 1 | 4 | 0.0566 | 0.0183 to 0.1580 | 56.55 | 10.506 |
| 27 | omega_elementwise_r3_n1_identity | efb05eff | 8960 | [1, 35, 256] | 1 | 3 | 0.0473 | 0.0363 to 0.0650 | 47.30 | 10.553 |
| 28 | omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__subtract_s0_s1_fused_multiply | 5c56afee | 2048 | [1, 8, 256] | 7 | 4 | 0.0412 | 0.0027 to 0.0586 | 5.89 | 10.548 |
| 29 | omega_reduce_r3_o2_n2_multiply_add_zero_epi6_fused_identity_o6__identity_o0__multiply_s0_o4__add | 7310c707 | 2240 | [1, 35, 256] | 1 | 3 | 0.0412 | -0.0072 to 0.0915 | 41.20 | 10.534 |
| 30 | omega_reduce_r4_o3_n2_multiply_add_zero | fa4a9da9 | 4096 | [1, 1, 512, 1536] | 6 | 3 | 0.0359 | 0.0296 to 0.0496 | 5.98 | 10.534 |
| 31 | omega_elementwise_r2_n2_fused_identity_o0__multiply_s0_o1_g10 | d28d35ec | 8960 | [1, 8960] | 1 | 3 | 0.0284 | 0.0251 to 0.0603 | 28.40 | 10.556 |
| 32 | omega_reduce_r3_o2_n2_multiply_add_zero_epi3_fused_identity_o3__multiply_s0_o2__add_s1_o1__squar | 3d3a7294 | 128 | [1, 1, 512] | 3 | 4 | 0.0277 | 0.0147 to 0.0483 | 9.23 | 10.498 |
| 33 | omega_reduce_r3_o2_n2_multiply_add_zero_epi4_fused_identity_o4__multiply_s0_o3__add_s1_o2__squar | 2417697c | 128 | [1, 1, 512] | 3 | 4 | 0.0240 | -0.0172 to 0.0505 | 8.02 | 10.466 |
| 34 | omega_reduce_r2_o1_n1_identity_maximum_negative_infinity | a358f4cb | 128 | [1, 512] | 1 | 3 | 0.0225 | 0.0099 to 0.0331 | 22.50 | 10.521 |
| 35 | omega_reduce_r3_o2_n1_identity_maximum_negative_infinity | 565c16a0 | 65536 | [1, 512, 512] | 1 | 4 | 0.0174 | 0.0046 to 0.0555 | 17.35 | 10.532 |
| 36 | omega_reduce_r2_o1_n1_identity_minimum_positive_infinity | 2b18e2e4 | 128 | [1, 512] | 1 | 4 | 0.0161 | -0.0448 to 0.0316 | 16.10 | 10.465 |
| 37 | omega_elementwise_r2_n2_fused_identity_o0__multiply_s0_o1_g10 | d28d35ec | 1536 | [1, 1536] | 1 | 3 | 0.0123 | -0.0115 to 0.0843 | 12.30 | 10.525 |
| 38 | omega_elementwise_r1_n2_multiply | c5fcdbd8 | 512 | [512] | 1 | 3 | 0.0098 | 0.0014 to 0.0711 | 9.80 | 10.461 |
| 39 | omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__add_s0_s1_fused_multiply_o0_o | 96b2c794 | 128 | [1, 1, 128] | 12 | 3 | 0.0095 | -0.0029 to 0.0332 | 0.79 | 10.581 |
| 40 | omega_reduce_r3_o2_n5_fused_equal_o0_o1__add_o2_o3__select_s0_s1_o4_minimum_positive_infinity | 8f117d35 | 65536 | [1, 512, 512] | 1 | 3 | 0.0059 | -0.0114 to 0.0281 | 5.90 | 10.532 |
| 41 | omega_reduce_r3_o2_n4_fused_greater_o0_o1__greater_o2_o3__multiply_s0_s1_add_zero | b7329559 | 65536 | [1, 512, 512] | 1 | 3 | 0.0044 | -0.0165 to 0.0104 | 4.40 | 10.537 |
| 42 | omega_iota_r1 | 2ed77224 | 512 | [512] | 2 | 4 | 0.0033 | -0.0067 to 0.0770 | 1.67 | 10.504 |
| 43 | omega_elementwise_r2_n1_identity_g1 | 6ea595c3 | 1536 | [1, 1536] | 1 | 3 | 0.0001 | -0.0072 to 0.0146 | 0.10 | 10.524 |
| 44 | omega_elementwise_r3_n1_identity | efb05eff | 262144 | [1, 512, 512] | 1 | 3 | -0.0043 | -0.0167 to 0.0397 | -4.30 | 10.533 |
| 45 | omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__add_s0_s1_fused_multiply_o0_o | 96b2c794 | 256 | [1, 1, 256] | 3 | 3 | -0.0415 | -0.0553 to -0.0075 | -13.83 | 10.521 |
