in-situ cost by omission (the step with the group removed against the whole step, program order, one command buffer, no cache flush); us per dispatch, per process p1 / p2 / p3
| entry | extents | dispatches | base us per dispatch | tip us per dispatch | delta per dispatch (median of per-process pairs) | delta per step us (x dispatches) |
|---|---|---|---|---|---|---|
| omega_cached_attention_q1_h1_g8_d256_s3f800000_ln511_up0_x4_e1_k | [1, 1, 8, 256] | 28 | 16.17 / 14.85 / 16.50 | 16.09 / 2.88 / 16.73 | -0.08 | -2.2 |
| omega_cached_attention_q1_h1_g8_d256_s3f800000_ln511_up0_x4_e1_k | [1, 1, 8, 256] | 28 | 5.82 / 8.69 / 6.11 | 7.96 / 4.63 / 4.83 | -1.28 | -35.8 |
| omega_cached_attention_q1_h1_g8_d512_s3f800000_ln922337203685477 | [1, 1, 8, 512] | 7 | 40.06 / 35.64 / 37.56 | 36.14 / 37.95 / 36.27 | -1.29 | -9.0 |
| omega_cached_attention_q1_h1_g8_d512_s3f800000_ln922337203685477 | [1, 1, 8, 512] | 7 | 7.15 / -0.35 / 9.55 | 9.92 / 8.63 / 7.57 | +2.77 | +19.4 |
| omega_elementwise_r1_n2_multiply | [512] | 1 | 25.29 / 19.33 / 5.79 | 26.46 / -7.46 / 72.92 | +1.17 | +1.2 |
| omega_elementwise_r2_n1_identity_g1 | [1, 1536] | 1 | 3.38 / -50.67 / 23.38 | -313.54 / -22.67 / 96.46 | +28.00 | +28.0 |
| omega_elementwise_r2_n2_fused_identity_o0__multiply_s0_o1_g10 | [1, 1536] | 1 | 33.87 / 16.87 / 51.83 | 9.04 / 18.75 / -53.62 | -24.83 | -24.8 |
| omega_elementwise_r2_n2_fused_identity_o0__multiply_s0_o1_g10 | [1, 8960] | 1 | 23.83 / 40.96 / 9.04 | 13.08 / 4.50 / -17.13 | -26.17 | -26.2 |
| omega_elementwise_r3_n1_identity | [1, 35, 256] | 1 | -445.87 / 3.08 / 3.92 | -1.04 / -130.58 / 10.00 | +6.08 | +6.1 |
| omega_elementwise_r3_n1_identity | [1, 512, 512] | 1 | 51.38 / -17.63 / 33.38 | 21.71 / 1.58 / 14.58 | -18.80 | -18.8 |
| omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o | [1, 1, 128] | 12 | 3.17 / 2.32 / 1.93 | 1.51 / 5.51 / 13.94 | +3.19 | +38.3 |
| omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o | [1, 1, 256] | 3 | 5.94 / 0.58 / 4.51 | 6.67 / -177.43 / -10.46 | -14.97 | -44.9 |
| omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o | [1, 8, 128] | 28 | 3.32 / 0.46 / 3.69 | 3.69 / 4.50 / 2.37 | +0.37 | +10.4 |
| omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o | [1, 8, 256] | 7 | 2.59 / 15.36 / 2.70 | 1.98 / 0.97 / 4.55 | -0.61 | -4.3 |
| omega_iota_r1 | [512] | 2 | -6.92 / 38.02 / 1.06 | 1.44 / 23.19 / 1.06 | +0.00 | +0.0 |
| omega_reduce_r2_o1_n1_identity_add_zero | [1, 512] | 1 | -23.12 / -14.75 / -264.92 | -6.04 / -23.21 / -6.13 | +17.08 | +17.1 |
| omega_reduce_r2_o1_n1_identity_maximum_negative_infinity | [1, 512] | 1 | -16.17 / 27.71 / -9.88 | 17.21 / 5.17 / 4.50 | +14.38 | +14.4 |
| omega_reduce_r2_o1_n1_identity_minimum_positive_infinity | [1, 512] | 1 | 101.17 / -982.75 / -1.58 | -0.25 / -154.29 / 7.37 | +8.95 | +8.9 |
| omega_reduce_r2_o1_n2_multiply_add_zero_epi4_fused_identity_o4__ | [1, 1536] | 71 | 8.35 / 15.91 / 8.49 | 6.68 / 6.77 / 6.21 | -2.28 | -161.9 |
| omega_reduce_r2_o1_n2_multiply_add_zero_epi5_fused_identity_o5__ | [1, 1536] | 70 | 10.32 / 9.44 / 14.84 | 9.07 / 7.46 / 7.40 | -1.98 | -138.6 |
| omega_reduce_r2_o1_n2_multiply_add_zero_epi6_fused_identity_o6__ | [1, 1536] | 35 | 10.42 / 11.32 / 10.01 | 8.10 / 8.45 / 8.27 | -2.32 | -81.2 |
| omega_reduce_r3_o2_n1_identity_maximum_negative_infinity | [1, 512, 512] | 1 | 149.25 / 1.87 / 77.33 | -6.54 / 15.62 / 2.04 | -75.29 | -75.3 |
| omega_reduce_r3_o2_n2_multiply_add_zero | [1, 12288, 1536] | 20 | 44.20 / 44.70 / 45.01 | 42.83 / 8.19 / 43.36 | -1.65 | -33.0 |
| omega_reduce_r3_o2_n2_multiply_add_zero | [1, 1536, 12288] | 20 | 39.08 / 41.34 / 40.97 | 50.03 / 40.71 / 44.72 | +3.75 | +75.0 |
| omega_reduce_r3_o2_n2_multiply_add_zero | [1, 1536, 6144] | 15 | 24.63 / 22.10 / 23.98 | 13.19 / -28.18 / 22.26 | -11.44 | -171.6 |
| omega_reduce_r3_o2_n2_multiply_add_zero | [1, 256, 1536] | 35 | 2.03 / 3.65 / 4.44 | 14.66 / -6.39 / 5.09 | +0.65 | +22.7 |
| omega_reduce_r3_o2_n2_multiply_add_zero | [1, 6144, 1536] | 15 | 18.54 / 23.52 / 23.73 | 24.47 / 28.11 / 24.88 | +4.59 | +68.8 |
| omega_reduce_r3_o2_n2_multiply_add_zero_epi1_fused_identity_o1__ | [1, 1536, 8960] | 1 | 80.62 / 392.21 / 118.75 | 255.96 / 128.96 / 101.25 | -17.50 | -17.5 |
| omega_reduce_r3_o2_n2_multiply_add_zero_epi2_fused_identity_o2__ | [1, 1536, 262144] | 1 | 1106.29 / 1201.75 / 1113.79 | 1085.00 / 1092.71 / 1090.12 | -23.67 | -23.7 |
| omega_reduce_r3_o2_n2_multiply_add_zero_epi3_fused_identity_o3__ | [1, 1, 256] | 12 | 7.25 / 16.82 / 9.75 | 7.77 / -27.59 / 7.72 | -2.03 | -24.4 |
| omega_reduce_r3_o2_n2_multiply_add_zero_epi3_fused_identity_o3__ | [1, 1, 512] | 3 | 16.12 / 9.33 / 18.97 | 71.83 / 9.75 / 22.53 | +3.56 | +10.7 |
| omega_reduce_r3_o2_n2_multiply_add_zero_epi4_fused_identity_o4__ | [1, 1, 256] | 12 | 8.20 / 8.63 / 7.93 | 6.83 / 36.11 / 7.37 | -0.56 | -6.7 |
| omega_reduce_r3_o2_n2_multiply_add_zero_epi4_fused_identity_o4__ | [1, 1, 512] | 3 | 1.26 / 190.62 / 1.53 | -4.21 / -52.07 / -49.42 | -50.95 | -152.9 |
| omega_reduce_r3_o2_n2_multiply_add_zero_epi4_fused_identity_o4__ | [1, 8, 256] | 28 | 7.47 / 8.01 / 8.72 | 7.59 / 7.21 / 6.78 | -0.80 | -22.4 |
| omega_reduce_r3_o2_n2_multiply_add_zero_epi4_fused_identity_o4__ | [1, 8, 512] | 7 | 9.23 / 15.33 / 8.94 | 5.51 / -42.91 / -37.74 | -46.68 | -326.8 |
| omega_reduce_r3_o2_n2_multiply_add_zero_epi5_fused_identity_o5__ | [1, 1536, 256] | 35 | 16.23 / 6.58 / 8.64 | 9.18 / 16.70 / 8.98 | +0.34 | +11.9 |
| omega_reduce_r3_o2_n2_multiply_add_zero_epi6_fused_identity_o6__ | [1, 35, 256] | 1 | 38.67 / -2.92 / 904.17 | 17.62 / 42.83 / 9.92 | -21.05 | -21.1 |
| omega_reduce_r3_o2_n2_multiply_add_zero_epi9_fused_identity_o9__ | [1, 1536, 12288] | 20 | 43.91 / 45.11 / 44.42 | 42.70 / 39.48 / 41.40 | -3.02 | -60.4 |
| omega_reduce_r3_o2_n2_multiply_add_zero_epi9_fused_identity_o9__ | [1, 1536, 6144] | 15 | 24.94 / 4.86 / 25.32 | 28.19 / 34.22 / 24.22 | +3.25 | +48.8 |
| omega_reduce_r3_o2_n4_fused_greater_o0_o1__greater_o2_o3__multip | [1, 512, 512] | 1 | 8.67 / -1.46 / 0.37 | 17.63 / 1.04 / 5.67 | +5.30 | +5.3 |
| omega_reduce_r3_o2_n5_fused_equal_o0_o1__add_o2_o3__select_s0_s1 | [1, 512, 512] | 1 | 4.58 / 73.38 / 50.08 | 23.21 / 31.58 / 28.50 | -21.58 | -21.6 |
| omega_reduce_r4_o3_n2_multiply_add_zero | [1, 1, 256, 1536] | 24 | -11.46 / 6.65 / 7.35 | 7.07 / 4.59 / 6.80 | -0.55 | -13.2 |
| omega_reduce_r4_o3_n2_multiply_add_zero | [1, 1, 512, 1536] | 6 | 19.97 / 3.64 / 9.48 | 11.47 / -17.79 / 8.47 | -8.50 | -51.0 |
| omega_reduce_r4_o3_n2_multiply_add_zero | [1, 8, 256, 1536] | 28 | 13.75 / 13.99 / -0.35 | 13.16 / 11.68 / 11.77 | -0.59 | -16.5 |
| omega_reduce_r4_o3_n2_multiply_add_zero | [1, 8, 512, 1536] | 7 | 20.93 / 19.71 / 21.85 | 16.04 / 16.68 / 16.74 | -4.89 | -34.2 |
| omega_reduce_r5_o2_n2_multiply_add_zero | [1, 1, 8, 256, 1536] | 28 | 12.33 / 11.82 / 5.25 | 14.92 / 11.74 / 12.03 | +2.59 | +72.5 |
| omega_reduce_r5_o2_n2_multiply_add_zero | [1, 1, 8, 512, 1536] | 7 | 17.63 / 16.53 / 26.54 | 21.16 / 16.74 / 16.57 | +0.21 | +1.5 |
sum over groups: base 10141.5 us (sum of group costs, not the step), delta -1159.0 us
