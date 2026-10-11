in-situ cost by omission (the step with the group removed against the whole step, program order, one command buffer, no cache flush); us per dispatch, per process p1 / p2 / p3
| entry | extents | dispatches | base us per dispatch | tip us per dispatch | delta per dispatch (median of per-process pairs) | delta per step us (x dispatches) |
|---|---|---|---|---|---|---|
| omega_cached_attention_q1_h8_g2_d64_s3c800000_ln9223372036854775 | [1, 8, 2, 64] | 24 | 48.23 / 36.77 / 23.89 | 23.98 / 24.92 / 19.67 | -11.85 | -284.4 |
| omega_cached_attention_q1_h8_g2_d64_s3c800000_ln9223372036854775 | [1, 8, 2, 64] | 24 | 4.13 / 9.62 / 18.63 | 11.61 / 18.40 / 10.49 | +7.48 | +179.5 |
| omega_elementwise_r1_n2_multiply | [113] | 1 | 65.21 / -171.75 / 0.67 | -174.50 / 84.17 / 151.75 | +151.08 | +151.1 |
| omega_elementwise_r2_n1_identity_g1 | [1, 1024] | 1 | 127.08 / -6.00 / 8.25 | -815.33 / -148.29 / -199.87 | -208.12 | -208.1 |
| omega_elementwise_r2_n2_fused_identity_o0__multiply_s0_o1_g10 | [1, 1024] | 1 | 40.00 / 29.96 / -189.92 | 827.67 / -11.79 / -533.33 | -41.75 | -41.8 |
| omega_elementwise_r3_n1_identity | [1, 113, 435] | 1 | 229.83 / 84.62 / -285.75 | -2264.79 / 183.92 / -96.83 | +99.30 | +99.3 |
| omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o | [1, 8, 32] | 24 | -25.53 / 10.06 / 3.70 | 0.94 / 4.79 / 31.22 | +26.47 | +635.3 |
| omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o | [1, 16, 32] | 24 | -61.39 / -3.65 / 9.14 | 5.61 / 3.87 / 2.04 | +7.52 | +180.5 |
| omega_iota_r1 | [113] | 1 | -751.87 / -80.58 / -62.46 | -142.04 / 277.58 / -199.50 | +358.16 | +358.2 |
| omega_iota_r1 | [435] | 1 | -199.67 / -27.42 / -131.13 | 57.71 / 403.00 / -609.25 | +257.38 | +257.4 |
| omega_moe_topk_e32_k8_stacked | [1] | 24 | 9.98 / 9.25 / 7.45 | 44.64 / 8.24 / 9.85 | +2.40 | +57.6 |
| omega_reduce_r2_o1_n1_identity_add_zero | [1, 113] | 1 | 125.38 / 34.62 / 0.54 | 783.88 / 2.50 / -327.08 | -32.12 | -32.1 |
| omega_reduce_r2_o1_n1_identity_maximum_negative_infinity | [1, 113] | 1 | 714.04 / -10.96 / 7.96 | -582.08 / -4.71 / 411.83 | +6.25 | +6.3 |
| omega_reduce_r2_o1_n1_identity_minimum_positive_infinity | [1, 113] | 1 | -538.96 / 3.00 / -2.58 | -87.08 / 243.83 / -39.00 | +240.83 | +240.8 |
| omega_reduce_r2_o1_n2_multiply_add_zero_epi4_fused_identity_o4__ | [1, 1024] | 49 | 13.03 / 11.60 / 10.47 | 8.36 / 15.61 / 12.14 | +1.67 | +81.8 |
| omega_reduce_r3_o2_n1_identity_maximum_negative_infinity | [1, 113, 435] | 1 | 24.21 / 215.00 / 211.17 | 1124.17 / 168.58 / -134.08 | -46.42 | -46.4 |
| omega_reduce_r3_o2_n2_multiply_add_zero | [1, 1024, 32] | 24 | 10.60 / 19.74 / 5.60 | 64.98 / 6.41 / 5.26 | -0.34 | -8.2 |
| omega_reduce_r3_o2_n2_multiply_add_zero_epi10_fused_identity_o10 | [1, 8, 1024] | 24 | 50.78 / 16.22 / 10.04 | 15.56 / 11.60 / 13.02 | -4.62 | -110.9 |
| omega_reduce_r3_o2_n2_multiply_add_zero_epi1_fused_identity_o1__ | [1, 1024, 49155] | 1 | 138.50 / 158.04 / 349.92 | 414.29 / 657.75 / 475.21 | +275.79 | +275.8 |
| omega_reduce_r3_o2_n4_fused_greater_o0_o1__greater_o2_o3__multip | [1, 113, 435] | 1 | -112.25 / -87.25 / 305.04 | -19.38 / 151.46 / -255.75 | +92.87 | +92.9 |
| omega_reduce_r3_o2_n5_fused_equal_o0_o1__add_o2_o3__select_s0_s1 | [1, 113, 435] | 1 | 31.46 / 78.87 / 6.58 | -19.62 / -156.96 / 159.92 | -51.08 | -51.1 |
| omega_reduce_r4_o3_n2_multiply_add_zero | [1, 16, 64, 1024] | 24 | -42.91 / 26.85 / 8.04 | 13.21 / 7.75 / 17.49 | +9.45 | +226.8 |
| omega_reduce_r4_o3_n2_multiply_add_zero | [1, 8, 64, 1024] | 48 | 21.83 / 7.33 / 7.15 | 10.16 / 13.93 / 8.03 | +0.88 | +42.2 |
| omega_reduce_r4_o3_n2_multiply_add_zero_epi3_fused_identity_o3__ | [1, 8, 1024, 512] | 24 | 48.57 / 32.65 / 33.75 | 60.38 / 51.21 / 41.16 | +11.81 | +283.4 |
| omega_reduce_r4_o3_n2_multiply_add_zero_g10 | [1, 8, 1024, 512] | 24 | 25.91 / 20.28 / 24.52 | 34.60 / 47.82 / 32.53 | +8.69 | +208.6 |
| omega_reduce_r4_o3_n2_multiply_add_zero_g10 | [1, 8, 512, 1024] | 24 | 53.18 / 52.97 / 31.41 | 59.02 / 39.72 / 45.12 | +5.84 | +140.2 |
| omega_reduce_r5_o2_n2_multiply_add_zero_epi2_fused_identity_o2__ | [1, 8, 2, 64, 1024] | 24 | -1.08 / 18.48 / 15.66 | 14.36 / 20.50 / 6.33 | +2.02 | +48.5 |
sum over groups: base 6404.3 us (sum of group costs, not the step), delta +2783.1 us
