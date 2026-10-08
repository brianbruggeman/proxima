proxima timed dispatches=383 sum(own cb)=198.68 ms
llama ops per request=1051 sum(own cb)=157.56 ms floor_us=4

| class | proxima ops | proxima ms (own cb) | proxima ms (marginal) | llama ops | llama ms (own cb) | llama ms (net of floor) | gap ms (own cb) | ratio |
|---|---|---|---|---|---|---|---|---|
| matmul (weights) | 218 | 167.69 | 167.21 | 381 | 131.89 | 130.37 | +35.80 | 1.27x |
| attention core | 27 | 26.06 | 26.29 | 95 | 12.93 | 12.55 | +13.13 | 2.02x |
| rms norm | 49 | 2.92 | 2.72 | 96 | 1.77 | 1.39 | +1.15 | 1.65x |
| moe routing | 24 | 0.31 | 0.24 | 0 | 0.00 | 0.00 | +0.31 | n/a |
| rope | 0 | 0.00 | 0.00 | 96 | 1.71 | 1.32 | -1.71 | 0.00x |
| elementwise, copy, other | 65 | 1.70 | 1.57 | 383 | 9.27 | 7.73 | -7.57 | 0.18x |

| matmul shape | proxima ops | proxima ms (own cb) | llama ops | llama ms (own cb) | gap ms | ratio |
|---|---|---|---|---|---|---|
| Q8_0 4194304 | 72 | 134.03 | 0 | 0.00 | +134.03 | n/a |
| F32 8192 | 24 | 5.31 | 0 | 0.00 | +5.31 | n/a |
| F32 32768 (1024x32) | 24 | 5.10 | 47 | 2.17 | +2.93 | 2.34x |
| Q8_0 1048576 (1024x1024) | 48 | 15.12 | 96 | 15.08 | +0.03 | 1.00x |
| F32 49155 | 1 | 0.01 | 0 | 0.00 | +0.01 | n/a |
| Q8_0 50334720 (1024x49155) | 1 | 0.16 | 1 | 0.22 | -0.06 | 0.72x |
| Q8_0 524288 (1024x512) | 48 | 7.97 | 237 | 114.41 | -106.44 | 0.07x |

| proxima census label | ops | ms (own cb) | ms (marginal) | us per op (own cb) |
|---|---|---|---|---|
| matvec Q8_0 | 169 | 157.27 | 156.93 | 930.6 |
| cached attention partial | 24 | 26.04 | 26.28 | 1085.1 |
| matvec f32 | 49 | 10.42 | 10.28 | 212.6 |
| RMSNorm sumsq + fused epilogue | 49 | 2.92 | 2.72 | 59.6 |
| UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__subtract_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__add_s0_s1 | 24 | 0.77 | 0.72 | 32.3 |
| UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__add_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__subtract_s0_s1 | 24 | 0.49 | 0.44 | 20.2 |
| elementwise fused_identity_o0__multiply_s0_o1_g10 | 1 | 0.36 | 0.36 | 363.6 |
| UNCLASSIFIED omega_moe_topk_e32_k8_stacked | 24 | 0.31 | 0.24 | 12.7 |
| constant | 9 | 0.04 | 0.02 | 4.0 |
| softmax max | 2 | 0.01 | 0.01 | 6.3 |
| UNCLASSIFIED omega_reduce_r3_o2_n5_fused_equal_o0_o1__add_o2_o3__select_s0_s1_o4_minimum_positive_infinity | 1 | 0.01 | 0.00 | 9.8 |
| iota | 2 | 0.01 | 0.00 | 3.8 |

