proxima timed dispatches=383 sum(own cb)=199.89 ms
llama ops per request=1051 sum(own cb)=157.56 ms floor_us=4

| class | proxima ops | proxima ms (own cb) | proxima ms (marginal) | llama ops | llama ms (own cb) | llama ms (net of floor) | gap ms (own cb) | ratio |
|---|---|---|---|---|---|---|---|---|
| matmul (weights) | 218 | 169.00 | 168.58 | 381 | 131.89 | 130.37 | +37.11 | 1.28x |
| attention core | 27 | 26.01 | 26.34 | 95 | 12.93 | 12.55 | +13.08 | 2.01x |
| rms norm | 49 | 2.91 | 2.70 | 96 | 1.77 | 1.39 | +1.13 | 1.64x |
| moe routing | 24 | 0.29 | 0.21 | 0 | 0.00 | 0.00 | +0.29 | n/a |
| rope | 0 | 0.00 | 0.00 | 96 | 1.71 | 1.32 | -1.71 | 0.00x |
| elementwise, copy, other | 65 | 1.69 | 1.56 | 383 | 9.27 | 7.73 | -7.57 | 0.18x |

| matmul shape | proxima ops | proxima ms (own cb) | llama ops | llama ms (own cb) | gap ms | ratio |
|---|---|---|---|---|---|---|
| Q8_0 4194304 | 72 | 135.33 | 0 | 0.00 | +135.33 | n/a |
| F32 8192 | 24 | 5.31 | 0 | 0.00 | +5.31 | n/a |
| F32 32768 (1024x32) | 24 | 5.10 | 47 | 2.17 | +2.93 | 2.35x |
| Q8_0 1048576 (1024x1024) | 48 | 15.12 | 96 | 15.08 | +0.04 | 1.00x |
| F32 49155 | 1 | 0.01 | 0 | 0.00 | +0.01 | n/a |
| Q8_0 50334720 (1024x49155) | 1 | 0.16 | 1 | 0.22 | -0.06 | 0.73x |
| Q8_0 524288 (1024x512) | 48 | 7.96 | 237 | 114.41 | -106.45 | 0.07x |

| proxima census label | ops | ms (own cb) | ms (marginal) | us per op (own cb) |
|---|---|---|---|---|
| matvec Q8_0 | 169 | 158.57 | 158.30 | 938.3 |
| cached attention partial | 24 | 25.99 | 26.33 | 1083.0 |
| matvec f32 | 49 | 10.42 | 10.28 | 212.7 |
| RMSNorm sumsq + fused epilogue | 49 | 2.91 | 2.70 | 59.3 |
| UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__subtract_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__add_s0_s1 | 24 | 0.77 | 0.72 | 32.2 |
| UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__add_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__subtract_s0_s1 | 24 | 0.48 | 0.44 | 20.0 |
| elementwise fused_identity_o0__multiply_s0_o1_g10 | 1 | 0.36 | 0.36 | 363.1 |
| UNCLASSIFIED omega_moe_topk_e32_k8_stacked | 24 | 0.29 | 0.21 | 12.0 |
| constant | 9 | 0.04 | 0.02 | 4.0 |
| softmax max | 2 | 0.01 | 0.01 | 6.2 |
| UNCLASSIFIED omega_reduce_r3_o2_n5_fused_equal_o0_o1__add_o2_o3__select_s0_s1_o4_minimum_positive_infinity | 1 | 0.01 | 0.01 | 10.5 |
| iota | 2 | 0.01 | 0.00 | 3.9 |

