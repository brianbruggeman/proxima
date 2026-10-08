proxima timed dispatches=383 sum(own cb)=200.46 ms
llama ops per request=1051 sum(own cb)=157.56 ms floor_us=4

| class | proxima ops | proxima ms (own cb) | proxima ms (marginal) | llama ops | llama ms (own cb) | llama ms (net of floor) | gap ms (own cb) | ratio |
|---|---|---|---|---|---|---|---|---|
| matmul (weights) | 218 | 169.01 | 168.37 | 381 | 131.89 | 130.37 | +37.12 | 1.28x |
| attention core | 27 | 26.31 | 26.38 | 95 | 12.93 | 12.55 | +13.38 | 2.04x |
| rms norm | 49 | 2.90 | 2.60 | 96 | 1.77 | 1.39 | +1.13 | 1.64x |
| moe routing | 24 | 0.31 | 0.24 | 0 | 0.00 | 0.00 | +0.31 | n/a |
| rope | 0 | 0.00 | 0.00 | 96 | 1.71 | 1.32 | -1.71 | 0.00x |
| elementwise, copy, other | 65 | 1.94 | 1.50 | 383 | 9.27 | 7.73 | -7.33 | 0.21x |

| matmul shape | proxima ops | proxima ms (own cb) | llama ops | llama ms (own cb) | gap ms | ratio |
|---|---|---|---|---|---|---|
| Q8_0 4194304 | 72 | 135.33 | 0 | 0.00 | +135.33 | n/a |
| F32 8192 | 24 | 5.31 | 0 | 0.00 | +5.31 | n/a |
| F32 32768 (1024x32) | 24 | 5.09 | 47 | 2.17 | +2.91 | 2.34x |
| Q8_0 1048576 (1024x1024) | 48 | 15.17 | 96 | 15.08 | +0.08 | 1.01x |
| F32 49155 | 1 | 0.01 | 0 | 0.00 | +0.01 | n/a |
| Q8_0 50334720 (1024x49155) | 1 | 0.16 | 1 | 0.22 | -0.06 | 0.72x |
| Q8_0 524288 (1024x512) | 48 | 7.94 | 237 | 114.41 | -106.47 | 0.07x |

| proxima census label | ops | ms (own cb) | ms (marginal) | us per op (own cb) |
|---|---|---|---|---|
| matvec Q8_0 | 169 | 158.60 | 158.13 | 938.5 |
| cached attention partial | 24 | 26.29 | 26.37 | 1095.3 |
| matvec f32 | 49 | 10.41 | 10.24 | 212.4 |
| RMSNorm sumsq + fused epilogue | 49 | 2.90 | 2.60 | 59.2 |
| UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__subtract_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__add_s0_s1 | 24 | 0.77 | 0.72 | 32.2 |
| elementwise fused_identity_o0__multiply_s0_o1_g10 | 1 | 0.60 | 0.34 | 598.4 |
| UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__add_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__subtract_s0_s1 | 24 | 0.48 | 0.39 | 19.9 |
| UNCLASSIFIED omega_moe_topk_e32_k8_stacked | 24 | 0.31 | 0.24 | 12.8 |
| constant | 9 | 0.04 | 0.02 | 4.9 |
| softmax max | 2 | 0.01 | 0.01 | 6.2 |
| UNCLASSIFIED omega_reduce_r3_o2_n5_fused_equal_o0_o1__add_o2_o3__select_s0_s1_o4_minimum_positive_infinity | 1 | 0.01 | 0.01 | 11.0 |
| iota | 2 | 0.01 | 0.00 | 4.1 |

