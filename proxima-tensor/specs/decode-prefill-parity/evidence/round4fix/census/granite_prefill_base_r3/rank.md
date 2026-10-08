proxima timed dispatches=383 sum(own cb)=200.02 ms
llama ops per request=1051 sum(own cb)=157.56 ms floor_us=4

| class | proxima ops | proxima ms (own cb) | proxima ms (marginal) | llama ops | llama ms (own cb) | llama ms (net of floor) | gap ms (own cb) | ratio |
|---|---|---|---|---|---|---|---|---|
| matmul (weights) | 218 | 168.89 | 168.20 | 381 | 131.89 | 130.37 | +37.00 | 1.28x |
| attention core | 27 | 26.24 | 26.33 | 95 | 12.93 | 12.55 | +13.32 | 2.03x |
| rms norm | 49 | 2.90 | 2.71 | 96 | 1.77 | 1.39 | +1.13 | 1.64x |
| moe routing | 24 | 0.31 | 0.21 | 0 | 0.00 | 0.00 | +0.31 | n/a |
| rope | 0 | 0.00 | 0.00 | 96 | 1.71 | 1.32 | -1.71 | 0.00x |
| elementwise, copy, other | 65 | 1.68 | 1.48 | 383 | 9.27 | 7.73 | -7.59 | 0.18x |

| matmul shape | proxima ops | proxima ms (own cb) | llama ops | llama ms (own cb) | gap ms | ratio |
|---|---|---|---|---|---|---|
| Q8_0 4194304 | 72 | 135.35 | 0 | 0.00 | +135.35 | n/a |
| F32 8192 | 24 | 5.30 | 0 | 0.00 | +5.30 | n/a |
| F32 32768 (1024x32) | 24 | 5.00 | 47 | 2.17 | +2.83 | 2.30x |
| Q8_0 1048576 (1024x1024) | 48 | 15.10 | 96 | 15.08 | +0.02 | 1.00x |
| F32 49155 | 1 | 0.01 | 0 | 0.00 | +0.01 | n/a |
| Q8_0 50334720 (1024x49155) | 1 | 0.16 | 1 | 0.22 | -0.06 | 0.73x |
| Q8_0 524288 (1024x512) | 48 | 7.96 | 237 | 114.41 | -106.45 | 0.07x |

| proxima census label | ops | ms (own cb) | ms (marginal) | us per op (own cb) |
|---|---|---|---|---|
| matvec Q8_0 | 169 | 158.58 | 158.07 | 938.3 |
| cached attention partial | 24 | 26.23 | 26.33 | 1092.7 |
| matvec f32 | 49 | 10.31 | 10.14 | 210.4 |
| RMSNorm sumsq + fused epilogue | 49 | 2.90 | 2.71 | 59.2 |
| UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__subtract_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__add_s0_s1 | 24 | 0.77 | 0.68 | 32.1 |
| UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__add_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__subtract_s0_s1 | 24 | 0.47 | 0.40 | 19.7 |
| elementwise fused_identity_o0__multiply_s0_o1_g10 | 1 | 0.36 | 0.36 | 362.5 |
| UNCLASSIFIED omega_moe_topk_e32_k8_stacked | 24 | 0.31 | 0.21 | 12.8 |
| constant | 9 | 0.04 | 0.02 | 3.9 |
| softmax max | 2 | 0.01 | 0.01 | 6.3 |
| UNCLASSIFIED omega_reduce_r3_o2_n5_fused_equal_o0_o1__add_o2_o3__select_s0_s1_o4_minimum_positive_infinity | 1 | 0.01 | 0.01 | 10.4 |
| iota | 2 | 0.01 | 0.00 | 4.0 |

