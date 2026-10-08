proxima timed dispatches=383 sum(own cb)=200.72 ms
llama ops per request=1051 sum(own cb)=157.56 ms floor_us=4

| class | proxima ops | proxima ms (own cb) | proxima ms (marginal) | llama ops | llama ms (own cb) | llama ms (net of floor) | gap ms (own cb) | ratio |
|---|---|---|---|---|---|---|---|---|
| matmul (weights) | 218 | 169.30 | 168.74 | 381 | 131.89 | 130.37 | +37.41 | 1.28x |
| attention core | 27 | 26.49 | 26.68 | 95 | 12.93 | 12.55 | +13.56 | 2.05x |
| rms norm | 49 | 2.92 | 2.72 | 96 | 1.77 | 1.39 | +1.15 | 1.65x |
| moe routing | 24 | 0.30 | 0.24 | 0 | 0.00 | 0.00 | +0.30 | n/a |
| rope | 0 | 0.00 | 0.00 | 96 | 1.71 | 1.32 | -1.71 | 0.00x |
| elementwise, copy, other | 65 | 1.71 | 1.57 | 383 | 9.27 | 7.73 | -7.56 | 0.18x |

| matmul shape | proxima ops | proxima ms (own cb) | llama ops | llama ms (own cb) | gap ms | ratio |
|---|---|---|---|---|---|---|
| Q8_0 4194304 | 72 | 135.10 | 0 | 0.00 | +135.10 | n/a |
| F32 8192 | 24 | 5.34 | 0 | 0.00 | +5.34 | n/a |
| F32 32768 (1024x32) | 24 | 5.02 | 47 | 2.17 | +2.84 | 2.31x |
| Q8_0 1048576 (1024x1024) | 48 | 15.69 | 96 | 15.08 | +0.61 | 1.04x |
| F32 49155 | 1 | 0.01 | 0 | 0.00 | +0.01 | n/a |
| Q8_0 50334720 (1024x49155) | 1 | 0.16 | 1 | 0.22 | -0.06 | 0.72x |
| Q8_0 524288 (1024x512) | 48 | 7.99 | 237 | 114.41 | -106.42 | 0.07x |

| proxima census label | ops | ms (own cb) | ms (marginal) | us per op (own cb) |
|---|---|---|---|---|
| matvec Q8_0 | 169 | 158.94 | 158.51 | 940.5 |
| cached attention partial | 24 | 26.47 | 26.67 | 1102.9 |
| matvec f32 | 49 | 10.36 | 10.23 | 211.5 |
| RMSNorm sumsq + fused epilogue | 49 | 2.92 | 2.72 | 59.6 |
| UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__subtract_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__add_s0_s1 | 24 | 0.78 | 0.73 | 32.4 |
| UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__add_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__subtract_s0_s1 | 24 | 0.49 | 0.44 | 20.3 |
| elementwise fused_identity_o0__multiply_s0_o1_g10 | 1 | 0.36 | 0.36 | 363.6 |
| UNCLASSIFIED omega_moe_topk_e32_k8_stacked | 24 | 0.30 | 0.24 | 12.5 |
| constant | 9 | 0.04 | 0.02 | 4.2 |
| softmax max | 2 | 0.01 | 0.01 | 6.1 |
| UNCLASSIFIED omega_reduce_r3_o2_n5_fused_equal_o0_o1__add_o2_o3__select_s0_s1_o4_minimum_positive_infinity | 1 | 0.01 | 0.01 | 10.6 |
| iota | 2 | 0.01 | 0.00 | 4.3 |

