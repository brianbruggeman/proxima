proxima timed dispatches=383 sum(own cb)=200.48 ms
llama ops per request=1051 sum(own cb)=157.56 ms floor_us=4

| class | proxima ops | proxima ms (own cb) | proxima ms (marginal) | llama ops | llama ms (own cb) | llama ms (net of floor) | gap ms (own cb) | ratio |
|---|---|---|---|---|---|---|---|---|
| matmul (weights) | 218 | 169.07 | 168.35 | 381 | 131.89 | 130.37 | +37.18 | 1.28x |
| attention core | 27 | 26.54 | 26.65 | 95 | 12.93 | 12.55 | +13.62 | 2.05x |
| rms norm | 49 | 2.88 | 2.64 | 96 | 1.77 | 1.39 | +1.11 | 1.63x |
| moe routing | 24 | 0.30 | 0.22 | 0 | 0.00 | 0.00 | +0.30 | n/a |
| rope | 0 | 0.00 | 0.00 | 96 | 1.71 | 1.32 | -1.71 | 0.00x |
| elementwise, copy, other | 65 | 1.69 | 1.47 | 383 | 9.27 | 7.73 | -7.58 | 0.18x |

| matmul shape | proxima ops | proxima ms (own cb) | llama ops | llama ms (own cb) | gap ms | ratio |
|---|---|---|---|---|---|---|
| Q8_0 4194304 | 72 | 134.95 | 0 | 0.00 | +134.95 | n/a |
| F32 8192 | 24 | 5.33 | 0 | 0.00 | +5.33 | n/a |
| F32 32768 (1024x32) | 24 | 5.07 | 47 | 2.17 | +2.90 | 2.33x |
| Q8_0 1048576 (1024x1024) | 48 | 15.61 | 96 | 15.08 | +0.53 | 1.04x |
| F32 49155 | 1 | 0.01 | 0 | 0.00 | +0.01 | n/a |
| Q8_0 50334720 (1024x49155) | 1 | 0.16 | 1 | 0.22 | -0.06 | 0.72x |
| Q8_0 524288 (1024x512) | 48 | 7.94 | 237 | 114.41 | -106.47 | 0.07x |

| proxima census label | ops | ms (own cb) | ms (marginal) | us per op (own cb) |
|---|---|---|---|---|
| matvec Q8_0 | 169 | 158.66 | 158.17 | 938.8 |
| cached attention partial | 24 | 26.52 | 26.64 | 1105.1 |
| matvec f32 | 49 | 10.41 | 10.17 | 212.5 |
| RMSNorm sumsq + fused epilogue | 49 | 2.88 | 2.64 | 58.8 |
| UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__subtract_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__add_s0_s1 | 24 | 0.77 | 0.68 | 31.9 |
| UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__add_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__subtract_s0_s1 | 24 | 0.48 | 0.39 | 20.0 |
| elementwise fused_identity_o0__multiply_s0_o1_g10 | 1 | 0.36 | 0.36 | 364.1 |
| UNCLASSIFIED omega_moe_topk_e32_k8_stacked | 24 | 0.30 | 0.22 | 12.7 |
| constant | 9 | 0.04 | 0.02 | 3.9 |
| softmax max | 2 | 0.01 | 0.01 | 6.2 |
| UNCLASSIFIED omega_reduce_r3_o2_n5_fused_equal_o0_o1__add_o2_o3__select_s0_s1_o4_minimum_positive_infinity | 1 | 0.01 | 0.01 | 10.6 |
| iota | 2 | 0.01 | 0.00 | 4.0 |

