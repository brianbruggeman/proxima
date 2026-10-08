proxima timed dispatches=383 sum(own cb)=203.54 ms
llama ops per request=1051 sum(own cb)=157.56 ms floor_us=4

| class | proxima ops | proxima ms (own cb) | proxima ms (marginal) | llama ops | llama ms (own cb) | llama ms (net of floor) | gap ms (own cb) | ratio |
|---|---|---|---|---|---|---|---|---|
| matmul (weights) | 218 | 167.72 | 167.38 | 381 | 131.89 | 130.37 | +35.83 | 1.27x |
| attention core | 27 | 30.88 | 30.55 | 95 | 12.93 | 12.55 | +17.95 | 2.39x |
| rms norm | 49 | 2.93 | 2.69 | 96 | 1.77 | 1.39 | +1.16 | 1.65x |
| moe routing | 24 | 0.31 | 0.24 | 0 | 0.00 | 0.00 | +0.31 | n/a |
| rope | 0 | 0.00 | 0.00 | 96 | 1.71 | 1.32 | -1.71 | 0.00x |
| elementwise, copy, other | 65 | 1.71 | 1.56 | 383 | 9.27 | 7.73 | -7.56 | 0.18x |

| matmul shape | proxima ops | proxima ms (own cb) | llama ops | llama ms (own cb) | gap ms | ratio |
|---|---|---|---|---|---|---|
| Q8_0 4194304 | 72 | 134.07 | 0 | 0.00 | +134.07 | n/a |
| F32 8192 | 24 | 5.31 | 0 | 0.00 | +5.31 | n/a |
| F32 32768 (1024x32) | 24 | 5.09 | 47 | 2.17 | +2.92 | 2.34x |
| Q8_0 1048576 (1024x1024) | 48 | 15.12 | 96 | 15.08 | +0.04 | 1.00x |
| F32 49155 | 1 | 0.01 | 0 | 0.00 | +0.01 | n/a |
| Q8_0 50334720 (1024x49155) | 1 | 0.16 | 1 | 0.22 | -0.06 | 0.71x |
| Q8_0 524288 (1024x512) | 48 | 7.96 | 237 | 114.41 | -106.45 | 0.07x |

| proxima census label | ops | ms (own cb) | ms (marginal) | us per op (own cb) |
|---|---|---|---|---|
| matvec Q8_0 | 169 | 157.31 | 157.09 | 930.8 |
| cached attention partial | 24 | 30.86 | 30.54 | 1285.8 |
| matvec f32 | 49 | 10.41 | 10.29 | 212.5 |
| RMSNorm sumsq + fused epilogue | 49 | 2.93 | 2.69 | 59.8 |
| UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__subtract_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__add_s0_s1 | 24 | 0.78 | 0.73 | 32.4 |
| UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__add_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__subtract_s0_s1 | 24 | 0.49 | 0.43 | 20.3 |
| elementwise fused_identity_o0__multiply_s0_o1_g10 | 1 | 0.36 | 0.36 | 363.4 |
| UNCLASSIFIED omega_moe_topk_e32_k8_stacked | 24 | 0.31 | 0.24 | 12.8 |
| constant | 9 | 0.04 | 0.02 | 4.1 |
| softmax max | 2 | 0.01 | 0.01 | 6.2 |
| UNCLASSIFIED omega_reduce_r3_o2_n5_fused_equal_o0_o1__add_o2_o3__select_s0_s1_o4_minimum_positive_infinity | 1 | 0.01 | 0.01 | 10.4 |
| iota | 2 | 0.01 | 0.00 | 4.6 |

