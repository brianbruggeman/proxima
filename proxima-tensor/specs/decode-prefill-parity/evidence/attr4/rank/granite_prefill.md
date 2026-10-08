proxima timed dispatches=383 sum(own cb)=231.06 ms
llama ops per request=1051 sum(own cb)=157.56 ms floor_us=4

| class | proxima ops | proxima ms (own cb) | proxima ms (marginal) | llama ops | llama ms (own cb) | llama ms (net of floor) | gap ms (own cb) | ratio |
|---|---|---|---|---|---|---|---|---|
| matmul (weights) | 218 | 198.63 | 198.17 | 381 | 131.89 | 130.37 | +66.74 | 1.51x |
| attention core | 27 | 27.50 | 27.35 | 95 | 12.93 | 12.55 | +14.58 | 2.13x |
| rms norm | 49 | 2.92 | 2.71 | 96 | 1.77 | 1.39 | +1.15 | 1.65x |
| moe routing | 24 | 0.30 | 0.24 | 0 | 0.00 | 0.00 | +0.30 | n/a |
| rope | 0 | 0.00 | 0.00 | 96 | 1.71 | 1.32 | -1.71 | 0.00x |
| elementwise, copy, other | 65 | 1.71 | 1.56 | 383 | 9.27 | 7.73 | -7.56 | 0.18x |

| matmul shape | proxima ops | proxima ms (own cb) | llama ops | llama ms (own cb) | gap ms | ratio |
|---|---|---|---|---|---|---|
| Q8_0 4194304 | 72 | 165.00 | 0 | 0.00 | +165.00 | n/a |
| F32 8192 | 24 | 5.31 | 0 | 0.00 | +5.31 | n/a |
| F32 32768 (1024x32) | 24 | 5.08 | 47 | 2.17 | +2.91 | 2.34x |
| Q8_0 1048576 (1024x1024) | 48 | 15.12 | 96 | 15.08 | +0.03 | 1.00x |
| F32 49155 | 1 | 0.01 | 0 | 0.00 | +0.01 | n/a |
| Q8_0 50334720 (1024x49155) | 1 | 0.16 | 1 | 0.22 | -0.06 | 0.73x |
| Q8_0 524288 (1024x512) | 48 | 7.94 | 237 | 114.41 | -106.47 | 0.07x |

| proxima census label | ops | ms (own cb) | ms (marginal) | us per op (own cb) |
|---|---|---|---|---|
| matvec Q8_0 | 169 | 188.22 | 187.93 | 1113.7 |
| cached attention partial | 24 | 27.49 | 27.35 | 1145.2 |
| matvec f32 | 49 | 10.41 | 10.24 | 212.4 |
| RMSNorm sumsq + fused epilogue | 49 | 2.92 | 2.71 | 59.6 |
| UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__subtract_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__add_s0_s1 | 24 | 0.78 | 0.72 | 32.4 |
| UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__add_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__subtract_s0_s1 | 24 | 0.47 | 0.43 | 19.8 |
| elementwise fused_identity_o0__multiply_s0_o1_g10 | 1 | 0.36 | 0.36 | 364.1 |
| UNCLASSIFIED omega_moe_topk_e32_k8_stacked | 24 | 0.30 | 0.24 | 12.5 |
| constant | 9 | 0.05 | 0.02 | 5.6 |
| softmax max | 2 | 0.01 | 0.01 | 6.3 |
| UNCLASSIFIED omega_reduce_r3_o2_n5_fused_equal_o0_o1__add_o2_o3__select_s0_s1_o4_minimum_positive_infinity | 1 | 0.01 | 0.01 | 10.2 |
| iota | 2 | 0.01 | 0.00 | 4.1 |

