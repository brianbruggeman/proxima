proxima timed dispatches=383 sum(own cb)=230.90 ms
llama ops per request=1051 sum(own cb)=157.56 ms floor_us=4

| class | proxima ops | proxima ms (own cb) | proxima ms (marginal) | llama ops | llama ms (own cb) | llama ms (net of floor) | gap ms (own cb) | ratio |
|---|---|---|---|---|---|---|---|---|
| matmul (weights) | 218 | 198.52 | 198.21 | 381 | 131.89 | 130.37 | +66.63 | 1.51x |
| attention core | 27 | 27.43 | 27.36 | 95 | 12.93 | 12.55 | +14.51 | 2.12x |
| rms norm | 49 | 2.93 | 2.70 | 96 | 1.77 | 1.39 | +1.16 | 1.66x |
| moe routing | 24 | 0.31 | 0.25 | 0 | 0.00 | 0.00 | +0.31 | n/a |
| rope | 0 | 0.00 | 0.00 | 96 | 1.71 | 1.32 | -1.71 | 0.00x |
| elementwise, copy, other | 65 | 1.71 | 1.56 | 383 | 9.27 | 7.73 | -7.56 | 0.18x |

| matmul shape | proxima ops | proxima ms (own cb) | llama ops | llama ms (own cb) | gap ms | ratio |
|---|---|---|---|---|---|---|
| Q8_0 4194304 | 72 | 164.84 | 0 | 0.00 | +164.84 | n/a |
| F32 8192 | 24 | 5.31 | 0 | 0.00 | +5.31 | n/a |
| F32 32768 (1024x32) | 24 | 5.10 | 47 | 2.17 | +2.92 | 2.34x |
| Q8_0 1048576 (1024x1024) | 48 | 15.15 | 96 | 15.08 | +0.07 | 1.00x |
| F32 49155 | 1 | 0.01 | 0 | 0.00 | +0.01 | n/a |
| Q8_0 50334720 (1024x49155) | 1 | 0.16 | 1 | 0.22 | -0.06 | 0.71x |
| Q8_0 524288 (1024x512) | 48 | 7.96 | 237 | 114.41 | -106.45 | 0.07x |

| proxima census label | ops | ms (own cb) | ms (marginal) | us per op (own cb) |
|---|---|---|---|---|
| matvec Q8_0 | 169 | 188.11 | 187.91 | 1113.1 |
| cached attention partial | 24 | 27.42 | 27.36 | 1142.3 |
| matvec f32 | 49 | 10.41 | 10.30 | 212.5 |
| RMSNorm sumsq + fused epilogue | 49 | 2.93 | 2.70 | 59.8 |
| UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__subtract_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__add_s0_s1 | 24 | 0.78 | 0.72 | 32.4 |
| UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__add_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__subtract_s0_s1 | 24 | 0.49 | 0.44 | 20.2 |
| elementwise fused_identity_o0__multiply_s0_o1_g10 | 1 | 0.36 | 0.36 | 363.2 |
| UNCLASSIFIED omega_moe_topk_e32_k8_stacked | 24 | 0.31 | 0.25 | 12.9 |
| constant | 9 | 0.04 | 0.02 | 4.1 |
| softmax max | 2 | 0.01 | 0.01 | 6.8 |
| UNCLASSIFIED omega_reduce_r3_o2_n5_fused_equal_o0_o1__add_o2_o3__select_s0_s1_o4_minimum_positive_infinity | 1 | 0.01 | 0.01 | 10.6 |
| iota | 2 | 0.01 | 0.00 | 4.4 |

