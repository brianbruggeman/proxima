proxima timed dispatches=359 sum(own cb)=396.29 ms
llama ops per request=1051 sum(own cb)=157.56 ms floor_us=4

| class | proxima ops | proxima ms (own cb) | proxima ms (marginal) | llama ops | llama ms (own cb) | llama ms (net of floor) | gap ms (own cb) | ratio |
|---|---|---|---|---|---|---|---|---|
| matmul (weights) | 218 | 330.09 | 327.24 | 381 | 131.89 | 130.37 | +198.20 | 2.50x |
| elementwise, copy, other | 65 | 35.66 | 35.51 | 383 | 9.27 | 7.73 | +26.39 | 3.85x |
| attention core | 27 | 27.68 | 27.21 | 95 | 12.93 | 12.55 | +14.76 | 2.14x |
| rms norm | 49 | 2.87 | 2.68 | 96 | 1.77 | 1.39 | +1.10 | 1.62x |
| moe routing | 24 | 0.00 | 0.00 | 0 | 0.00 | 0.00 | +0.00 | n/a |
| rope | 0 | 0.00 | 0.00 | 96 | 1.71 | 1.32 | -1.71 | 0.00x |

| matmul shape | proxima ops | proxima ms (own cb) | llama ops | llama ms (own cb) | gap ms | ratio |
|---|---|---|---|---|---|---|
| Q8_0 4194304 | 72 | 194.88 | 0 | 0.00 | +194.88 | n/a |
| F32 8192 | 24 | 99.23 | 0 | 0.00 | +99.23 | n/a |
| F32 32768 (1024x32) | 24 | 11.55 | 47 | 2.17 | +9.37 | 5.31x |
| Q8_0 1048576 (1024x1024) | 48 | 16.09 | 96 | 15.08 | +1.01 | 1.07x |
| F32 49155 | 1 | 0.01 | 0 | 0.00 | +0.01 | n/a |
| Q8_0 50334720 (1024x49155) | 1 | 0.16 | 1 | 0.22 | -0.06 | 0.71x |
| Q8_0 524288 (1024x512) | 48 | 8.17 | 237 | 114.41 | -106.24 | 0.07x |

| proxima census label | ops | ms (own cb) | ms (marginal) | us per op (own cb) |
|---|---|---|---|---|
| matvec Q8_0 | 169 | 219.30 | 216.76 | 1297.7 |
| matvec f32 | 49 | 110.79 | 110.47 | 2260.9 |
| cached attention partial | 24 | 27.66 | 27.20 | 1152.6 |
| UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__subtract_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__add_s0_s1 | 24 | 23.37 | 23.33 | 973.9 |
| UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__add_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__subtract_s0_s1 | 24 | 11.83 | 11.77 | 492.9 |
| RMSNorm sumsq + fused epilogue | 49 | 2.87 | 2.68 | 58.5 |
| elementwise fused_identity_o0__multiply_s0_o1_g10 | 1 | 0.37 | 0.37 | 372.0 |
| constant | 9 | 0.04 | 0.02 | 4.7 |
| softmax max | 2 | 0.01 | 0.01 | 6.1 |
| UNCLASSIFIED omega_reduce_r3_o2_n5_fused_equal_o0_o1__add_o2_o3__select_s0_s1_o4_minimum_positive_infinity | 1 | 0.01 | 0.01 | 9.8 |
| iota | 2 | 0.01 | 0.00 | 4.2 |
| elementwise identity_g1 | 1 | 0.01 | 0.00 | 6.4 |

