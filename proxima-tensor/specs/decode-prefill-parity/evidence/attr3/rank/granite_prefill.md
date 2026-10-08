proxima timed dispatches=359 sum(own cb)=285.10 ms
llama ops per request=1051 sum(own cb)=157.56 ms floor_us=4

| class | proxima ops | proxima ms (own cb) | proxima ms (marginal) | llama ops | llama ms (own cb) | llama ms (net of floor) | gap ms (own cb) | ratio |
|---|---|---|---|---|---|---|---|---|
| matmul (weights) | 218 | 218.41 | 219.60 | 381 | 131.89 | 130.37 | +86.52 | 1.66x |
| elementwise, copy, other | 65 | 36.19 | 36.02 | 383 | 9.27 | 7.73 | +26.92 | 3.91x |
| attention core | 27 | 27.59 | 27.24 | 95 | 12.93 | 12.55 | +14.67 | 2.13x |
| rms norm | 49 | 2.90 | 2.60 | 96 | 1.77 | 1.39 | +1.13 | 1.64x |
| moe routing | 24 | 0.00 | 0.00 | 0 | 0.00 | 0.00 | +0.00 | n/a |
| rope | 0 | 0.00 | 0.00 | 96 | 1.71 | 1.32 | -1.71 | 0.00x |

| matmul shape | proxima ops | proxima ms (own cb) | llama ops | llama ms (own cb) | gap ms | ratio |
|---|---|---|---|---|---|---|
| Q8_0 4194304 | 72 | 169.99 | 0 | 0.00 | +169.99 | n/a |
| F32 8192 | 24 | 13.63 | 0 | 0.00 | +13.63 | n/a |
| F32 32768 (1024x32) | 24 | 11.52 | 47 | 2.17 | +9.35 | 5.30x |
| Q8_0 1048576 (1024x1024) | 48 | 15.13 | 96 | 15.08 | +0.05 | 1.00x |
| F32 49155 | 1 | 0.01 | 0 | 0.00 | +0.01 | n/a |
| Q8_0 50334720 (1024x49155) | 1 | 0.16 | 1 | 0.22 | -0.06 | 0.72x |
| Q8_0 524288 (1024x512) | 48 | 7.97 | 237 | 114.41 | -106.44 | 0.07x |

| proxima census label | ops | ms (own cb) | ms (marginal) | us per op (own cb) |
|---|---|---|---|---|
| matvec Q8_0 | 169 | 193.25 | 194.59 | 1143.5 |
| cached attention partial | 24 | 27.58 | 27.23 | 1149.0 |
| matvec f32 | 49 | 25.16 | 25.02 | 513.4 |
| UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__subtract_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__add_s0_s1 | 24 | 23.75 | 23.69 | 989.5 |
| UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__add_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__subtract_s0_s1 | 24 | 12.00 | 11.93 | 500.1 |
| RMSNorm sumsq + fused epilogue | 49 | 2.90 | 2.60 | 59.2 |
| elementwise fused_identity_o0__multiply_s0_o1_g10 | 1 | 0.36 | 0.36 | 362.4 |
| constant | 9 | 0.04 | 0.02 | 4.0 |
| softmax max | 2 | 0.01 | 0.01 | 6.3 |
| UNCLASSIFIED omega_reduce_r3_o2_n5_fused_equal_o0_o1__add_o2_o3__select_s0_s1_o4_minimum_positive_infinity | 1 | 0.01 | 0.00 | 10.6 |
| iota | 2 | 0.01 | 0.00 | 4.1 |
| elementwise identity_g1 | 1 | 0.01 | 0.00 | 6.4 |

