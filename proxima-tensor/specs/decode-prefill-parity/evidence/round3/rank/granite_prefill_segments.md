proxima timed dispatches=383 sum(own cb)=232.92 ms
llama ops per request=1051 sum(own cb)=157.56 ms floor_us=4

| class | proxima ops | proxima ms (own cb) | proxima ms (marginal) | llama ops | llama ms (own cb) | llama ms (net of floor) | gap ms (own cb) | ratio |
|---|---|---|---|---|---|---|---|---|
| matmul (weights) | 218 | 200.27 | 198.64 | 381 | 131.89 | 130.37 | +68.38 | 1.52x |
| attention core | 27 | 27.54 | 27.26 | 95 | 12.93 | 12.55 | +14.61 | 2.13x |
| rms norm | 49 | 2.90 | 2.73 | 96 | 1.77 | 1.39 | +1.13 | 1.64x |
| moe routing | 24 | 0.31 | 0.24 | 0 | 0.00 | 0.00 | +0.31 | n/a |
| rope | 0 | 0.00 | 0.00 | 96 | 1.71 | 1.32 | -1.71 | 0.00x |
| elementwise, copy, other | 65 | 1.91 | 2.02 | 383 | 9.27 | 7.73 | -7.36 | 0.21x |

| matmul shape | proxima ops | proxima ms (own cb) | llama ops | llama ms (own cb) | gap ms | ratio |
|---|---|---|---|---|---|---|
| Q8_0 4194304 | 72 | 166.61 | 0 | 0.00 | +166.61 | n/a |
| F32 8192 | 24 | 5.32 | 0 | 0.00 | +5.32 | n/a |
| F32 32768 (1024x32) | 24 | 5.09 | 47 | 2.17 | +2.92 | 2.34x |
| Q8_0 1048576 (1024x1024) | 48 | 15.13 | 96 | 15.08 | +0.04 | 1.00x |
| F32 49155 | 1 | 0.01 | 0 | 0.00 | +0.01 | n/a |
| Q8_0 50334720 (1024x49155) | 1 | 0.16 | 1 | 0.22 | -0.06 | 0.72x |
| Q8_0 524288 (1024x512) | 48 | 7.95 | 237 | 114.41 | -106.46 | 0.07x |

| proxima census label | ops | ms (own cb) | ms (marginal) | us per op (own cb) |
|---|---|---|---|---|
| matvec Q8_0 | 169 | 189.85 | 188.37 | 1123.4 |
| cached attention partial | 24 | 27.52 | 27.25 | 1146.6 |
| matvec f32 | 49 | 10.42 | 10.27 | 212.6 |
| RMSNorm sumsq + fused epilogue | 49 | 2.90 | 2.73 | 59.1 |
| UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__subtract_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__add_s0_s1 | 24 | 0.98 | 1.19 | 40.6 |
| UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__add_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__subtract_s0_s1 | 24 | 0.49 | 0.43 | 20.4 |
| elementwise fused_identity_o0__multiply_s0_o1_g10 | 1 | 0.36 | 0.36 | 364.0 |
| UNCLASSIFIED omega_moe_topk_e32_k8_stacked | 24 | 0.31 | 0.24 | 12.9 |
| constant | 9 | 0.04 | 0.02 | 4.0 |
| softmax max | 2 | 0.01 | 0.01 | 6.6 |
| UNCLASSIFIED omega_reduce_r3_o2_n5_fused_equal_o0_o1__add_o2_o3__select_s0_s1_o4_minimum_positive_infinity | 1 | 0.01 | 0.01 | 11.1 |
| iota | 2 | 0.01 | 0.00 | 4.4 |

