proxima timed dispatches=398 sum(own cb)=6.30 ms
llama ops per request=534 sum(own cb)=6.49 ms floor_us=4

| class | proxima ops | proxima ms (own cb) | proxima ms (marginal) | llama ops | llama ms (own cb) | llama ms (net of floor) | gap ms (own cb) | ratio |
|---|---|---|---|---|---|---|---|---|
| attention core | 51 | 1.38 | 1.24 | 48 | 0.73 | 0.53 | +0.66 | 1.90x |
| moe routing | 24 | 0.20 | 0.12 | 0 | 0.00 | 0.00 | +0.20 | n/a |
| rms norm | 49 | 0.53 | 0.22 | 49 | 0.37 | 0.17 | +0.16 | 1.44x |
| matmul (weights) | 218 | 3.84 | 2.89 | 193 | 3.99 | 3.22 | -0.15 | 0.96x |
| rope | 0 | 0.00 | 0.00 | 48 | 0.32 | 0.13 | -0.32 | 0.00x |
| elementwise, copy, other | 56 | 0.35 | 0.20 | 196 | 1.09 | 0.31 | -0.75 | 0.32x |

| matmul shape | proxima ops | proxima ms (own cb) | llama ops | llama ms (own cb) | gap ms | ratio |
|---|---|---|---|---|---|---|
| Q8_0 4194304 | 72 | 2.15 | 0 | 0.00 | +2.15 | n/a |
| F32 8192 | 24 | 0.28 | 0 | 0.00 | +0.28 | n/a |
| F32 32768 (1024x32) | 24 | 0.21 | 24 | 0.17 | +0.03 | 1.20x |
| Q8_0 1048576 (1024x1024) | 48 | 0.54 | 48 | 0.52 | +0.02 | 1.05x |
| F32 49155 | 1 | 0.01 | 0 | 0.00 | +0.01 | n/a |
| Q8_0 50334720 (1024x49155) | 1 | 0.16 | 1 | 0.24 | -0.08 | 0.67x |
| Q8_0 524288 (1024x512) | 48 | 0.51 | 120 | 3.06 | -2.56 | 0.16x |

| proxima census label | ops | ms (own cb) | ms (marginal) | us per op (own cb) |
|---|---|---|---|---|
| matvec Q8_0 | 169 | 3.35 | 2.52 | 19.8 |
| cached attention partial | 24 | 1.20 | 1.13 | 50.0 |
| RMSNorm sumsq + fused epilogue | 49 | 0.53 | 0.22 | 10.8 |
| matvec f32 | 49 | 0.49 | 0.37 | 10.1 |
| UNCLASSIFIED omega_moe_topk_e32_k8_stacked | 24 | 0.20 | 0.12 | 8.4 |
| cached attention merge | 24 | 0.16 | 0.10 | 6.8 |
| UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__add_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__subtract_s0_s1 | 24 | 0.14 | 0.08 | 6.0 |
| UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__subtract_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__add_s0_s1 | 24 | 0.14 | 0.09 | 5.9 |
| elementwise fused_identity_o0__multiply_s0_o1_g10 | 1 | 0.02 | 0.02 | 18.4 |
| softmax max | 2 | 0.01 | 0.01 | 6.1 |
| UNCLASSIFIED omega_reduce_r3_o2_n5_fused_equal_o0_o1__add_o2_o3__select_s0_s1_o4_minimum_positive_infinity | 1 | 0.01 | 0.01 | 10.5 |
| iota | 2 | 0.01 | 0.00 | 4.3 |

