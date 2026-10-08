proxima timed dispatches=374 sum(own cb)=6.27 ms
llama ops per request=534 sum(own cb)=6.49 ms floor_us=4

| class | proxima ops | proxima ms (own cb) | proxima ms (marginal) | llama ops | llama ms (own cb) | llama ms (net of floor) | gap ms (own cb) | ratio |
|---|---|---|---|---|---|---|---|---|
| attention core | 51 | 1.38 | 1.24 | 48 | 0.73 | 0.53 | +0.66 | 1.90x |
| rms norm | 49 | 0.56 | 0.24 | 49 | 0.37 | 0.17 | +0.19 | 1.51x |
| moe routing | 24 | 0.00 | 0.00 | 0 | 0.00 | 0.00 | +0.00 | n/a |
| matmul (weights) | 218 | 3.87 | 2.80 | 193 | 3.99 | 3.22 | -0.12 | 0.97x |
| rope | 0 | 0.00 | 0.00 | 48 | 0.32 | 0.13 | -0.32 | 0.00x |
| elementwise, copy, other | 56 | 0.47 | 0.32 | 196 | 1.09 | 0.31 | -0.62 | 0.43x |

| matmul shape | proxima ops | proxima ms (own cb) | llama ops | llama ms (own cb) | gap ms | ratio |
|---|---|---|---|---|---|---|
| Q8_0 4194304 | 72 | 2.11 | 0 | 0.00 | +2.11 | n/a |
| F32 8192 | 24 | 0.30 | 0 | 0.00 | +0.30 | n/a |
| Q8_0 1048576 (1024x1024) | 48 | 0.55 | 48 | 0.52 | +0.03 | 1.06x |
| F32 32768 (1024x32) | 24 | 0.20 | 24 | 0.17 | +0.03 | 1.17x |
| F32 49155 | 1 | 0.01 | 0 | 0.00 | +0.01 | n/a |
| Q8_0 50334720 (1024x49155) | 1 | 0.16 | 1 | 0.24 | -0.08 | 0.66x |
| Q8_0 524288 (1024x512) | 48 | 0.54 | 120 | 3.06 | -2.52 | 0.18x |

| proxima census label | ops | ms (own cb) | ms (marginal) | us per op (own cb) |
|---|---|---|---|---|
| matvec Q8_0 | 169 | 3.35 | 2.46 | 19.8 |
| cached attention partial | 24 | 1.20 | 1.14 | 50.1 |
| RMSNorm sumsq + fused epilogue | 49 | 0.56 | 0.24 | 11.3 |
| matvec f32 | 49 | 0.52 | 0.34 | 10.5 |
| UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__subtract_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__add_s0_s1 | 24 | 0.23 | 0.17 | 9.5 |
| UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__add_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__subtract_s0_s1 | 24 | 0.18 | 0.10 | 7.6 |
| cached attention merge | 24 | 0.16 | 0.09 | 6.6 |
| elementwise fused_identity_o0__multiply_s0_o1_g10 | 1 | 0.02 | 0.02 | 18.8 |
| softmax max | 2 | 0.01 | 0.01 | 6.4 |
| UNCLASSIFIED omega_reduce_r3_o2_n5_fused_equal_o0_o1__add_o2_o3__select_s0_s1_o4_minimum_positive_infinity | 1 | 0.01 | 0.01 | 10.3 |
| iota | 2 | 0.01 | 0.00 | 3.9 |
| elementwise identity_g1 | 1 | 0.01 | 0.00 | 6.1 |

