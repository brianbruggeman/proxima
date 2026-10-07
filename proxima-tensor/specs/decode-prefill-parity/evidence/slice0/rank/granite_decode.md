proxima timed dispatches=902 sum(own cb)=12.45 ms
llama ops per request=534 sum(own cb)=6.49 ms floor_us=4

| class | proxima ops | proxima ms (own cb) | proxima ms (marginal) | llama ops | llama ms (own cb) | llama ms (net of floor) | gap ms (own cb) | ratio |
|---|---|---|---|---|---|---|---|---|
| matmul (weights) | 698 | 9.68 | 5.84 | 193 | 3.99 | 3.22 | +5.69 | 2.43x |
| attention core | 51 | 1.39 | 1.23 | 48 | 0.73 | 0.53 | +0.66 | 1.91x |
| rope | 96 | 0.75 | 0.42 | 48 | 0.32 | 0.13 | +0.43 | 2.34x |
| rms norm | 49 | 0.56 | 0.29 | 49 | 0.37 | 0.17 | +0.20 | 1.54x |
| moe routing | 24 | 0.00 | 0.00 | 0 | 0.00 | 0.00 | +0.00 | n/a |
| elementwise, copy, other | 8 | 0.06 | 0.04 | 196 | 1.09 | 0.31 | -1.03 | 0.06x |

| matmul shape | proxima ops | proxima ms (own cb) | llama ops | llama ms (own cb) | gap ms | ratio |
|---|---|---|---|---|---|---|
| Q8_0 524288 (1024x512) | 624 | 8.20 | 120 | 3.06 | +5.14 | 2.68x |
| Q8_0 1048576 (1024x1024) | 48 | 0.90 | 48 | 0.52 | +0.39 | 1.75x |
| Q8_0 50334720 (1024x49155) | 1 | 0.35 | 1 | 0.24 | +0.11 | 1.48x |
| F32 32768 (1024x32) | 24 | 0.22 | 24 | 0.17 | +0.04 | 1.24x |
| F32 49155 | 1 | 0.01 | 0 | 0.00 | +0.01 | n/a |

| proxima census label | ops | ms (own cb) | ms (marginal) | us per op (own cb) |
|---|---|---|---|---|
| matvec Q8_0 | 673 | 9.46 | 5.69 | 14.1 |
| cached attention partial | 24 | 1.20 | 1.12 | 50.1 |
| RoPE | 96 | 0.75 | 0.42 | 7.8 |
| RMSNorm sumsq + fused epilogue | 49 | 0.56 | 0.29 | 11.5 |
| matvec f32 | 25 | 0.22 | 0.15 | 8.9 |
| cached attention merge | 24 | 0.17 | 0.11 | 6.9 |
| elementwise fused_identity_o0__multiply_s0_o1_g10 | 1 | 0.02 | 0.02 | 18.8 |
| softmax max | 2 | 0.01 | 0.01 | 6.8 |
| UNCLASSIFIED omega_reduce_r3_o2_n5_fused_equal_o0_o1__add_o2_o3__select_s0_s1_o4_minimum_positive_infinity | 1 | 0.01 | 0.01 | 10.8 |
| iota | 2 | 0.01 | 0.00 | 4.2 |
| identity copy | 1 | 0.01 | 0.00 | 6.2 |
| elementwise identity_g1 | 1 | 0.01 | 0.00 | 6.0 |

