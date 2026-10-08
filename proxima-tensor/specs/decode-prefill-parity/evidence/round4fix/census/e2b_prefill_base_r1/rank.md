proxima timed dispatches=842 sum(own cb)=737.54 ms
llama ops per request=2444 sum(own cb)=568.86 ms floor_us=4

| class | proxima ops | proxima ms (own cb) | proxima ms (marginal) | llama ops | llama ms (own cb) | llama ms (net of floor) | gap ms (own cb) | ratio |
|---|---|---|---|---|---|---|---|---|
| matmul (weights) | 277 | 657.33 | 689.79 | 828 | 487.35 | 484.03 | +169.99 | 1.35x |
| attention core | 38 | 50.65 | 53.98 | 105 | 41.31 | 40.89 | +9.33 | 1.23x |
| rms norm | 242 | 20.55 | 19.64 | 726 | 13.68 | 10.78 | +6.87 | 1.50x |
| output head | 1 | 1.09 | 1.09 | 1 | 0.94 | 0.94 | +0.15 | 1.16x |
| rope | 0 | 0.00 | 0.00 | 150 | 3.37 | 2.77 | -3.37 | 0.00x |
| elementwise, copy, other | 284 | 7.92 | 7.14 | 634 | 22.21 | 19.67 | -14.28 | 0.36x |

| matmul shape | proxima ops | proxima ms (own cb) | llama ops | llama ms (own cb) | gap ms | ratio |
|---|---|---|---|---|---|---|
| Q4_0 18874368 (1536x12288) | 60 | 401.25 | 180 | 284.91 | +116.34 | 1.41x |
| Q4_0 9437184 (1536x6144) | 45 | 161.77 | 135 | 110.18 | +51.59 | 1.47x |
| Q4_0 3145728 (1536x2048) | 56 | 51.32 | 168 | 47.83 | +3.48 | 1.07x |
| Q4_0 6291456 (1536x4096) | 14 | 23.16 | 42 | 22.83 | +0.34 | 1.01x |
| F32 262144 | 1 | 0.01 | 0 | 0.00 | +0.01 | n/a |
| Q4_0 786432 (1536x512) | 6 | 1.43 | 18 | 1.57 | -0.15 | 0.91x |
| Q4_0 393216 (1536x256) | 94 | 14.94 | 282 | 15.67 | -0.73 | 0.95x |
| F16 13762560 (1536x8960) | 1 | 3.45 | 3 | 4.35 | -0.90 | 0.79x |

| proxima census label | ops | ms (own cb) | ms (marginal) | us per op (own cb) |
|---|---|---|---|---|
| matvec Q4_0 | 275 | 653.87 | 686.32 | 2377.7 |
| cached attention partial | 35 | 50.63 | 53.97 | 1446.4 |
| RMSNorm sumsq + fused epilogue | 242 | 20.55 | 19.64 | 84.9 |
| elementwise fused_identity_o0__multiply_s0_o1_g10 | 2 | 4.07 | 4.06 | 2033.3 |
| matvec Float16 | 1 | 3.45 | 3.45 | 3451.1 |
| UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__subtract_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__add_s0_s1 | 35 | 2.23 | 2.13 | 63.8 |
| constant | 224 | 1.23 | 0.61 | 5.5 |
| head | 1 | 1.09 | 1.09 | 1090.3 |
| UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__add_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__subtract_s0_s1 | 15 | 0.20 | 0.17 | 13.4 |
| identity copy | 2 | 0.15 | 0.14 | 74.5 |
| UNCLASSIFIED omega_reduce_r3_o2_n5_fused_equal_o0_o1__add_o2_o3__select_s0_s1_o4_minimum_positive_infinity | 1 | 0.02 | 0.01 | 15.3 |
| softmax max | 2 | 0.01 | 0.01 | 7.4 |

