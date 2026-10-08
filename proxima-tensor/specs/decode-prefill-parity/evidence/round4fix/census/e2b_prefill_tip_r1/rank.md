proxima timed dispatches=842 sum(own cb)=569.01 ms
llama ops per request=2444 sum(own cb)=568.86 ms floor_us=4

| class | proxima ops | proxima ms (own cb) | proxima ms (marginal) | llama ops | llama ms (own cb) | llama ms (net of floor) | gap ms (own cb) | ratio |
|---|---|---|---|---|---|---|---|---|
| matmul (weights) | 277 | 497.89 | 497.38 | 828 | 487.35 | 484.03 | +10.55 | 1.02x |
| rms norm | 242 | 18.23 | 17.12 | 726 | 13.68 | 10.78 | +4.55 | 1.33x |
| attention core | 38 | 44.20 | 44.33 | 105 | 41.31 | 40.89 | +2.89 | 1.07x |
| output head | 1 | 1.09 | 1.09 | 1 | 0.94 | 0.94 | +0.15 | 1.16x |
| rope | 0 | 0.00 | 0.00 | 150 | 3.37 | 2.77 | -3.37 | 0.00x |
| elementwise, copy, other | 284 | 7.59 | 6.97 | 634 | 22.21 | 19.67 | -14.62 | 0.34x |

| matmul shape | proxima ops | proxima ms (own cb) | llama ops | llama ms (own cb) | gap ms | ratio |
|---|---|---|---|---|---|---|
| Q4_0 18874368 (1536x12288) | 60 | 298.41 | 180 | 284.91 | +13.51 | 1.05x |
| Q4_0 9437184 (1536x6144) | 45 | 112.46 | 135 | 110.18 | +2.28 | 1.02x |
| F32 262144 | 1 | 0.01 | 0 | 0.00 | +0.01 | n/a |
| Q4_0 6291456 (1536x4096) | 14 | 22.82 | 42 | 22.83 | -0.01 | 1.00x |
| Q4_0 786432 (1536x512) | 6 | 1.42 | 18 | 1.57 | -0.16 | 0.90x |
| F16 13762560 (1536x8960) | 1 | 3.45 | 3 | 4.35 | -0.90 | 0.79x |
| Q4_0 3145728 (1536x2048) | 56 | 46.37 | 168 | 47.83 | -1.46 | 0.97x |
| Q4_0 393216 (1536x256) | 94 | 12.94 | 282 | 15.67 | -2.73 | 0.83x |

| proxima census label | ops | ms (own cb) | ms (marginal) | us per op (own cb) |
|---|---|---|---|---|
| matvec Q4_0 | 275 | 494.43 | 493.92 | 1797.9 |
| cached attention partial | 35 | 44.18 | 44.32 | 1262.3 |
| RMSNorm sumsq + fused epilogue | 242 | 18.23 | 17.12 | 75.3 |
| elementwise fused_identity_o0__multiply_s0_o1_g10 | 2 | 4.07 | 4.07 | 2033.5 |
| matvec Float16 | 1 | 3.45 | 3.45 | 3452.2 |
| UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__subtract_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__add_s0_s1 | 35 | 2.24 | 2.17 | 63.9 |
| head | 1 | 1.09 | 1.09 | 1092.5 |
| constant | 224 | 0.90 | 0.43 | 4.0 |
| UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__add_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__subtract_s0_s1 | 15 | 0.19 | 0.14 | 12.9 |
| identity copy | 2 | 0.15 | 0.14 | 74.3 |
| UNCLASSIFIED omega_reduce_r3_o2_n5_fused_equal_o0_o1__add_o2_o3__select_s0_s1_o4_minimum_positive_infinity | 1 | 0.01 | 0.01 | 14.9 |
| softmax max | 2 | 0.01 | 0.01 | 7.4 |

