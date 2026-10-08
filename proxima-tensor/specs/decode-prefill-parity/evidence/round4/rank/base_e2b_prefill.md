proxima timed dispatches=842 sum(own cb)=568.98 ms
llama ops per request=2444 sum(own cb)=568.86 ms floor_us=4

| class | proxima ops | proxima ms (own cb) | proxima ms (marginal) | llama ops | llama ms (own cb) | llama ms (net of floor) | gap ms (own cb) | ratio |
|---|---|---|---|---|---|---|---|---|
| matmul (weights) | 277 | 497.89 | 497.75 | 828 | 487.35 | 484.03 | +10.54 | 1.02x |
| rms norm | 242 | 18.22 | 17.12 | 726 | 13.68 | 10.78 | +4.53 | 1.33x |
| attention core | 38 | 44.19 | 44.22 | 105 | 41.31 | 40.89 | +2.88 | 1.07x |
| output head | 1 | 1.09 | 1.08 | 1 | 0.94 | 0.94 | +0.14 | 1.15x |
| rope | 0 | 0.00 | 0.00 | 150 | 3.37 | 2.77 | -3.37 | 0.00x |
| elementwise, copy, other | 284 | 7.60 | 6.98 | 634 | 22.21 | 19.67 | -14.61 | 0.34x |

| matmul shape | proxima ops | proxima ms (own cb) | llama ops | llama ms (own cb) | gap ms | ratio |
|---|---|---|---|---|---|---|
| Q4_0 18874368 (1536x12288) | 60 | 298.32 | 180 | 284.91 | +13.41 | 1.05x |
| Q4_0 9437184 (1536x6144) | 45 | 112.48 | 135 | 110.18 | +2.30 | 1.02x |
| Q4_0 6291456 (1536x4096) | 14 | 22.87 | 42 | 22.83 | +0.04 | 1.00x |
| F32 262144 | 1 | 0.01 | 0 | 0.00 | +0.01 | n/a |
| Q4_0 786432 (1536x512) | 6 | 1.42 | 18 | 1.57 | -0.15 | 0.90x |
| F16 13762560 (1536x8960) | 1 | 3.45 | 3 | 4.35 | -0.90 | 0.79x |
| Q4_0 3145728 (1536x2048) | 56 | 46.37 | 168 | 47.83 | -1.46 | 0.97x |
| Q4_0 393216 (1536x256) | 94 | 12.96 | 282 | 15.67 | -2.70 | 0.83x |

| proxima census label | ops | ms (own cb) | ms (marginal) | us per op (own cb) |
|---|---|---|---|---|
| matvec Q4_0 | 275 | 494.42 | 494.29 | 1797.9 |
| cached attention partial | 35 | 44.17 | 44.21 | 1262.1 |
| RMSNorm sumsq + fused epilogue | 242 | 18.22 | 17.12 | 75.3 |
| elementwise fused_identity_o0__multiply_s0_o1_g10 | 2 | 4.06 | 4.06 | 2032.4 |
| matvec Float16 | 1 | 3.45 | 3.45 | 3453.1 |
| UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__subtract_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__add_s0_s1 | 35 | 2.24 | 2.17 | 63.9 |
| head | 1 | 1.09 | 1.08 | 1086.4 |
| constant | 224 | 0.91 | 0.44 | 4.1 |
| UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__add_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__subtract_s0_s1 | 15 | 0.19 | 0.15 | 13.0 |
| identity copy | 2 | 0.15 | 0.14 | 72.7 |
| UNCLASSIFIED omega_reduce_r3_o2_n5_fused_equal_o0_o1__add_o2_o3__select_s0_s1_o4_minimum_positive_infinity | 1 | 0.02 | 0.01 | 15.5 |
| softmax max | 2 | 0.02 | 0.01 | 7.6 |

