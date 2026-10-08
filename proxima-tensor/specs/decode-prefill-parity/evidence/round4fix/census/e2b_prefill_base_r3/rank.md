proxima timed dispatches=842 sum(own cb)=651.71 ms
llama ops per request=2444 sum(own cb)=568.86 ms floor_us=4

| class | proxima ops | proxima ms (own cb) | proxima ms (marginal) | llama ops | llama ms (own cb) | llama ms (net of floor) | gap ms (own cb) | ratio |
|---|---|---|---|---|---|---|---|---|
| matmul (weights) | 277 | 573.92 | 595.49 | 828 | 487.35 | 484.03 | +86.58 | 1.18x |
| attention core | 38 | 50.05 | 49.87 | 105 | 41.31 | 40.89 | +8.73 | 1.21x |
| rms norm | 242 | 19.02 | 17.78 | 726 | 13.68 | 10.78 | +5.33 | 1.39x |
| output head | 1 | 1.10 | 1.09 | 1 | 0.94 | 0.94 | +0.15 | 1.16x |
| rope | 0 | 0.00 | 0.00 | 150 | 3.37 | 2.77 | -3.37 | 0.00x |
| elementwise, copy, other | 284 | 7.63 | 6.99 | 634 | 22.21 | 19.67 | -14.58 | 0.34x |

| matmul shape | proxima ops | proxima ms (own cb) | llama ops | llama ms (own cb) | gap ms | ratio |
|---|---|---|---|---|---|---|
| Q4_0 18874368 (1536x12288) | 60 | 357.69 | 180 | 284.91 | +72.78 | 1.26x |
| Q4_0 9437184 (1536x6144) | 45 | 124.92 | 135 | 110.18 | +14.74 | 1.13x |
| Q4_0 6291456 (1536x4096) | 14 | 24.68 | 42 | 22.83 | +1.86 | 1.08x |
| Q4_0 786432 (1536x512) | 6 | 1.89 | 18 | 1.57 | +0.31 | 1.20x |
| F32 262144 | 1 | 0.01 | 0 | 0.00 | +0.01 | n/a |
| Q4_0 393216 (1536x256) | 94 | 14.93 | 282 | 15.67 | -0.74 | 0.95x |
| F16 13762560 (1536x8960) | 1 | 3.45 | 3 | 4.35 | -0.90 | 0.79x |
| Q4_0 3145728 (1536x2048) | 56 | 46.35 | 168 | 47.83 | -1.48 | 0.97x |

| proxima census label | ops | ms (own cb) | ms (marginal) | us per op (own cb) |
|---|---|---|---|---|
| matvec Q4_0 | 275 | 570.46 | 592.03 | 2074.4 |
| cached attention partial | 35 | 50.02 | 49.86 | 1429.3 |
| RMSNorm sumsq + fused epilogue | 242 | 19.02 | 17.78 | 78.6 |
| elementwise fused_identity_o0__multiply_s0_o1_g10 | 2 | 4.07 | 4.06 | 2032.7 |
| matvec Float16 | 1 | 3.45 | 3.45 | 3450.9 |
| UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__subtract_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__add_s0_s1 | 35 | 2.24 | 2.17 | 64.0 |
| head | 1 | 1.10 | 1.09 | 1097.0 |
| constant | 224 | 0.93 | 0.44 | 4.1 |
| UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__add_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__subtract_s0_s1 | 15 | 0.20 | 0.16 | 13.5 |
| identity copy | 2 | 0.15 | 0.14 | 74.4 |
| UNCLASSIFIED omega_reduce_r3_o2_n5_fused_equal_o0_o1__add_o2_o3__select_s0_s1_o4_minimum_positive_infinity | 1 | 0.02 | 0.01 | 15.4 |
| softmax max | 2 | 0.01 | 0.01 | 7.4 |

