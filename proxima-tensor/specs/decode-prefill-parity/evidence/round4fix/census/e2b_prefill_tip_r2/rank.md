proxima timed dispatches=842 sum(own cb)=684.79 ms
llama ops per request=2444 sum(own cb)=568.86 ms floor_us=4

| class | proxima ops | proxima ms (own cb) | proxima ms (marginal) | llama ops | llama ms (own cb) | llama ms (net of floor) | gap ms (own cb) | ratio |
|---|---|---|---|---|---|---|---|---|
| matmul (weights) | 277 | 612.44 | 639.40 | 828 | 487.35 | 484.03 | +125.10 | 1.26x |
| rms norm | 242 | 18.94 | 17.69 | 726 | 13.68 | 10.78 | +5.26 | 1.38x |
| attention core | 38 | 44.64 | 44.68 | 105 | 41.31 | 40.89 | +3.32 | 1.08x |
| output head | 1 | 1.09 | 1.09 | 1 | 0.94 | 0.94 | +0.15 | 1.16x |
| rope | 0 | 0.00 | 0.00 | 150 | 3.37 | 2.77 | -3.37 | 0.00x |
| elementwise, copy, other | 284 | 7.67 | 6.98 | 634 | 22.21 | 19.67 | -14.54 | 0.35x |

| matmul shape | proxima ops | proxima ms (own cb) | llama ops | llama ms (own cb) | gap ms | ratio |
|---|---|---|---|---|---|---|
| Q4_0 18874368 (1536x12288) | 60 | 376.98 | 180 | 284.91 | +92.07 | 1.32x |
| Q4_0 9437184 (1536x6144) | 45 | 143.66 | 135 | 110.18 | +33.48 | 1.30x |
| Q4_0 6291456 (1536x4096) | 14 | 25.23 | 42 | 22.83 | +2.40 | 1.11x |
| Q4_0 786432 (1536x512) | 6 | 1.65 | 18 | 1.57 | +0.08 | 1.05x |
| F32 262144 | 1 | 0.01 | 0 | 0.00 | +0.01 | n/a |
| Q4_0 393216 (1536x256) | 94 | 14.97 | 282 | 15.67 | -0.70 | 0.96x |
| F16 13762560 (1536x8960) | 1 | 3.45 | 3 | 4.35 | -0.90 | 0.79x |
| Q4_0 3145728 (1536x2048) | 56 | 46.50 | 168 | 47.83 | -1.34 | 0.97x |

| proxima census label | ops | ms (own cb) | ms (marginal) | us per op (own cb) |
|---|---|---|---|---|
| matvec Q4_0 | 275 | 608.98 | 635.94 | 2214.5 |
| cached attention partial | 35 | 44.62 | 44.67 | 1274.8 |
| RMSNorm sumsq + fused epilogue | 242 | 18.94 | 17.69 | 78.3 |
| elementwise fused_identity_o0__multiply_s0_o1_g10 | 2 | 4.06 | 4.06 | 2032.1 |
| matvec Float16 | 1 | 3.45 | 3.45 | 3452.2 |
| UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__subtract_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__add_s0_s1 | 35 | 2.23 | 2.16 | 63.8 |
| head | 1 | 1.09 | 1.09 | 1094.8 |
| constant | 224 | 0.99 | 0.43 | 4.4 |
| UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__add_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__subtract_s0_s1 | 15 | 0.20 | 0.17 | 13.3 |
| identity copy | 2 | 0.15 | 0.14 | 73.4 |
| UNCLASSIFIED omega_reduce_r3_o2_n5_fused_equal_o0_o1__add_o2_o3__select_s0_s1_o4_minimum_positive_infinity | 1 | 0.01 | 0.01 | 14.7 |
| softmax max | 2 | 0.01 | 0.01 | 6.8 |

