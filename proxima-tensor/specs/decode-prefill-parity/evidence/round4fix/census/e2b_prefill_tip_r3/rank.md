proxima timed dispatches=842 sum(own cb)=628.51 ms
llama ops per request=2444 sum(own cb)=568.86 ms floor_us=4

| class | proxima ops | proxima ms (own cb) | proxima ms (marginal) | llama ops | llama ms (own cb) | llama ms (net of floor) | gap ms (own cb) | ratio |
|---|---|---|---|---|---|---|---|---|
| matmul (weights) | 277 | 556.13 | 577.35 | 828 | 487.35 | 484.03 | +68.78 | 1.14x |
| rms norm | 242 | 18.99 | 17.63 | 726 | 13.68 | 10.78 | +5.31 | 1.39x |
| attention core | 38 | 44.72 | 44.61 | 105 | 41.31 | 40.89 | +3.41 | 1.08x |
| output head | 1 | 1.09 | 1.09 | 1 | 0.94 | 0.94 | +0.15 | 1.16x |
| rope | 0 | 0.00 | 0.00 | 150 | 3.37 | 2.77 | -3.37 | 0.00x |
| elementwise, copy, other | 284 | 7.58 | 6.89 | 634 | 22.21 | 19.67 | -14.63 | 0.34x |

| matmul shape | proxima ops | proxima ms (own cb) | llama ops | llama ms (own cb) | gap ms | ratio |
|---|---|---|---|---|---|---|
| Q4_0 18874368 (1536x12288) | 60 | 348.61 | 180 | 284.91 | +63.70 | 1.22x |
| Q4_0 9437184 (1536x6144) | 45 | 116.75 | 135 | 110.18 | +6.57 | 1.06x |
| Q4_0 6291456 (1536x4096) | 14 | 24.29 | 42 | 22.83 | +1.47 | 1.06x |
| Q4_0 786432 (1536x512) | 6 | 1.81 | 18 | 1.57 | +0.24 | 1.15x |
| F32 262144 | 1 | 0.01 | 0 | 0.00 | +0.01 | n/a |
| Q4_0 393216 (1536x256) | 94 | 14.83 | 282 | 15.67 | -0.84 | 0.95x |
| F16 13762560 (1536x8960) | 1 | 3.45 | 3 | 4.35 | -0.90 | 0.79x |
| Q4_0 3145728 (1536x2048) | 56 | 46.38 | 168 | 47.83 | -1.45 | 0.97x |

| proxima census label | ops | ms (own cb) | ms (marginal) | us per op (own cb) |
|---|---|---|---|---|
| matvec Q4_0 | 275 | 552.66 | 573.89 | 2009.7 |
| cached attention partial | 35 | 44.70 | 44.60 | 1277.1 |
| RMSNorm sumsq + fused epilogue | 242 | 18.99 | 17.63 | 78.5 |
| elementwise fused_identity_o0__multiply_s0_o1_g10 | 2 | 4.07 | 4.06 | 2033.2 |
| matvec Float16 | 1 | 3.45 | 3.45 | 3452.1 |
| UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__subtract_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__add_s0_s1 | 35 | 2.24 | 2.12 | 64.0 |
| head | 1 | 1.09 | 1.09 | 1092.7 |
| constant | 224 | 0.88 | 0.38 | 3.9 |
| UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__add_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__subtract_s0_s1 | 15 | 0.20 | 0.16 | 13.4 |
| identity copy | 2 | 0.15 | 0.14 | 75.6 |
| UNCLASSIFIED omega_reduce_r3_o2_n5_fused_equal_o0_o1__add_o2_o3__select_s0_s1_o4_minimum_positive_infinity | 1 | 0.02 | 0.01 | 15.4 |
| softmax max | 2 | 0.01 | 0.01 | 7.4 |

