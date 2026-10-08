proxima timed dispatches=842 sum(own cb)=683.24 ms
llama ops per request=2444 sum(own cb)=568.86 ms floor_us=4

| class | proxima ops | proxima ms (own cb) | proxima ms (marginal) | llama ops | llama ms (own cb) | llama ms (net of floor) | gap ms (own cb) | ratio |
|---|---|---|---|---|---|---|---|---|
| matmul (weights) | 277 | 605.68 | 634.58 | 828 | 487.35 | 484.03 | +118.33 | 1.24x |
| attention core | 38 | 49.81 | 50.39 | 105 | 41.31 | 40.89 | +8.49 | 1.21x |
| rms norm | 242 | 18.93 | 17.78 | 726 | 13.68 | 10.78 | +5.25 | 1.38x |
| output head | 1 | 1.09 | 1.09 | 1 | 0.94 | 0.94 | +0.15 | 1.16x |
| rope | 0 | 0.00 | 0.00 | 150 | 3.37 | 2.77 | -3.37 | 0.00x |
| elementwise, copy, other | 284 | 7.73 | 7.06 | 634 | 22.21 | 19.67 | -14.48 | 0.35x |

| matmul shape | proxima ops | proxima ms (own cb) | llama ops | llama ms (own cb) | gap ms | ratio |
|---|---|---|---|---|---|---|
| Q4_0 18874368 (1536x12288) | 60 | 371.21 | 180 | 284.91 | +86.30 | 1.30x |
| Q4_0 9437184 (1536x6144) | 45 | 144.62 | 135 | 110.18 | +34.44 | 1.31x |
| Q4_0 6291456 (1536x4096) | 14 | 23.66 | 42 | 22.83 | +0.84 | 1.04x |
| F32 262144 | 1 | 0.01 | 0 | 0.00 | +0.01 | n/a |
| Q4_0 786432 (1536x512) | 6 | 1.42 | 18 | 1.57 | -0.16 | 0.90x |
| Q4_0 393216 (1536x256) | 94 | 14.93 | 282 | 15.67 | -0.74 | 0.95x |
| F16 13762560 (1536x8960) | 1 | 3.45 | 3 | 4.35 | -0.90 | 0.79x |
| Q4_0 3145728 (1536x2048) | 56 | 46.37 | 168 | 47.83 | -1.47 | 0.97x |

| proxima census label | ops | ms (own cb) | ms (marginal) | us per op (own cb) |
|---|---|---|---|---|
| matvec Q4_0 | 275 | 602.21 | 631.12 | 2189.9 |
| cached attention partial | 35 | 49.79 | 50.38 | 1422.4 |
| RMSNorm sumsq + fused epilogue | 242 | 18.93 | 17.78 | 78.2 |
| elementwise fused_identity_o0__multiply_s0_o1_g10 | 2 | 4.06 | 4.06 | 2032.4 |
| matvec Float16 | 1 | 3.45 | 3.45 | 3452.7 |
| UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__subtract_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__add_s0_s1 | 35 | 2.24 | 2.18 | 63.9 |
| head | 1 | 1.09 | 1.09 | 1094.4 |
| constant | 224 | 1.04 | 0.50 | 4.6 |
| UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__add_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__subtract_s0_s1 | 15 | 0.20 | 0.16 | 13.3 |
| identity copy | 2 | 0.15 | 0.14 | 73.9 |
| UNCLASSIFIED omega_reduce_r3_o2_n5_fused_equal_o0_o1__add_o2_o3__select_s0_s1_o4_minimum_positive_infinity | 1 | 0.02 | 0.01 | 15.4 |
| softmax max | 2 | 0.01 | 0.01 | 7.1 |

