proxima timed dispatches=842 sum(own cb)=573.52 ms
llama ops per request=2444 sum(own cb)=568.86 ms floor_us=4

| class | proxima ops | proxima ms (own cb) | proxima ms (marginal) | llama ops | llama ms (own cb) | llama ms (net of floor) | gap ms (own cb) | ratio |
|---|---|---|---|---|---|---|---|---|
| matmul (weights) | 277 | 496.53 | 496.06 | 828 | 487.35 | 484.03 | +9.18 | 1.02x |
| attention core | 38 | 50.07 | 49.78 | 105 | 41.31 | 40.89 | +8.76 | 1.21x |
| rms norm | 242 | 18.23 | 17.16 | 726 | 13.68 | 10.78 | +4.55 | 1.33x |
| output head | 1 | 1.09 | 1.09 | 1 | 0.94 | 0.94 | +0.15 | 1.16x |
| rope | 0 | 0.00 | 0.00 | 150 | 3.37 | 2.77 | -3.37 | 0.00x |
| elementwise, copy, other | 284 | 7.59 | 6.88 | 634 | 22.21 | 19.67 | -14.62 | 0.34x |

| matmul shape | proxima ops | proxima ms (own cb) | llama ops | llama ms (own cb) | gap ms | ratio |
|---|---|---|---|---|---|---|
| Q4_0 18874368 (1536x12288) | 60 | 297.16 | 180 | 284.91 | +12.25 | 1.04x |
| Q4_0 9437184 (1536x6144) | 45 | 112.41 | 135 | 110.18 | +2.23 | 1.02x |
| F32 262144 | 1 | 0.01 | 0 | 0.00 | +0.01 | n/a |
| Q4_0 6291456 (1536x4096) | 14 | 22.77 | 42 | 22.83 | -0.06 | 1.00x |
| Q4_0 786432 (1536x512) | 6 | 1.42 | 18 | 1.57 | -0.15 | 0.90x |
| F16 13762560 (1536x8960) | 1 | 3.45 | 3 | 4.35 | -0.90 | 0.79x |
| Q4_0 3145728 (1536x2048) | 56 | 46.35 | 168 | 47.83 | -1.48 | 0.97x |
| Q4_0 393216 (1536x256) | 94 | 12.96 | 282 | 15.67 | -2.71 | 0.83x |

| proxima census label | ops | ms (own cb) | ms (marginal) | us per op (own cb) |
|---|---|---|---|---|
| matvec Q4_0 | 275 | 493.06 | 492.59 | 1793.0 |
| cached attention partial | 35 | 50.05 | 49.76 | 1430.0 |
| RMSNorm sumsq + fused epilogue | 242 | 18.23 | 17.16 | 75.3 |
| elementwise fused_identity_o0__multiply_s0_o1_g10 | 2 | 4.07 | 4.07 | 2032.9 |
| matvec Float16 | 1 | 3.45 | 3.46 | 3453.9 |
| UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__subtract_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__add_s0_s1 | 35 | 2.24 | 2.12 | 63.9 |
| head | 1 | 1.09 | 1.09 | 1094.0 |
| constant | 224 | 0.90 | 0.38 | 4.0 |
| UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__add_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__subtract_s0_s1 | 15 | 0.20 | 0.14 | 13.1 |
| identity copy | 2 | 0.15 | 0.15 | 75.6 |
| UNCLASSIFIED omega_reduce_r3_o2_n5_fused_equal_o0_o1__add_o2_o3__select_s0_s1_o4_minimum_positive_infinity | 1 | 0.02 | 0.01 | 15.3 |
| softmax max | 2 | 0.01 | 0.01 | 7.5 |

