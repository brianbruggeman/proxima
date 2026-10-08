proxima timed dispatches=842 sum(own cb)=575.70 ms
llama ops per request=2444 sum(own cb)=568.86 ms floor_us=4

| class | proxima ops | proxima ms (own cb) | proxima ms (marginal) | llama ops | llama ms (own cb) | llama ms (net of floor) | gap ms (own cb) | ratio |
|---|---|---|---|---|---|---|---|---|
| matmul (weights) | 277 | 498.61 | 498.28 | 828 | 487.35 | 484.03 | +11.27 | 1.02x |
| attention core | 38 | 50.03 | 49.83 | 105 | 41.31 | 40.89 | +8.71 | 1.21x |
| rms norm | 242 | 18.33 | 17.20 | 726 | 13.68 | 10.78 | +4.65 | 1.34x |
| output head | 1 | 1.09 | 1.08 | 1 | 0.94 | 0.94 | +0.14 | 1.15x |
| rope | 0 | 0.00 | 0.00 | 150 | 3.37 | 2.77 | -3.37 | 0.00x |
| elementwise, copy, other | 284 | 7.64 | 7.04 | 634 | 22.21 | 19.67 | -14.57 | 0.34x |

| matmul shape | proxima ops | proxima ms (own cb) | llama ops | llama ms (own cb) | gap ms | ratio |
|---|---|---|---|---|---|---|
| Q4_0 18874368 (1536x12288) | 60 | 299.09 | 180 | 284.91 | +14.18 | 1.05x |
| Q4_0 9437184 (1536x6144) | 45 | 112.43 | 135 | 110.18 | +2.25 | 1.02x |
| Q4_0 6291456 (1536x4096) | 14 | 22.87 | 42 | 22.83 | +0.04 | 1.00x |
| F32 262144 | 1 | 0.01 | 0 | 0.00 | +0.01 | n/a |
| Q4_0 786432 (1536x512) | 6 | 1.42 | 18 | 1.57 | -0.15 | 0.90x |
| F16 13762560 (1536x8960) | 1 | 3.45 | 3 | 4.35 | -0.90 | 0.79x |
| Q4_0 3145728 (1536x2048) | 56 | 46.36 | 168 | 47.83 | -1.47 | 0.97x |
| Q4_0 393216 (1536x256) | 94 | 12.98 | 282 | 15.67 | -2.69 | 0.83x |

| proxima census label | ops | ms (own cb) | ms (marginal) | us per op (own cb) |
|---|---|---|---|---|
| matvec Q4_0 | 275 | 495.15 | 494.82 | 1800.5 |
| cached attention partial | 35 | 50.01 | 49.82 | 1428.7 |
| RMSNorm sumsq + fused epilogue | 242 | 18.33 | 17.20 | 75.8 |
| elementwise fused_identity_o0__multiply_s0_o1_g10 | 2 | 4.06 | 4.06 | 2031.5 |
| matvec Float16 | 1 | 3.45 | 3.45 | 3451.0 |
| UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__subtract_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__add_s0_s1 | 35 | 2.24 | 2.17 | 64.1 |
| head | 1 | 1.09 | 1.08 | 1086.9 |
| constant | 224 | 0.95 | 0.49 | 4.2 |
| UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__add_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__subtract_s0_s1 | 15 | 0.20 | 0.16 | 13.4 |
| identity copy | 2 | 0.14 | 0.14 | 71.3 |
| UNCLASSIFIED omega_reduce_r3_o2_n5_fused_equal_o0_o1__add_o2_o3__select_s0_s1_o4_minimum_positive_infinity | 1 | 0.02 | 0.01 | 15.6 |
| softmax max | 2 | 0.02 | 0.01 | 7.5 |

