proxima timed dispatches=653 sum(own cb)=11.24 ms
llama ops per request=818 sum(own cb)=11.10 ms floor_us=4

| class | proxima ops | proxima ms (own cb) | proxima ms (marginal) | llama ops | llama ms (own cb) | llama ms (net of floor) | gap ms (own cb) | ratio |
|---|---|---|---|---|---|---|---|---|
| rms norm | 242 | 2.95 | 1.43 | 242 | 1.80 | 0.83 | +1.15 | 1.64x |
| matmul (weights) | 277 | 5.49 | 4.25 | 276 | 5.30 | 4.19 | +0.19 | 1.04x |
| output head | 1 | 1.09 | 1.09 | 1 | 0.94 | 0.94 | +0.14 | 1.15x |
| attention core | 73 | 1.29 | 1.08 | 35 | 1.21 | 1.07 | +0.08 | 1.07x |
| rope | 0 | 0.00 | 0.00 | 50 | 0.32 | 0.12 | -0.32 | 0.00x |
| elementwise, copy, other | 60 | 0.42 | 0.30 | 214 | 1.52 | 0.67 | -1.10 | 0.28x |

| matmul shape | proxima ops | proxima ms (own cb) | llama ops | llama ms (own cb) | gap ms | ratio |
|---|---|---|---|---|---|---|
| F16 13762560 (1536x8960) | 1 | 0.31 | 1 | 0.08 | +0.22 | 3.64x |
| Q4_0 393216 (1536x256) | 94 | 0.90 | 94 | 0.84 | +0.06 | 1.07x |
| Q4_0 3145728 (1536x2048) | 56 | 0.74 | 56 | 0.71 | +0.04 | 1.05x |
| Q4_0 6291456 (1536x4096) | 14 | 0.27 | 14 | 0.26 | +0.01 | 1.05x |
| F32 262144 | 1 | 0.01 | 0 | 0.00 | +0.01 | n/a |
| Q4_0 786432 (1536x512) | 6 | 0.06 | 6 | 0.06 | +0.00 | 1.01x |
| Q4_0 9437184 (1536x6144) | 45 | 1.01 | 45 | 1.06 | -0.05 | 0.95x |
| Q4_0 18874368 (1536x12288) | 60 | 2.19 | 60 | 2.30 | -0.11 | 0.95x |

| proxima census label | ops | ms (own cb) | ms (marginal) | us per op (own cb) |
|---|---|---|---|---|
| matvec Q4_0 | 275 | 5.17 | 4.10 | 18.8 |
| RMSNorm sumsq + fused epilogue | 242 | 2.95 | 1.43 | 12.2 |
| head | 1 | 1.09 | 1.09 | 1087.7 |
| cached attention partial | 35 | 0.99 | 0.86 | 28.2 |
| matvec Float16 | 1 | 0.31 | 0.14 | 305.9 |
| cached attention merge | 35 | 0.28 | 0.21 | 8.1 |
| UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__subtract_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__add_s0_s1 | 35 | 0.22 | 0.16 | 6.3 |
| UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__add_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__subtract_s0_s1 | 15 | 0.07 | 0.04 | 4.9 |
| elementwise fused_identity_o0__multiply_s0_o1_g10 | 2 | 0.07 | 0.07 | 35.3 |
| identity copy | 2 | 0.02 | 0.01 | 7.8 |
| softmax max | 2 | 0.01 | 0.01 | 7.5 |
| UNCLASSIFIED omega_reduce_r3_o2_n5_fused_equal_o0_o1__add_o2_o3__select_s0_s1_o4_minimum_positive_infinity | 1 | 0.01 | 0.01 | 14.9 |

