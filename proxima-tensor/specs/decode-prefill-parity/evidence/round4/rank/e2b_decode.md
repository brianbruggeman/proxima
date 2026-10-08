proxima timed dispatches=653 sum(own cb)=11.22 ms
llama ops per request=818 sum(own cb)=11.10 ms floor_us=4

| class | proxima ops | proxima ms (own cb) | proxima ms (marginal) | llama ops | llama ms (own cb) | llama ms (net of floor) | gap ms (own cb) | ratio |
|---|---|---|---|---|---|---|---|---|
| rms norm | 242 | 2.96 | 1.46 | 242 | 1.80 | 0.83 | +1.16 | 1.65x |
| matmul (weights) | 277 | 5.50 | 4.57 | 276 | 5.30 | 4.19 | +0.20 | 1.04x |
| output head | 1 | 1.09 | 1.08 | 1 | 0.94 | 0.94 | +0.14 | 1.15x |
| attention core | 73 | 1.29 | 1.03 | 35 | 1.21 | 1.07 | +0.08 | 1.07x |
| rope | 0 | 0.00 | 0.00 | 50 | 0.32 | 0.12 | -0.32 | 0.00x |
| elementwise, copy, other | 60 | 0.39 | 0.27 | 214 | 1.52 | 0.67 | -1.13 | 0.26x |

| matmul shape | proxima ops | proxima ms (own cb) | llama ops | llama ms (own cb) | gap ms | ratio |
|---|---|---|---|---|---|---|
| Q4_0 393216 (1536x256) | 94 | 0.93 | 94 | 0.84 | +0.09 | 1.11x |
| Q4_0 18874368 (1536x12288) | 60 | 2.37 | 60 | 2.30 | +0.07 | 1.03x |
| F16 13762560 (1536x8960) | 1 | 0.12 | 1 | 0.08 | +0.03 | 1.39x |
| Q4_0 3145728 (1536x2048) | 56 | 0.73 | 56 | 0.71 | +0.02 | 1.03x |
| F32 262144 | 1 | 0.02 | 0 | 0.00 | +0.02 | n/a |
| Q4_0 6291456 (1536x4096) | 14 | 0.27 | 14 | 0.26 | +0.01 | 1.05x |
| Q4_0 786432 (1536x512) | 6 | 0.06 | 6 | 0.06 | -0.00 | 0.99x |
| Q4_0 9437184 (1536x6144) | 45 | 1.01 | 45 | 1.06 | -0.05 | 0.95x |

| proxima census label | ops | ms (own cb) | ms (marginal) | us per op (own cb) |
|---|---|---|---|---|
| matvec Q4_0 | 275 | 5.36 | 4.44 | 19.5 |
| RMSNorm sumsq + fused epilogue | 242 | 2.96 | 1.46 | 12.2 |
| head | 1 | 1.09 | 1.08 | 1086.6 |
| cached attention partial | 35 | 0.98 | 0.82 | 28.1 |
| cached attention merge | 35 | 0.28 | 0.19 | 8.1 |
| UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__subtract_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__add_s0_s1 | 35 | 0.22 | 0.16 | 6.4 |
| matvec Float16 | 1 | 0.12 | 0.12 | 117.0 |
| UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__add_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__subtract_s0_s1 | 15 | 0.07 | 0.04 | 4.9 |
| elementwise fused_identity_o0__multiply_s0_o1_g10 | 2 | 0.04 | 0.03 | 18.1 |
| matvec f32 | 1 | 0.02 | 0.02 | 19.2 |
| UNCLASSIFIED omega_reduce_r3_o2_n5_fused_equal_o0_o1__add_o2_o3__select_s0_s1_o4_minimum_positive_infinity | 1 | 0.01 | 0.01 | 14.9 |
| identity copy | 2 | 0.01 | 0.01 | 7.3 |

