proxima timed dispatches=653 sum(own cb)=11.32 ms
llama ops per request=818 sum(own cb)=11.10 ms floor_us=4

| class | proxima ops | proxima ms (own cb) | proxima ms (marginal) | llama ops | llama ms (own cb) | llama ms (net of floor) | gap ms (own cb) | ratio |
|---|---|---|---|---|---|---|---|---|
| rms norm | 242 | 2.92 | 1.31 | 242 | 1.80 | 0.83 | +1.12 | 1.62x |
| output head | 1 | 1.08 | 1.08 | 1 | 0.94 | 0.94 | +0.14 | 1.15x |
| attention core | 73 | 1.29 | 1.08 | 35 | 1.21 | 1.07 | +0.08 | 1.07x |
| matmul (weights) | 277 | 5.29 | 4.21 | 276 | 5.30 | 4.19 | -0.01 | 1.00x |
| rope | 0 | 0.00 | 0.00 | 50 | 0.32 | 0.12 | -0.32 | 0.00x |
| elementwise, copy, other | 60 | 0.74 | 0.63 | 214 | 1.52 | 0.67 | -0.79 | 0.48x |

| matmul shape | proxima ops | proxima ms (own cb) | llama ops | llama ms (own cb) | gap ms | ratio |
|---|---|---|---|---|---|---|
| Q4_0 393216 (1536x256) | 94 | 0.90 | 94 | 0.84 | +0.06 | 1.07x |
| F16 13762560 (1536x8960) | 1 | 0.12 | 1 | 0.08 | +0.03 | 1.39x |
| Q4_0 3145728 (1536x2048) | 56 | 0.72 | 56 | 0.71 | +0.02 | 1.02x |
| F32 262144 | 1 | 0.02 | 0 | 0.00 | +0.02 | n/a |
| Q4_0 6291456 (1536x4096) | 14 | 0.26 | 14 | 0.26 | +0.01 | 1.03x |
| Q4_0 786432 (1536x512) | 6 | 0.06 | 6 | 0.06 | -0.00 | 0.94x |
| Q4_0 9437184 (1536x6144) | 45 | 1.01 | 45 | 1.06 | -0.04 | 0.96x |
| Q4_0 18874368 (1536x12288) | 60 | 2.20 | 60 | 2.30 | -0.09 | 0.96x |

| proxima census label | ops | ms (own cb) | ms (marginal) | us per op (own cb) |
|---|---|---|---|---|
| matvec Q4_0 | 275 | 5.15 | 4.08 | 18.7 |
| RMSNorm sumsq + fused epilogue | 242 | 2.92 | 1.31 | 12.1 |
| head | 1 | 1.08 | 1.08 | 1084.3 |
| cached attention partial | 35 | 0.99 | 0.85 | 28.2 |
| UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__subtract_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__add_s0_s1 | 35 | 0.55 | 0.50 | 15.8 |
| cached attention merge | 35 | 0.28 | 0.21 | 8.1 |
| matvec Float16 | 1 | 0.12 | 0.11 | 116.9 |
| UNCLASSIFIED omega_elementwise_twin_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__add_s0_s1_fused_multiply_o0_o3__multiply_o2_o1__subtract_s0_s1 | 15 | 0.09 | 0.05 | 5.8 |
| elementwise fused_identity_o0__multiply_s0_o1_g10 | 2 | 0.04 | 0.04 | 20.5 |
| matvec f32 | 1 | 0.02 | 0.01 | 15.1 |
| identity copy | 2 | 0.01 | 0.01 | 7.3 |
| UNCLASSIFIED omega_reduce_r3_o2_n5_fused_equal_o0_o1__add_o2_o3__select_s0_s1_o4_minimum_positive_infinity | 1 | 0.01 | 0.01 | 14.6 |

