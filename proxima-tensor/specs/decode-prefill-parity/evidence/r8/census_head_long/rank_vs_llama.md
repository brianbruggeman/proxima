proxima timed dispatches=941 sum(own cb)=13.79 ms
llama ops per request=818 sum(own cb)=11.10 ms floor_us=4

| class | proxima ops | proxima ms (own cb) | proxima ms (marginal) | llama ops | llama ms (own cb) | llama ms (net of floor) | gap ms (own cb) | ratio |
|---|---|---|---|---|---|---|---|---|
| rms norm | 446 | 3.90 | 2.01 | 242 | 1.80 | 0.83 | +2.10 | 2.17x |
| matmul (weights) | 277 | 6.39 | 5.46 | 276 | 5.30 | 4.19 | +1.09 | 1.21x |
| rope | 100 | 0.62 | 0.40 | 50 | 0.32 | 0.12 | +0.30 | 1.93x |
| attention core | 73 | 1.48 | 1.38 | 35 | 1.21 | 1.07 | +0.27 | 1.23x |
| output head | 1 | 1.12 | 1.12 | 1 | 0.94 | 0.94 | +0.18 | 1.19x |
| elementwise, copy, other | 44 | 0.27 | 0.16 | 214 | 1.52 | 0.67 | -1.25 | 0.18x |

| matmul shape | proxima ops | proxima ms (own cb) | llama ops | llama ms (own cb) | gap ms | ratio |
|---|---|---|---|---|---|---|
| Q4_0 3145728 (1536x2048) | 56 | 1.31 | 56 | 0.71 | +0.60 | 1.85x |
| Q4_0 18874368 (1536x12288) | 60 | 2.52 | 60 | 2.30 | +0.22 | 1.10x |
| Q4_0 9437184 (1536x6144) | 45 | 1.19 | 45 | 1.06 | +0.14 | 1.13x |
| Q4_0 393216 (1536x256) | 94 | 0.91 | 94 | 0.84 | +0.07 | 1.08x |
| F16 13762560 (1536x8960) | 1 | 0.12 | 1 | 0.08 | +0.03 | 1.40x |
| Q4_0 6291456 (1536x4096) | 14 | 0.28 | 14 | 0.26 | +0.02 | 1.08x |
| F32 262144 | 1 | 0.01 | 0 | 0.00 | +0.01 | n/a |
| Q4_0 786432 (1536x512) | 6 | 0.06 | 6 | 0.06 | +0.00 | 1.00x |

| proxima census label | ops | ms (own cb) | ms (marginal) | us per op (own cb) |
|---|---|---|---|---|
| matvec Q4_0 | 275 | 6.26 | 5.33 | 22.8 |
| RMSNorm sumsq + fused epilogue | 106 | 1.45 | 0.74 | 13.7 |
| RMSNorm sumsq | 170 | 1.33 | 0.58 | 7.8 |
| norm apply | 170 | 1.12 | 0.69 | 6.6 |
| head | 1 | 1.12 | 1.12 | 1122.2 |
| cached attention partial | 35 | 1.02 | 1.00 | 29.1 |
| RoPE | 100 | 0.62 | 0.40 | 6.2 |
| cached attention merge | 35 | 0.44 | 0.37 | 12.7 |
| identity copy | 36 | 0.19 | 0.10 | 5.3 |
| matvec Float16 | 1 | 0.12 | 0.12 | 117.4 |
| elementwise fused_identity_o0__multiply_s0_o1_g10 | 2 | 0.04 | 0.03 | 18.2 |
| UNCLASSIFIED omega_reduce_r3_o2_n5_fused_equal_o0_o1__add_o2_o3__select_s0_s1_o4_minimum_positive_infinity | 1 | 0.02 | 0.01 | 15.5 |

