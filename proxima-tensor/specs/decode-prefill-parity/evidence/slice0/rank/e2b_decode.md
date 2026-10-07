proxima timed dispatches=941 sum(own cb)=16.27 ms
llama ops per request=818 sum(own cb)=11.10 ms floor_us=4

| class | proxima ops | proxima ms (own cb) | proxima ms (marginal) | llama ops | llama ms (own cb) | llama ms (net of floor) | gap ms (own cb) | ratio |
|---|---|---|---|---|---|---|---|---|
| matmul (weights) | 277 | 8.71 | 7.58 | 276 | 5.30 | 4.19 | +3.41 | 1.64x |
| rms norm | 446 | 3.97 | 2.18 | 242 | 1.80 | 0.83 | +2.17 | 2.21x |
| attention core | 73 | 1.56 | 1.27 | 35 | 1.21 | 1.07 | +0.35 | 1.29x |
| rope | 100 | 0.61 | 0.40 | 50 | 0.32 | 0.12 | +0.29 | 1.89x |
| output head | 1 | 1.12 | 1.12 | 1 | 0.94 | 0.94 | +0.18 | 1.19x |
| elementwise, copy, other | 44 | 0.30 | 0.18 | 214 | 1.52 | 0.67 | -1.22 | 0.20x |

| matmul shape | proxima ops | proxima ms (own cb) | llama ops | llama ms (own cb) | gap ms | ratio |
|---|---|---|---|---|---|---|
| Q4_0 18874368 (1536x12288) | 60 | 4.02 | 60 | 2.30 | +1.73 | 1.75x |
| Q4_0 9437184 (1536x6144) | 45 | 1.83 | 45 | 1.06 | +0.78 | 1.73x |
| Q4_0 3145728 (1536x2048) | 56 | 1.32 | 56 | 0.71 | +0.61 | 1.87x |
| Q4_0 393216 (1536x256) | 94 | 1.02 | 94 | 0.84 | +0.19 | 1.22x |
| Q4_0 6291456 (1536x4096) | 14 | 0.30 | 14 | 0.26 | +0.05 | 1.19x |
| F16 13762560 (1536x8960) | 1 | 0.12 | 1 | 0.08 | +0.03 | 1.39x |
| Q4_0 786432 (1536x512) | 6 | 0.08 | 6 | 0.06 | +0.02 | 1.29x |
| F32 262144 | 1 | 0.01 | 0 | 0.00 | +0.01 | n/a |

| proxima census label | ops | ms (own cb) | ms (marginal) | us per op (own cb) |
|---|---|---|---|---|
| matvec Q4_0 | 275 | 8.58 | 7.46 | 31.2 |
| RMSNorm sumsq + fused epilogue | 106 | 1.61 | 0.85 | 15.2 |
| RMSNorm sumsq | 170 | 1.41 | 0.72 | 8.3 |
| head | 1 | 1.12 | 1.12 | 1122.5 |
| cached attention partial | 35 | 1.06 | 0.95 | 30.3 |
| norm apply | 170 | 0.95 | 0.62 | 5.6 |
| RoPE | 100 | 0.61 | 0.40 | 6.1 |
| cached attention merge | 35 | 0.48 | 0.31 | 13.6 |
| identity copy | 36 | 0.20 | 0.11 | 5.6 |
| matvec Float16 | 1 | 0.12 | 0.12 | 117.3 |
| elementwise fused_identity_o0__multiply_s0_o1_g10 | 2 | 0.06 | 0.05 | 27.7 |
| UNCLASSIFIED omega_reduce_r3_o2_n5_fused_equal_o0_o1__add_o2_o3__select_s0_s1_o4_minimum_positive_infinity | 1 | 0.02 | 0.01 | 15.8 |

