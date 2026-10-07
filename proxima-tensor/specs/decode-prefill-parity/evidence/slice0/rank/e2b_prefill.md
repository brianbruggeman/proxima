proxima timed dispatches=1568 sum(own cb)=2381.38 ms
llama ops per request=2444 sum(own cb)=568.86 ms floor_us=4

| class | proxima ops | proxima ms (own cb) | proxima ms (marginal) | llama ops | llama ms (own cb) | llama ms (net of floor) | gap ms (own cb) | ratio |
|---|---|---|---|---|---|---|---|---|
| matmul (weights) | 277 | 1587.45 | 1591.28 | 828 | 487.35 | 484.03 | +1100.10 | 3.26x |
| attention core | 346 | 749.70 | 714.72 | 105 | 41.31 | 40.89 | +708.38 | 18.15x |
| rms norm | 446 | 26.58 | 24.37 | 726 | 13.68 | 10.78 | +12.90 | 1.94x |
| rope | 128 | 9.40 | 9.18 | 150 | 3.37 | 2.77 | +6.03 | 2.79x |
| output head | 1 | 1.12 | 1.12 | 1 | 0.94 | 0.94 | +0.18 | 1.19x |
| elementwise, copy, other | 370 | 7.13 | 6.16 | 634 | 22.21 | 19.67 | -15.08 | 0.32x |

| matmul shape | proxima ops | proxima ms (own cb) | llama ops | llama ms (own cb) | gap ms | ratio |
|---|---|---|---|---|---|---|
| Q4_0 18874368 (1536x12288) | 60 | 902.90 | 180 | 284.91 | +617.99 | 3.17x |
| Q4_0 9437184 (1536x6144) | 45 | 333.67 | 135 | 110.18 | +223.49 | 3.03x |
| F16 13762560 (1536x8960) | 1 | 107.77 | 3 | 4.35 | +103.42 | 24.76x |
| Q4_0 3145728 (1536x2048) | 56 | 136.75 | 168 | 47.83 | +88.91 | 2.86x |
| Q4_0 6291456 (1536x4096) | 14 | 68.34 | 42 | 22.83 | +45.51 | 2.99x |
| Q4_0 393216 (1536x256) | 94 | 33.78 | 282 | 15.67 | +18.12 | 2.16x |
| Q4_0 786432 (1536x512) | 6 | 4.22 | 18 | 1.57 | +2.65 | 2.68x |
| F32 262144 | 1 | 0.01 | 0 | 0.00 | +0.01 | n/a |

| proxima census label | ops | ms (own cb) | ms (marginal) | us per op (own cb) |
|---|---|---|---|---|
| matvec Q4_0 | 275 | 1479.66 | 1483.47 | 5380.6 |
| cached attention partial | 7 | 344.70 | 343.56 | 49243.4 |
| attention dot | 112 | 212.16 | 178.72 | 1894.3 |
| attention AV | 56 | 145.11 | 145.16 | 2591.3 |
| matvec Float16 | 1 | 107.77 | 107.80 | 107774.1 |
| softmax exp | 56 | 22.87 | 22.71 | 408.3 |
| softmax sum | 57 | 16.30 | 15.99 | 285.9 |
| norm apply | 170 | 16.27 | 15.34 | 95.7 |
| RoPE | 128 | 9.40 | 9.18 | 73.4 |
| softmax max | 58 | 8.56 | 8.58 | 147.6 |
| RMSNorm sumsq + fused epilogue | 106 | 6.57 | 5.81 | 62.0 |
| elementwise fused_identity_o0__multiply_s0_o1_g10 | 2 | 4.08 | 4.07 | 2038.6 |

