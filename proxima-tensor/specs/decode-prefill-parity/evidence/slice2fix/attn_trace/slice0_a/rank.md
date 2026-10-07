proxima timed dispatches=1568 sum(own cb)=2324.08 ms
llama ops per request=2444 sum(own cb)=568.86 ms floor_us=4

| class | proxima ops | proxima ms (own cb) | proxima ms (marginal) | llama ops | llama ms (own cb) | llama ms (net of floor) | gap ms (own cb) | ratio |
|---|---|---|---|---|---|---|---|---|
| matmul (weights) | 277 | 1590.99 | 1590.25 | 828 | 487.35 | 484.03 | +1103.64 | 3.26x |
| attention core | 346 | 691.66 | 690.57 | 105 | 41.31 | 40.89 | +650.34 | 16.74x |
| rms norm | 446 | 24.14 | 22.96 | 726 | 13.68 | 10.78 | +10.45 | 1.76x |
| rope | 128 | 9.36 | 9.19 | 150 | 3.37 | 2.77 | +6.00 | 2.78x |
| output head | 1 | 1.12 | 1.12 | 1 | 0.94 | 0.94 | +0.18 | 1.19x |
| elementwise, copy, other | 370 | 6.82 | 5.92 | 634 | 22.21 | 19.67 | -15.39 | 0.31x |

| matmul shape | proxima ops | proxima ms (own cb) | llama ops | llama ms (own cb) | gap ms | ratio |
|---|---|---|---|---|---|---|
| Q4_0 18874368 (1536x12288) | 60 | 905.61 | 180 | 284.91 | +620.70 | 3.18x |
| Q4_0 9437184 (1536x6144) | 45 | 334.29 | 135 | 110.18 | +224.11 | 3.03x |
| F16 13762560 (1536x8960) | 1 | 107.73 | 3 | 4.35 | +103.38 | 24.75x |
| Q4_0 3145728 (1536x2048) | 56 | 136.90 | 168 | 47.83 | +89.07 | 2.86x |
| Q4_0 6291456 (1536x4096) | 14 | 68.42 | 42 | 22.83 | +45.59 | 3.00x |
| Q4_0 393216 (1536x256) | 94 | 33.79 | 282 | 15.67 | +18.12 | 2.16x |
| Q4_0 786432 (1536x512) | 6 | 4.23 | 18 | 1.57 | +2.66 | 2.69x |
| F32 262144 | 1 | 0.01 | 0 | 0.00 | +0.01 | n/a |

| proxima census label | ops | ms (own cb) | ms (marginal) | us per op (own cb) |
|---|---|---|---|---|
| matvec Q4_0 | 275 | 1483.25 | 1482.53 | 5393.6 |
| cached attention partial | 7 | 337.79 | 337.85 | 48255.3 |
| attention dot | 112 | 178.40 | 177.92 | 1592.9 |
| attention AV | 56 | 144.74 | 144.32 | 2584.6 |
| matvec Float16 | 1 | 107.73 | 107.71 | 107730.1 |
| norm apply | 170 | 14.93 | 14.72 | 87.8 |
| softmax exp | 56 | 13.83 | 13.82 | 246.9 |
| RoPE | 128 | 9.36 | 9.19 | 73.2 |
| softmax max | 58 | 8.54 | 8.44 | 147.3 |
| softmax sum | 57 | 8.36 | 8.21 | 146.6 |
| RMSNorm sumsq + fused epilogue | 106 | 5.45 | 4.99 | 51.5 |
| elementwise fused_identity_o0__multiply_s0_o1_g10 | 2 | 4.08 | 4.08 | 2038.5 |

