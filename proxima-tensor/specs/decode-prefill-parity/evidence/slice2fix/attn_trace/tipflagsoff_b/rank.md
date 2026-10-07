proxima timed dispatches=1568 sum(own cb)=2321.92 ms
llama ops per request=2444 sum(own cb)=568.86 ms floor_us=4

| class | proxima ops | proxima ms (own cb) | proxima ms (marginal) | llama ops | llama ms (own cb) | llama ms (net of floor) | gap ms (own cb) | ratio |
|---|---|---|---|---|---|---|---|---|
| matmul (weights) | 277 | 1589.88 | 1590.07 | 828 | 487.35 | 484.03 | +1102.54 | 3.26x |
| attention core | 346 | 690.73 | 691.13 | 105 | 41.31 | 40.89 | +649.42 | 16.72x |
| rms norm | 446 | 24.03 | 22.63 | 726 | 13.68 | 10.78 | +10.34 | 1.76x |
| rope | 128 | 9.35 | 9.20 | 150 | 3.37 | 2.77 | +5.98 | 2.78x |
| output head | 1 | 1.12 | 1.12 | 1 | 0.94 | 0.94 | +0.18 | 1.19x |
| elementwise, copy, other | 370 | 6.81 | 5.91 | 634 | 22.21 | 19.67 | -15.40 | 0.31x |

| matmul shape | proxima ops | proxima ms (own cb) | llama ops | llama ms (own cb) | gap ms | ratio |
|---|---|---|---|---|---|---|
| Q4_0 18874368 (1536x12288) | 60 | 904.02 | 180 | 284.91 | +619.12 | 3.17x |
| Q4_0 9437184 (1536x6144) | 45 | 334.40 | 135 | 110.18 | +224.22 | 3.04x |
| F16 13762560 (1536x8960) | 1 | 107.71 | 3 | 4.35 | +103.36 | 24.74x |
| Q4_0 3145728 (1536x2048) | 56 | 137.07 | 168 | 47.83 | +89.24 | 2.87x |
| Q4_0 6291456 (1536x4096) | 14 | 68.68 | 42 | 22.83 | +45.85 | 3.01x |
| Q4_0 393216 (1536x256) | 94 | 33.75 | 282 | 15.67 | +18.08 | 2.15x |
| Q4_0 786432 (1536x512) | 6 | 4.23 | 18 | 1.57 | +2.66 | 2.69x |
| F32 262144 | 1 | 0.01 | 0 | 0.00 | +0.01 | n/a |

| proxima census label | ops | ms (own cb) | ms (marginal) | us per op (own cb) |
|---|---|---|---|---|
| matvec Q4_0 | 275 | 1482.16 | 1482.35 | 5389.7 |
| cached attention partial | 7 | 337.40 | 337.92 | 48200.5 |
| attention dot | 112 | 178.96 | 178.63 | 1597.9 |
| attention AV | 56 | 143.62 | 144.02 | 2564.6 |
| matvec Float16 | 1 | 107.71 | 107.71 | 107712.7 |
| norm apply | 170 | 14.90 | 14.61 | 87.6 |
| softmax exp | 56 | 13.84 | 13.82 | 247.1 |
| RoPE | 128 | 9.35 | 9.20 | 73.0 |
| softmax max | 58 | 8.55 | 8.45 | 147.4 |
| softmax sum | 57 | 8.36 | 8.28 | 146.7 |
| RMSNorm sumsq + fused epilogue | 106 | 5.41 | 4.81 | 51.0 |
| elementwise fused_identity_o0__multiply_s0_o1_g10 | 2 | 4.08 | 4.08 | 2038.1 |

