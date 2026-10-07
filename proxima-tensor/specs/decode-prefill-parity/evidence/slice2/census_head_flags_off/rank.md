proxima timed dispatches=1568 sum(own cb)=2319.92 ms
llama ops per request=2444 sum(own cb)=568.86 ms floor_us=4

| class | proxima ops | proxima ms (own cb) | proxima ms (marginal) | llama ops | llama ms (own cb) | llama ms (net of floor) | gap ms (own cb) | ratio |
|---|---|---|---|---|---|---|---|---|
| matmul (weights) | 277 | 1587.01 | 1589.24 | 828 | 487.35 | 484.03 | +1099.66 | 3.26x |
| attention core | 346 | 691.59 | 690.00 | 105 | 41.31 | 40.89 | +650.28 | 16.74x |
| rms norm | 446 | 24.09 | 22.53 | 726 | 13.68 | 10.78 | +10.41 | 1.76x |
| rope | 128 | 9.36 | 9.05 | 150 | 3.37 | 2.77 | +6.00 | 2.78x |
| output head | 1 | 1.12 | 1.12 | 1 | 0.94 | 0.94 | +0.18 | 1.19x |
| elementwise, copy, other | 370 | 6.75 | 5.74 | 634 | 22.21 | 19.67 | -15.46 | 0.30x |

| matmul shape | proxima ops | proxima ms (own cb) | llama ops | llama ms (own cb) | gap ms | ratio |
|---|---|---|---|---|---|---|
| Q4_0 18874368 (1536x12288) | 60 | 902.06 | 180 | 284.91 | +617.16 | 3.17x |
| Q4_0 9437184 (1536x6144) | 45 | 332.71 | 135 | 110.18 | +222.53 | 3.02x |
| F16 13762560 (1536x8960) | 1 | 107.77 | 3 | 4.35 | +103.42 | 24.76x |
| Q4_0 3145728 (1536x2048) | 56 | 137.78 | 168 | 47.83 | +89.95 | 2.88x |
| Q4_0 6291456 (1536x4096) | 14 | 68.67 | 42 | 22.83 | +45.84 | 3.01x |
| Q4_0 393216 (1536x256) | 94 | 33.77 | 282 | 15.67 | +18.10 | 2.16x |
| Q4_0 786432 (1536x512) | 6 | 4.23 | 18 | 1.57 | +2.66 | 2.69x |
| F32 262144 | 1 | 0.01 | 0 | 0.00 | +0.01 | n/a |

| proxima census label | ops | ms (own cb) | ms (marginal) | us per op (own cb) |
|---|---|---|---|---|
| matvec Q4_0 | 275 | 1479.22 | 1481.47 | 5379.0 |
| cached attention partial | 7 | 337.02 | 337.49 | 48146.3 |
| attention dot | 112 | 178.91 | 178.21 | 1597.4 |
| attention AV | 56 | 144.89 | 143.94 | 2587.3 |
| matvec Float16 | 1 | 107.77 | 107.76 | 107769.4 |
| norm apply | 170 | 14.90 | 14.53 | 87.7 |
| softmax exp | 56 | 13.84 | 13.69 | 247.1 |
| RoPE | 128 | 9.36 | 9.05 | 73.1 |
| softmax max | 58 | 8.55 | 8.39 | 147.4 |
| softmax sum | 57 | 8.39 | 8.28 | 147.1 |
| RMSNorm sumsq + fused epilogue | 106 | 5.44 | 4.80 | 51.3 |
| elementwise fused_identity_o0__multiply_s0_o1_g10 | 2 | 4.08 | 4.07 | 2039.2 |

