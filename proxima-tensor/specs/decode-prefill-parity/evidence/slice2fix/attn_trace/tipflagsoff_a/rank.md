proxima timed dispatches=1568 sum(own cb)=2319.84 ms
llama ops per request=2444 sum(own cb)=568.86 ms floor_us=4

| class | proxima ops | proxima ms (own cb) | proxima ms (marginal) | llama ops | llama ms (own cb) | llama ms (net of floor) | gap ms (own cb) | ratio |
|---|---|---|---|---|---|---|---|---|
| matmul (weights) | 277 | 1586.75 | 1590.22 | 828 | 487.35 | 484.03 | +1099.41 | 3.26x |
| attention core | 346 | 691.02 | 690.40 | 105 | 41.31 | 40.89 | +649.71 | 16.73x |
| rms norm | 446 | 24.10 | 22.97 | 726 | 13.68 | 10.78 | +10.42 | 1.76x |
| rope | 128 | 10.00 | 9.76 | 150 | 3.37 | 2.77 | +6.64 | 2.97x |
| output head | 1 | 1.12 | 1.12 | 1 | 0.94 | 0.94 | +0.18 | 1.19x |
| elementwise, copy, other | 370 | 6.83 | 5.97 | 634 | 22.21 | 19.67 | -15.37 | 0.31x |

| matmul shape | proxima ops | proxima ms (own cb) | llama ops | llama ms (own cb) | gap ms | ratio |
|---|---|---|---|---|---|---|
| Q4_0 18874368 (1536x12288) | 60 | 902.20 | 180 | 284.91 | +617.29 | 3.17x |
| Q4_0 9437184 (1536x6144) | 45 | 333.63 | 135 | 110.18 | +223.45 | 3.03x |
| F16 13762560 (1536x8960) | 1 | 107.72 | 3 | 4.35 | +103.37 | 24.75x |
| Q4_0 3145728 (1536x2048) | 56 | 136.73 | 168 | 47.83 | +88.90 | 2.86x |
| Q4_0 6291456 (1536x4096) | 14 | 68.49 | 42 | 22.83 | +45.66 | 3.00x |
| Q4_0 393216 (1536x256) | 94 | 33.72 | 282 | 15.67 | +18.06 | 2.15x |
| Q4_0 786432 (1536x512) | 6 | 4.24 | 18 | 1.57 | +2.66 | 2.69x |
| F32 262144 | 1 | 0.01 | 0 | 0.00 | +0.01 | n/a |

| proxima census label | ops | ms (own cb) | ms (marginal) | us per op (own cb) |
|---|---|---|---|---|
| matvec Q4_0 | 275 | 1479.02 | 1482.49 | 5378.2 |
| cached attention partial | 7 | 337.58 | 338.19 | 48225.6 |
| attention dot | 112 | 178.66 | 178.21 | 1595.2 |
| attention AV | 56 | 144.04 | 143.47 | 2572.2 |
| matvec Float16 | 1 | 107.72 | 107.71 | 107722.9 |
| norm apply | 170 | 14.91 | 14.72 | 87.7 |
| softmax exp | 56 | 13.83 | 13.82 | 247.0 |
| RoPE | 128 | 10.00 | 9.76 | 78.1 |
| softmax max | 58 | 8.53 | 8.43 | 147.1 |
| softmax sum | 57 | 8.37 | 8.28 | 146.8 |
| RMSNorm sumsq + fused epilogue | 106 | 5.44 | 5.00 | 51.4 |
| elementwise fused_identity_o0__multiply_s0_o1_g10 | 2 | 4.08 | 4.07 | 2038.5 |

