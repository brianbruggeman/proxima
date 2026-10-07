proxima timed dispatches=1568 sum(own cb)=2318.78 ms
llama ops per request=2444 sum(own cb)=568.86 ms floor_us=4

| class | proxima ops | proxima ms (own cb) | proxima ms (marginal) | llama ops | llama ms (own cb) | llama ms (net of floor) | gap ms (own cb) | ratio |
|---|---|---|---|---|---|---|---|---|
| matmul (weights) | 277 | 1586.36 | 1590.57 | 828 | 487.35 | 484.03 | +1099.01 | 3.26x |
| attention core | 346 | 691.07 | 690.61 | 105 | 41.31 | 40.89 | +649.76 | 16.73x |
| rms norm | 446 | 24.13 | 22.96 | 726 | 13.68 | 10.78 | +10.45 | 1.76x |
| rope | 128 | 9.36 | 9.13 | 150 | 3.37 | 2.77 | +5.99 | 2.78x |
| output head | 1 | 1.12 | 1.12 | 1 | 0.94 | 0.94 | +0.18 | 1.19x |
| elementwise, copy, other | 370 | 6.73 | 5.85 | 634 | 22.21 | 19.67 | -15.48 | 0.30x |

| matmul shape | proxima ops | proxima ms (own cb) | llama ops | llama ms (own cb) | gap ms | ratio |
|---|---|---|---|---|---|---|
| Q4_0 18874368 (1536x12288) | 60 | 901.63 | 180 | 284.91 | +616.72 | 3.16x |
| Q4_0 9437184 (1536x6144) | 45 | 334.22 | 135 | 110.18 | +224.04 | 3.03x |
| F16 13762560 (1536x8960) | 1 | 107.71 | 3 | 4.35 | +103.36 | 24.74x |
| Q4_0 3145728 (1536x2048) | 56 | 136.36 | 168 | 47.83 | +88.53 | 2.85x |
| Q4_0 6291456 (1536x4096) | 14 | 68.37 | 42 | 22.83 | +45.54 | 3.00x |
| Q4_0 393216 (1536x256) | 94 | 33.82 | 282 | 15.67 | +18.15 | 2.16x |
| Q4_0 786432 (1536x512) | 6 | 4.23 | 18 | 1.57 | +2.66 | 2.69x |
| F32 262144 | 1 | 0.01 | 0 | 0.00 | +0.01 | n/a |

| proxima census label | ops | ms (own cb) | ms (marginal) | us per op (own cb) |
|---|---|---|---|---|
| matvec Q4_0 | 275 | 1478.63 | 1482.86 | 5376.9 |
| cached attention partial | 7 | 337.33 | 337.75 | 48189.4 |
| attention dot | 112 | 178.77 | 178.48 | 1596.2 |
| attention AV | 56 | 144.21 | 143.87 | 2575.2 |
| matvec Float16 | 1 | 107.71 | 107.69 | 107712.8 |
| norm apply | 170 | 14.93 | 14.73 | 87.8 |
| softmax exp | 56 | 13.85 | 13.82 | 247.4 |
| RoPE | 128 | 9.36 | 9.13 | 73.1 |
| softmax max | 58 | 8.55 | 8.43 | 147.3 |
| softmax sum | 57 | 8.37 | 8.27 | 146.9 |
| RMSNorm sumsq + fused epilogue | 106 | 5.46 | 5.00 | 51.5 |
| elementwise fused_identity_o0__multiply_s0_o1_g10 | 2 | 4.08 | 4.07 | 2038.2 |

