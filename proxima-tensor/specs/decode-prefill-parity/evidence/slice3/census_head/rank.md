proxima timed dispatches=1568 sum(own cb)=1339.71 ms
llama ops per request=2444 sum(own cb)=568.86 ms floor_us=4

| class | proxima ops | proxima ms (own cb) | proxima ms (marginal) | llama ops | llama ms (own cb) | llama ms (net of floor) | gap ms (own cb) | ratio |
|---|---|---|---|---|---|---|---|---|
| attention core | 346 | 700.44 | 697.25 | 105 | 41.31 | 40.89 | +659.12 | 16.95x |
| matmul (weights) | 277 | 597.91 | 597.39 | 828 | 487.35 | 484.03 | +110.56 | 1.23x |
| rms norm | 446 | 24.08 | 22.77 | 726 | 13.68 | 10.78 | +10.40 | 1.76x |
| rope | 128 | 9.36 | 9.25 | 150 | 3.37 | 2.77 | +6.00 | 2.78x |
| output head | 1 | 1.12 | 1.12 | 1 | 0.94 | 0.94 | +0.18 | 1.19x |
| elementwise, copy, other | 370 | 6.80 | 6.02 | 634 | 22.21 | 19.67 | -15.41 | 0.31x |

| matmul shape | proxima ops | proxima ms (own cb) | llama ops | llama ms (own cb) | gap ms | ratio |
|---|---|---|---|---|---|---|
| F16 13762560 (1536x8960) | 1 | 107.78 | 3 | 4.35 | +103.42 | 24.76x |
| Q4_0 18874368 (1536x12288) | 60 | 296.90 | 180 | 284.91 | +11.99 | 1.04x |
| Q4_0 9437184 (1536x6144) | 45 | 111.67 | 135 | 110.18 | +1.49 | 1.01x |
| F32 262144 | 1 | 0.01 | 0 | 0.00 | +0.01 | n/a |
| Q4_0 786432 (1536x512) | 6 | 1.38 | 18 | 1.57 | -0.20 | 0.87x |
| Q4_0 6291456 (1536x4096) | 14 | 22.40 | 42 | 22.83 | -0.43 | 0.98x |
| Q4_0 3145728 (1536x2048) | 56 | 45.63 | 168 | 47.83 | -2.20 | 0.95x |
| Q4_0 393216 (1536x256) | 94 | 12.14 | 282 | 15.67 | -3.52 | 0.78x |

| proxima census label | ops | ms (own cb) | ms (marginal) | us per op (own cb) |
|---|---|---|---|---|
| matvec Q4_0 | 275 | 490.12 | 489.58 | 1782.3 |
| cached attention partial | 7 | 346.13 | 343.85 | 49447.8 |
| attention dot | 112 | 179.05 | 178.37 | 1598.6 |
| attention AV | 56 | 144.49 | 144.50 | 2580.1 |
| matvec Float16 | 1 | 107.78 | 107.80 | 107775.7 |
| norm apply | 170 | 14.87 | 14.53 | 87.4 |
| softmax exp | 56 | 13.84 | 13.82 | 247.1 |
| RoPE | 128 | 9.36 | 9.25 | 73.1 |
| softmax max | 58 | 8.54 | 8.46 | 147.3 |
| softmax sum | 57 | 8.38 | 8.26 | 147.1 |
| RMSNorm sumsq + fused epilogue | 106 | 5.49 | 5.02 | 51.8 |
| elementwise fused_identity_o0__multiply_s0_o1_g10 | 2 | 4.08 | 4.08 | 2039.3 |

