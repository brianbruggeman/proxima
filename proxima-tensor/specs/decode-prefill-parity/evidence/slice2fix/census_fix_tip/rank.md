proxima timed dispatches=1568 sum(own cb)=1325.28 ms
llama ops per request=2444 sum(own cb)=568.86 ms floor_us=4

| class | proxima ops | proxima ms (own cb) | proxima ms (marginal) | llama ops | llama ms (own cb) | llama ms (net of floor) | gap ms (own cb) | ratio |
|---|---|---|---|---|---|---|---|---|
| attention core | 346 | 686.30 | 685.23 | 105 | 41.31 | 40.89 | +644.98 | 16.61x |
| matmul (weights) | 277 | 597.74 | 596.89 | 828 | 487.35 | 484.03 | +110.39 | 1.23x |
| rms norm | 446 | 24.03 | 22.54 | 726 | 13.68 | 10.78 | +10.35 | 1.76x |
| rope | 128 | 9.34 | 9.03 | 150 | 3.37 | 2.77 | +5.98 | 2.78x |
| output head | 1 | 1.12 | 1.12 | 1 | 0.94 | 0.94 | +0.18 | 1.19x |
| elementwise, copy, other | 370 | 6.75 | 5.77 | 634 | 22.21 | 19.67 | -15.46 | 0.30x |

| matmul shape | proxima ops | proxima ms (own cb) | llama ops | llama ms (own cb) | gap ms | ratio |
|---|---|---|---|---|---|---|
| F16 13762560 (1536x8960) | 1 | 107.73 | 3 | 4.35 | +103.38 | 24.75x |
| Q4_0 18874368 (1536x12288) | 60 | 296.87 | 180 | 284.91 | +11.96 | 1.04x |
| Q4_0 9437184 (1536x6144) | 45 | 111.65 | 135 | 110.18 | +1.47 | 1.01x |
| F32 262144 | 1 | 0.01 | 0 | 0.00 | +0.01 | n/a |
| Q4_0 786432 (1536x512) | 6 | 1.38 | 18 | 1.57 | -0.20 | 0.87x |
| Q4_0 6291456 (1536x4096) | 14 | 22.39 | 42 | 22.83 | -0.43 | 0.98x |
| Q4_0 3145728 (1536x2048) | 56 | 45.60 | 168 | 47.83 | -2.24 | 0.95x |
| Q4_0 393216 (1536x256) | 94 | 12.10 | 282 | 15.67 | -3.56 | 0.77x |

| proxima census label | ops | ms (own cb) | ms (marginal) | us per op (own cb) |
|---|---|---|---|---|
| matvec Q4_0 | 275 | 489.99 | 489.17 | 1781.8 |
| cached attention partial | 7 | 337.71 | 337.95 | 48244.5 |
| attention dot | 112 | 177.09 | 176.30 | 1581.2 |
| attention AV | 56 | 140.75 | 140.73 | 2513.4 |
| matvec Float16 | 1 | 107.73 | 107.71 | 107732.4 |
| norm apply | 170 | 14.88 | 14.52 | 87.5 |
| softmax exp | 56 | 13.83 | 13.70 | 247.0 |
| RoPE | 128 | 9.34 | 9.03 | 73.0 |
| softmax max | 58 | 8.53 | 8.33 | 147.1 |
| softmax sum | 57 | 8.38 | 8.23 | 147.1 |
| RMSNorm sumsq + fused epilogue | 106 | 5.45 | 4.95 | 51.4 |
| elementwise fused_identity_o0__multiply_s0_o1_g10 | 2 | 4.08 | 4.07 | 2038.6 |

