proxima timed dispatches=1568 sum(own cb)=1325.43 ms
llama ops per request=2444 sum(own cb)=568.86 ms floor_us=4

| class | proxima ops | proxima ms (own cb) | proxima ms (marginal) | llama ops | llama ms (own cb) | llama ms (net of floor) | gap ms (own cb) | ratio |
|---|---|---|---|---|---|---|---|---|
| attention core | 346 | 686.30 | 686.22 | 105 | 41.31 | 40.89 | +644.98 | 16.61x |
| matmul (weights) | 277 | 597.73 | 597.26 | 828 | 487.35 | 484.03 | +110.38 | 1.23x |
| rms norm | 446 | 24.09 | 22.99 | 726 | 13.68 | 10.78 | +10.41 | 1.76x |
| rope | 128 | 9.37 | 9.20 | 150 | 3.37 | 2.77 | +6.01 | 2.78x |
| output head | 1 | 1.12 | 1.12 | 1 | 0.94 | 0.94 | +0.18 | 1.19x |
| elementwise, copy, other | 370 | 6.83 | 5.95 | 634 | 22.21 | 19.67 | -15.38 | 0.31x |

| matmul shape | proxima ops | proxima ms (own cb) | llama ops | llama ms (own cb) | gap ms | ratio |
|---|---|---|---|---|---|---|
| F16 13762560 (1536x8960) | 1 | 107.72 | 3 | 4.35 | +103.37 | 24.75x |
| Q4_0 18874368 (1536x12288) | 60 | 296.85 | 180 | 284.91 | +11.94 | 1.04x |
| Q4_0 9437184 (1536x6144) | 45 | 111.62 | 135 | 110.18 | +1.44 | 1.01x |
| F32 262144 | 1 | 0.01 | 0 | 0.00 | +0.01 | n/a |
| Q4_0 786432 (1536x512) | 6 | 1.38 | 18 | 1.57 | -0.20 | 0.88x |
| Q4_0 6291456 (1536x4096) | 14 | 22.40 | 42 | 22.83 | -0.43 | 0.98x |
| Q4_0 3145728 (1536x2048) | 56 | 45.60 | 168 | 47.83 | -2.23 | 0.95x |
| Q4_0 393216 (1536x256) | 94 | 12.14 | 282 | 15.67 | -3.53 | 0.77x |

| proxima census label | ops | ms (own cb) | ms (marginal) | us per op (own cb) |
|---|---|---|---|---|
| matvec Q4_0 | 275 | 489.99 | 489.54 | 1781.8 |
| cached attention partial | 7 | 337.65 | 337.87 | 48236.3 |
| attention dot | 112 | 176.91 | 176.51 | 1579.6 |
| attention AV | 56 | 140.95 | 141.30 | 2516.9 |
| matvec Float16 | 1 | 107.72 | 107.71 | 107723.4 |
| norm apply | 170 | 14.91 | 14.74 | 87.7 |
| softmax exp | 56 | 13.85 | 13.82 | 247.3 |
| RoPE | 128 | 9.37 | 9.20 | 73.2 |
| softmax max | 58 | 8.55 | 8.45 | 147.4 |
| softmax sum | 57 | 8.38 | 8.26 | 147.1 |
| RMSNorm sumsq + fused epilogue | 106 | 5.44 | 5.00 | 51.4 |
| elementwise fused_identity_o0__multiply_s0_o1_g10 | 2 | 4.08 | 4.07 | 2038.5 |

