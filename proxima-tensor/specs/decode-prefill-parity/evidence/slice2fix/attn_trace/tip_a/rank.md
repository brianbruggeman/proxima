proxima timed dispatches=1568 sum(own cb)=1325.96 ms
llama ops per request=2444 sum(own cb)=568.86 ms floor_us=4

| class | proxima ops | proxima ms (own cb) | proxima ms (marginal) | llama ops | llama ms (own cb) | llama ms (net of floor) | gap ms (own cb) | ratio |
|---|---|---|---|---|---|---|---|---|
| attention core | 346 | 686.55 | 685.62 | 105 | 41.31 | 40.89 | +645.24 | 16.62x |
| matmul (weights) | 277 | 597.90 | 597.06 | 828 | 487.35 | 484.03 | +110.55 | 1.23x |
| rms norm | 446 | 24.11 | 22.67 | 726 | 13.68 | 10.78 | +10.43 | 1.76x |
| rope | 128 | 9.44 | 9.10 | 150 | 3.37 | 2.77 | +6.07 | 2.80x |
| output head | 1 | 1.12 | 1.12 | 1 | 0.94 | 0.94 | +0.18 | 1.19x |
| elementwise, copy, other | 370 | 6.83 | 5.92 | 634 | 22.21 | 19.67 | -15.37 | 0.31x |

| matmul shape | proxima ops | proxima ms (own cb) | llama ops | llama ms (own cb) | gap ms | ratio |
|---|---|---|---|---|---|---|
| F16 13762560 (1536x8960) | 1 | 107.72 | 3 | 4.35 | +103.37 | 24.75x |
| Q4_0 18874368 (1536x12288) | 60 | 296.96 | 180 | 284.91 | +12.05 | 1.04x |
| Q4_0 9437184 (1536x6144) | 45 | 111.67 | 135 | 110.18 | +1.49 | 1.01x |
| F32 262144 | 1 | 0.01 | 0 | 0.00 | +0.01 | n/a |
| Q4_0 786432 (1536x512) | 6 | 1.38 | 18 | 1.57 | -0.20 | 0.87x |
| Q4_0 6291456 (1536x4096) | 14 | 22.40 | 42 | 22.83 | -0.43 | 0.98x |
| Q4_0 3145728 (1536x2048) | 56 | 45.62 | 168 | 47.83 | -2.21 | 0.95x |
| Q4_0 393216 (1536x256) | 94 | 12.14 | 282 | 15.67 | -3.53 | 0.77x |

| proxima census label | ops | ms (own cb) | ms (marginal) | us per op (own cb) |
|---|---|---|---|---|
| matvec Q4_0 | 275 | 490.17 | 489.35 | 1782.4 |
| cached attention partial | 7 | 337.84 | 337.75 | 48263.1 |
| attention dot | 112 | 176.63 | 176.42 | 1577.0 |
| attention AV | 56 | 141.29 | 140.92 | 2523.0 |
| matvec Float16 | 1 | 107.72 | 107.71 | 107721.2 |
| norm apply | 170 | 14.92 | 14.65 | 87.8 |
| softmax exp | 56 | 13.85 | 13.82 | 247.4 |
| RoPE | 128 | 9.44 | 9.10 | 73.7 |
| softmax max | 58 | 8.55 | 8.43 | 147.4 |
| softmax sum | 57 | 8.39 | 8.27 | 147.2 |
| RMSNorm sumsq + fused epilogue | 106 | 5.44 | 4.81 | 51.3 |
| elementwise fused_identity_o0__multiply_s0_o1_g10 | 2 | 4.08 | 4.07 | 2038.0 |

