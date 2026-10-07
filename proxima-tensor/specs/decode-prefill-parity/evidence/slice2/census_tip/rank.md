proxima timed dispatches=1568 sum(own cb)=1418.36 ms
llama ops per request=2444 sum(own cb)=568.86 ms floor_us=4

| class | proxima ops | proxima ms (own cb) | proxima ms (marginal) | llama ops | llama ms (own cb) | llama ms (net of floor) | gap ms (own cb) | ratio |
|---|---|---|---|---|---|---|---|---|
| attention core | 346 | 691.31 | 689.52 | 105 | 41.31 | 40.89 | +650.00 | 16.73x |
| matmul (weights) | 277 | 685.52 | 685.09 | 828 | 487.35 | 484.03 | +198.17 | 1.41x |
| rms norm | 446 | 24.17 | 22.93 | 726 | 13.68 | 10.78 | +10.49 | 1.77x |
| rope | 128 | 9.40 | 9.16 | 150 | 3.37 | 2.77 | +6.04 | 2.79x |
| output head | 1 | 1.12 | 1.12 | 1 | 0.94 | 0.94 | +0.18 | 1.19x |
| elementwise, copy, other | 370 | 6.84 | 5.95 | 634 | 22.21 | 19.67 | -15.37 | 0.31x |

| matmul shape | proxima ops | proxima ms (own cb) | llama ops | llama ms (own cb) | gap ms | ratio |
|---|---|---|---|---|---|---|
| F16 13762560 (1536x8960) | 1 | 107.73 | 3 | 4.35 | +103.38 | 24.75x |
| Q4_0 18874368 (1536x12288) | 60 | 348.66 | 180 | 284.91 | +63.75 | 1.22x |
| Q4_0 9437184 (1536x6144) | 45 | 131.31 | 135 | 110.18 | +21.13 | 1.19x |
| Q4_0 3145728 (1536x2048) | 56 | 54.50 | 168 | 47.83 | +6.67 | 1.14x |
| Q4_0 6291456 (1536x4096) | 14 | 26.72 | 42 | 22.83 | +3.89 | 1.17x |
| Q4_0 786432 (1536x512) | 6 | 1.66 | 18 | 1.57 | +0.08 | 1.05x |
| F32 262144 | 1 | 0.01 | 0 | 0.00 | +0.01 | n/a |
| Q4_0 393216 (1536x256) | 94 | 14.92 | 282 | 15.67 | -0.75 | 0.95x |

| proxima census label | ops | ms (own cb) | ms (marginal) | us per op (own cb) |
|---|---|---|---|---|
| matvec Q4_0 | 275 | 577.77 | 577.37 | 2101.0 |
| cached attention partial | 7 | 337.99 | 338.01 | 48284.8 |
| attention dot | 112 | 177.41 | 176.65 | 1584.0 |
| attention AV | 56 | 145.18 | 144.34 | 2592.4 |
| matvec Float16 | 1 | 107.73 | 107.72 | 107734.3 |
| norm apply | 170 | 14.97 | 14.71 | 88.1 |
| softmax exp | 56 | 13.83 | 13.81 | 247.0 |
| RoPE | 128 | 9.40 | 9.16 | 73.4 |
| softmax max | 58 | 8.54 | 8.44 | 147.2 |
| softmax sum | 57 | 8.36 | 8.27 | 146.7 |
| RMSNorm sumsq + fused epilogue | 106 | 5.46 | 4.98 | 51.5 |
| elementwise fused_identity_o0__multiply_s0_o1_g10 | 2 | 4.08 | 4.08 | 2038.6 |

