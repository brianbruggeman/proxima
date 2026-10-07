proxima timed dispatches=1568 sum(own cb)=1348.90 ms
llama ops per request=2444 sum(own cb)=568.86 ms floor_us=4

| class | proxima ops | proxima ms (own cb) | proxima ms (marginal) | llama ops | llama ms (own cb) | llama ms (net of floor) | gap ms (own cb) | ratio |
|---|---|---|---|---|---|---|---|---|
| attention core | 346 | 689.23 | 687.32 | 105 | 41.31 | 40.89 | +647.92 | 16.68x |
| matmul (weights) | 277 | 618.43 | 617.67 | 828 | 487.35 | 484.03 | +131.08 | 1.27x |
| rms norm | 446 | 24.03 | 22.53 | 726 | 13.68 | 10.78 | +10.35 | 1.76x |
| rope | 128 | 9.34 | 9.07 | 150 | 3.37 | 2.77 | +5.98 | 2.78x |
| output head | 1 | 1.12 | 1.12 | 1 | 0.94 | 0.94 | +0.18 | 1.19x |
| elementwise, copy, other | 370 | 6.75 | 5.81 | 634 | 22.21 | 19.67 | -15.46 | 0.30x |

| matmul shape | proxima ops | proxima ms (own cb) | llama ops | llama ms (own cb) | gap ms | ratio |
|---|---|---|---|---|---|---|
| F16 13762560 (1536x8960) | 1 | 107.72 | 3 | 4.35 | +103.36 | 24.74x |
| Q4_0 18874368 (1536x12288) | 60 | 308.53 | 180 | 284.91 | +23.62 | 1.08x |
| Q4_0 9437184 (1536x6144) | 45 | 116.13 | 135 | 110.18 | +5.95 | 1.05x |
| Q4_0 6291456 (1536x4096) | 14 | 23.50 | 42 | 22.83 | +0.67 | 1.03x |
| Q4_0 3145728 (1536x2048) | 56 | 48.03 | 168 | 47.83 | +0.20 | 1.00x |
| F32 262144 | 1 | 0.01 | 0 | 0.00 | +0.01 | n/a |
| Q4_0 786432 (1536x512) | 6 | 1.45 | 18 | 1.57 | -0.13 | 0.92x |
| Q4_0 393216 (1536x256) | 94 | 13.06 | 282 | 15.67 | -2.60 | 0.83x |

| proxima census label | ops | ms (own cb) | ms (marginal) | us per op (own cb) |
|---|---|---|---|---|
| matvec Q4_0 | 275 | 510.70 | 509.96 | 1857.1 |
| cached attention partial | 7 | 337.75 | 337.80 | 48249.8 |
| attention dot | 112 | 176.85 | 175.87 | 1579.1 |
| attention AV | 56 | 143.88 | 143.31 | 2569.3 |
| matvec Float16 | 1 | 107.72 | 107.70 | 107717.7 |
| norm apply | 170 | 14.88 | 14.40 | 87.5 |
| softmax exp | 56 | 13.84 | 13.74 | 247.1 |
| RoPE | 128 | 9.34 | 9.07 | 73.0 |
| softmax max | 58 | 8.55 | 8.40 | 147.4 |
| softmax sum | 57 | 8.36 | 8.20 | 146.7 |
| RMSNorm sumsq + fused epilogue | 106 | 5.45 | 4.94 | 51.4 |
| elementwise fused_identity_o0__multiply_s0_o1_g10 | 2 | 4.08 | 4.07 | 2038.4 |

