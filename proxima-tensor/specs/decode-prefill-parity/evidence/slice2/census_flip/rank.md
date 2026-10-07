proxima timed dispatches=1568 sum(own cb)=1591.05 ms
llama ops per request=2444 sum(own cb)=568.86 ms floor_us=4

| class | proxima ops | proxima ms (own cb) | proxima ms (marginal) | llama ops | llama ms (own cb) | llama ms (net of floor) | gap ms (own cb) | ratio |
|---|---|---|---|---|---|---|---|---|
| attention core | 346 | 688.25 | 686.63 | 105 | 41.31 | 40.89 | +646.94 | 16.66x |
| matmul (weights) | 277 | 861.46 | 863.82 | 828 | 487.35 | 484.03 | +374.11 | 1.77x |
| rms norm | 446 | 24.08 | 22.65 | 726 | 13.68 | 10.78 | +10.39 | 1.76x |
| rope | 128 | 9.36 | 9.08 | 150 | 3.37 | 2.77 | +6.00 | 2.78x |
| output head | 1 | 1.12 | 1.12 | 1 | 0.94 | 0.94 | +0.18 | 1.19x |
| elementwise, copy, other | 370 | 6.78 | 5.84 | 634 | 22.21 | 19.67 | -15.43 | 0.31x |

| matmul shape | proxima ops | proxima ms (own cb) | llama ops | llama ms (own cb) | gap ms | ratio |
|---|---|---|---|---|---|---|
| Q4_0 18874368 (1536x12288) | 60 | 451.35 | 180 | 284.91 | +166.44 | 1.58x |
| F16 13762560 (1536x8960) | 1 | 107.76 | 3 | 4.35 | +103.41 | 24.76x |
| Q4_0 9437184 (1536x6144) | 45 | 171.13 | 135 | 110.18 | +60.94 | 1.55x |
| Q4_0 3145728 (1536x2048) | 56 | 72.20 | 168 | 47.83 | +24.37 | 1.51x |
| Q4_0 6291456 (1536x4096) | 14 | 35.36 | 42 | 22.83 | +12.53 | 1.55x |
| Q4_0 393216 (1536x256) | 94 | 21.39 | 282 | 15.67 | +5.72 | 1.36x |
| Q4_0 786432 (1536x512) | 6 | 2.25 | 18 | 1.57 | +0.68 | 1.43x |
| F32 262144 | 1 | 0.01 | 0 | 0.00 | +0.01 | n/a |

| proxima census label | ops | ms (own cb) | ms (marginal) | us per op (own cb) |
|---|---|---|---|---|
| matvec Q4_0 | 275 | 753.68 | 756.05 | 2740.7 |
| cached attention partial | 7 | 337.67 | 337.79 | 48238.2 |
| attention dot | 112 | 176.55 | 175.96 | 1576.3 |
| attention AV | 56 | 143.28 | 142.51 | 2558.6 |
| matvec Float16 | 1 | 107.76 | 107.76 | 107764.5 |
| norm apply | 170 | 14.92 | 14.60 | 87.7 |
| softmax exp | 56 | 13.84 | 13.80 | 247.1 |
| RoPE | 128 | 9.36 | 9.08 | 73.1 |
| softmax max | 58 | 8.54 | 8.38 | 147.2 |
| softmax sum | 57 | 8.38 | 8.18 | 147.0 |
| RMSNorm sumsq + fused epilogue | 106 | 5.45 | 4.95 | 51.4 |
| elementwise fused_identity_o0__multiply_s0_o1_g10 | 2 | 4.08 | 4.07 | 2038.1 |

