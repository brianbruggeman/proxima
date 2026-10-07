proxima timed dispatches=1568 sum(own cb)=1338.43 ms
llama ops per request=2444 sum(own cb)=568.86 ms floor_us=4

| class | proxima ops | proxima ms (own cb) | proxima ms (marginal) | llama ops | llama ms (own cb) | llama ms (net of floor) | gap ms (own cb) | ratio |
|---|---|---|---|---|---|---|---|---|
| attention core | 346 | 686.46 | 686.37 | 105 | 41.31 | 40.89 | +645.15 | 16.62x |
| matmul (weights) | 277 | 610.75 | 610.00 | 828 | 487.35 | 484.03 | +123.40 | 1.25x |
| rms norm | 446 | 24.03 | 22.56 | 726 | 13.68 | 10.78 | +10.35 | 1.76x |
| rope | 128 | 9.34 | 9.11 | 150 | 3.37 | 2.77 | +5.98 | 2.78x |
| output head | 1 | 1.12 | 1.12 | 1 | 0.94 | 0.94 | +0.18 | 1.19x |
| elementwise, copy, other | 370 | 6.73 | 5.84 | 634 | 22.21 | 19.67 | -15.48 | 0.30x |

| matmul shape | proxima ops | proxima ms (own cb) | llama ops | llama ms (own cb) | gap ms | ratio |
|---|---|---|---|---|---|---|
| F16 13762560 (1536x8960) | 1 | 107.70 | 3 | 4.35 | +103.35 | 24.74x |
| Q4_0 18874368 (1536x12288) | 60 | 304.85 | 180 | 284.91 | +19.94 | 1.07x |
| Q4_0 9437184 (1536x6144) | 45 | 114.66 | 135 | 110.18 | +4.48 | 1.04x |
| Q4_0 6291456 (1536x4096) | 14 | 23.00 | 42 | 22.83 | +0.17 | 1.01x |
| F32 262144 | 1 | 0.01 | 0 | 0.00 | +0.01 | n/a |
| Q4_0 786432 (1536x512) | 6 | 1.41 | 18 | 1.57 | -0.17 | 0.90x |
| Q4_0 3145728 (1536x2048) | 56 | 46.81 | 168 | 47.83 | -1.03 | 0.98x |
| Q4_0 393216 (1536x256) | 94 | 12.30 | 282 | 15.67 | -3.36 | 0.79x |

| proxima census label | ops | ms (own cb) | ms (marginal) | us per op (own cb) |
|---|---|---|---|---|
| matvec Q4_0 | 275 | 503.03 | 502.30 | 1829.2 |
| cached attention partial | 7 | 337.57 | 338.42 | 48224.3 |
| attention dot | 112 | 176.49 | 176.05 | 1575.8 |
| attention AV | 56 | 141.68 | 141.50 | 2529.9 |
| matvec Float16 | 1 | 107.70 | 107.69 | 107702.3 |
| norm apply | 170 | 14.88 | 14.55 | 87.5 |
| softmax exp | 56 | 13.83 | 13.78 | 247.0 |
| RoPE | 128 | 9.34 | 9.11 | 73.0 |
| softmax max | 58 | 8.54 | 8.37 | 147.2 |
| softmax sum | 57 | 8.37 | 8.26 | 146.8 |
| RMSNorm sumsq + fused epilogue | 106 | 5.43 | 4.89 | 51.3 |
| elementwise fused_identity_o0__multiply_s0_o1_g10 | 2 | 4.08 | 4.07 | 2038.1 |

