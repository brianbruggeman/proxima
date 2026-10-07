proxima timed dispatches=1847 sum(own cb)=6299.90 ms
llama ops per request=1051 sum(own cb)=157.56 ms floor_us=4

| class | proxima ops | proxima ms (own cb) | proxima ms (marginal) | llama ops | llama ms (own cb) | llama ms (net of floor) | gap ms (own cb) | ratio |
|---|---|---|---|---|---|---|---|---|
| matmul (weights) | 698 | 6003.64 | 6001.44 | 381 | 131.89 | 130.37 | +5871.75 | 45.52x |
| attention core | 603 | 286.28 | 284.41 | 95 | 12.93 | 12.55 | +273.36 | 22.15x |
| rope | 96 | 4.07 | 3.71 | 96 | 1.71 | 1.32 | +2.36 | 2.38x |
| rms norm | 49 | 2.90 | 2.69 | 96 | 1.77 | 1.39 | +1.13 | 1.64x |
| elementwise, copy, other | 401 | 3.00 | 1.84 | 383 | 9.27 | 7.73 | -6.26 | 0.32x |

| matmul shape | proxima ops | proxima ms (own cb) | llama ops | llama ms (own cb) | gap ms | ratio |
|---|---|---|---|---|---|---|
| Q8_0 524288 (1024x512) | 624 | 5873.64 | 237 | 114.41 | +5759.23 | 51.34x |
| Q8_0 1048576 (1024x1024) | 48 | 118.21 | 96 | 15.08 | +103.13 | 7.84x |
| F32 32768 (1024x32) | 24 | 11.62 | 47 | 2.17 | +9.44 | 5.34x |
| F32 49155 | 1 | 0.01 | 0 | 0.00 | +0.01 | n/a |
| Q8_0 50334720 (1024x49155) | 1 | 0.16 | 1 | 0.22 | -0.06 | 0.74x |

| proxima census label | ops | ms (own cb) | ms (marginal) | us per op (own cb) |
|---|---|---|---|---|
| matvec Q8_0 | 673 | 5992.02 | 5989.94 | 8903.4 |
| cached attention partial | 24 | 282.17 | 282.10 | 11756.9 |
| matvec f32 | 25 | 11.63 | 11.50 | 465.0 |
| RoPE | 96 | 4.07 | 3.71 | 42.4 |
| softmax max | 386 | 3.19 | 1.77 | 8.3 |
| RMSNorm sumsq + fused epilogue | 49 | 2.90 | 2.69 | 59.2 |
| elementwise select | 168 | 1.06 | 0.59 | 6.3 |
| elementwise equal | 168 | 1.06 | 0.75 | 6.3 |
| softmax exp | 192 | 0.92 | 0.54 | 4.8 |
| elementwise fused_identity_o0__multiply_s0_o1_g10 | 1 | 0.62 | 0.39 | 615.3 |
| constant | 33 | 0.13 | 0.05 | 3.9 |
| iota | 26 | 0.11 | 0.05 | 4.1 |

