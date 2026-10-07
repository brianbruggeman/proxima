proxima timed dispatches=1847 sum(own cb)=6297.81 ms
llama ops per request=1051 sum(own cb)=157.56 ms floor_us=4

| class | proxima ops | proxima ms (own cb) | proxima ms (marginal) | llama ops | llama ms (own cb) | llama ms (net of floor) | gap ms (own cb) | ratio |
|---|---|---|---|---|---|---|---|---|
| matmul (weights) | 698 | 6002.88 | 5999.83 | 381 | 131.89 | 130.37 | +5870.99 | 45.51x |
| attention core | 603 | 286.30 | 284.32 | 95 | 12.93 | 12.55 | +273.37 | 22.15x |
| rope | 96 | 3.12 | 2.81 | 96 | 1.71 | 1.32 | +1.41 | 1.83x |
| rms norm | 49 | 2.87 | 3.43 | 96 | 1.77 | 1.39 | +1.10 | 1.62x |
| elementwise, copy, other | 401 | 2.65 | 1.54 | 383 | 9.27 | 7.73 | -6.62 | 0.29x |

| matmul shape | proxima ops | proxima ms (own cb) | llama ops | llama ms (own cb) | gap ms | ratio |
|---|---|---|---|---|---|---|
| Q8_0 524288 (1024x512) | 624 | 5872.94 | 237 | 114.41 | +5758.53 | 51.33x |
| Q8_0 1048576 (1024x1024) | 48 | 118.18 | 96 | 15.08 | +103.10 | 7.84x |
| F32 32768 (1024x32) | 24 | 11.58 | 47 | 2.17 | +9.41 | 5.33x |
| F32 49155 | 1 | 0.01 | 0 | 0.00 | +0.01 | n/a |
| Q8_0 50334720 (1024x49155) | 1 | 0.16 | 1 | 0.22 | -0.06 | 0.73x |

| proxima census label | ops | ms (own cb) | ms (marginal) | us per op (own cb) |
|---|---|---|---|---|
| matvec Q8_0 | 673 | 5991.29 | 5988.79 | 8902.4 |
| cached attention partial | 24 | 282.32 | 282.19 | 11763.4 |
| matvec f32 | 25 | 11.59 | 11.04 | 463.7 |
| RoPE | 96 | 3.12 | 2.81 | 32.5 |
| softmax max | 386 | 3.09 | 1.72 | 8.0 |
| RMSNorm sumsq + fused epilogue | 49 | 2.87 | 3.43 | 58.6 |
| elementwise select | 168 | 1.03 | 0.56 | 6.1 |
| elementwise equal | 168 | 0.96 | 0.49 | 5.7 |
| softmax exp | 192 | 0.88 | 0.42 | 4.6 |
| elementwise fused_identity_o0__multiply_s0_o1_g10 | 1 | 0.37 | 0.37 | 373.4 |
| constant | 33 | 0.14 | 0.06 | 4.1 |
| iota | 26 | 0.11 | 0.05 | 4.2 |

