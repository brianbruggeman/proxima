proxima timed dispatches=1847 sum(own cb)=1024.71 ms
llama ops per request=1051 sum(own cb)=157.56 ms floor_us=4

| class | proxima ops | proxima ms (own cb) | proxima ms (marginal) | llama ops | llama ms (own cb) | llama ms (net of floor) | gap ms (own cb) | ratio |
|---|---|---|---|---|---|---|---|---|
| matmul (weights) | 698 | 729.24 | 798.84 | 381 | 131.89 | 130.37 | +597.35 | 5.53x |
| attention core | 603 | 286.73 | 284.87 | 95 | 12.93 | 12.55 | +273.80 | 22.18x |
| rope | 96 | 3.12 | 2.87 | 96 | 1.71 | 1.32 | +1.41 | 1.82x |
| rms norm | 49 | 2.87 | 2.58 | 96 | 1.77 | 1.39 | +1.10 | 1.62x |
| elementwise, copy, other | 401 | 2.75 | 1.66 | 383 | 9.27 | 7.73 | -6.52 | 0.30x |

| matmul shape | proxima ops | proxima ms (own cb) | llama ops | llama ms (own cb) | gap ms | ratio |
|---|---|---|---|---|---|---|
| Q8_0 524288 (1024x512) | 624 | 599.36 | 237 | 114.41 | +484.95 | 5.24x |
| Q8_0 1048576 (1024x1024) | 48 | 118.14 | 96 | 15.08 | +103.06 | 7.83x |
| F32 32768 (1024x32) | 24 | 11.57 | 47 | 2.17 | +9.39 | 5.32x |
| F32 49155 | 1 | 0.01 | 0 | 0.00 | +0.01 | n/a |
| Q8_0 50334720 (1024x49155) | 1 | 0.16 | 1 | 0.22 | -0.06 | 0.72x |

| proxima census label | ops | ms (own cb) | ms (marginal) | us per op (own cb) |
|---|---|---|---|---|
| matvec Q8_0 | 673 | 717.66 | 787.45 | 1066.4 |
| cached attention partial | 24 | 282.90 | 282.69 | 11787.4 |
| matvec f32 | 25 | 11.58 | 11.39 | 463.1 |
| RoPE | 96 | 3.12 | 2.87 | 32.5 |
| softmax max | 386 | 2.93 | 1.61 | 7.6 |
| RMSNorm sumsq + fused epilogue | 49 | 2.87 | 2.58 | 58.7 |
| elementwise select | 168 | 1.06 | 0.58 | 6.3 |
| elementwise equal | 168 | 1.04 | 0.58 | 6.2 |
| softmax exp | 192 | 0.89 | 0.56 | 4.6 |
| elementwise fused_identity_o0__multiply_s0_o1_g10 | 1 | 0.37 | 0.37 | 372.8 |
| constant | 33 | 0.13 | 0.06 | 4.1 |
| iota | 26 | 0.11 | 0.05 | 4.1 |

